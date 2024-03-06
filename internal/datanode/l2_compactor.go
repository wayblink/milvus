// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datanode

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/allocator"
	"github.com/milvus-io/milvus/internal/datanode/io"
	iterators "github.com/milvus-io/milvus/internal/datanode/iterators"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type level2CompactionTask struct {
	compactor
	io        io.BinlogIO
	stageIO   io.BinlogIO
	allocator allocator.Allocator
	metaCache metacache.MetaCache
	syncMgr   syncmgr.SyncManager

	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}
	tr     *timerecord.TimeRecorder

	plan        *datapb.CompactionPlan
	partitionID int64

	bufferLock      sync.Mutex
	bufferDatas     map[interface{}]*InsertData
	bufferSizes     map[interface{}]int64
	totalBufferSize atomic.Int64
	spilledDatas    map[interface{}][]SpilledData
	spilledSizes    map[interface{}]int64
	spillCount      atomic.Int64

	// spill when in memory data is larger than memoryBufferSize
	memoryBufferSize int64
	// partitionKey larger than preferSegmentSizeMax should be split by clustering
	preferSegmentSizeMax int64
	// partitionKey smaller than preferSegmentSizeMin should be merged together
	preferSegmentSizeMin int64
}

// SpilledData store spilled data like a segment
// todo: maybe we can optimize it, using a more efficient storage format
type SpilledData struct {
	data map[UniqueID]*datapb.FieldBinlog
}

func newLevel2CompactionTask(
	ctx context.Context,
	binlogIO io.BinlogIO,
	stagingIO io.BinlogIO,
	alloc allocator.Allocator,
	metaCache metacache.MetaCache,
	syncMgr syncmgr.SyncManager,
	plan *datapb.CompactionPlan,
) *level2CompactionTask {
	ctx, cancel := context.WithCancel(ctx)
	return &level2CompactionTask{
		ctx:                  ctx,
		cancel:               cancel,
		io:                   binlogIO,
		stageIO:              stagingIO,
		allocator:            alloc,
		metaCache:            metaCache,
		syncMgr:              syncMgr,
		plan:                 plan,
		tr:                   timerecord.NewTimeRecorder("level2 compaction"),
		done:                 make(chan struct{}, 1),
		bufferDatas:          make(map[interface{}]*InsertData),
		bufferSizes:          make(map[interface{}]int64),
		spilledDatas:         make(map[interface{}][]SpilledData),
		spilledSizes:         make(map[interface{}]int64),
		totalBufferSize:      atomic.Int64{},
		spillCount:           atomic.Int64{},
		memoryBufferSize:     Params.DataNodeCfg.MajorCompactionMemoryBuffer.GetAsSize(),
		preferSegmentSizeMax: Params.DataNodeCfg.MajorCompactionPreferSegmentSizeMax.GetAsSize(),
		preferSegmentSizeMin: Params.DataNodeCfg.MajorCompactionPreferSegmentSizeMin.GetAsSize(),
	}
}

func (t *level2CompactionTask) complete() {
	t.done <- struct{}{}
}

func (t *level2CompactionTask) stop() {
	t.cancel()
	<-t.done
}

func (t *level2CompactionTask) getPlanID() UniqueID {
	return t.plan.GetPlanID()
}

func (t *level2CompactionTask) getChannelName() string {
	return t.plan.GetChannel()
}

func (t *level2CompactionTask) getCollection() int64 {
	return t.metaCache.Collection()
}

// injectDone unlock the segments
func (t *level2CompactionTask) injectDone() {
	for _, binlog := range t.plan.SegmentBinlogs {
		t.syncMgr.Unblock(binlog.SegmentID)
	}
}

func (t *level2CompactionTask) compact() (*datapb.CompactionPlanResult, error) {
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(t.ctx, fmt.Sprintf("L2Compact-%d", t.getPlanID()))
	defer span.End()
	log := log.With(zap.Int64("planID", t.plan.GetPlanID()), zap.String("type", t.plan.GetType().String()))
	if t.plan.GetType() != datapb.CompactionType_MajorCompaction {
		// this shouldn't be reached
		log.Warn("compact wrong, illegal compaction type")
		return nil, errIllegalCompactionPlan
	}
	log.Info("L2 compaction", zap.Duration("wait in queue elapse", t.tr.RecordSpan()))
	if !funcutil.CheckCtxValid(ctx) {
		log.Warn("compact wrong, task context done or timeout")
		return nil, errContext
	}

	ctxTimeout, cancelAll := context.WithTimeout(ctx, time.Duration(t.plan.GetTimeoutInSeconds())*time.Second)
	defer cancelAll()
	log.Info("compact start", zap.Int32("timeout in seconds", t.plan.GetTimeoutInSeconds()))

	err := binlog.DecompressCompactionBinlogs(t.plan.GetSegmentBinlogs())
	if err != nil {
		log.Warn("DecompressCompactionBinlogs fails", zap.Error(err))
		return nil, err
	}
	segIDs := make([]UniqueID, 0, len(t.plan.GetSegmentBinlogs()))
	for _, s := range t.plan.GetSegmentBinlogs() {
		segIDs = append(segIDs, s.GetSegmentID())
	}

	collectionID, partitionID, meta, err := t.getSegmentMeta(segIDs[0])
	if err != nil {
		log.Warn("compact wrong", zap.Error(err))
		return nil, err
	}
	t.partitionID = partitionID

	var pkField *schemapb.FieldSchema
	for _, fs := range meta.GetSchema().GetFields() {
		if fs.GetIsPrimaryKey() && fs.GetFieldID() >= 100 && typeutil.IsPrimaryFieldType(fs.GetDataType()) {
			pkField = fs
		}
	}
	if pkField == nil {
		log.Warn("failed to get pk field from schema")
		// todo wrap error
		return nil, fmt.Errorf("no pk field in schema")
	}

	// todo: check if it is necessary
	// Inject to stop flush
	injectStart := time.Now()
	for _, segID := range segIDs {
		t.syncMgr.Block(segID)
	}
	log.Info("compact inject elapse", zap.Duration("elapse", time.Since(injectStart)))
	defer func() {
		if err != nil {
			for _, segID := range segIDs {
				t.syncMgr.Unblock(segID)
			}
		}
	}()

	// 1, download delta logs to build deltaMap
	deltaPk2Ts, err := t.loadDeltaMap(ctxTimeout, t.plan.GetSegmentBinlogs())
	if err != nil {
		return nil, err
	}

	// 2, load PartitionStats
	// todo：based on existing PartitionStats, we can save some operations

	// 3, mapStep
	err = t.mapStep(ctxTimeout, meta, pkField, deltaPk2Ts, nil)
	if err != nil {
		return nil, err
	}

	// reduceStep
	uploadSegments, partitionStats, err := t.reduceStep(ctx, meta, pkField)
	if err != nil {
		return nil, err
	}

	// collect partition stats
	err = t.uploadPartitionStats(ctx, collectionID, partitionID, partitionStats)
	if err != nil {
		return nil, err
	}

	// clean up
	t.cleanUpStep(ctx)

	// assemble CompactionPlanResult
	planResult := &datapb.CompactionPlanResult{
		State:    commonpb.CompactionState_Completed,
		PlanID:   t.getPlanID(),
		Segments: uploadSegments,
		Type:     t.plan.GetType(),
		Channel:  t.plan.GetChannel(),
	}

	metrics.DataNodeCompactionLatency.
		WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), t.plan.GetType().String()).
		Observe(float64(t.tr.ElapseSpan().Milliseconds()))
	log.Info("L2 compaction finished",
		zap.Duration("elapse", t.tr.ElapseSpan()),
		zap.Int64("spillCount", t.spillCount.Load()))

	return planResult, nil
}

// mapStep read and map input segments iteratively, spill data if need
func (t *level2CompactionTask) mapStep(ctx context.Context, meta *etcdpb.CollectionMeta, pkField *schemapb.FieldSchema, deltaPk2Ts map[interface{}]Timestamp, partitionStats *storage.PartitionStatsSnapshot) error {
	flightSegments := t.plan.GetSegmentBinlogs()
	mapStart := time.Now()
	for _, segment := range flightSegments {
		err := t.split(ctx, meta, pkField, segment, deltaPk2Ts)
		if err != nil {
			return err
		}
	}
	// first spill all in memory data to disk, actually we don't need to do this, just for ease
	err := t.spillAll(ctx, meta)
	if err != nil {
		return err
	}
	log.Info("compact map step end",
		zap.Int("segments", len(flightSegments)),
		zap.Duration("map elapse", time.Since(mapStart)))
	return nil
}

func (t *level2CompactionTask) uploadPartitionStats(ctx context.Context, collectionID, partitionID UniqueID, partitionStats *storage.PartitionStatsSnapshot) error {
	partitionStatsBytes, err := storage.SerializePartitionStatsSnapshot(partitionStats)
	if err != nil {
		return err
	}
	// use allocID as partitionStats file name
	newVersion, err := t.allocator.AllocOne()
	if err != nil {
		return err
	}
	newStatsPath := t.io.JoinFullPath(common.PartitionStatsPath,
		metautil.JoinIDPath(collectionID, partitionID, newVersion))
	kv := map[string][]byte{
		newStatsPath: partitionStatsBytes,
	}
	err = t.io.Upload(ctx, kv)
	if err != nil {
		return err
	}
	log.Info("Finish upload PartitionStats file", zap.String("key", newStatsPath), zap.Int("length", len(partitionStatsBytes)))
	return nil
}

// cleanUpStep try best to clean all temp datas
func (t *level2CompactionTask) cleanUpStep(ctx context.Context) {
	stagePath := t.stageIO.JoinFullPath(common.CompactionStagePath, metautil.JoinIDPath(t.plan.PlanID))
	err := t.stageIO.Remove(ctx, stagePath)
	if err != nil {
		log.Warn("Fail to remove staging data", zap.String("key", stagePath), zap.Error(err))
	}
}

// reduceStep merge and subdivide the mapped data. Generate new segments and partitionStats
func (t *level2CompactionTask) reduceStep(ctx context.Context, meta *etcdpb.CollectionMeta, pkField *schemapb.FieldSchema) ([]*datapb.CompactionSegment, *storage.PartitionStatsSnapshot, error) {
	mergeResults := make([]*datapb.CompactionSegment, 0)
	partitionStats := storage.NewPartitionStatsSnapshot()
	toSplitAgainKeys := make([]interface{}, 0)
	toMergeKeys := make([]interface{}, 0)
	singleKeys := make([]interface{}, 0)
	for key, size := range t.spilledSizes {
		if size > t.preferSegmentSizeMax {
			toSplitAgainKeys = append(toSplitAgainKeys, key)
		} else if size < t.preferSegmentSizeMin {
			toMergeKeys = append(toMergeKeys, key)
		} else {
			singleKeys = append(singleKeys, key)
		}
	}

	pkID := pkField.GetFieldID()
	pkType := pkField.GetDataType()

	// merge data of multi partitionKey into one segment, collect the stats
	mergeFunc := func(keys []interface{}) (*datapb.CompactionSegment, *storage.SegmentStats, error) {
		writeBuffer, err := storage.NewInsertData(meta.GetSchema())
		if err != nil {
			return nil, nil, err
		}
		// todo: the constructor don't accept row_num=0, actually rowNum has no use in primaryKeyStats
		stats, err := storage.NewPrimaryKeyStats(pkID, int64(pkType), 1)
		if err != nil {
			return nil, nil, err
		}
		fieldStats, err := storage.NewFieldStats(pkID, pkType, 0)
		if err != nil {
			return nil, nil, err
		}
		var numRows int64 = 0
		for _, key := range keys {
			spilledDatas, exist := t.spilledDatas[key]
			if !exist {
				return nil, nil, errors.New("Can't find spill data")
			}
			for _, spilledData := range spilledDatas {
				paths := make([]string, 0)
				for _, fieldData := range spilledData.data {
					paths = append(paths, fieldData.GetBinlogs()[0].GetLogPath())
				}
				data, err := t.stageIO.Download(ctx, paths)
				if err != nil {
					log.Warn("download staging log wrong", zap.Strings("paths", paths), zap.Error(err))
					return nil, nil, err
				}

				iter, err := iterators.NewInsertBinlogIterator(data, pkID, pkType, nil)
				if err != nil {
					log.Warn("new insert binlogs Itr wrong", zap.Strings("paths", paths), zap.Error(err))
					return nil, nil, err
				}

				for iter.HasNext() {
					vInter, err := iter.Next()
					if err != nil {
						return nil, nil, err
					}
					err = writeBuffer.Append(vInter.GetData().(*iterators.InsertRow).GetValue())
					if err != nil {
						return nil, nil, err
					}
					numRows++
					stats.Update(vInter.GetPk())
					fieldStats.Update(storage.NewScalarFieldValue(vInter.GetPk()))
				}
			}
		}
		newSegmentID, err := t.allocator.AllocOne()
		if err != nil {
			return nil, nil, err
		}
		fieldBinlogs, statslogs, err := t.uploadSegment(ctx, meta, t.partitionID, newSegmentID, writeBuffer, stats, numRows)
		if err != nil {
			return nil, nil, err
		}
		return &datapb.CompactionSegment{
			PlanID:              t.plan.GetPlanID(),
			SegmentID:           newSegmentID,
			NumOfRows:           numRows,
			InsertLogs:          fieldBinlogs,
			Field2StatslogPaths: statslogs,
			Channel:             t.plan.GetChannel(),
		}, &storage.SegmentStats{[]storage.FieldStats{*fieldStats}}, nil
	}

	buckets := make([][]interface{}, 0)
	currentBucket := make([]interface{}, 0)
	var currentBucketSize int64 = 0
	for _, toMergeKey := range toMergeKeys {
		if currentBucketSize+t.spilledSizes[toMergeKey] > t.preferSegmentSizeMin {
			buckets = append(buckets, currentBucket)
			currentBucket = make([]interface{}, 0)
			currentBucketSize = 0
		} else {
			currentBucket = append(currentBucket, toMergeKey)
			currentBucketSize += t.spilledSizes[toMergeKey]
		}
	}
	// remain currentBucket
	buckets = append(buckets, currentBucket)

	for _, bucket := range buckets {
		newSegment, segmentStats, err := mergeFunc(bucket)
		if err != nil {
			return nil, nil, err
		}
		partitionStats.UpdateSegmentStats(newSegment.SegmentID, *segmentStats)
		mergeResults = append(mergeResults, newSegment)
	}

	for _, singleKey := range singleKeys {
		newSegment, segmentStats, err := mergeFunc([]interface{}{singleKey})
		if err != nil {
			return nil, nil, err
		}
		partitionStats.UpdateSegmentStats(newSegment.SegmentID, *segmentStats)
		mergeResults = append(mergeResults, newSegment)
	}

	// todo subdivided by vector clustering
	for _, singleKey := range toSplitAgainKeys {
		newSegment, segmentStats, err := mergeFunc([]interface{}{singleKey})
		if err != nil {
			return nil, nil, err
		}
		partitionStats.UpdateSegmentStats(newSegment.SegmentID, *segmentStats)
		mergeResults = append(mergeResults, newSegment)
	}

	log.Info("finish reduce step",
		zap.Int64("planID", t.getPlanID()),
		zap.Int("toSplitAgainKeys", len(toSplitAgainKeys)),
		zap.Int("toMergeKeys", len(toMergeKeys)),
		zap.Int("singleKeys", len(singleKeys)),
		zap.Int("mergeSegments", len(mergeResults)),
	)
	return mergeResults, partitionStats, nil
}

// read insert log of one segment, split it into buckets according to partitionKey. Spill data to file when necessary
func (t *level2CompactionTask) split(
	ctx context.Context,
	meta *etcdpb.CollectionMeta,
	pkField *schemapb.FieldSchema, // must not be nil
	segment *datapb.CompactionSegmentBinlogs,
	delta map[interface{}]Timestamp,
) error {
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, fmt.Sprintf("Compact-Map-%d", t.getPlanID()))
	defer span.End()
	log := log.With(zap.Int64("planID", t.getPlanID()))

	// vars
	processStart := time.Now()
	fieldBinlogPaths := make([][]string, 0)
	currentTs := tsoutil.GetCurrentTime()
	// initial timestampFrom, timestampTo = -1, -1 is an illegal value, only to mark initial state
	var (
		timestampTo   int64 = -1
		timestampFrom int64 = -1
		expired       int64 = 0
		deleted       int64 = 0
		remained      int64 = 0
	)

	isDeletedValue := func(v *storage.Value) bool {
		ts, ok := delta[v.PK.GetValue()]
		// insert task and delete task has the same ts when upsert
		// here should be < instead of <=
		// to avoid the upsert data to be deleted after compact
		if ok && uint64(v.Timestamp) < ts {
			return true
		}
		return false
	}

	// Get the number of field binlog files from non-empty segment
	var binlogNum int
	for _, b := range segment.GetFieldBinlogs() {
		if b != nil {
			binlogNum = len(b.GetBinlogs())
			break
		}
	}
	// Unable to deal with all empty segments cases, so return error
	if binlogNum == 0 {
		log.Warn("compact wrong, all segments' binlogs are empty")
		return errIllegalCompactionPlan
	}
	for idx := 0; idx < binlogNum; idx++ {
		var ps []string
		for _, f := range segment.GetFieldBinlogs() {
			ps = append(ps, f.GetBinlogs()[idx].GetLogPath())
		}
		fieldBinlogPaths = append(fieldBinlogPaths, ps)
	}

	pkID := pkField.GetFieldID()
	pkType := pkField.GetDataType()

	// todo: concurrent
	for _, path := range fieldBinlogPaths {
		bytesArr, err := t.io.Download(ctx, path)
		blobs := make([]*Blob, len(bytesArr))
		for i := range bytesArr {
			blobs[i] = &Blob{Value: bytesArr[i]}
		}
		if err != nil {
			log.Warn("download insertlogs wrong", zap.Strings("path", path), zap.Error(err))
			return err
		}

		pkIter, err := storage.NewInsertBinlogIterator(blobs, pkID, pkType)
		if err != nil {
			log.Warn("new insert binlogs Itr wrong", zap.Strings("path", path), zap.Error(err))
			return err
		}

		var rowSize int
		if pkIter.RowNum() != 0 {
			// calculate a average size of one row
			rowSize = pkIter.DataSize() / pkIter.RowNum()
		}
		for pkIter.HasNext() {
			vInter, _ := pkIter.Next()
			v, ok := vInter.(*storage.Value)
			if !ok {
				log.Warn("transfer interface to Value wrong", zap.Strings("path", path))
				return errors.New("unexpected error")
			}

			if isDeletedValue(v) {
				deleted++
				continue
			}

			// Filtering expired entity
			ts := Timestamp(v.Timestamp)
			if IsExpiredEntity(t.plan.GetCollectionTtl(), ts, currentTs) {
				expired++
				continue
			}

			// Update timestampFrom, timestampTo
			if v.Timestamp < timestampFrom || timestampFrom == -1 {
				timestampFrom = v.Timestamp
			}
			if v.Timestamp > timestampTo || timestampFrom == -1 {
				timestampTo = v.Timestamp
			}

			row, ok := v.Value.(map[UniqueID]interface{})
			if !ok {
				log.Warn("transfer interface to map wrong", zap.Strings("path", path))
				return errors.New("unexpected error")
			}

			log.Debug("size check", zap.Int("size", rowSize), zap.Int("reflectSize", int(reflect.TypeOf(row).Size())))
			err = t.addToBuffer(ctx, meta, row, rowSize)
			if err != nil {
				return err
			}
			remained++
		}
	}

	log.Info("compact map segment end",
		zap.Int64("remained entities", remained),
		zap.Int64("deleted entities", deleted),
		zap.Int64("expired entities", expired),
		zap.Duration("map elapse", time.Since(processStart)))
	return nil
}

func (t *level2CompactionTask) addToBuffer(ctx context.Context, meta *etcdpb.CollectionMeta, row map[int64]interface{}, rowSize int) error {
	t.bufferLock.Lock()
	defer t.bufferLock.Unlock()
	key := row[t.plan.PartitionKeyId]
	buffer, ok := t.bufferDatas[key]
	if !ok {
		writeBuffer, err := storage.NewInsertData(meta.GetSchema())
		if err != nil {
			return err
		}
		buffer = writeBuffer
		t.bufferDatas[key] = buffer
	}
	buffer.Append(row)
	_, exist := t.bufferSizes[key]
	if exist {
		t.bufferSizes[key] = t.bufferSizes[key] + int64(rowSize)
	} else {
		t.bufferSizes[key] = int64(rowSize)
	}
	t.totalBufferSize.Add(int64(rowSize))

	// trigger spill
	if t.totalBufferSize.Load() >= t.memoryBufferSize {
		err := t.spillTopKeys(ctx, meta)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *level2CompactionTask) spillTopKeys(ctx context.Context, meta *etcdpb.CollectionMeta) error {
	type KeySize struct {
		key  interface{}
		size int64
	}
	keysSize := lo.MapToSlice(t.bufferSizes, func(key interface{}, size int64) KeySize {
		return KeySize{key: key, size: size}
	})
	sort.Slice(keysSize, func(i, j int) bool {
		return keysSize[i].size > keysSize[j].size
	})
	var spillSize int64 = 0
	for _, value := range keysSize {
		segID, err := t.allocator.AllocOne()
		if err != nil {
			return err
		}
		err = t.spill(ctx, meta, segID, value.key, t.bufferDatas[value.key])
		if err != nil {
			return err
		}
		spillSize += value.size
		if spillSize >= t.memoryBufferSize/2 {
			break
		}
	}
	return nil
}

func (t *level2CompactionTask) spillAll(ctx context.Context, meta *etcdpb.CollectionMeta) error {
	for pkey, data := range t.bufferDatas {
		// todo: not use allocID to save some RPC
		segID, err := t.allocator.AllocOne()
		if err != nil {
			return err
		}
		err = t.spill(ctx, meta, segID, pkey, data)
		if err != nil {
			log.Error("spill fail", zap.Int64("segId", segID))
			return err
		}
		log.Debug("spill data", zap.Any("pkey", pkey), zap.Int64("segId", segID))
	}
	t.totalBufferSize.Store(0)
	return nil
}

func (t *level2CompactionTask) spill(ctx context.Context, meta *etcdpb.CollectionMeta, segmentID int64, pkey interface{}, iData *storage.InsertData) error {
	iCodec := storage.NewInsertCodecWithSchema(meta)
	fieldsBlobs, err := iCodec.Serialize(t.partitionID, segmentID, iData)
	if err != nil {
		return err
	}

	fieldLogs := make(map[UniqueID]*datapb.FieldBinlog)
	notifyGenIdx := make(chan struct{})
	defer close(notifyGenIdx)

	generator, err := t.allocator.GetGenerator(len(fieldsBlobs), notifyGenIdx)
	if err != nil {
		return err
	}

	spillKVs := make(map[string][]byte)
	for _, blob := range fieldsBlobs {
		// Blob Key is generated by Serialize from int64 fieldID in collection schema, which won't raise error in ParseInt
		fID, _ := strconv.ParseInt(blob.GetKey(), 10, 64)
		idPath := metautil.JoinIDPath(t.getPlanID(), segmentID, fID, <-generator)
		key := t.stageIO.JoinFullPath(common.CompactionStagePath, idPath)
		value := blob.GetValue()
		fileLen := len(value)
		spillKVs[key] = value
		fieldLogs[fID] = &datapb.FieldBinlog{
			FieldID: fID,
			Binlogs: []*datapb.Binlog{{LogSize: int64(fileLen), LogPath: key, EntriesNum: blob.RowNum}},
		}
	}

	err = t.stageIO.Upload(ctx, spillKVs)
	if err != nil {
		return err
	}

	if _, ok := t.spilledDatas[pkey]; ok {
		t.spilledDatas[pkey] = append(t.spilledDatas[pkey], SpilledData{data: fieldLogs})
		t.spilledSizes[pkey] = t.spilledSizes[pkey] + int64(iData.GetMemorySize())
	} else {
		t.spilledDatas[pkey] = []SpilledData{{data: fieldLogs}}
		t.spilledSizes[pkey] = int64(iData.GetMemorySize())
	}
	delete(t.bufferDatas, pkey)
	delete(t.bufferSizes, pkey)
	t.totalBufferSize.Add(-int64(iData.GetMemorySize()))
	t.spillCount.Add(1)
	return nil
}

func (t *level2CompactionTask) uploadSegment(ctx context.Context, meta *etcdpb.CollectionMeta, partitionID, segmentID int64, insertData *storage.InsertData, stats *storage.PrimaryKeyStats, totalRows int64) ([]*datapb.FieldBinlog, []*datapb.FieldBinlog, error) {
	iCodec := storage.NewInsertCodecWithSchema(meta)
	inlogs, err := iCodec.Serialize(t.partitionID, segmentID, insertData)
	if err != nil {
		return nil, nil, err
	}

	uploadFieldBinlogs := make([]*datapb.FieldBinlog, 0)
	notifyGenIdx := make(chan struct{})
	defer close(notifyGenIdx)

	generator, err := t.allocator.GetGenerator(len(inlogs)+1, notifyGenIdx)
	if err != nil {
		return nil, nil, err
	}

	uploadInsertKVs := make(map[string][]byte)
	for _, blob := range inlogs {
		// Blob Key is generated by Serialize from int64 fieldID in collection schema, which won't raise error in ParseInt
		fID, _ := strconv.ParseInt(blob.GetKey(), 10, 64)
		idPath := metautil.JoinIDPath(meta.GetID(), partitionID, segmentID, fID, <-generator)
		key := t.io.JoinFullPath(common.SegmentInsertLogPath, idPath)
		value := blob.GetValue()
		fileLen := len(value)
		uploadInsertKVs[key] = value
		uploadFieldBinlogs = append(uploadFieldBinlogs, &datapb.FieldBinlog{
			FieldID: fID,
			Binlogs: []*datapb.Binlog{{LogSize: int64(fileLen), LogPath: key, EntriesNum: blob.RowNum}},
		})
		log.Debug("upload segment insert log", zap.String("key", key))
	}

	err = t.io.Upload(ctx, uploadInsertKVs)
	if err != nil {
		return nil, nil, err
	}

	uploadStatsKVs := make(map[string][]byte)
	statBlob, err := iCodec.SerializePkStats(stats, totalRows)
	if err != nil {
		return nil, nil, err
	}
	fID, _ := strconv.ParseInt(statBlob.GetKey(), 10, 64)
	idPath := metautil.JoinIDPath(meta.GetID(), partitionID, segmentID, fID, <-generator)
	key := t.io.JoinFullPath(common.SegmentStatslogPath, idPath)
	value := statBlob.GetValue()
	uploadStatsKVs[key] = value
	fileLen := len(value)
	err = t.io.Upload(ctx, uploadStatsKVs)
	log.Debug("upload segment stats log", zap.String("key", key))
	if err != nil {
		return nil, nil, err
	}
	uploadStatslogs := []*datapb.FieldBinlog{
		{
			FieldID: fID,
			Binlogs: []*datapb.Binlog{{LogSize: int64(fileLen), LogPath: key, EntriesNum: totalRows}},
		},
	}

	return uploadFieldBinlogs, uploadStatslogs, nil
}

// todo: move it into metaCache as it is duplicate code, see also in compactionTask.getSegmentMeta
func (t *level2CompactionTask) getSegmentMeta(segID UniqueID) (UniqueID, UniqueID, *etcdpb.CollectionMeta, error) {
	collID := t.metaCache.Collection()
	seg, ok := t.metaCache.GetSegmentByID(segID)
	if !ok {
		return -1, -1, nil, merr.WrapErrSegmentNotFound(segID)
	}
	partID := seg.PartitionID()
	sch := t.metaCache.Schema()

	meta := &etcdpb.CollectionMeta{
		ID:     collID,
		Schema: sch,
	}
	return collID, partID, meta, nil
}

func (t *level2CompactionTask) loadDeltaMap(ctx context.Context, segments []*datapb.CompactionSegmentBinlogs) (map[interface{}]Timestamp, error) {
	downloadStart := time.Now()
	dblobs := make(map[UniqueID][]*Blob)
	for _, s := range segments {
		// Get the number of field binlog files from non-empty segment
		var binlogNum int
		for _, b := range s.GetFieldBinlogs() {
			if b != nil {
				binlogNum = len(b.GetBinlogs())
				break
			}
		}
		// Unable to deal with all empty segments cases, so return error
		if binlogNum == 0 {
			log.Warn("compact wrong, all segments' binlogs are empty")
			return nil, errIllegalCompactionPlan
		}

		segID := s.GetSegmentID()
		paths := make([]string, 0)
		for _, d := range s.GetDeltalogs() {
			for _, l := range d.GetBinlogs() {
				path := l.GetLogPath()
				paths = append(paths, path)
			}
		}

		if len(paths) != 0 {
			bytesArr, err := t.io.Download(ctx, paths)
			blobs := make([]*Blob, len(bytesArr))
			for i := range bytesArr {
				blobs[i] = &Blob{Value: bytesArr[i]}
			}
			if err != nil {
				log.Warn("compact download deltalogs wrong", zap.Int64("segment", segID), zap.Strings("path", paths), zap.Error(err))
				return nil, err
			}
			dblobs[segID] = append(dblobs[segID], blobs...)
		}
	}
	log.Info("compact download deltalogs elapse", zap.Duration("elapse", time.Since(downloadStart)))

	deltaPk2Ts, err := MergeDeltalogs(dblobs)
	if err != nil {
		return nil, err
	}

	return deltaPk2Ts, nil
}
