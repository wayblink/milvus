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
	"path"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/golang/protobuf/proto"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/allocator"
	"github.com/milvus-io/milvus/internal/datanode/io"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/hardware"
	"github.com/milvus-io/milvus/pkg/util/lock"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type majorCompactionTask struct {
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

	plan *datapb.CompactionPlan

	// schedule
	totalBufferSize atomic.Int64
	spillChan       chan SpillSignal
	spillCount      atomic.Int64
	pool            *conc.Pool[any]

	// inner field
	collectionID          int64
	partitionID           int64
	collectionMeta        *etcdpb.CollectionMeta
	currentTs             typeutil.Timestamp // for TTL
	isVectorClusteringKey bool
	clusteringKeyField    *schemapb.FieldSchema
	primaryKeyField       *schemapb.FieldSchema

	spillMutex         sync.Mutex
	memoryBufferSize   int64
	clusterBuffers     []*ClusterBuffer
	clusterBufferLocks *lock.KeyLock[interface{}]
	// scalar
	keyToBufferFunc func(interface{}) *ClusterBuffer
	// vector
	segmentIDOffsetMapping map[int64]string
	offsetToBufferFunc     func(int64, []uint32) *ClusterBuffer
}

type ClusterBuffer struct {
	id                  int
	bufferSize          int64
	buffer              *InsertData
	bufferRowNum        int64
	bufferTimeStampFrom int64
	bufferTimeStampTo   int64

	currentSegmentID      int64
	currentSpillSize      int64
	currentSpillRowNum    int64
	currentSpillBinlogs   map[UniqueID]*datapb.FieldBinlog
	currentSpillStatslogs []*datapb.FieldBinlog
	currentPKStats        *storage.PrimaryKeyStats

	uploadedSegments     []*datapb.CompactionSegment
	uploadedSegmentStats map[UniqueID]storage.SegmentStats

	clusteringKeyFieldStats *storage.FieldStats
}

type SpillSignal struct {
	buffer *ClusterBuffer
}

func newMajorCompactionTask(
	ctx context.Context,
	binlogIO io.BinlogIO,
	stagingIO io.BinlogIO,
	alloc allocator.Allocator,
	metaCache metacache.MetaCache,
	syncMgr syncmgr.SyncManager,
	plan *datapb.CompactionPlan,
) *majorCompactionTask {
	ctx, cancel := context.WithCancel(ctx)
	return &majorCompactionTask{
		ctx:                ctx,
		cancel:             cancel,
		io:                 binlogIO,
		stageIO:            stagingIO,
		allocator:          alloc,
		metaCache:          metaCache,
		syncMgr:            syncMgr,
		plan:               plan,
		tr:                 timerecord.NewTimeRecorder("major_compaction"),
		done:               make(chan struct{}, 1),
		totalBufferSize:    atomic.Int64{},
		spillChan:          make(chan SpillSignal, 100),
		pool:               conc.NewPool[any](hardware.GetCPUNum() * 2),
		clusterBuffers:     make([]*ClusterBuffer, 0),
		clusterBufferLocks: lock.NewKeyLock[interface{}](),
	}
}

func (t *majorCompactionTask) complete() {
	t.done <- struct{}{}
}

func (t *majorCompactionTask) stop() {
	t.cancel()
	<-t.done
}

func (t *majorCompactionTask) getPlanID() UniqueID {
	return t.plan.GetPlanID()
}

func (t *majorCompactionTask) getChannelName() string {
	return t.plan.GetChannel()
}

func (t *majorCompactionTask) getCollection() int64 {
	return t.metaCache.Collection()
}

// injectDone unlock the segments
func (t *majorCompactionTask) injectDone() {
	for _, binlog := range t.plan.SegmentBinlogs {
		t.syncMgr.Unblock(binlog.SegmentID)
	}
}

func (t *majorCompactionTask) init() error {
	segIDs := make([]UniqueID, 0, len(t.plan.GetSegmentBinlogs()))
	for _, s := range t.plan.GetSegmentBinlogs() {
		segIDs = append(segIDs, s.GetSegmentID())
	}
	collectionID, partitionID, meta, err := t.getSegmentMeta(segIDs[0])
	if err != nil {
		return err
	}
	t.collectionID = collectionID
	t.partitionID = partitionID
	t.collectionMeta = meta

	clusteringKeyID := t.plan.GetClusteringKeyId()
	var clusteringKeyField *schemapb.FieldSchema
	var pkField *schemapb.FieldSchema
	for _, field := range meta.Schema.Fields {
		if field.FieldID == clusteringKeyID {
			clusteringKeyField = field
		}
		if field.GetIsPrimaryKey() && field.GetFieldID() >= 100 && typeutil.IsPrimaryFieldType(field.GetDataType()) {
			pkField = field
		}
	}
	t.clusteringKeyField = clusteringKeyField
	t.primaryKeyField = pkField
	t.isVectorClusteringKey = typeutil.IsVectorType(t.clusteringKeyField.DataType)
	t.currentTs = tsoutil.GetCurrentTime()
	t.memoryBufferSize = t.getMemoryBufferSize()
	log.Info("major compaction memory buffer", zap.Int64("size", t.memoryBufferSize))

	return nil
}

func (t *majorCompactionTask) compact() (*datapb.CompactionPlanResult, error) {
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

	err := t.init()
	if err != nil {
		return nil, err
	}
	log.Info("compact start", zap.Int32("timeout in seconds", t.plan.GetTimeoutInSeconds()))

	err = binlog.DecompressCompactionBinlogs(t.plan.GetSegmentBinlogs())
	if err != nil {
		log.Warn("DecompressCompactionBinlogs fails", zap.Error(err))
		return nil, err
	}
	segIDs := make([]UniqueID, 0, len(t.plan.GetSegmentBinlogs()))
	for _, s := range t.plan.GetSegmentBinlogs() {
		segIDs = append(segIDs, s.GetSegmentID())
	}

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
	deltaBlobs, _, err := loadDeltaMap(ctxTimeout, t.io, t.plan.GetSegmentBinlogs())
	if err != nil {
		return nil, err
	}
	deltaPk2Ts, err := MergeDeltalogs(deltaBlobs)
	if err != nil {
		return nil, err
	}

	// 2, final clean up
	defer t.cleanUp(ctx)

	// todo move analyze to indexnode Analysis method
	if t.isVectorClusteringKey {
		analyzeResultPath := t.plan.AnalyzeResultPath
		centroidFilePath := t.io.JoinFullPath(common.AnalyzeStatsPath, analyzeResultPath, metautil.JoinIDPath(t.collectionID, t.partitionID, t.clusteringKeyField.FieldID), "centroids")
		centroidBytes, err := t.io.Download(ctx, []string{centroidFilePath})
		if err != nil {
			return nil, err
		}
		centroids := &segcorepb.ClusteringCentroidsStats{}
		err = proto.Unmarshal(centroidBytes[0], centroids)
		if err != nil {
			return nil, err
		}
		log.Debug("read clustering centroids stats", zap.String("path", centroidFilePath), zap.Int("centroidNum", len(centroids.GetCentroids())))
		offsetMappingFiles := make(map[int64]string, 0)
		for _, segmentID := range t.plan.AnalyzeSegmentIds {
			path := t.io.JoinFullPath(common.AnalyzeStatsPath, analyzeResultPath, metautil.JoinIDPath(t.collectionID, t.partitionID, t.clusteringKeyField.FieldID, segmentID), "offsets_mapping")
			offsetMappingFiles[segmentID] = path
			log.Debug("read segment offset mapping file", zap.Int64("segmentID", segmentID), zap.String("path", path))
		}
		t.segmentIDOffsetMapping = offsetMappingFiles

		for id, centroid := range centroids.GetCentroids() {
			fieldStats, err := storage.NewFieldStats(t.clusteringKeyField.FieldID, t.clusteringKeyField.DataType, 0)
			if err != nil {
				return nil, err
			}
			fieldStats.SetVectorCentroids(storage.NewVectorFieldValue(t.clusteringKeyField.DataType, centroid))
			clusterBuffer := &ClusterBuffer{
				id:                      id,
				uploadedSegments:        make([]*datapb.CompactionSegment, 0),
				uploadedSegmentStats:    make(map[UniqueID]storage.SegmentStats, 0),
				clusteringKeyFieldStats: fieldStats,
			}
			t.clusterBuffers = append(t.clusterBuffers, clusterBuffer)
		}
		t.offsetToBufferFunc = func(offset int64, idMapping []uint32) *ClusterBuffer {
			return t.clusterBuffers[idMapping[offset]]
		}
	} else {
		analyzeDict, err := t.scalarAnalyze(ctx)
		if err != nil {
			return nil, err
		}
		plan := t.scalarPlan(analyzeDict)
		scalarToClusterBufferMap := make(map[interface{}]*ClusterBuffer, 0)
		for id, bucket := range plan {
			fieldStats, err := storage.NewFieldStats(t.clusteringKeyField.FieldID, t.clusteringKeyField.DataType, 0)
			if err != nil {
				return nil, err
			}
			for _, key := range bucket {
				fieldStats.UpdateMinMax(storage.NewScalarFieldValue(t.clusteringKeyField.DataType, key))
			}
			buffer := &ClusterBuffer{
				id:                      id,
				uploadedSegments:        make([]*datapb.CompactionSegment, 0),
				uploadedSegmentStats:    make(map[UniqueID]storage.SegmentStats, 0),
				clusteringKeyFieldStats: fieldStats,
			}
			t.clusterBuffers = append(t.clusterBuffers, buffer)
			for _, key := range bucket {
				scalarToClusterBufferMap[key] = buffer
			}
		}
		t.keyToBufferFunc = func(key interface{}) *ClusterBuffer {
			// todo: if keys are too many, the map will be quite large, we should mark the range of each buffer and select buffer by range
			return scalarToClusterBufferMap[key]
		}
	}

	// 3, mapping
	log.Info("L2 compaction start mapping")
	uploadSegments, partitionStats, err := t.mapping(ctx, deltaPk2Ts)
	if err != nil {
		return nil, err
	}

	// 4, collect partition stats
	err = t.uploadPartitionStats(ctx, t.collectionID, t.partitionID, partitionStats)
	if err != nil {
		return nil, err
	}

	// 5, assemble CompactionPlanResult
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
	log.Info("L2 compaction finished", zap.Duration("elapse", t.tr.ElapseSpan()))

	return planResult, nil
}

// mapping read and split input segments into buffers
func (t *majorCompactionTask) mapping(ctx context.Context,
	deltaPk2Ts map[interface{}]Timestamp,
) ([]*datapb.CompactionSegment, *storage.PartitionStatsSnapshot, error) {
	inputSegments := t.plan.GetSegmentBinlogs()
	mapStart := time.Now()

	// start spill goroutine
	go t.backgroundSpill(ctx)

	futures := make([]*conc.Future[any], 0, len(inputSegments))
	for _, segment := range inputSegments {
		segmentClone := &datapb.CompactionSegmentBinlogs{
			SegmentID: segment.SegmentID,
			// only FieldBinlogs needed
			FieldBinlogs: segment.FieldBinlogs,
		}
		future := t.pool.Submit(func() (any, error) {
			err := t.mappingSegment(ctx, segmentClone, deltaPk2Ts)
			return struct{}{}, err
		})
		futures = append(futures, future)
	}
	if err := conc.AwaitAll(futures...); err != nil {
		return nil, nil, err
	}

	// force spill all buffers
	err := t.spillAll(ctx)
	if err != nil {
		return nil, nil, err
	}

	resultSegments := make([]*datapb.CompactionSegment, 0)
	resultPartitionStats := &storage.PartitionStatsSnapshot{
		SegmentStats: make(map[UniqueID]storage.SegmentStats, 0),
	}
	for _, buffer := range t.clusterBuffers {
		for _, seg := range buffer.uploadedSegments {
			se := &datapb.CompactionSegment{
				PlanID:              seg.GetPlanID(),
				SegmentID:           seg.GetSegmentID(),
				NumOfRows:           seg.GetNumOfRows(),
				InsertLogs:          seg.GetInsertLogs(),
				Field2StatslogPaths: seg.GetField2StatslogPaths(),
				Deltalogs:           seg.GetDeltalogs(),
				Channel:             seg.GetChannel(),
			}
			log.Debug("put segment into final compaction result", zap.String("segment", se.String()))
			resultSegments = append(resultSegments, se)
		}
		for segID, segmentStat := range buffer.uploadedSegmentStats {
			log.Debug("put segment into final partition stats", zap.Int64("segmentID", segID), zap.Any("stats", segmentStat))
			resultPartitionStats.SegmentStats[segID] = segmentStat
		}
	}

	log.Info("mapping end",
		zap.Int64("collectionID", t.getCollection()),
		zap.Int64("partitionID", t.partitionID),
		zap.Int("segmentFrom", len(inputSegments)),
		zap.Int("segmentTo", len(resultSegments)),
		zap.Duration("elapse", time.Since(mapStart)))

	return resultSegments, resultPartitionStats, nil
}

// read insert log of one segment, mappingSegment it into buckets according to partitionKey. Spill data to file when necessary
func (t *majorCompactionTask) mappingSegment(
	ctx context.Context,
	segment *datapb.CompactionSegmentBinlogs,
	delta map[interface{}]Timestamp,
) error {
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, fmt.Sprintf("Compact-Map-%d", t.getPlanID()))
	defer span.End()
	log := log.With(zap.Int64("planID", t.getPlanID()),
		zap.Int64("collectionID", t.getCollection()),
		zap.Int64("partitionID", t.partitionID),
		zap.Int64("segmentID", segment.GetSegmentID()))
	log.Info("mapping segment start")
	processStart := time.Now()
	fieldBinlogPaths := make([][]string, 0)
	var (
		expired  int64 = 0
		deleted  int64 = 0
		remained int64 = 0
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

	mappingStats := &segcorepb.ClusteringCentroidIdMappingStats{}
	if t.isVectorClusteringKey {
		offSetPath := t.segmentIDOffsetMapping[segment.SegmentID]
		offsetBytes, err := t.io.Download(ctx, []string{offSetPath})
		if err != nil {
			return err
		}
		err = proto.Unmarshal(offsetBytes[0], mappingStats)
		if err != nil {
			return err
		}
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

		pkIter, err := storage.NewInsertBinlogIterator(blobs, t.primaryKeyField.GetFieldID(), t.primaryKeyField.GetDataType())
		if err != nil {
			log.Warn("new insert binlogs Itr wrong", zap.Strings("path", path), zap.Error(err))
			return err
		}

		var offset int64 = -1
		for pkIter.HasNext() {
			vInter, _ := pkIter.Next()
			v, ok := vInter.(*storage.Value)
			if !ok {
				log.Warn("transfer interface to Value wrong", zap.Strings("path", path))
				return errors.New("unexpected error")
			}
			offset++

			// Filtering deleted entity
			if isDeletedValue(v) {
				deleted++
				continue
			}
			// Filtering expired entity
			ts := Timestamp(v.Timestamp)
			if IsExpiredEntity(t.plan.GetCollectionTtl(), ts, t.currentTs) {
				expired++
				continue
			}

			row, ok := v.Value.(map[UniqueID]interface{})
			if !ok {
				log.Warn("transfer interface to map wrong", zap.Strings("path", path))
				return errors.New("unexpected error")
			}

			clusteringKey := row[t.plan.GetClusteringKeyId()]
			var clusterBuffer *ClusterBuffer
			if t.isVectorClusteringKey {
				clusterBuffer = t.offsetToBufferFunc(offset, mappingStats.GetCentroidIdMapping())
			} else {
				clusterBuffer = t.keyToBufferFunc(clusteringKey)
			}
			err = t.writeToBuffer(ctx, clusterBuffer, v)
			if err != nil {
				return err
			}
			remained++

			// trigger spill
			if clusterBuffer.bufferRowNum > t.plan.GetMaxSegmentRows() ||
				clusterBuffer.bufferSize > int64(Params.DataNodeCfg.BinLogMaxSize.GetAsInt()) {
				t.spillChan <- SpillSignal{
					buffer: clusterBuffer,
				}
			} else if t.totalBufferSize.Load() >= t.memoryBufferSize {
				t.spillChan <- SpillSignal{}
			}
			// block here, wait for memory release by spill
			currentSize := t.totalBufferSize.Load()
			if currentSize > t.getSpillMemorySizeThreshold() {
			loop:
				for {
					select {
					case <-ctx.Done():
						log.Warn("stop waiting for memory buffer release as context done")
						return nil
					case <-t.done:
						log.Warn("stop waiting for memory buffer release as task chan done")
						return nil
					default:
						currentSize := t.totalBufferSize.Load()
						if currentSize < t.getSpillMemorySizeThreshold() {
							break loop
						}
						time.Sleep(time.Millisecond * 50)
					}
				}
			}
		}
	}

	log.Info("mapping segment end",
		zap.Int64("remained_entities", remained),
		zap.Int64("deleted_entities", deleted),
		zap.Int64("expired_entities", expired),
		zap.Duration("elapse", time.Since(processStart)))
	return nil
}

func (t *majorCompactionTask) writeToBuffer(ctx context.Context, clusterBuffer *ClusterBuffer, value *storage.Value) error {
	pk := value.PK
	timestamp := value.Timestamp
	row, ok := value.Value.(map[UniqueID]interface{})
	if !ok {
		log.Warn("transfer interface to map wrong")
		return errors.New("unexpected error")
	}
	rowSize := int(reflect.TypeOf(row).Size())

	t.clusterBufferLocks.Lock(clusterBuffer.id)
	defer t.clusterBufferLocks.Unlock(clusterBuffer.id)
	// prepare
	if clusterBuffer.currentSegmentID == 0 {
		segmentID, err := t.allocator.AllocOne()
		if err != nil {
			return err
		}
		clusterBuffer.currentSegmentID = segmentID
		clusterBuffer.currentSpillBinlogs = make(map[UniqueID]*datapb.FieldBinlog, 0)
		clusterBuffer.currentSpillStatslogs = make([]*datapb.FieldBinlog, 0)
		clusterBuffer.bufferTimeStampFrom = -1
		clusterBuffer.bufferTimeStampTo = -1
	}
	if clusterBuffer.currentPKStats == nil {
		stats, err := storage.NewPrimaryKeyStats(t.primaryKeyField.FieldID, int64(t.primaryKeyField.DataType), 1)
		if err != nil {
			return err
		}
		clusterBuffer.currentPKStats = stats
	}
	if clusterBuffer.buffer == nil {
		writeBuffer, err := storage.NewInsertData(t.collectionMeta.GetSchema())
		if err != nil {
			return err
		}
		clusterBuffer.buffer = writeBuffer
	}

	// Update timestampFrom, timestampTo
	if timestamp < clusterBuffer.bufferTimeStampFrom || clusterBuffer.bufferTimeStampFrom == -1 {
		clusterBuffer.bufferTimeStampFrom = timestamp
	}
	if timestamp > clusterBuffer.bufferTimeStampTo || clusterBuffer.bufferTimeStampFrom == -1 {
		clusterBuffer.bufferTimeStampTo = timestamp
	}
	clusterBuffer.buffer.Append(row)
	clusterBuffer.bufferSize = clusterBuffer.bufferSize + int64(rowSize)
	clusterBuffer.bufferRowNum = clusterBuffer.bufferRowNum + 1
	clusterBuffer.currentPKStats.Update(pk)
	t.totalBufferSize.Add(int64(rowSize))
	return nil
}

// getMemoryBufferSize return memoryBufferSize
func (t *majorCompactionTask) getMemoryBufferSize() int64 {
	return int64(float64(hardware.GetMemoryCount()) * Params.DataNodeCfg.MajorCompactionMemoryBufferRatio.GetAsFloat())
}

func (t *majorCompactionTask) getSpillMemorySizeThreshold() int64 {
	return int64(float64(t.memoryBufferSize) * 0.8)
}

func (t *majorCompactionTask) backgroundSpill(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			log.Info("major compaction task context exit")
			return
		case <-t.done:
			log.Info("major compaction task done")
			return
		case signal := <-t.spillChan:
			var err error
			if signal.buffer == nil {
				err = t.spillLargestBuffers(ctx)
			} else {
				err = func() error {
					t.clusterBufferLocks.Lock(signal.buffer.id)
					defer t.clusterBufferLocks.Unlock(signal.buffer.id)
					return t.spill(ctx, signal.buffer)
				}()
			}
			if err != nil {
				log.Warn("fail to spill data", zap.Error(err))
				// todo handle error
			}
		}
	}
}

func (t *majorCompactionTask) spillLargestBuffers(ctx context.Context) error {
	// only one spillLargestBuffers or spillAll should do at the same time
	t.spillMutex.Lock()
	defer t.spillMutex.Unlock()
	bufferIDs := make([]int, 0)
	for _, buffer := range t.clusterBuffers {
		bufferIDs = append(bufferIDs, buffer.id)
		t.clusterBufferLocks.Lock(buffer.id)
	}
	defer func() {
		for _, buffer := range t.clusterBuffers {
			t.clusterBufferLocks.Unlock(buffer.id)
		}
	}()
	sort.Slice(bufferIDs, func(i, j int) bool {
		return t.clusterBuffers[i].bufferSize > t.clusterBuffers[j].bufferSize
	})
	for index, id := range bufferIDs {
		err := t.spill(ctx, t.clusterBuffers[id])
		if err != nil {
			return err
		}
		if index >= len(bufferIDs) {
			break
		}
	}
	return nil
}

func (t *majorCompactionTask) spillAll(ctx context.Context) error {
	// only one spillLargestBuffers or spillAll should do at the same time
	t.spillMutex.Lock()
	defer t.spillMutex.Unlock()
	for _, buffer := range t.clusterBuffers {
		t.clusterBufferLocks.Lock(buffer.id)
	}
	defer func() {
		for _, buffer := range t.clusterBuffers {
			t.clusterBufferLocks.Unlock(buffer.id)
		}
	}()
	for _, buffer := range t.clusterBuffers {
		err := func() error {
			err := t.spill(ctx, buffer)
			if err != nil {
				log.Error("spill fail")
				return err
			}
			err = t.packBuffersToSegments(ctx, buffer)
			return err
		}()
		if err != nil {
			return err
		}
	}
	t.totalBufferSize.Store(0)
	return nil
}

func (t *majorCompactionTask) packBuffersToSegments(ctx context.Context, buffer *ClusterBuffer) error {
	if len(buffer.currentSpillBinlogs) == 0 {
		return nil
	}

	insertLogs := make([]*datapb.FieldBinlog, 0)
	for _, fieldBinlog := range buffer.currentSpillBinlogs {
		insertLogs = append(insertLogs, fieldBinlog)
	}
	iCodec := storage.NewInsertCodecWithSchema(t.collectionMeta)
	statPaths, err := uploadStatsLog(ctx, t.io, t.allocator, t.collectionID, t.partitionID, buffer.currentSegmentID, buffer.currentPKStats, buffer.currentSpillRowNum, iCodec)
	if err != nil {
		log.Error("fail to upload stats log", zap.Int("bufferID", buffer.id), zap.Any("buffer", buffer), zap.Error(err))
		return err
	}
	statsLogs := make([]*datapb.FieldBinlog, 0)
	for _, fieldBinlog := range statPaths {
		statsLogs = append(statsLogs, fieldBinlog)
	}

	// pack current spill data into a segment
	seg := &datapb.CompactionSegment{
		PlanID:              t.plan.GetPlanID(),
		SegmentID:           buffer.currentSegmentID,
		NumOfRows:           buffer.currentSpillRowNum,
		InsertLogs:          insertLogs,
		Field2StatslogPaths: statsLogs,
		Channel:             t.plan.GetChannel(),
	}
	buffer.uploadedSegments = append(buffer.uploadedSegments, seg)
	segmentStats := storage.SegmentStats{
		FieldStats: []storage.FieldStats{{
			FieldID:   buffer.clusteringKeyFieldStats.FieldID,
			Type:      buffer.clusteringKeyFieldStats.Type,
			Max:       buffer.clusteringKeyFieldStats.Max,
			Min:       buffer.clusteringKeyFieldStats.Min,
			BF:        buffer.clusteringKeyFieldStats.BF,
			Centroids: buffer.clusteringKeyFieldStats.Centroids,
		}},
		NumRows: int(buffer.currentSpillRowNum),
	}
	buffer.uploadedSegmentStats[buffer.currentSegmentID] = segmentStats
	log.Info("pack segment", zap.Int64("segID", buffer.currentSegmentID), zap.String("seg", seg.String()), zap.Any("segStats", segmentStats))

	// refresh
	buffer.currentSpillRowNum = 0
	buffer.currentSpillSize = 0
	buffer.currentPKStats = nil
	segmentID, err := t.allocator.AllocOne()
	if err != nil {
		return err
	}
	buffer.currentSegmentID = segmentID
	buffer.currentSpillBinlogs = make(map[UniqueID]*datapb.FieldBinlog, 0)
	buffer.currentSpillStatslogs = make([]*datapb.FieldBinlog, 0)
	return nil
}

func (t *majorCompactionTask) spill(ctx context.Context, buffer *ClusterBuffer) error {
	if buffer.currentSpillRowNum > t.plan.GetMaxSegmentRows() {
		t.packBuffersToSegments(ctx, buffer)
	}

	if buffer.buffer.IsEmpty() {
		return nil
	}

	iCodec := storage.NewInsertCodecWithSchema(t.collectionMeta)
	inPaths, err := uploadInsertLog(ctx, t.io, t.allocator, t.collectionID, t.partitionID, buffer.currentSegmentID, buffer.buffer, iCodec)
	if err != nil {
		return err
	}

	for fID, path := range inPaths {
		for _, binlog := range path.GetBinlogs() {
			binlog.TimestampTo = uint64(buffer.bufferTimeStampTo)
			binlog.TimestampFrom = uint64(buffer.bufferTimeStampFrom)
		}
		tmpBinlog, ok := buffer.currentSpillBinlogs[fID]
		if !ok {
			tmpBinlog = path
		} else {
			tmpBinlog.Binlogs = append(tmpBinlog.Binlogs, path.GetBinlogs()...)
		}
		buffer.currentSpillBinlogs[fID] = tmpBinlog
	}
	buffer.currentSpillRowNum = buffer.currentSpillRowNum + buffer.bufferRowNum

	// clean buffer
	buffer.buffer = nil
	buffer.bufferSize = 0
	buffer.bufferRowNum = 0
	t.totalBufferSize.Add(-buffer.bufferSize)

	return nil
}

func (t *majorCompactionTask) getSegmentMeta(segID UniqueID) (UniqueID, UniqueID, *etcdpb.CollectionMeta, error) {
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

func (t *majorCompactionTask) uploadPartitionStats(ctx context.Context, collectionID, partitionID UniqueID, partitionStats *storage.PartitionStatsSnapshot) error {
	// use allocID as partitionStats file name
	newVersion, err := t.allocator.AllocOne()
	if err != nil {
		return err
	}
	partitionStats.Version = newVersion
	partitionStatsBytes, err := storage.SerializePartitionStatsSnapshot(partitionStats)
	if err != nil {
		return err
	}
	newStatsPath := t.io.JoinFullPath(common.PartitionStatsPath,
		path.Join(strconv.FormatInt(collectionID, 10), strconv.FormatInt(partitionID, 10), t.plan.GetChannel(), strconv.FormatInt(newVersion, 10)))
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

// cleanUp try best to clean all temp datas
func (t *majorCompactionTask) cleanUp(ctx context.Context) {
	//stagePath := t.stageIO.JoinFullPath(common.CompactionStagePath, metautil.JoinIDPath(t.plan.PlanID))
	//err := t.stageIO.Remove(ctx, stagePath)
	//if err != nil {
	//	log.Warn("Fail to remove staging data", zap.String("key", stagePath), zap.Error(err))
	//}
}

func (t *majorCompactionTask) scalarAnalyze(ctx context.Context) (map[interface{}]int64, error) {
	inputSegments := t.plan.GetSegmentBinlogs()
	futures := make([]*conc.Future[any], 0, len(inputSegments))
	mapStart := time.Now()
	var mutex sync.Mutex
	analyzeDict := make(map[interface{}]int64, 0)
	for _, segment := range inputSegments {
		segmentClone := &datapb.CompactionSegmentBinlogs{
			SegmentID:           segment.SegmentID,
			FieldBinlogs:        segment.FieldBinlogs,
			Field2StatslogPaths: segment.Field2StatslogPaths,
			Deltalogs:           segment.Deltalogs,
			InsertChannel:       segment.InsertChannel,
			Level:               segment.Level,
			CollectionID:        segment.CollectionID,
			PartitionID:         segment.PartitionID,
		}
		future := t.pool.Submit(func() (any, error) {
			analyzeResult, err := t.scalarAnalyzeSegment(ctx, segmentClone)
			mutex.Lock()
			defer mutex.Unlock()
			for key, v := range analyzeResult {
				if _, exist := analyzeDict[key]; exist {
					analyzeDict[key] = analyzeDict[key] + v
				} else {
					analyzeDict[key] = v
				}
			}
			return struct{}{}, err
		})
		futures = append(futures, future)
	}
	if err := conc.AwaitAll(futures...); err != nil {
		return nil, err
	}
	log.Info("analyze end",
		zap.Int64("collectionID", t.getCollection()),
		zap.Int64("partitionID", t.partitionID),
		zap.Int("segments", len(inputSegments)),
		zap.Duration("elapse", time.Since(mapStart)))
	return analyzeDict, nil
}

func (t *majorCompactionTask) scalarAnalyzeSegment(
	ctx context.Context,
	segment *datapb.CompactionSegmentBinlogs,
) (map[interface{}]int64, error) {
	log := log.With(zap.Int64("planID", t.getPlanID()), zap.Int64("segmentID", segment.GetSegmentID()))

	// vars
	processStart := time.Now()
	fieldBinlogPaths := make([][]string, 0)
	// initial timestampFrom, timestampTo = -1, -1 is an illegal value, only to mark initial state
	var (
		timestampTo   int64                 = -1
		timestampFrom int64                 = -1
		expired       int64                 = 0
		deleted       int64                 = 0
		remained      int64                 = 0
		analyzeResult map[interface{}]int64 = make(map[interface{}]int64, 0)
	)

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
		return nil, errIllegalCompactionPlan
	}
	log.Debug("binlogNum", zap.Int("binlogNum", binlogNum))
	for idx := 0; idx < binlogNum; idx++ {
		var ps []string
		for _, f := range segment.GetFieldBinlogs() {
			// todo add a new reader only read one column
			if f.FieldID == t.primaryKeyField.GetFieldID() || f.FieldID == t.clusteringKeyField.GetFieldID() || f.FieldID == common.RowIDField || f.FieldID == common.TimeStampField {
				ps = append(ps, f.GetBinlogs()[idx].GetLogPath())
			}
		}
		fieldBinlogPaths = append(fieldBinlogPaths, ps)
	}

	for _, path := range fieldBinlogPaths {
		bytesArr, err := t.io.Download(ctx, path)
		blobs := make([]*Blob, len(bytesArr))
		for i := range bytesArr {
			blobs[i] = &Blob{Value: bytesArr[i]}
		}
		if err != nil {
			log.Warn("download insertlogs wrong", zap.Strings("path", path), zap.Error(err))
			return nil, err
		}

		pkIter, err := storage.NewInsertBinlogIterator(blobs, t.primaryKeyField.GetFieldID(), t.primaryKeyField.GetDataType())
		if err != nil {
			log.Warn("new insert binlogs Itr wrong", zap.Strings("path", path), zap.Error(err))
			return nil, err
		}

		// log.Info("pkIter.RowNum()", zap.Int("pkIter.RowNum()", pkIter.RowNum()), zap.Bool("hasNext", pkIter.HasNext()))
		for pkIter.HasNext() {
			vIter, _ := pkIter.Next()
			v, ok := vIter.(*storage.Value)
			if !ok {
				log.Warn("transfer interface to Value wrong", zap.Strings("path", path))
				return nil, errors.New("unexpected error")
			}

			// Filtering expired entity
			ts := Timestamp(v.Timestamp)
			if IsExpiredEntity(t.plan.GetCollectionTtl(), ts, t.currentTs) {
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
			// rowValue := vIter.GetData().(*iterators.InsertRow).GetValue()
			row, ok := v.Value.(map[UniqueID]interface{})
			if !ok {
				log.Warn("transfer interface to map wrong", zap.Strings("path", path))
				return nil, errors.New("unexpected error")
			}
			key := row[t.clusteringKeyField.GetFieldID()]
			if _, exist := analyzeResult[key]; exist {
				analyzeResult[key] = analyzeResult[key] + 1
			} else {
				analyzeResult[key] = 1
			}
			remained++
		}
	}

	log.Info("analyze segment end",
		zap.Int64("remained entities", remained),
		zap.Int64("deleted entities", deleted),
		zap.Int64("expired entities", expired),
		zap.Duration("map elapse", time.Since(processStart)))
	return analyzeResult, nil
}

func (t *majorCompactionTask) scalarPlan(dict map[interface{}]int64) [][]interface{} {
	keys := lo.MapToSlice(dict, func(k interface{}, _ int64) interface{} {
		return k
	})
	sort.Slice(keys, func(i, j int) bool {
		return storage.NewScalarFieldValue(t.clusteringKeyField.DataType, keys[i]).LE(storage.NewScalarFieldValue(t.clusteringKeyField.DataType, keys[j]))
	})

	buckets := make([][]interface{}, 0)
	currentBucket := make([]interface{}, 0)
	var currentBucketSize int64 = 0
	maxRows := t.plan.MaxSegmentRows
	preferRows := t.plan.PreferSegmentRows
	for _, key := range keys {
		// todo can optimize
		if dict[key] > preferRows {
			buckets = append(buckets, currentBucket)
			buckets = append(buckets, []interface{}{key})
			currentBucket = make([]interface{}, 0)
			currentBucketSize = 0
		} else if currentBucketSize+dict[key] > maxRows {
			buckets = append(buckets, currentBucket)
			currentBucket = []interface{}{key}
			currentBucketSize = dict[key]
		} else if currentBucketSize+dict[key] > preferRows {
			currentBucket = append(currentBucket, key)
			buckets = append(buckets, currentBucket)
			currentBucket = make([]interface{}, 0)
			currentBucketSize = 0
		} else {
			currentBucket = append(currentBucket, key)
			currentBucketSize += dict[key]
		}
	}
	buckets = append(buckets, currentBucket)
	return buckets
}
