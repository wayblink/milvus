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
	"math"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/datanode/allocator"
	"github.com/milvus-io/milvus/internal/datanode/compaction"
	"github.com/milvus-io/milvus/internal/datanode/io"
	iter "github.com/milvus-io/milvus/internal/datanode/iterators"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/hardware"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type levelZeroCompactionTask struct {
	io.BinlogIO
	allocator allocator.Allocator
	cm        storage.ChunkManager

	plan *datapb.CompactionPlan

	ctx    context.Context
	cancel context.CancelFunc

	done chan struct{}
	tr   *timerecord.TimeRecorder
}

// make sure compactionTask implements compactor interface
var _ compaction.Compactor = (*levelZeroCompactionTask)(nil)

func newLevelZeroCompactionTask(
	ctx context.Context,
	binlogIO io.BinlogIO,
	alloc allocator.Allocator,
	cm storage.ChunkManager,
	plan *datapb.CompactionPlan,
) *levelZeroCompactionTask {
	ctx, cancel := context.WithCancel(ctx)
	return &levelZeroCompactionTask{
		ctx:    ctx,
		cancel: cancel,

		BinlogIO:  binlogIO,
		allocator: alloc,
		cm:        cm,
		plan:      plan,
		tr:        timerecord.NewTimeRecorder("levelzero compaction"),
		done:      make(chan struct{}, 1),
	}
}

func (t *levelZeroCompactionTask) Complete() {
	t.done <- struct{}{}
}

func (t *levelZeroCompactionTask) Stop() {
	t.cancel()
	<-t.done
}

func (t *levelZeroCompactionTask) GetPlanID() UniqueID {
	return t.plan.GetPlanID()
}

func (t *levelZeroCompactionTask) GetChannelName() string {
	return t.plan.GetChannel()
}

func (t *levelZeroCompactionTask) GetCollection() int64 {
	// The length of SegmentBinlogs is checked before task enqueueing.
	return t.plan.GetSegmentBinlogs()[0].GetCollectionID()
}

func (t *levelZeroCompactionTask) Compact() (*datapb.CompactionPlanResult, error) {
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(t.ctx, "L0Compact")
	defer span.End()
	log := log.Ctx(t.ctx).With(zap.Int64("planID", t.plan.GetPlanID()), zap.String("type", t.plan.GetType().String()))
	log.Info("L0 compaction", zap.Duration("wait in queue elapse", t.tr.RecordSpan()))

	if !funcutil.CheckCtxValid(ctx) {
		log.Warn("compact wrong, task context done or timeout")
		return nil, ctx.Err()
	}

	ctxTimeout, cancelAll := context.WithTimeout(ctx, time.Duration(t.plan.GetTimeoutInSeconds())*time.Second)
	defer cancelAll()

	l0Segments := lo.Filter(t.plan.GetSegmentBinlogs(), func(s *datapb.CompactionSegmentBinlogs, _ int) bool {
		return s.Level == datapb.SegmentLevel_L0
	})

	targetSegments := lo.Filter(t.plan.GetSegmentBinlogs(), func(s *datapb.CompactionSegmentBinlogs, _ int) bool {
		return s.Level != datapb.SegmentLevel_L0
	})
	if len(targetSegments) == 0 {
		log.Warn("compact wrong, not target sealed segments")
		return nil, errors.New("illegal compaction plan with empty target segments")
	}
	err := binlog.DecompressCompactionBinlogs(l0Segments)
	if err != nil {
		log.Warn("DecompressCompactionBinlogs failed", zap.Error(err))
		return nil, err
	}

	var (
		totalSize      int64
		totalDeltalogs = make(map[UniqueID][]string)
	)
	for _, s := range l0Segments {
		paths := []string{}
		for _, d := range s.GetDeltalogs() {
			for _, l := range d.GetBinlogs() {
				paths = append(paths, l.GetLogPath())
				totalSize += l.GetMemorySize()
			}
		}
		if len(paths) > 0 {
			totalDeltalogs[s.GetSegmentID()] = paths
		}
	}

	var resultSegments []*datapb.CompactionSegment

	if float64(hardware.GetFreeMemoryCount())*paramtable.Get().DataNodeCfg.L0BatchMemoryRatio.GetAsFloat() < float64(totalSize) {
		resultSegments, err = t.linearProcess(ctxTimeout, targetSegments, totalDeltalogs)
	} else {
		resultSegments, err = t.batchProcess(ctxTimeout, targetSegments, lo.Values(totalDeltalogs)...)
	}
	if err != nil {
		return nil, err
	}

	result := &datapb.CompactionPlanResult{
		PlanID:   t.plan.GetPlanID(),
		State:    datapb.CompactionTaskState_completed,
		Segments: resultSegments,
		Channel:  t.plan.GetChannel(),
		Type:     t.plan.GetType(),
	}

	metrics.DataNodeCompactionLatency.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), t.plan.GetType().String()).
		Observe(float64(t.tr.ElapseSpan().Milliseconds()))
	log.Info("L0 compaction finished", zap.Duration("elapse", t.tr.ElapseSpan()))

	return result, nil
}

func (t *levelZeroCompactionTask) linearProcess(ctx context.Context, targetSegments []*datapb.CompactionSegmentBinlogs, totalDeltalogs map[int64][]string) ([]*datapb.CompactionSegment, error) {
	log := log.Ctx(t.ctx).With(
		zap.Int64("planID", t.plan.GetPlanID()),
		zap.String("type", t.plan.GetType().String()),
		zap.Int("target segment counts", len(targetSegments)),
	)

	// just for logging
	targetSegmentIDs := lo.Map(targetSegments, func(segment *datapb.CompactionSegmentBinlogs, _ int) int64 {
		return segment.GetSegmentID()
	})

	var (
		resultSegments  = make(map[int64]*datapb.CompactionSegment)
		alteredSegments = make(map[int64]*storage.DeleteData)
	)

	segmentBFs, err := t.loadBF(ctx, targetSegments)
	if err != nil {
		return nil, err
	}
	for segID, deltaLogs := range totalDeltalogs {
		log := log.With(zap.Int64("levelzero segment", segID))

		log.Info("Linear L0 compaction start processing segment")
		allIters, err := t.loadDelta(ctx, deltaLogs)
		if err != nil {
			log.Warn("Linear L0 compaction loadDelta fail", zap.Int64s("target segments", targetSegmentIDs), zap.Error(err))
			return nil, err
		}

		t.splitDelta(ctx, allIters, alteredSegments, segmentBFs)

		err = t.uploadByCheck(ctx, true, alteredSegments, resultSegments)
		if err != nil {
			log.Warn("Linear L0 compaction upload buffer fail", zap.Int64s("target segments", targetSegmentIDs), zap.Error(err))
			return nil, err
		}
	}

	err = t.uploadByCheck(ctx, false, alteredSegments, resultSegments)
	if err != nil {
		log.Warn("Linear L0 compaction upload all buffer fail", zap.Int64s("target segment", targetSegmentIDs), zap.Error(err))
		return nil, err
	}
	log.Info("Linear L0 compaction finished", zap.Duration("elapse", t.tr.RecordSpan()))
	return lo.Values(resultSegments), nil
}

func (t *levelZeroCompactionTask) batchProcess(ctx context.Context, targetSegments []*datapb.CompactionSegmentBinlogs, deltaLogs ...[]string) ([]*datapb.CompactionSegment, error) {
	log := log.Ctx(t.ctx).With(
		zap.Int64("planID", t.plan.GetPlanID()),
		zap.String("type", t.plan.GetType().String()),
		zap.Int("target segment counts", len(targetSegments)),
	)

	// just for logging
	targetSegmentIDs := lo.Map(targetSegments, func(segment *datapb.CompactionSegmentBinlogs, _ int) int64 {
		return segment.GetSegmentID()
	})

	log.Info("Batch L0 compaction start processing")
	resultSegments := make(map[int64]*datapb.CompactionSegment)

	iters, err := t.loadDelta(ctx, lo.Flatten(deltaLogs))
	if err != nil {
		log.Warn("Batch L0 compaction loadDelta fail", zap.Int64s("target segments", targetSegmentIDs), zap.Error(err))
		return nil, err
	}

	segmentBFs, err := t.loadBF(ctx, targetSegments)
	if err != nil {
		return nil, err
	}

	alteredSegments := make(map[int64]*storage.DeleteData)
	t.splitDelta(ctx, iters, alteredSegments, segmentBFs)

	err = t.uploadByCheck(ctx, false, alteredSegments, resultSegments)
	if err != nil {
		log.Warn("Batch L0 compaction upload fail", zap.Int64s("target segments", targetSegmentIDs), zap.Error(err))
		return nil, err
	}
	log.Info("Batch L0 compaction finished", zap.Duration("elapse", t.tr.RecordSpan()))
	return lo.Values(resultSegments), nil
}

func (t *levelZeroCompactionTask) loadDelta(ctx context.Context, deltaLogs ...[]string) ([]*iter.DeltalogIterator, error) {
	allIters := make([]*iter.DeltalogIterator, 0)

	for _, paths := range deltaLogs {
		blobs, err := t.Download(ctx, paths)
		if err != nil {
			return nil, err
		}

		allIters = append(allIters, iter.NewDeltalogIterator(blobs, nil))
	}
	return allIters, nil
}

func (t *levelZeroCompactionTask) splitDelta(
	ctx context.Context,
	allIters []*iter.DeltalogIterator,
	targetSegBuffer map[int64]*storage.DeleteData,
	segmentBfs map[int64]*metacache.BloomFilterSet,
) {
	_, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, "L0Compact splitDelta")
	defer span.End()

	split := func(pk storage.PrimaryKey) []int64 {
		lc := storage.NewLocationsCache(pk)
		predicts := make([]int64, 0, len(segmentBfs))
		for segmentID, bf := range segmentBfs {
			if bf.PkExists(lc) {
				predicts = append(predicts, segmentID)
			}
		}
		return predicts
	}

	// spilt all delete data to segments
	for _, deltaIter := range allIters {
		for deltaIter.HasNext() {
			// checked by HasNext, no error here
			labeled, _ := deltaIter.Next()

			predicted := split(labeled.GetPk())

			for _, gotSeg := range predicted {
				delBuffer, ok := targetSegBuffer[gotSeg]
				if !ok {
					delBuffer = &storage.DeleteData{}
					targetSegBuffer[gotSeg] = delBuffer
				}

				delBuffer.Append(labeled.GetPk(), labeled.GetTimestamp())
			}
		}
	}
}

func (t *levelZeroCompactionTask) composeDeltalog(segmentID int64, dData *storage.DeleteData) (map[string][]byte, *datapb.Binlog, error) {
	segment, ok := lo.Find(t.plan.GetSegmentBinlogs(), func(segment *datapb.CompactionSegmentBinlogs) bool {
		return segment.GetSegmentID() == segmentID
	})
	if !ok {
		return nil, nil, merr.WrapErrSegmentNotFound(segmentID, "cannot find segment in compaction plan")
	}

	var (
		collectionID = segment.GetCollectionID()
		partitionID  = segment.GetPartitionID()
		uploadKv     = make(map[string][]byte)
	)

	blob, err := storage.NewDeleteCodec().Serialize(collectionID, partitionID, segmentID, dData)
	if err != nil {
		return nil, nil, err
	}

	logID, err := t.allocator.AllocOne()
	if err != nil {
		return nil, nil, err
	}

	blobKey := metautil.JoinIDPath(collectionID, partitionID, segmentID, logID)
	blobPath := t.BinlogIO.JoinFullPath(common.SegmentDeltaLogPath, blobKey)

	uploadKv[blobPath] = blob.GetValue()

	minTs := uint64(math.MaxUint64)
	maxTs := uint64(0)
	for _, ts := range dData.Tss {
		if ts > maxTs {
			maxTs = ts
		}
		if ts < minTs {
			minTs = ts
		}
	}

	deltalog := &datapb.Binlog{
		EntriesNum:    dData.RowCount,
		LogSize:       int64(len(blob.GetValue())),
		LogPath:       blobPath,
		LogID:         logID,
		TimestampFrom: minTs,
		TimestampTo:   maxTs,
		MemorySize:    dData.Size(),
	}

	return uploadKv, deltalog, nil
}

func (t *levelZeroCompactionTask) uploadByCheck(ctx context.Context, requireCheck bool, alteredSegments map[int64]*storage.DeleteData, resultSegments map[int64]*datapb.CompactionSegment) error {
	allBlobs := make(map[string][]byte)
	tmpResults := make(map[int64]*datapb.CompactionSegment)
	for segID, dData := range alteredSegments {
		if !requireCheck || (dData.Size() >= paramtable.Get().DataNodeCfg.FlushDeleteBufferBytes.GetAsInt64()) {
			blobs, binlog, err := t.composeDeltalog(segID, dData)
			if err != nil {
				log.Warn("L0 compaction composeDelta fail", zap.Int64("segmentID", segID), zap.Error(err))
				return err
			}
			allBlobs = lo.Assign(blobs, allBlobs)
			tmpResults[segID] = &datapb.CompactionSegment{
				SegmentID: segID,
				Deltalogs: []*datapb.FieldBinlog{{Binlogs: []*datapb.Binlog{binlog}}},
				Channel:   t.plan.GetChannel(),
			}
			delete(alteredSegments, segID)
		}
	}

	if len(allBlobs) == 0 {
		return nil
	}

	if err := t.Upload(ctx, allBlobs); err != nil {
		log.Warn("L0 compaction upload blobs fail", zap.Error(err))
		return err
	}

	for segID, compSeg := range tmpResults {
		if _, ok := resultSegments[segID]; !ok {
			resultSegments[segID] = compSeg
		} else {
			binlog := compSeg.Deltalogs[0].Binlogs[0]
			resultSegments[segID].Deltalogs[0].Binlogs = append(resultSegments[segID].Deltalogs[0].Binlogs, binlog)
		}
	}

	return nil
}

func (t *levelZeroCompactionTask) loadBF(ctx context.Context, targetSegments []*datapb.CompactionSegmentBinlogs) (map[int64]*metacache.BloomFilterSet, error) {
	_, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, "L0Compact loadBF")
	defer span.End()

	var (
		futures = make([]*conc.Future[any], 0, len(targetSegments))
		pool    = getOrCreateStatsPool()

		mu  = &sync.Mutex{}
		bfs = make(map[int64]*metacache.BloomFilterSet)
	)

	for _, segment := range targetSegments {
		segment := segment
		innerCtx := ctx
		future := pool.Submit(func() (any, error) {
			_ = binlog.DecompressBinLog(storage.StatsBinlog, segment.GetCollectionID(),
				segment.GetPartitionID(), segment.GetSegmentID(), segment.GetField2StatslogPaths())
			pks, err := loadStats(innerCtx, t.cm, t.plan.GetSchema(), segment.GetSegmentID(), segment.GetField2StatslogPaths())
			if err != nil {
				log.Warn("failed to load segment stats log",
					zap.Int64("planID", t.plan.GetPlanID()),
					zap.String("type", t.plan.GetType().String()),
					zap.Error(err))
				return err, err
			}
			bf := metacache.NewBloomFilterSet(pks...)
			mu.Lock()
			defer mu.Unlock()
			bfs[segment.GetSegmentID()] = bf
			return nil, nil
		})
		futures = append(futures, future)
	}

	err := conc.AwaitAll(futures...)
	return bfs, err
}
