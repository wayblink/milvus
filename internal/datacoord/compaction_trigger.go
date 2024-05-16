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

package datacoord

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/util/clustering"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/indexparamcheck"
	"github.com/milvus-io/milvus/pkg/util/lock"
	"github.com/milvus-io/milvus/pkg/util/logutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type compactTime struct {
	expireTime    Timestamp
	collectionTTL time.Duration
}

type trigger interface {
	start()
	stop()
	// triggerCompaction triggers a compaction if any compaction condition satisfy.
	triggerCompaction() error
	// triggerSingleCompaction triggers a compaction bundled with collection-partition-channel-segment
	triggerSingleCompaction(collectionID, partitionID, segmentID int64, channel string, blockToSendSignal bool) error
	// triggerManualCompaction force to start a compaction
	triggerManualCompaction(collectionID int64, clusteringCompaction bool) (UniqueID, error)
}

type compactionSignal struct {
	id           UniqueID
	isForce      bool
	isGlobal     bool
	isClustering bool
	collectionID UniqueID
	partitionID  UniqueID
	channel      string
	segmentID    UniqueID
	pos          *msgpb.MsgPosition
}

var _ trigger = (*compactionTrigger)(nil)

type compactionTrigger struct {
	ctx               context.Context
	handler           Handler
	meta              *meta
	allocator         allocator
	signals           chan *compactionSignal
	compactionHandler compactionPlanContext
	globalTrigger     *time.Ticker
	forceMu           lock.Mutex
	quit              chan struct{}
	wg                sync.WaitGroup

	clusteringCompactionManager *ClusteringCompactionManager

	indexEngineVersionManager IndexEngineVersionManager

	estimateNonDiskSegmentPolicy calUpperLimitPolicy
	estimateDiskSegmentPolicy    calUpperLimitPolicy
	// A sloopy hack, so we can test with different segment row count without worrying that
	// they are re-calculated in every compaction.
	testingOnly bool
}

func newCompactionTrigger(
	ctx context.Context,
	meta *meta,
	compactionHandler compactionPlanContext,
	allocator allocator,
	handler Handler,
	indexVersionManager IndexEngineVersionManager,
	clusteringCompactionManager *ClusteringCompactionManager,
) *compactionTrigger {
	return &compactionTrigger{
		ctx:                          ctx,
		meta:                         meta,
		allocator:                    allocator,
		signals:                      make(chan *compactionSignal, 100),
		compactionHandler:            compactionHandler,
		indexEngineVersionManager:    indexVersionManager,
		estimateDiskSegmentPolicy:    calBySchemaPolicyWithDiskIndex,
		estimateNonDiskSegmentPolicy: calBySchemaPolicy,
		handler:                      handler,
		clusteringCompactionManager:  clusteringCompactionManager,
	}
}

func (t *compactionTrigger) start() {
	t.quit = make(chan struct{})
	t.globalTrigger = time.NewTicker(Params.DataCoordCfg.GlobalCompactionInterval.GetAsDuration(time.Second))
	t.wg.Add(3)
	go func() {
		defer logutil.LogPanic()
		defer t.wg.Done()

		for {
			select {
			case <-t.quit:
				log.Info("compaction trigger quit")
				return
			case signal := <-t.signals:
				switch {
				case signal.isClustering:
					err := t.handleClusteringCompactionSignal(signal)
					if err != nil {
						log.Warn("unable to handleClusteringCompactionSignal", zap.Error(err))
					}
				case signal.isGlobal:
					// ManualCompaction also use use handleGlobalSignal
					// so throw err here
					err := t.handleGlobalSignal(signal)
					if err != nil {
						log.Warn("unable to handleGlobalSignal", zap.Error(err))
					}
				default:
					// no need to handle err in handleSignal
					t.handleSignal(signal)
					// shouldn't reset, otherwise a frequent flushed collection will affect other collections
					// t.globalTrigger.Reset(Params.DataCoordCfg.GlobalCompactionInterval)
				}
			}
		}
	}()

	// As major compaction has states, related segments must be set compacting when datacoord restart.
	// So clusteringCompactionManager must start before common compaction loop starts
	if t.clusteringCompactionManager != nil {
		t.clusteringCompactionManager.start()
	}
	go t.startClusteringCompactionLoop()
	go t.startGlobalCompactionLoop()
}

func (t *compactionTrigger) startGlobalCompactionLoop() {
	defer logutil.LogPanic()
	defer t.wg.Done()

	// If AutoCompaction disabled, global loop will not start
	if !Params.DataCoordCfg.EnableAutoCompaction.GetAsBool() {
		return
	}

	for {
		select {
		case <-t.quit:
			t.globalTrigger.Stop()
			log.Info("global compaction loop exit")
			return
		case <-t.globalTrigger.C:
			err := t.triggerCompaction()
			if err != nil {
				log.Warn("unable to triggerCompaction", zap.Error(err))
			}
		}
	}
}

func (t *compactionTrigger) startClusteringCompactionLoop() {
	defer logutil.LogPanic()
	defer t.wg.Done()
	clusteringCompactionTicker := time.NewTicker(paramtable.Get().DataCoordCfg.ClusteringCompactionTriggerInterval.GetAsDuration(time.Second))
	defer clusteringCompactionTicker.Stop()
	for {
		select {
		case <-t.quit:
			clusteringCompactionTicker.Stop()
			log.Info("clustering compaction loop exit")
			return
		case <-clusteringCompactionTicker.C:
			err := t.triggerClusteringCompaction()
			if err != nil {
				log.Warn("unable to triggerClusteringCompaction", zap.Error(err))
			}
			clusteringCompactionTicker.Reset(paramtable.Get().DataCoordCfg.ClusteringCompactionTriggerInterval.GetAsDuration(time.Second))
		}
	}
}

func (t *compactionTrigger) stop() {
	if t.clusteringCompactionManager != nil {
		t.clusteringCompactionManager.stop()
	}
	close(t.quit)
	t.wg.Wait()
}

func (t *compactionTrigger) allocTs() (Timestamp, error) {
	cctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ts, err := t.allocator.allocTimestamp(cctx)
	if err != nil {
		return 0, err
	}

	return ts, nil
}

func (t *compactionTrigger) getCollection(collectionID UniqueID) (*collectionInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	coll, err := t.handler.GetCollection(ctx, collectionID)
	if err != nil {
		return nil, fmt.Errorf("collection ID %d not found, err: %w", collectionID, err)
	}
	return coll, nil
}

func (t *compactionTrigger) isCollectionAutoCompactionEnabled(coll *collectionInfo) bool {
	enabled, err := getCollectionAutoCompactionEnabled(coll.Properties)
	if err != nil {
		log.Warn("collection properties auto compaction not valid, returning false", zap.Error(err))
		return false
	}
	return enabled
}

func (t *compactionTrigger) isChannelCheckpointHealthy(vchanName string) bool {
	if paramtable.Get().DataCoordCfg.ChannelCheckpointMaxLag.GetAsInt64() <= 0 {
		return true
	}
	checkpoint := t.meta.GetChannelCheckpoint(vchanName)
	if checkpoint == nil {
		log.Warn("channel checkpoint not found", zap.String("channel", vchanName))
		return false
	}

	cpTime := tsoutil.PhysicalTime(checkpoint.GetTimestamp())
	return time.Since(cpTime) < paramtable.Get().DataCoordCfg.ChannelCheckpointMaxLag.GetAsDuration(time.Second)
}

func getCompactTime(ts Timestamp, coll *collectionInfo) (*compactTime, error) {
	collectionTTL, err := getCollectionTTL(coll.Properties)
	if err != nil {
		return nil, err
	}

	pts, _ := tsoutil.ParseTS(ts)

	if collectionTTL > 0 {
		ttexpired := pts.Add(-collectionTTL)
		ttexpiredLogic := tsoutil.ComposeTS(ttexpired.UnixNano()/int64(time.Millisecond), 0)
		return &compactTime{ttexpiredLogic, collectionTTL}, nil
	}

	// no expiration time
	return &compactTime{0, 0}, nil
}

// triggerCompaction trigger a compaction if any compaction condition satisfy.
func (t *compactionTrigger) triggerCompaction() error {
	id, err := t.allocSignalID()
	if err != nil {
		return err
	}
	signal := &compactionSignal{
		id:       id,
		isForce:  false,
		isGlobal: true,
	}
	t.signals <- signal
	return nil
}

// triggerClusteringCompaction trigger clustering compaction.
func (t *compactionTrigger) triggerClusteringCompaction() error {
	if Params.DataCoordCfg.ClusteringCompactionEnable.GetAsBool() &&
		Params.DataCoordCfg.ClusteringCompactionAutoEnable.GetAsBool() {
		collections := t.meta.GetCollections()
		isStart, _, err := t.allocator.allocN(int64(len(collections)))
		if err != nil {
			return err
		}
		id := isStart
		for _, collection := range collections {
			clusteringKeyField := clustering.GetClusteringKeyField(collection.Schema)
			if clusteringKeyField != nil {
				signal := &compactionSignal{
					id:           id,
					isForce:      false,
					isGlobal:     true,
					isClustering: true,
					collectionID: collection.ID,
				}
				t.signals <- signal
				id++
			}
		}
	}
	return nil
}

// triggerSingleCompaction trigger a compaction bundled with collection-partition-channel-segment
func (t *compactionTrigger) triggerSingleCompaction(collectionID, partitionID, segmentID int64, channel string, blockToSendSignal bool) error {
	// If AutoCompaction disabled, flush request will not trigger compaction
	if !Params.DataCoordCfg.EnableAutoCompaction.GetAsBool() {
		return nil
	}

	id, err := t.allocSignalID()
	if err != nil {
		return err
	}
	signal := &compactionSignal{
		id:           id,
		isForce:      false,
		isGlobal:     false,
		collectionID: collectionID,
		partitionID:  partitionID,
		segmentID:    segmentID,
		channel:      channel,
	}
	if blockToSendSignal {
		t.signals <- signal
		return nil
	}
	select {
	case t.signals <- signal:
	default:
		log.Info("no space to send compaction signal", zap.Int64("collectionID", collectionID), zap.Int64("segmentID", segmentID), zap.String("channel", channel))
	}

	return nil
}

// triggerManualCompaction force to start a compaction
// invoked by user `ManualCompaction` operation
func (t *compactionTrigger) triggerManualCompaction(collectionID int64, clusteringCompaction bool) (UniqueID, error) {
	id, err := t.allocSignalID()
	if err != nil {
		return -1, err
	}
	signal := &compactionSignal{
		id:           id,
		isForce:      true,
		isGlobal:     true,
		isClustering: clusteringCompaction,
		collectionID: collectionID,
	}

	if clusteringCompaction {
		compacting, triggerID := t.clusteringCompactionManager.collectionIsClusteringCompacting(signal.collectionID)
		if compacting {
			log.Info("collection is clustering compacting", zap.Int64("collectionID", signal.collectionID), zap.Int64("triggerID", triggerID))
			return triggerID, nil
		}
		err = t.handleClusteringCompactionSignal(signal)
	} else {
		err = t.handleGlobalSignal(signal)
	}
	if err != nil {
		log.Warn("unable to handle compaction signal", zap.Error(err))
		return -1, err
	}

	return id, nil
}

func (t *compactionTrigger) allocSignalID() (UniqueID, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return t.allocator.allocID(ctx)
}

func (t *compactionTrigger) getExpectedSegmentSize(collectionID int64) int64 {
	indexInfos := t.meta.indexMeta.GetIndexesForCollection(collectionID, "")

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	collMeta, err := t.handler.GetCollection(ctx, collectionID)
	if err != nil {
		log.Warn("failed to get collection", zap.Int64("collectionID", collectionID), zap.Error(err))
		return Params.DataCoordCfg.SegmentMaxSize.GetAsInt64() * 1024 * 1024
	}

	vectorFields := typeutil.GetVectorFieldSchemas(collMeta.Schema)
	fieldIndexTypes := lo.SliceToMap(indexInfos, func(t *model.Index) (int64, indexparamcheck.IndexType) {
		return t.FieldID, GetIndexType(t.IndexParams)
	})
	vectorFieldsWithDiskIndex := lo.Filter(vectorFields, func(field *schemapb.FieldSchema, _ int) bool {
		if indexType, ok := fieldIndexTypes[field.FieldID]; ok {
			return indexparamcheck.IsDiskIndex(indexType)
		}
		return false
	})

	allDiskIndex := len(vectorFields) == len(vectorFieldsWithDiskIndex)
	if allDiskIndex {
		// Only if all vector fields index type are DiskANN, recalc segment max size here.
		return Params.DataCoordCfg.DiskSegmentMaxSize.GetAsInt64() * 1024 * 1024
	}
	// If some vector fields index type are not DiskANN, recalc segment max size using default policy.
	return Params.DataCoordCfg.SegmentMaxSize.GetAsInt64() * 1024 * 1024
}

func (t *compactionTrigger) handleGlobalSignal(signal *compactionSignal) error {
	t.forceMu.Lock()
	defer t.forceMu.Unlock()

	log := log.With(zap.Int64("compactionID", signal.id),
		zap.Int64("signal.collectionID", signal.collectionID),
		zap.Int64("signal.partitionID", signal.partitionID),
		zap.Int64("signal.segmentID", signal.segmentID))
	m := t.meta.GetSegmentsChanPart(func(segment *SegmentInfo) bool {
		return (signal.collectionID == 0 || segment.CollectionID == signal.collectionID) &&
			isSegmentHealthy(segment) &&
			isFlush(segment) &&
			!segment.isCompacting && // not compacting now
			!segment.GetIsImporting() && // not importing now
			segment.GetLevel() != datapb.SegmentLevel_L0 && // ignore level zero segments
			segment.GetLevel() != datapb.SegmentLevel_L2 // ignore l2 segment
	}) // m is list of chanPartSegments, which is channel-partition organized segments

	if len(m) == 0 {
		log.Info("the length of SegmentsChanPart is 0, skip to handle compaction")
		return nil
	}

	ts, err := t.allocTs()
	if err != nil {
		log.Warn("allocate ts failed, skip to handle compaction")
		return err
	}

	channelCheckpointOK := make(map[string]bool)
	isChannelCPOK := func(channelName string) bool {
		cached, ok := channelCheckpointOK[channelName]
		if ok {
			return cached
		}
		return t.isChannelCheckpointHealthy(channelName)
	}

	for _, group := range m {
		log := log.With(zap.Int64("collectionID", group.collectionID),
			zap.Int64("partitionID", group.partitionID),
			zap.String("channel", group.channelName))
		if !signal.isForce && t.compactionHandler.isFull() {
			log.Warn("compaction plan skipped due to handler full")
			break
		}
		if !isChannelCPOK(group.channelName) && !signal.isForce {
			log.Warn("compaction plan skipped due to channel checkpoint lag", zap.String("channel", signal.channel))
			continue
		}

		if Params.DataCoordCfg.IndexBasedCompaction.GetAsBool() {
			group.segments = FilterInIndexedSegments(t.handler, t.meta, group.segments...)
		}

		coll, err := t.getCollection(group.collectionID)
		if err != nil {
			log.Warn("get collection info failed, skip handling compaction", zap.Error(err))
			return err
		}

		if !signal.isForce && !t.isCollectionAutoCompactionEnabled(coll) {
			log.RatedInfo(20, "collection auto compaction disabled",
				zap.Int64("collectionID", group.collectionID),
			)
			return nil
		}

		ct, err := getCompactTime(ts, coll)
		if err != nil {
			log.Warn("get compact time failed, skip to handle compaction",
				zap.Int64("collectionID", group.collectionID),
				zap.Int64("partitionID", group.partitionID),
				zap.String("channel", group.channelName))
			return err
		}

		plans := t.generatePlans(group.segments, signal.isForce, ct)
		for _, plan := range plans {
			segIDs := fetchSegIDs(plan.GetSegmentBinlogs())

			if !signal.isForce && t.compactionHandler.isFull() {
				log.Warn("compaction plan skipped due to handler full",
					zap.Int64("collectionID", signal.collectionID),
					zap.Int64s("segmentIDs", segIDs))
				break
			}
			start := time.Now()
			if err := fillOriginPlan(t.allocator, plan); err != nil {
				log.Warn("failed to fill plan",
					zap.Int64("collectionID", signal.collectionID),
					zap.Int64s("segmentIDs", segIDs),
					zap.Error(err))
				continue
			}
			err := t.compactionHandler.execCompactionPlan(signal, plan)
			if err != nil {
				log.Warn("failed to execute compaction plan",
					zap.Int64("collectionID", signal.collectionID),
					zap.Int64("planID", plan.PlanID),
					zap.Int64s("segmentIDs", segIDs),
					zap.Error(err))
				continue
			}

			log.Info("time cost of generating global compaction",
				zap.Int64("planID", plan.PlanID),
				zap.Int64("time cost", time.Since(start).Milliseconds()),
				zap.Int64("collectionID", signal.collectionID),
				zap.String("channel", group.channelName),
				zap.Int64("partitionID", group.partitionID),
				zap.Int64s("segmentIDs", segIDs))
		}
	}
	return nil
}

func (t *compactionTrigger) handleClusteringCompactionSignal(signal *compactionSignal) error {
	if !Params.DataCoordCfg.ClusteringCompactionEnable.GetAsBool() {
		err := merr.WrapErrClusteringCompactionClusterNotSupport()
		log.Warn(err.Error())
		return err
	}
	// for ut
	if t.clusteringCompactionManager == nil {
		err := merr.WrapErrClusteringCompactionClusterNotSupport()
		log.Warn("clustering compaction manager is nil")
		return err
	}

	ts, err := t.allocTs()
	if err != nil {
		log.Warn("allocate ts failed, skip to handle compaction")
		return err
	}

	t.forceMu.Lock()
	defer t.forceMu.Unlock()

	log := log.With(zap.Int64("compactionID", signal.id), zap.Int64("collectionID", signal.collectionID))

	coll, err := t.getCollection(signal.collectionID)
	if err != nil {
		log.Warn("get collection info failed, skip handling compaction", zap.Error(err))
		return err
	}
	clusteringKeyField := clustering.GetClusteringKeyField(coll.Schema)
	if clusteringKeyField == nil {
		err := merr.WrapErrClusteringCompactionCollectionNotSupport(fmt.Sprint(signal.collectionID))
		log.Debug(err.Error())
		return err
	}

	partSegments := t.meta.GetSegmentsChanPart(func(segment *SegmentInfo) bool {
		return (signal.collectionID == 0 || segment.CollectionID == signal.collectionID) &&
			isSegmentHealthy(segment) &&
			isFlush(segment) &&
			!segment.isCompacting && // not compacting now
			!segment.GetIsImporting() && // not importing now
			segment.GetLevel() != datapb.SegmentLevel_L0 // ignore level zero segments
	}) // partSegments is list of chanPartSegments, which is channel-partition organized segments

	if len(partSegments) == 0 {
		log.Info("the length of SegmentsChanPart is 0, skip to handle compaction")
		return nil
	}

	clusteringCompactionTasks := make([]*datapb.CompactionTask, 0)
	for _, group := range partSegments {
		log := log.With(zap.Int64("collectionID", group.collectionID),
			zap.Int64("partitionID", group.partitionID),
			zap.String("channel", group.channelName))

		ct, err := getCompactTime(ts, coll)
		if err != nil {
			log.Warn("get compact time failed, skip to handle compaction")
			return err
		}

		if len(group.segments) == 0 {
			log.Info("the length of SegmentsChanPart is 0, skip to handle compaction")
			continue
		}

		if !signal.isForce {
			execute, err := triggerCompactionPolicy(t.ctx, t.meta, group.collectionID, group.partitionID, group.channelName, group.segments)
			if err != nil {
				log.Warn("failed to trigger clustering compaction", zap.Error(err))
				continue
			}
			if !execute {
				continue
			}
		}

		clusteringCompactionTask := &datapb.CompactionTask{
			TriggerID:          signal.id,
			State:              datapb.CompactionTaskState_init,
			StartTime:          ts,
			Type:               datapb.CompactionType_ClusteringCompaction,
			CollectionTtl:      ct.collectionTTL.Nanoseconds(),
			CollectionID:       signal.collectionID,
			PartitionID:        group.partitionID,
			Channel:            group.channelName,
			ClusteringKeyField: clusteringKeyField,
		}
		t.clusteringCompactionManager.fillClusteringCompactionTask(clusteringCompactionTask, group.segments)
		// mark all segments prepare for clustering compaction
		for _, seg := range group.segments {
			t.meta.SetSegmentCompacting(seg.ID, true)
		}
		clusteringCompactionTasks = append(clusteringCompactionTasks, clusteringCompactionTask)
	}
	if len(clusteringCompactionTasks) > 0 {
		t.clusteringCompactionManager.submit(clusteringCompactionTasks)
	}
	return nil
}

func (t *compactionTrigger) setSegmentsCompacting(plans []*datapb.CompactionPlan, compacting bool) {
	for _, plan := range plans {
		for _, segmentBinlogs := range plan.GetSegmentBinlogs() {
			t.meta.SetSegmentCompacting(segmentBinlogs.GetSegmentID(), compacting)
		}
	}
}

// handleSignal processes segment flush caused partition-chan level compaction signal
func (t *compactionTrigger) handleSignal(signal *compactionSignal) {
	t.forceMu.Lock()
	defer t.forceMu.Unlock()

	// 1. check whether segment's binlogs should be compacted or not
	if t.compactionHandler.isFull() {
		log.Warn("compaction plan skipped due to handler full")
		return
	}

	if !t.isChannelCheckpointHealthy(signal.channel) {
		log.Warn("compaction plan skipped due to channel checkpoint lag", zap.String("channel", signal.channel))
		return
	}

	segment := t.meta.GetHealthySegment(signal.segmentID)
	if segment == nil {
		log.Warn("segment in compaction signal not found in meta", zap.Int64("segmentID", signal.segmentID))
		return
	}

	channel := segment.GetInsertChannel()
	partitionID := segment.GetPartitionID()
	collectionID := segment.GetCollectionID()
	segments := t.getCandidateSegments(channel, partitionID)

	if len(segments) == 0 {
		log.Info("the number of candidate segments is 0, skip to handle compaction")
		return
	}

	ts, err := t.allocTs()
	if err != nil {
		log.Warn("allocate ts failed, skip to handle compaction", zap.Int64("collectionID", signal.collectionID),
			zap.Int64("partitionID", signal.partitionID), zap.Int64("segmentID", signal.segmentID))
		return
	}

	coll, err := t.getCollection(collectionID)
	if err != nil {
		log.Warn("get collection info failed, skip handling compaction",
			zap.Int64("collectionID", collectionID),
			zap.Int64("partitionID", partitionID),
			zap.String("channel", channel),
			zap.Error(err),
		)
		return
	}

	if !signal.isForce && !t.isCollectionAutoCompactionEnabled(coll) {
		log.RatedInfo(20, "collection auto compaction disabled",
			zap.Int64("collectionID", collectionID),
		)
		return
	}

	ct, err := getCompactTime(ts, coll)
	if err != nil {
		log.Warn("get compact time failed, skip to handle compaction", zap.Int64("collectionID", segment.GetCollectionID()),
			zap.Int64("partitionID", partitionID), zap.String("channel", channel))
		return
	}

	plans := t.generatePlans(segments, signal.isForce, ct)
	for _, plan := range plans {
		if t.compactionHandler.isFull() {
			log.Warn("compaction plan skipped due to handler full", zap.Int64("collection", signal.collectionID), zap.Int64("planID", plan.PlanID))
			break
		}
		start := time.Now()
		if err := fillOriginPlan(t.allocator, plan); err != nil {
			log.Warn("failed to fill plan", zap.Error(err))
			continue
		}
		if err := t.compactionHandler.execCompactionPlan(signal, plan); err != nil {
			log.Warn("failed to execute compaction plan",
				zap.Int64("collection", signal.collectionID),
				zap.Int64("planID", plan.PlanID),
				zap.Int64s("segmentIDs", fetchSegIDs(plan.GetSegmentBinlogs())),
				zap.Error(err))
			continue
		}
		log.Info("time cost of generating compaction",
			zap.Int64("planID", plan.PlanID),
			zap.Int64("time cost", time.Since(start).Milliseconds()),
			zap.Int64("collectionID", signal.collectionID),
			zap.String("channel", channel),
			zap.Int64("partitionID", partitionID),
			zap.Int64s("segmentIDs", fetchSegIDs(plan.GetSegmentBinlogs())))
	}
}

func (t *compactionTrigger) generatePlans(segments []*SegmentInfo, force bool, compactTime *compactTime) []*datapb.CompactionPlan {
	if len(segments) == 0 {
		log.Warn("the number of candidate segments is 0, skip to generate compaction plan")
		return []*datapb.CompactionPlan{}
	}

	// find segments need internal compaction
	// TODO add low priority candidates, for example if the segment is smaller than full 0.9 * max segment size but larger than small segment boundary, we only execute compaction when there are no compaction running actively
	var prioritizedCandidates []*SegmentInfo
	var smallCandidates []*SegmentInfo
	var nonPlannedSegments []*SegmentInfo

	expectedSize := t.getExpectedSegmentSize(segments[0].CollectionID)

	// TODO, currently we lack of the measurement of data distribution, there should be another compaction help on redistributing segment based on scalar/vector field distribution
	for _, segment := range segments {
		segment := segment.ShadowClone()
		// TODO should we trigger compaction periodically even if the segment has no obvious reason to be compacted?
		if force || t.ShouldDoSingleCompaction(segment, compactTime) {
			prioritizedCandidates = append(prioritizedCandidates, segment)
		} else if t.isSmallSegment(segment, expectedSize) {
			smallCandidates = append(smallCandidates, segment)
		} else {
			nonPlannedSegments = append(nonPlannedSegments, segment)
		}
	}

	buckets := [][]*SegmentInfo{}
	// sort segment from large to small
	sort.Slice(prioritizedCandidates, func(i, j int) bool {
		if prioritizedCandidates[i].getSegmentSize() != prioritizedCandidates[j].getSegmentSize() {
			return prioritizedCandidates[i].getSegmentSize() > prioritizedCandidates[j].getSegmentSize()
		}
		return prioritizedCandidates[i].GetID() < prioritizedCandidates[j].GetID()
	})

	sort.Slice(smallCandidates, func(i, j int) bool {
		if smallCandidates[i].getSegmentSize() != smallCandidates[j].getSegmentSize() {
			return smallCandidates[i].getSegmentSize() > smallCandidates[j].getSegmentSize()
		}
		return smallCandidates[i].GetID() < smallCandidates[j].GetID()
	})

	// Sort non-planned from small to large.
	sort.Slice(nonPlannedSegments, func(i, j int) bool {
		if nonPlannedSegments[i].getSegmentSize() != nonPlannedSegments[j].getSegmentSize() {
			return nonPlannedSegments[i].getSegmentSize() < nonPlannedSegments[j].getSegmentSize()
		}
		return nonPlannedSegments[i].GetID() > nonPlannedSegments[j].GetID()
	})

	// greedy pick from large segment to small, the goal is to fill each segment to reach 512M
	// we must ensure all prioritized candidates is in a plan
	// TODO the compaction selection policy should consider if compaction workload is high
	for len(prioritizedCandidates) > 0 {
		var bucket []*SegmentInfo
		// pop out the first element
		segment := prioritizedCandidates[0]
		bucket = append(bucket, segment)
		prioritizedCandidates = prioritizedCandidates[1:]

		// only do single file compaction if segment is already large enough
		if segment.getSegmentSize() < expectedSize {
			var result []*SegmentInfo
			free := expectedSize - segment.getSegmentSize()
			maxNum := Params.DataCoordCfg.MaxSegmentToMerge.GetAsInt() - 1
			prioritizedCandidates, result, free = greedySelect(prioritizedCandidates, free, maxNum)
			bucket = append(bucket, result...)
			maxNum -= len(result)
			if maxNum > 0 {
				smallCandidates, result, _ = greedySelect(smallCandidates, free, maxNum)
				bucket = append(bucket, result...)
			}
		}
		// since this is priority compaction, we will execute even if there is only segment
		log.Info("pick priority candidate for compaction",
			zap.Int64("prioritized segmentID", segment.GetID()),
			zap.Int64s("picked segmentIDs", lo.Map(bucket, func(s *SegmentInfo, _ int) int64 { return s.GetID() })),
			zap.Int64("target size", lo.SumBy(bucket, func(s *SegmentInfo) int64 { return s.getSegmentSize() })),
			zap.Int64("target count", lo.SumBy(bucket, func(s *SegmentInfo) int64 { return s.GetNumOfRows() })),
		)
		buckets = append(buckets, bucket)
	}

	var remainingSmallSegs []*SegmentInfo
	// check if there are small candidates left can be merged into large segments
	for len(smallCandidates) > 0 {
		var bucket []*SegmentInfo
		// pop out the first element
		segment := smallCandidates[0]
		bucket = append(bucket, segment)
		smallCandidates = smallCandidates[1:]

		var result []*SegmentInfo
		free := expectedSize - segment.getSegmentSize()
		// for small segment merge, we pick one largest segment and merge as much as small segment together with it
		// Why reverse?	 try to merge as many segments as expected.
		// for instance, if a 255M and 255M is the largest small candidates, they will never be merged because of the MinSegmentToMerge limit.
		smallCandidates, result, _ = reverseGreedySelect(smallCandidates, free, Params.DataCoordCfg.MaxSegmentToMerge.GetAsInt()-1)
		bucket = append(bucket, result...)

		// only merge if candidate number is large than MinSegmentToMerge or if target size is large enough
		targetSize := lo.SumBy(bucket, func(s *SegmentInfo) int64 { return s.getSegmentSize() })
		if len(bucket) >= Params.DataCoordCfg.MinSegmentToMerge.GetAsInt() ||
			len(bucket) > 1 && t.isCompactableSegment(targetSize, expectedSize) {
			buckets = append(buckets, bucket)
		} else {
			remainingSmallSegs = append(remainingSmallSegs, bucket...)
		}
	}

	remainingSmallSegs = t.squeezeSmallSegmentsToBuckets(remainingSmallSegs, buckets, expectedSize)

	// If there are still remaining small segments, try adding them to non-planned segments.
	for _, npSeg := range nonPlannedSegments {
		bucket := []*SegmentInfo{npSeg}
		targetSize := npSeg.getSegmentSize()
		for i := len(remainingSmallSegs) - 1; i >= 0; i-- {
			// Note: could also simply use MaxRowNum as limit.
			if targetSize+remainingSmallSegs[i].getSegmentSize() <=
				int64(Params.DataCoordCfg.SegmentExpansionRate.GetAsFloat()*float64(expectedSize)) {
				bucket = append(bucket, remainingSmallSegs[i])
				targetSize += remainingSmallSegs[i].getSegmentSize()
				remainingSmallSegs = append(remainingSmallSegs[:i], remainingSmallSegs[i+1:]...)
			}
		}
		if len(bucket) > 1 {
			buckets = append(buckets, bucket)
		}
	}

	plans := make([]*datapb.CompactionPlan, len(buckets))
	for i, b := range buckets {
		plans[i] = segmentsToPlan(b, datapb.CompactionType_MixCompaction, compactTime)
	}
	return plans
}

func segmentsToPlan(segments []*SegmentInfo, compactionType datapb.CompactionType, compactTime *compactTime) *datapb.CompactionPlan {
	plan := &datapb.CompactionPlan{
		Type:          compactionType,
		Channel:       segments[0].GetInsertChannel(),
		CollectionTtl: compactTime.collectionTTL.Nanoseconds(),
	}

	var size int64
	for _, s := range segments {
		segmentBinlogs := &datapb.CompactionSegmentBinlogs{
			SegmentID:           s.GetID(),
			FieldBinlogs:        s.GetBinlogs(),
			Field2StatslogPaths: s.GetStatslogs(),
			Deltalogs:           s.GetDeltalogs(),
			CollectionID:        s.GetCollectionID(),
			PartitionID:         s.GetPartitionID(),
		}
		plan.TotalRows += s.GetNumOfRows()
		size += s.getSegmentSize()
		plan.SegmentBinlogs = append(plan.SegmentBinlogs, segmentBinlogs)
	}

	log.Info("generate a plan for priority candidates", zap.Any("plan", plan),
		zap.Int64("target segment row", plan.TotalRows), zap.Int64("target segment size", size))
	return plan
}

func greedySelect(candidates []*SegmentInfo, free int64, maxSegment int) ([]*SegmentInfo, []*SegmentInfo, int64) {
	var result []*SegmentInfo

	for i := 0; i < len(candidates); {
		candidate := candidates[i]
		if len(result) < maxSegment && candidate.getSegmentSize() < free {
			result = append(result, candidate)
			free -= candidate.getSegmentSize()
			candidates = append(candidates[:i], candidates[i+1:]...)
		} else {
			i++
		}
	}

	return candidates, result, free
}

func reverseGreedySelect(candidates []*SegmentInfo, free int64, maxSegment int) ([]*SegmentInfo, []*SegmentInfo, int64) {
	var result []*SegmentInfo

	for i := len(candidates) - 1; i >= 0; i-- {
		candidate := candidates[i]
		if (len(result) < maxSegment) && (candidate.getSegmentSize() < free) {
			result = append(result, candidate)
			free -= candidate.getSegmentSize()
			candidates = append(candidates[:i], candidates[i+1:]...)
		}
	}
	return candidates, result, free
}

func (t *compactionTrigger) getCandidateSegments(channel string, partitionID UniqueID) []*SegmentInfo {
	segments := t.meta.GetSegmentsByChannel(channel)
	if Params.DataCoordCfg.IndexBasedCompaction.GetAsBool() {
		segments = FilterInIndexedSegments(t.handler, t.meta, segments...)
	}

	var res []*SegmentInfo
	for _, s := range segments {
		if !isSegmentHealthy(s) ||
			!isFlush(s) ||
			s.GetInsertChannel() != channel ||
			s.GetPartitionID() != partitionID ||
			s.isCompacting ||
			s.GetIsImporting() ||
			s.GetLevel() == datapb.SegmentLevel_L0 {
			continue
		}
		res = append(res, s)
	}

	return res
}

func (t *compactionTrigger) isSmallSegment(segment *SegmentInfo, expectedSize int64) bool {
	return segment.getSegmentSize() < int64(float64(expectedSize)*Params.DataCoordCfg.SegmentSmallProportion.GetAsFloat())
}

func (t *compactionTrigger) isCompactableSegment(targetSize, expectedSize int64) bool {
	smallProportion := Params.DataCoordCfg.SegmentSmallProportion.GetAsFloat()
	compactableProportion := Params.DataCoordCfg.SegmentCompactableProportion.GetAsFloat()

	// avoid invalid single segment compaction
	if compactableProportion < smallProportion {
		compactableProportion = smallProportion
	}

	return targetSize > int64(float64(expectedSize)*compactableProportion)
}

func isExpandableSmallSegment(segment *SegmentInfo, expectedSize int64) bool {
	return segment.getSegmentSize() < int64(float64(expectedSize)*(Params.DataCoordCfg.SegmentExpansionRate.GetAsFloat()-1))
}

func (t *compactionTrigger) ShouldDoSingleCompaction(segment *SegmentInfo, compactTime *compactTime) bool {
	// no longer restricted binlog numbers because this is now related to field numbers

	binlogCount := GetBinlogCount(segment.GetBinlogs())
	deltaLogCount := GetBinlogCount(segment.GetDeltalogs())
	if deltaLogCount > Params.DataCoordCfg.SingleCompactionDeltalogMaxNum.GetAsInt() {
		log.Info("total delta number is too much, trigger compaction", zap.Int64("segmentID", segment.ID), zap.Int("Bin logs", binlogCount), zap.Int("Delta logs", deltaLogCount))
		return true
	}

	// if expire time is enabled, put segment into compaction candidate
	totalExpiredSize := int64(0)
	totalExpiredRows := 0
	for _, binlogs := range segment.GetBinlogs() {
		for _, l := range binlogs.GetBinlogs() {
			// TODO, we should probably estimate expired log entries by total rows in binlog and the ralationship of timeTo, timeFrom and expire time
			if l.TimestampTo < compactTime.expireTime {
				log.RatedDebug(10, "mark binlog as expired",
					zap.Int64("segmentID", segment.ID),
					zap.Int64("binlogID", l.GetLogID()),
					zap.Uint64("binlogTimestampTo", l.TimestampTo),
					zap.Uint64("compactExpireTime", compactTime.expireTime))
				totalExpiredRows += int(l.GetEntriesNum())
				totalExpiredSize += l.GetMemorySize()
			}
		}
	}

	if float64(totalExpiredRows)/float64(segment.GetNumOfRows()) >= Params.DataCoordCfg.SingleCompactionRatioThreshold.GetAsFloat() ||
		totalExpiredSize > Params.DataCoordCfg.SingleCompactionExpiredLogMaxSize.GetAsInt64() {
		log.Info("total expired entities is too much, trigger compaction", zap.Int64("segmentID", segment.ID),
			zap.Int("expiredRows", totalExpiredRows), zap.Int64("expiredLogSize", totalExpiredSize),
			zap.Bool("createdByCompaction", segment.CreatedByCompaction), zap.Int64s("compactionFrom", segment.CompactionFrom))
		return true
	}

	totalDeletedRows := 0
	totalDeleteLogSize := int64(0)
	for _, deltaLogs := range segment.GetDeltalogs() {
		for _, l := range deltaLogs.GetBinlogs() {
			totalDeletedRows += int(l.GetEntriesNum())
			totalDeleteLogSize += l.GetMemorySize()
		}
	}

	// currently delta log size and delete ratio policy is applied
	if float64(totalDeletedRows)/float64(segment.GetNumOfRows()) >= Params.DataCoordCfg.SingleCompactionRatioThreshold.GetAsFloat() || totalDeleteLogSize > Params.DataCoordCfg.SingleCompactionDeltaLogMaxSize.GetAsInt64() {
		log.Info("total delete entities is too much, trigger compaction",
			zap.Int64("segmentID", segment.ID),
			zap.Int64("numRows", segment.GetNumOfRows()),
			zap.Int("deleted rows", totalDeletedRows),
			zap.Int64("delete log size", totalDeleteLogSize))
		return true
	}

	if Params.DataCoordCfg.AutoUpgradeSegmentIndex.GetAsBool() {
		// index version of segment lower than current version and IndexFileKeys should have value, trigger compaction
		indexIDToSegIdxes := t.meta.indexMeta.GetSegmentIndexes(segment.CollectionID, segment.ID)
		for _, index := range indexIDToSegIdxes {
			if index.CurrentIndexVersion < t.indexEngineVersionManager.GetCurrentIndexEngineVersion() &&
				len(index.IndexFileKeys) > 0 {
				log.Info("index version is too old, trigger compaction",
					zap.Int64("segmentID", segment.ID),
					zap.Int64("indexID", index.IndexID),
					zap.Strings("indexFileKeys", index.IndexFileKeys),
					zap.Int32("currentIndexVersion", index.CurrentIndexVersion),
					zap.Int32("currentEngineVersion", t.indexEngineVersionManager.GetCurrentIndexEngineVersion()))
				return true
			}
		}
	}

	return false
}

func isFlush(segment *SegmentInfo) bool {
	return segment.GetState() == commonpb.SegmentState_Flushed || segment.GetState() == commonpb.SegmentState_Flushing
}

func fetchSegIDs(segBinLogs []*datapb.CompactionSegmentBinlogs) []int64 {
	var segIDs []int64
	for _, segBinLog := range segBinLogs {
		segIDs = append(segIDs, segBinLog.GetSegmentID())
	}
	return segIDs
}

// buckets will be updated inplace
func (t *compactionTrigger) squeezeSmallSegmentsToBuckets(small []*SegmentInfo, buckets [][]*SegmentInfo, expectedSize int64) (remaining []*SegmentInfo) {
	for i := len(small) - 1; i >= 0; i-- {
		s := small[i]
		if !isExpandableSmallSegment(s, expectedSize) {
			continue
		}
		// Try squeeze this segment into existing plans. This could cause segment size to exceed maxSize.
		for bidx, b := range buckets {
			totalSize := lo.SumBy(b, func(s *SegmentInfo) int64 { return s.getSegmentSize() })
			if totalSize+s.getSegmentSize() > int64(Params.DataCoordCfg.SegmentExpansionRate.GetAsFloat()*float64(expectedSize)) {
				continue
			}
			buckets[bidx] = append(buckets[bidx], s)

			small = append(small[:i], small[i+1:]...)
			break
		}
	}

	return small
}
