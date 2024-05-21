package datacoord

import (
	"context"
	"sync"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/lock"
	"github.com/milvus-io/milvus/pkg/util/logutil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

type CompactionTriggerType int8

const (
	TriggerTypeLevelZeroViewChange CompactionTriggerType = iota + 1
	TriggerTypeLevelZeroViewIDLE
	TriggerTypeSegmentSizeViewChange
	TriggerTypeClustering
)

const (
	L0CompactionPolicy         string = "L0CompactionPolicy"
	ClusteringCompactionPolicy string = "ClusteringCompactionPolicy"
)

type TriggerManager interface {
	Start()
	Stop()
	ManualTrigger(ctx context.Context, collectionID int64, clusteringCompaction bool) (UniqueID, error)
}

// CompactionTriggerManager registers Triggers to TriggerType
// so that when the certain TriggerType happens, the corresponding triggers can
// trigger the correct compaction plans.
// Trigger types:
// 1. Change of Views
//   - LevelZeroViewTrigger
//   - SegmentSizeViewTrigger
//
// 2. SystemIDLE & schedulerIDLE
// 3. Manual Compaction
type CompactionTriggerManager struct {
	scheduler         Scheduler
	compactionHandler compactionPlanContext // TODO replace with scheduler
	handler           Handler
	allocator         allocator

	view *FullViews
	// todo handle this lock
	viewGuard lock.RWMutex

	meta    *meta
	polices map[string]CompactionPolicy

	closeSig chan struct{}
	closeWg  sync.WaitGroup
}

func NewCompactionTriggerManager(alloc allocator, compactionHandler compactionPlanContext, meta *meta, handler Handler) *CompactionTriggerManager {
	m := &CompactionTriggerManager{
		allocator:         alloc,
		handler:           handler,
		compactionHandler: compactionHandler,
		view: &FullViews{
			collections: make(map[int64]*CollectionView),
		},
		meta:     meta,
		closeSig: make(chan struct{}),
		polices:  make(map[string]CompactionPolicy, 0),
	}
	m.polices[L0CompactionPolicy] = newL0CompactionPolicy(meta, m.view)
	m.polices[ClusteringCompactionPolicy] = newClusteringCompactionPolicy(meta, m.view, m.allocator, m.compactionHandler, m.handler)
	return m
}

func (m *CompactionTriggerManager) Start() {
	m.closeWg.Add(len(m.polices))
	for triggerType, policy := range m.polices {
		log.Info("Start compaction policy", zap.String("triggerType", triggerType))
		go m.startLoop(policy)
	}
}

func (m *CompactionTriggerManager) Close() {
	close(m.closeSig)
	m.closeWg.Wait()
}

func (m *CompactionTriggerManager) startLoop(policy CompactionPolicy) {
	defer logutil.LogPanic()
	defer m.closeWg.Done()

	defer policy.Stop()
	for {
		select {
		case <-m.closeSig:
			log.Info("Compaction View checkLoop quit")
			return
		case <-policy.Ticker().C:
			ctx := context.Background()
			events, err := policy.Trigger(ctx)
			if err != nil {
				log.Warn("Fail to trigger policy", zap.Error(err))
			}
			if len(events) > 0 {
				for triggerType, views := range events {
					m.notify(ctx, triggerType, views)
				}
			}
		}
	}
}

func (m *CompactionTriggerManager) ManualTrigger(ctx context.Context, collectionID int64, clusteringCompaction bool) (UniqueID, error) {
	log.Info("receive manual trigger", zap.Int64("collectionID", collectionID))
	views, triggerID, err := m.polices[ClusteringCompactionPolicy].(*clusteringCompactionPolicy).triggerOneCollection(context.Background(), collectionID, 0, true)
	if err != nil {
		return 0, err
	}
	events := make(map[CompactionTriggerType][]CompactionView, 0)
	events[TriggerTypeClustering] = views
	if len(events) > 0 {
		for triggerType, views := range events {
			m.notify(ctx, triggerType, views)
		}
	}
	return triggerID, nil
}

func (m *CompactionTriggerManager) notify(ctx context.Context, eventType CompactionTriggerType, views []CompactionView) {
	for _, view := range views {
		if m.compactionHandler.isFull() {
			log.RatedInfo(1.0, "Skip trigger compaction for scheduler is full")
			return
		}

		switch eventType {
		case TriggerTypeLevelZeroViewChange:
			log.Debug("Start to trigger a level zero compaction by TriggerTypeLevelZeroViewChange")
			outView, reason := view.Trigger()
			if outView != nil {
				log.Info("Success to trigger a LevelZeroCompaction output view, try to submit",
					zap.String("reason", reason),
					zap.String("output view", outView.String()))
				m.SubmitL0ViewToScheduler(ctx, outView)
			}
		case TriggerTypeLevelZeroViewIDLE:
			log.Debug("Start to trigger a level zero compaction by TriggerTypLevelZeroViewIDLE")
			outView, reason := view.Trigger()
			if outView == nil {
				log.Info("Start to force trigger a level zero compaction by TriggerTypLevelZeroViewIDLE")
				outView, reason = view.ForceTrigger()
			}

			if outView != nil {
				log.Info("Success to trigger a LevelZeroCompaction output view, try to submit",
					zap.String("reason", reason),
					zap.String("output view", outView.String()))
				m.SubmitL0ViewToScheduler(ctx, outView)
			}
		case TriggerTypeClustering:
			log.Debug("Start to trigger a clustering compaction by TriggerTypeClustering")
			outView, reason := view.Trigger()
			if outView != nil {
				log.Info("Success to trigger a ClusteringCompaction output view, try to submit",
					zap.String("reason", reason),
					zap.String("output view", outView.String()))
				m.SubmitClusteringViewToScheduler(ctx, outView)
			}
		}
	}
}

func (m *CompactionTriggerManager) SubmitL0ViewToScheduler(ctx context.Context, view CompactionView) {
	taskID, err := m.allocator.allocID(ctx)
	if err != nil {
		log.Warn("fail to submit compaction view to scheduler because allocate id fail", zap.String("view", view.String()))
		return
	}

	levelZeroSegs := lo.Map(view.GetSegmentsView(), func(segView *SegmentView, _ int) int64 {
		return segView.ID
	})

	task := &datapb.CompactionTask{
		TriggerID:        taskID, // inner trigger, use task id as trigger id
		PlanID:           taskID,
		Type:             datapb.CompactionType_Level0DeleteCompaction,
		InputSegments:    levelZeroSegs,
		Channel:          view.GetGroupLabel().Channel,
		CollectionID:     view.GetGroupLabel().CollectionID,
		PartitionID:      view.GetGroupLabel().PartitionID,
		Pos:              view.(*LevelZeroSegmentsView).earliestGrowingSegmentPos,
		TimeoutInSeconds: Params.DataCoordCfg.CompactionTimeoutInSeconds.GetAsInt32(),
	}

	m.compactionHandler.enqueueCompaction(&defaultCompactionTask{
		CompactionTask: task,
	})
	log.Info("Finish to submit a LevelZeroCompaction plan",
		zap.Int64("taskID", taskID),
		zap.String("type", task.GetType().String()),
	)
}

func (m *CompactionTriggerManager) SubmitClusteringViewToScheduler(ctx context.Context, view CompactionView) {
	taskID, analyzeID, err := m.allocator.allocN(2)
	if err != nil {
		log.Warn("fail to submit compaction view to scheduler because allocate id fail", zap.String("view", view.String()))
		return
	}
	view.GetSegmentsView()
	_, totalRows, maxSegmentRows, preferSegmentRows := summaryClusteringCompactionView(view)
	task := &datapb.CompactionTask{
		PlanID: taskID,
		//TriggerID:          signal.id,
		State:              datapb.CompactionTaskState_init,
		StartTime:          view.(*ClusteringSegmentsView).compactionTime.startTime,
		CollectionTtl:      view.(*ClusteringSegmentsView).compactionTime.collectionTTL.Nanoseconds(),
		TimeoutInSeconds:   Params.DataCoordCfg.ClusteringCompactionTimeoutInSeconds.GetAsInt32(),
		Type:               datapb.CompactionType_ClusteringCompaction,
		CollectionID:       view.GetGroupLabel().CollectionID,
		PartitionID:        view.GetGroupLabel().PartitionID,
		Channel:            view.GetGroupLabel().Channel,
		ClusteringKeyField: view.(*ClusteringSegmentsView).clusteringKeyField,
		InputSegments:      lo.Map(view.GetSegmentsView(), func(segmentView *SegmentView, _ int) int64 { return segmentView.ID }),
		MaxSegmentRows:     maxSegmentRows,
		PreferSegmentRows:  preferSegmentRows,
		TotalRows:          totalRows,
		AnalyzeTaskID:      analyzeID,
	}
	m.compactionHandler.enqueueCompaction(&clusteringCompactionTask{
		CompactionTask: task,
	})
	log.Info("Finish to submit a clustering compaction task",
		zap.Int64("taskID", taskID),
		zap.String("type", task.GetType().String()),
	)
}

func summaryClusteringCompactionView(view CompactionView) (segmentIDs []int64, totalRows, maxSegmentRows, preferSegmentRows int64) {
	for _, s := range view.GetSegmentsView() {
		totalRows += s.NumOfRows
		segmentIDs = append(segmentIDs, s.ID)
	}
	clusteringMaxSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionMaxSegmentSize.GetAsSize()
	clusteringPreferSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionPreferSegmentSize.GetAsSize()
	segmentMaxSize := paramtable.Get().DataCoordCfg.SegmentMaxSize.GetAsInt64() * 1024 * 1024
	maxSegmentRows = view.GetSegmentsView()[0].MaxRowNum * clusteringMaxSegmentSize / segmentMaxSize
	preferSegmentRows = view.GetSegmentsView()[0].MaxRowNum * clusteringPreferSegmentSize / segmentMaxSize
	return
}

// chanPartSegments is an internal result struct, which is aggregates of SegmentInfos with same collectionID, partitionID and channelName
type chanPartSegments struct {
	collectionID UniqueID
	partitionID  UniqueID
	channelName  string
	segments     []*SegmentInfo
}
