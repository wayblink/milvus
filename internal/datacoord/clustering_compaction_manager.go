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
	"path"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/logutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type ClusteringCompactionManager struct {
	ctx               context.Context
	meta              *meta
	allocator         allocator
	compactionHandler compactionPlanContext
	scheduler         Scheduler
	analyzeScheduler  *taskScheduler
	handler           Handler
	quit              chan struct{}
	wg                sync.WaitGroup
	signals           chan *compactionSignal
}

func newClusteringCompactionManager(
	ctx context.Context,
	meta *meta,
	allocator allocator,
	compactionHandler compactionPlanContext,
	analyzeScheduler *taskScheduler,
	handler Handler,
) *ClusteringCompactionManager {
	return &ClusteringCompactionManager{
		ctx:               ctx,
		meta:              meta,
		allocator:         allocator,
		compactionHandler: compactionHandler,
		analyzeScheduler:  analyzeScheduler,
		handler:           handler,
	}
}

func (t *ClusteringCompactionManager) start() {
	t.quit = make(chan struct{})
	t.wg.Add(2)
	go t.startJobCheckLoop()
	go t.startGCLoop()
}

func (t *ClusteringCompactionManager) stop() {
	close(t.quit)
	t.wg.Wait()
}

func (t *ClusteringCompactionManager) submit(tasks []*datapb.CompactionTask) error {
	log.Info("Insert clustering compaction tasks", zap.Int64("triggerID", tasks[0].TriggerId), zap.Int64("collectionID", tasks[0].CollectionId), zap.Int("task_num", len(tasks)))
	currentID, _, err := t.allocator.allocN(int64(2 * len(tasks)))
	if err != nil {
		return err
	}
	for _, task := range tasks {
		task.PlanId = currentID
		currentID++
		task.AnalyzeTaskId = currentID
		currentID++
		err := t.saveTask(task)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *ClusteringCompactionManager) getByTriggerId(triggerID int64) []*datapb.CompactionTask {
	return t.meta.GetClusteringCompactionTasksBySelector(func(task *datapb.CompactionTask) bool {
		return task.TriggerId == triggerID
	})
}

func (t *ClusteringCompactionManager) startJobCheckLoop() {
	defer logutil.LogPanic()
	defer t.wg.Done()
	ticker := time.NewTicker(paramtable.Get().DataCoordCfg.ClusteringCompactionStateCheckInterval.GetAsDuration(time.Second))
	defer ticker.Stop()
	for {
		select {
		case <-t.quit:
			log.Info("clustering compaction loop exit")
			return
		case <-ticker.C:
			err := t.processAllTasks()
			if err != nil {
				log.Warn("unable to triggerClusteringCompaction", zap.Error(err))
			}
			ticker.Reset(paramtable.Get().DataCoordCfg.ClusteringCompactionStateCheckInterval.GetAsDuration(time.Second))
		}
	}
}

func (t *ClusteringCompactionManager) startGCLoop() {
	defer logutil.LogPanic()
	defer t.wg.Done()
	ticker := time.NewTicker(paramtable.Get().DataCoordCfg.ClusteringCompactionGCInterval.GetAsDuration(time.Second))
	defer ticker.Stop()
	for {
		select {
		case <-t.quit:
			log.Info("clustering compaction gc loop exit")
			return
		case <-ticker.C:
			err := t.gc()
			if err != nil {
				log.Warn("fail to gc", zap.Error(err))
			}
			ticker.Reset(paramtable.Get().DataCoordCfg.ClusteringCompactionGCInterval.GetAsDuration(time.Second))
		}
	}
}

func (t *ClusteringCompactionManager) gc() error {
	log.Debug("start gc clustering compaction related meta and files")
	// gc clustering compaction tasks
	collections := t.meta.GetClusteringCompactionTasks()
	for _, collection := range collections {
		for _, trigger := range collection {
			for _, task := range trigger {
				// indexed is the final state of a clustering compaction task
				if task.State == datapb.CompactionTaskState_indexed || task.State == datapb.CompactionTaskState_failed || task.State == datapb.CompactionTaskState_timeout {
					if time.Since(tsoutil.PhysicalTime(task.StartTime)) > Params.DataCoordCfg.ClusteringCompactionDropTolerance.GetAsDuration(time.Second) {
						// skip handle this error, try best to delete meta
						err := t.dropTask(task)
						if err != nil {
							return err
						}
					}
				}
			}
		}
	}
	// gc partition stats
	channelPartitionStatsInfos := make(map[string][]*datapb.PartitionStatsInfo, 0)
	for _, partitionStatsInfo := range t.meta.partitionStatsInfos {
		channel := fmt.Sprintf("%d/%d/%s", partitionStatsInfo.CollectionID, partitionStatsInfo.PartitionID, partitionStatsInfo.VChannel)
		infos, exist := channelPartitionStatsInfos[channel]
		if exist {
			infos = append(infos, partitionStatsInfo)
			channelPartitionStatsInfos[channel] = infos
		} else {
			channelPartitionStatsInfos[channel] = []*datapb.PartitionStatsInfo{partitionStatsInfo}
		}
	}
	log.Debug("channels with PartitionStats meta", zap.Int("len", len(channelPartitionStatsInfos)))

	for channel, infos := range channelPartitionStatsInfos {
		sort.Slice(infos, func(i, j int) bool {
			return infos[i].Version > infos[j].Version
		})
		log.Debug("PartitionStats in channel", zap.String("channel", channel), zap.Int("len", len(infos)))
		if len(infos) > 2 {
			for i := 2; i < len(infos); i++ {
				info := infos[i]
				partitionStatsPath := path.Join(t.meta.chunkManager.RootPath(), common.PartitionStatsPath, metautil.JoinIDPath(info.CollectionID, info.PartitionID), info.GetVChannel(), strconv.FormatInt(info.GetVersion(), 10))
				err := t.meta.chunkManager.Remove(t.ctx, partitionStatsPath)
				log.Debug("remove partition stats file", zap.String("path", partitionStatsPath))
				if err != nil {
					return err
				}
				err = t.meta.DropPartitionStatsInfo(info)
				log.Debug("drop partition stats meta", zap.Any("info", info))
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (t *ClusteringCompactionManager) processAllTasks() error {
	collections := t.meta.GetClusteringCompactionTasks()
	for _, collection := range collections {
		for _, trigger := range collection {
			for _, task := range trigger {
				err := t.processTask(task)
				if err != nil {
					log.Error("fail in process task", zap.Int64("TriggerId", task.TriggerId), zap.Int64("collectionID", task.CollectionId), zap.Int64("planID", task.PlanId), zap.Error(err))
					task.State = datapb.CompactionTaskState_failed
					t.saveTask(task)
				}
			}
		}
	}
	return nil
}

func (t *ClusteringCompactionManager) processTask(task *datapb.CompactionTask) error {
	if task.State == datapb.CompactionTaskState_indexed || task.State == datapb.CompactionTaskState_failed || task.State == datapb.CompactionTaskState_timeout {
		return nil
	}
	coll, err := t.handler.GetCollection(t.ctx, task.GetCollectionId())
	if err != nil {
		log.Warn("fail to get collection", zap.Int64("collectionID", task.GetCollectionId()), zap.Error(err))
		return err
	}
	if coll == nil {
		log.Warn("collection not found, it may be dropped, stop clustering compaction task", zap.Int64("collectionID", task.GetCollectionId()))
		return merr.WrapErrCollectionNotFound(task.GetCollectionId())
	}

	switch task.State {
	case datapb.CompactionTaskState_pipelining:
		return t.processPipeliningTask(task)
	case datapb.CompactionTaskState_compacting:
		return t.processExecutingTask(task)
	case datapb.CompactionTaskState_compacted:
		return t.processCompactedTask(task)
	case datapb.CompactionTaskState_analyzing:
		return t.processAnalyzingTask(task)
	case datapb.CompactionTaskState_analyzed:
		return t.processAnalyzedTask(task)
	case datapb.CompactionTaskState_indexing:
		return t.processIndexingTask(task)
	case datapb.CompactionTaskState_timeout:
	case datapb.CompactionTaskState_failed:
	case datapb.CompactionTaskState_indexed:
		// indexed is the final state of a clustering compaction task
	}
	return nil
}

func (t *ClusteringCompactionManager) processPipeliningTask(task *datapb.CompactionTask) error {
	if typeutil.IsVectorType(task.GetClusteringKeyField().DataType) {
		err := t.submitToAnalyze(task)
		if err != nil {
			log.Warn("fail to submit analyze task", zap.Int64("jobID", task.TriggerId), zap.Error(err))
			return err
		}
	} else {
		err := t.submitToCompact(task)
		if err != nil {
			log.Warn("fail to submit compaction task", zap.Int64("jobID", task.TriggerId), zap.Error(err))
			return err
		}
	}
	return nil
}

func (t *ClusteringCompactionManager) processExecutingTask(task *datapb.CompactionTask) error {
	compactionTask := t.compactionHandler.getCompaction(task.GetPlanId())
	if compactionTask == nil {
		// if one compaction task is lost, mark it as failed, and the clustering compaction will be marked failed as well
		// todo: retry
		log.Warn("compaction task lost", zap.Int64("planID", task.GetPlanId()))
		return errors.New("compaction task lost")
	}
	log.Info("compaction task state", zap.Int64("planID", compactionTask.plan.PlanID), zap.Int32("state", int32(compactionTask.state)))
	task.State = compactionTaskStateV2(compactionTask.state)
	if task.State == datapb.CompactionTaskState_compacted {
		return t.processCompactedTask(task)
	}
	return nil
}

func (t *ClusteringCompactionManager) processCompactedTask(task *datapb.CompactionTask) error {
	if len(task.ResultSegments) == 0 {
		compactionTask := t.compactionHandler.getCompaction(task.GetPlanId())
		if compactionTask == nil {
			// if one compaction task is lost, mark it as failed, and the clustering compaction will be marked failed as well
			// todo: retry
			log.Warn("compaction task lost", zap.Int64("planID", task.GetPlanId()))
			return errors.New("compaction task lost")
		}
		segmentIDs := make([]int64, 0)
		for _, seg := range compactionTask.result.Segments {
			segmentIDs = append(segmentIDs, seg.GetSegmentID())
		}
		task.ResultSegments = segmentIDs
	}

	return t.processIndexingTask(task)
}

func (t *ClusteringCompactionManager) processIndexingTask(task *datapb.CompactionTask) error {
	// wait for segment indexed
	collectionIndexes := t.meta.indexMeta.GetIndexesForCollection(task.GetCollectionId(), "")
	indexed := func() bool {
		for _, collectionIndex := range collectionIndexes {
			for _, segmentID := range task.ResultSegments {
				segmentIndexState := t.meta.indexMeta.GetSegmentIndexState(task.GetCollectionId(), segmentID, collectionIndex.IndexID)
				if segmentIndexState.GetState() != commonpb.IndexState_Finished {
					return false
				}
			}
		}
		return true
	}()
	log.Info("check compaction result segments index states", zap.Bool("indexed", indexed), zap.Int64("planID", task.GetPlanId()), zap.Int64s("segments", task.ResultSegments))
	if indexed {
		err := t.meta.SavePartitionStatsInfo(&datapb.PartitionStatsInfo{
			CollectionID: task.GetCollectionId(),
			PartitionID:  task.GetPartitionId(),
			VChannel:     task.GetChannel(),
			Version:      task.GetPlanId(),
			SegmentIDs:   task.GetResultSegments(),
		})
		if err != nil {
			return err
		}

		task.State = datapb.CompactionTaskState_indexed
		ts, err := t.allocator.allocTimestamp(t.ctx)
		if err != nil {
			return err
		}
		task.EndTime = ts
		elapse := tsoutil.PhysicalTime(ts).UnixMilli() - tsoutil.PhysicalTime(task.StartTime).UnixMilli()
		log.Info("clustering compaction task elapse", zap.Int64("triggerID", task.GetTriggerId()), zap.Int64("collectionID", task.GetCollectionId()), zap.Int64("planID", task.GetPlanId()), zap.Int64("elapse", elapse))
		metrics.DataCoordCompactionLatency.
			WithLabelValues(fmt.Sprint(typeutil.IsVectorType(task.GetClusteringKeyField().DataType)), datapb.CompactionType_ClusteringCompaction.String()).
			Observe(float64(elapse))
	} else {
		task.State = datapb.CompactionTaskState_indexing
	}
	return nil
}

func (t *ClusteringCompactionManager) processAnalyzingTask(task *datapb.CompactionTask) error {
	analyzeTask := t.meta.analyzeMeta.GetTask(task.GetAnalyzeTaskId())
	log.Info("check analyze task state", zap.Int64("id", task.GetAnalyzeTaskId()), zap.String("state", analyzeTask.State.String()))
	switch analyzeTask.State {
	case indexpb.JobState_JobStateFinished:
		if analyzeTask.GetCentroidsFile() == "" && len(analyzeTask.GetOffsetMapping()) == 0 {
			task.State = datapb.CompactionTaskState_analyzed
			task.CentroidFilePath = analyzeTask.GetCentroidsFile()
			task.OffsetMappingFiles = analyzeTask.GetOffsetMapping()
		}
		t.processAnalyzedTask(task)
	case indexpb.JobState_JobStateFailed:
		log.Warn("analyze task fail", zap.Int64("analyzeID", task.GetAnalyzeTaskId()))
		// todo rethinking all the error flow
		return errors.New(analyzeTask.FailReason)
	default:
	}
	return nil
}

func (t *ClusteringCompactionManager) processAnalyzedTask(task *datapb.CompactionTask) error {
	return t.submitToCompact(task)
}

func (t *ClusteringCompactionManager) submitToAnalyze(task *datapb.CompactionTask) error {
	newAnalyzeTask := &indexpb.AnalyzeTask{
		CollectionID: task.GetCollectionId(),
		PartitionID:  task.GetPartitionId(),
		FieldID:      task.GetClusteringKeyField().FieldID,
		FieldName:    task.GetClusteringKeyField().Name,
		FieldType:    task.GetClusteringKeyField().DataType,
		SegmentIDs:   task.GetInputSegments(),
		TaskID:       task.GetAnalyzeTaskId(), // analyze id is pre allocated
		State:        indexpb.JobState_JobStateInit,
	}
	err := t.meta.analyzeMeta.AddAnalyzeTask(newAnalyzeTask)
	if err != nil {
		log.Warn("failed to create analyze task", zap.Int64("planID", task.GetPlanId()), zap.Error(err))
		return err
	}
	t.analyzeScheduler.enqueue(&analyzeTask{
		taskID: task.GetAnalyzeTaskId(),
		taskInfo: &indexpb.AnalyzeResult{
			TaskID: task.GetAnalyzeTaskId(),
			State:  indexpb.JobState_JobStateInit,
		},
	})
	task.State = datapb.CompactionTaskState_analyzing
	log.Info("submit analyze task", zap.Int64("planID", task.GetPlanId()), zap.Int64("triggerID", task.GetTriggerId()), zap.Int64("collectionID", task.GetCollectionId()), zap.Int64("id", task.GetAnalyzeTaskId()))
	return nil
}

func (t *ClusteringCompactionManager) submitToCompact(task *datapb.CompactionTask) error {
	trigger := &compactionSignal{
		id:           task.TriggerId,
		collectionID: task.CollectionId,
		partitionID:  task.PartitionId,
	}

	segments := make([]*SegmentInfo, 0)
	for _, segmentID := range task.InputSegments {
		segments = append(segments, t.meta.GetSegment(segmentID))
	}
	compactionPlan := segmentsToPlan(segments, datapb.CompactionType_ClusteringCompaction, &compactTime{collectionTTL: time.Duration(task.CollectionTtl)})
	compactionPlan.PlanID = task.GetPlanId()
	compactionPlan.StartTime = task.GetStartTime()
	compactionPlan.TimeoutInSeconds = task.GetTimeoutInSeconds()
	compactionPlan.Timetravel = task.GetTimetravel()
	compactionPlan.ClusteringKeyId = task.GetClusteringKeyField().FieldID
	compactionPlan.MaxSegmentRows = task.GetMaxSegmentRows()
	compactionPlan.PreferSegmentRows = task.GetPreferSegmentRows()
	compactionPlan.CentroidFilePath = task.GetCentroidFilePath()
	compactionPlan.OffsetMappingFiles = task.GetOffsetMappingFiles()
	err := t.compactionHandler.execCompactionPlan(trigger, compactionPlan)
	if err != nil {
		log.Warn("failed to execute compaction task", zap.Int64("planID", task.GetPlanId()), zap.Error(err))
		return err
	}
	task.State = datapb.CompactionTaskState_compacting
	log.Info("send compaction task to execute", zap.Int64("triggerID", task.GetTriggerId()),
		zap.Int64("planID", task.GetPlanId()),
		zap.Int64("collectionID", task.GetCollectionId()),
		zap.Int64("partitionID", task.GetPartitionId()),
		zap.Int64s("inputSegments", task.InputSegments))
	return nil
}

func (t *ClusteringCompactionManager) getCompactionJobState(compactionTasks []*datapb.CompactionTask) (state commonpb.CompactionState, executingCnt, completedCnt, failedCnt, timeoutCnt int) {
	for _, task := range compactionTasks {
		if task == nil {
			continue
		}
		switch task.State {
		case datapb.CompactionTaskState_indexed:
			completedCnt++
		case datapb.CompactionTaskState_failed:
			failedCnt++
		case datapb.CompactionTaskState_timeout:
			timeoutCnt++
		default:
			executingCnt++
		}
	}
	if executingCnt != 0 {
		state = commonpb.CompactionState_Executing
	} else {
		state = commonpb.CompactionState_Completed
	}
	return
}

// getClusteringCompactingJobs get clustering compaction info by collection id
func (t *ClusteringCompactionManager) collectionIsClusteringCompacting(collectionID UniqueID) (bool, int64) {
	tasks := t.meta.GetLatestClusteringCompactionTask(collectionID)
	if len(tasks) > 0 {
		state, _, _, _, _ := t.getCompactionJobState(tasks)
		return state == commonpb.CompactionState_Executing, tasks[0].TriggerId
	} else {
		return false, 0
	}
}

func (t *ClusteringCompactionManager) setSegmentsCompacting(plan *datapb.CompactionPlan, compacting bool) {
	for _, segmentBinlogs := range plan.GetSegmentBinlogs() {
		t.meta.SetSegmentCompacting(segmentBinlogs.GetSegmentID(), compacting)
	}
}

func (t *ClusteringCompactionManager) fillClusteringCompactionTask(task *datapb.CompactionTask, segments []*SegmentInfo) {
	var totalRows int64
	segmentIDs := make([]int64, 0)
	for _, s := range segments {
		totalRows += s.GetNumOfRows()
		segmentIDs = append(segmentIDs, s.GetID())
	}
	task.TotalRows = totalRows
	task.InputSegments = segmentIDs
	clusteringMaxSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionMaxSegmentSize.GetAsSize()
	clusteringPreferSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionPreferSegmentSize.GetAsSize()
	segmentMaxSize := paramtable.Get().DataCoordCfg.SegmentMaxSize.GetAsInt64() * 1024 * 1024
	task.MaxSegmentRows = segments[0].MaxRowNum * clusteringMaxSegmentSize / segmentMaxSize
	task.PreferSegmentRows = segments[0].MaxRowNum * clusteringPreferSegmentSize / segmentMaxSize
}

// dropTask drop clustering compaction task in meta
func (t *ClusteringCompactionManager) dropTask(task *datapb.CompactionTask) error {
	return t.meta.DropClusteringCompactionTask(task)
}

// saveTask update clustering compaction task in meta
func (t *ClusteringCompactionManager) saveTask(task *datapb.CompactionTask) error {
	return t.meta.SaveClusteringCompactionTask(task)
}

func triggerCompactionPolicy(ctx context.Context, meta *meta, collectionID int64, partitionID int64, channel string, segments []*SegmentInfo) (bool, error) {
	log := log.With(zap.Int64("collectionID", collectionID), zap.Int64("partitionID", partitionID))
	partitionStatsInfos := meta.ListPartitionStatsInfos(collectionID, partitionID, channel)
	sort.Slice(partitionStatsInfos, func(i, j int) bool {
		return partitionStatsInfos[i].Version > partitionStatsInfos[j].Version
	})

	if len(partitionStatsInfos) == 0 {
		var newDataSize int64 = 0
		for _, seg := range segments {
			newDataSize += seg.getSegmentSize()
		}
		if newDataSize > Params.DataCoordCfg.ClusteringCompactionNewDataSizeThreshold.GetAsSize() {
			log.Info("New data is larger than threshold, do compaction", zap.Int64("newDataSize", newDataSize))
			return true, nil
		}
		log.Info("No partition stats and no enough new data, skip compaction")
		return false, nil
	}

	partitionStats := partitionStatsInfos[0]
	version := partitionStats.Version
	pTime, _ := tsoutil.ParseTS(uint64(version))
	if time.Since(pTime) < Params.DataCoordCfg.ClusteringCompactionMinInterval.GetAsDuration(time.Second) {
		log.Debug("Too short time before last clustering compaction, skip compaction")
		return false, nil
	}
	if time.Since(pTime) > Params.DataCoordCfg.ClusteringCompactionMaxInterval.GetAsDuration(time.Second) {
		log.Debug("It is a long time after last clustering compaction, do compaction")
		return true, nil
	}

	var compactedSegmentSize int64 = 0
	var uncompactedSegmentSize int64 = 0
	for _, seg := range segments {
		if lo.Contains(partitionStats.SegmentIDs, seg.ID) {
			compactedSegmentSize += seg.getSegmentSize()
		} else {
			uncompactedSegmentSize += seg.getSegmentSize()
		}
	}

	// ratio based
	//ratio := float64(uncompactedSegmentSize) / float64(compactedSegmentSize)
	//if ratio > Params.DataCoordCfg.ClusteringCompactionNewDataRatioThreshold.GetAsFloat() {
	//	log.Info("New data is larger than threshold, do compaction", zap.Float64("ratio", ratio))
	//	return true, nil
	//}
	//log.Info("New data is smaller than threshold, skip compaction", zap.Float64("ratio", ratio))
	//return false, nil

	// size based
	if uncompactedSegmentSize > Params.DataCoordCfg.ClusteringCompactionNewDataSizeThreshold.GetAsSize() {
		log.Info("New data is larger than threshold, do compaction", zap.Int64("newDataSize", uncompactedSegmentSize))
		return true, nil
	}
	log.Info("New data is smaller than threshold, skip compaction", zap.Int64("newDataSize", uncompactedSegmentSize))
	return false, nil
}

func compactionTaskStateV2(state compactionTaskState) datapb.CompactionTaskState {
	switch state {
	case pipelining:
		return datapb.CompactionTaskState_pipelining
	case executing:
		return datapb.CompactionTaskState_compacting
	case completed:
		return datapb.CompactionTaskState_compacted
	case timeout:
		return datapb.CompactionTaskState_timeout
	case failed:
		return datapb.CompactionTaskState_failed
	default:
		return datapb.CompactionTaskState_unknown
	}
}
