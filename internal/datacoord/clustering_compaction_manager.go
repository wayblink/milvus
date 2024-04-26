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
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/logutil"
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

func (t *ClusteringCompactionManager) submit(job *ClusteringCompactionJob) error {
	log.Info("Insert clustering compaction job", zap.Int64("tiggerID", job.triggerID), zap.Int64("collectionID", job.collectionID))
	return t.saveJob(job)
}

func (t *ClusteringCompactionManager) getByTriggerId(triggerID int64) *ClusteringCompactionJob {
	clusteringInfos := t.meta.GetClusteringCompactionJobsByTriggerID(triggerID)
	// should be one
	if len(clusteringInfos) == 0 {
		return nil
	}
	return clusteringInfos[0]
}

func (t *ClusteringCompactionManager) getCompactionJobState(compactionJob *ClusteringCompactionJob) (state commonpb.CompactionState, executingCnt, completedCnt, failedCnt, timeoutCnt int) {
	tasks := make([]*compactionTask, 0)
	plans := compactionJob.subPlans
	for _, plan := range plans {
		task := t.compactionHandler.getCompaction(plan.GetPlanID())
		if task == nil {
			continue
		}
		tasks = append(tasks, task)
		switch task.state {
		case pipelining:
			executingCnt++
		case executing:
			executingCnt++
		case completed:
			completedCnt++
		case failed:
			failedCnt++
		case timeout:
			timeoutCnt++
		}
	}

	compactionState := compactionTaskState(compactionJob.state)
	if compactionState == pipelining || compactionState == executing {
		state = commonpb.CompactionState_Executing
	} else {
		state = commonpb.CompactionState_Completed
	}
	return
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
			err := t.checkAllJobState()
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
	// gc clustering compaction jobs
	jobs := t.GetAllJobs()
	log.Debug("clustering compaction job meta", zap.Int("len", len(jobs)))
	for _, job := range jobs {
		if job.state == clustering_completed || job.state == clustering_failed || job.state == clustering_timeout {
			if time.Since(tsoutil.PhysicalTime(job.startTime)) > Params.DataCoordCfg.ClusteringCompactionDropTolerance.GetAsDuration(time.Second) {
				// skip handle this error, try best to delete meta
				err := t.dropJob(job)
				if err != nil {
					return err
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

func (t *ClusteringCompactionManager) checkAllJobState() error {
	jobs := t.GetAllJobs()
	for _, job := range jobs {
		err := t.checkJobState(job)
		if err != nil {
			log.Error("fail in check job state", zap.Int64("jobID", job.triggerID), zap.Int64("collectionID", job.collectionID), zap.Error(err))
			job.state = clustering_failed
			t.saveJob(job)
		}
	}
	return nil
}

func (t *ClusteringCompactionManager) checkJobState(job *ClusteringCompactionJob) error {
	if job.state == clustering_completed || job.state == clustering_failed || job.state == clustering_timeout {
		return nil
	}
	coll, err := t.handler.GetCollection(t.ctx, job.collectionID)
	if err != nil {
		log.Warn("fail to get collection", zap.Int64("collectionID", job.collectionID), zap.Error(err))
		return err
	}
	if coll == nil {
		log.Warn("collection not found, it may be dropped, stop clustering compaction job", zap.Int64("collectionID", job.collectionID))
		job.state = clustering_failed
		return t.saveJob(job)
	}

	// pre submit analyze tasks to save some alloc RPC
	if job.state == clustering_pipelining {
		if typeutil.IsVectorType(job.clusteringKeyType) {
			err := t.submitToAnalyze(job)
			if err != nil {
				log.Warn("fail to submit analyze task", zap.Int64("jobID", job.triggerID), zap.Error(err))
				return err
			}
		} else {
			for idx, plan := range job.subPlans {
				err := t.submitToCompact(job, idx, plan)
				if err != nil {
					log.Warn("fail to submit compaction task", zap.Int64("jobID", job.triggerID), zap.Error(err))
					return err
				}
			}
		}
		job.state = clustering_executing
		err := t.saveJob(job)
		return err
	}

	var pipeliningCnt, analyzingCnt, analyzeFailedCnt, executingCnt, completedCnt, indexedCnt, failedCnt, timeoutCnt int
	for idx, plan := range job.subPlans {
		// check analyze task state, and submit to compact after analyze finished
		if typeutil.IsVectorType(job.clusteringKeyType) && plan.GetAnalyzeTaskId() != 0 && plan.GetAnalyzeTaskVersion() == 0 {
			analyzeTask := t.meta.analyzeMeta.GetTask(plan.GetAnalyzeTaskId())
			log.Info("check analysis task state", zap.Int64("id", plan.GetAnalyzeTaskId()), zap.String("state", analyzeTask.State.String()))
			switch analyzeTask.State {
			case indexpb.JobState_JobStateFinished:
				plan.AnalyzeTaskVersion = analyzeTask.Version
				err := t.submitToCompact(job, idx, plan)
				if err != nil {
					return err
				}
				executingCnt++
			case indexpb.JobState_JobStateFailed:
				log.Warn("analyze task fail", zap.Int64("analyzeID", plan.GetAnalyzeTaskId()))
				// todo rethinking all the error flow
				return errors.New(analyzeTask.FailReason)
				analyzeFailedCnt++
			default:
				analyzingCnt++
			}
			continue
		}

		// check compaction state
		switch compactionTaskState(plan.GetState()) {
		case pipelining, executing:
			compactionTask := t.compactionHandler.getCompaction(plan.GetPlanID())
			if compactionTask == nil {
				// if one compaction task is lost, mark it as failed, and the clustering compaction will be marked failed as well
				// todo: retry
				log.Warn("compaction task lost", zap.Int64("planID", plan.GetPlanID()))
				return errors.New("compaction task lost")
			}
			log.Info("compaction task state", zap.Int64("planID", compactionTask.plan.PlanID), zap.Int32("state", int32(compactionTask.state)))
			job.subPlans[idx].State = int32(compactionTask.state)
			switch compactionTask.state {
			case pipelining:
				pipeliningCnt++
			case executing:
				executingCnt++
			case completed:
				completedCnt++
			case failed:
				failedCnt++
			case timeout:
				timeoutCnt++
			}
		case completed:
			if plan.ResultIndexed {
				indexedCnt++
				continue
			}
			if len(plan.ResultSegments) == 0 {
				compactionTask := t.compactionHandler.getCompaction(plan.GetPlanID())
				if compactionTask == nil {
					// if one compaction task is lost, mark it as failed, and the clustering compaction will be marked failed as well
					// todo: retry
					log.Warn("compaction task lost", zap.Int64("planID", plan.GetPlanID()))
					return errors.New("compaction task lost")
				}
				segmentIDs := make([]int64, 0)
				for _, seg := range compactionTask.result.Segments {
					segmentIDs = append(segmentIDs, seg.GetSegmentID())
				}
				plan.ResultSegments = segmentIDs
			}

			// wait for segment indexed
			collectionIndexes := t.meta.indexMeta.GetIndexesForCollection(job.collectionID, "")
			indexed := func() bool {
				for _, collectionIndex := range collectionIndexes {
					for _, segmentID := range plan.ResultSegments {
						segmentIndexState := t.meta.indexMeta.GetSegmentIndexState(job.collectionID, segmentID, collectionIndex.IndexID)
						if segmentIndexState.GetState() != commonpb.IndexState_Finished {
							return false
						}
					}
				}
				return true
			}()
			log.Info("check compaction result segments index states", zap.Bool("indexed", indexed), zap.Int64("planID", plan.GetPlanID()), zap.Int64s("segments", plan.ResultSegments))
			if indexed {
				err := t.meta.SavePartitionStatsInfo(&datapb.PartitionStatsInfo{
					CollectionID: job.collectionID,
					PartitionID:  plan.PartitionID,
					VChannel:     plan.GetChannel(),
					Version:      plan.PlanID,
					SegmentIDs:   plan.ResultSegments,
				})
				if err != nil {
					return err
				}

				job.subPlans[idx].State = int32(completed)
				ts, err := t.allocator.allocTimestamp(t.ctx)
				if err != nil {
					return err
				}
				job.endTime = ts
				elapse := tsoutil.PhysicalTime(ts).UnixMilli() - tsoutil.PhysicalTime(job.startTime).UnixMilli()
				job.subPlans[idx].ResultIndexed = true
				log.Info("clustering compaction job elapse", zap.Int64("triggerID", job.triggerID), zap.Int64("collectionID", job.collectionID), zap.Int64("elapse", elapse))
				metrics.DataCoordCompactionLatency.
					WithLabelValues(fmt.Sprint(typeutil.IsVectorType(job.clusteringKeyType)), datapb.CompactionType_ClusteringCompaction.String()).
					Observe(float64(elapse))
				indexedCnt++
			} else {
				completedCnt++
			}
		case failed:
			// todo: retry sub tasks
			job.state = clustering_failed
			failedCnt++
		case timeout:
			// todo: retry sub tasks
			job.state = clustering_timeout
			timeoutCnt++
		}
	}

	if indexedCnt == len(job.subPlans) {
		job.state = clustering_completed
	}
	log.Info("Update clustering compaction job", zap.Int64("tiggerID", job.triggerID), zap.Int64("collectionID", job.collectionID),
		zap.Int("analyzingCnt", analyzingCnt),
		zap.Int("analyzeFailedCnt", analyzeFailedCnt),
		zap.Int("pipeliningCnt", pipeliningCnt),
		zap.Int("executingCnt", executingCnt),
		zap.Int("completedCnt", completedCnt),
		zap.Int("indexedCnt", indexedCnt),
		zap.Int("failedCnt", failedCnt),
		zap.Int("timeoutCnt", timeoutCnt),
	)
	return t.saveJob(job)
}

func (t *ClusteringCompactionManager) generateNewPlans(job *ClusteringCompactionJob) []*datapb.CompactionPlan {
	return nil
}

func (t *ClusteringCompactionManager) submitToAnalyze(job *ClusteringCompactionJob, plan ...*datapb.ClusteringCompactionPlan) error {
	var plans []*datapb.ClusteringCompactionPlan
	if plan != nil {
		plans = plan
	} else {
		plans = job.subPlans
	}
	// pre-alloc ids in batch
	currentID, _, err := t.allocator.allocN(int64(len(plans)))
	if err != nil {
		return err
	}

	for idx, plan := range plans {
		analyzeTaskID := currentID
		currentID++
		job.subPlans[idx].TimeoutInSeconds = Params.DataCoordCfg.ClusteringCompactionTimeoutInSeconds.GetAsInt32()

		// clustering compaction firstly analyze the plan, then decide whether to execute compaction
		newAnalyzeTask := &model.AnalyzeTask{
			CollectionID: job.collectionID,
			PartitionID:  plan.PartitionID,
			FieldID:      job.clusteringKeyID,
			FieldName:    job.clusteringKeyName,
			FieldType:    job.clusteringKeyType,
			SegmentIDs:   plan.InputSegments,
			TaskID:       analyzeTaskID,
		}
		err = t.meta.analyzeMeta.AddAnalyzeTask(newAnalyzeTask)
		if err != nil {
			log.Warn("failed to create analyze task", zap.Int64("planID", plan.PlanID), zap.Error(err))
			return err
		}
		t.analyzeScheduler.enqueue(&analyzeTask{
			taskID: analyzeTaskID,
			taskInfo: &indexpb.AnalyzeResult{
				TaskID: analyzeTaskID,
				State:  indexpb.JobState_JobStateInit,
			},
		})
		log.Info("submit analyze task", zap.Int64("planID", plan.GetPlanID()), zap.Int64("triggerID", job.triggerID), zap.Int64("collectionID", job.collectionID), zap.Int64("id", analyzeTaskID))
		job.subPlans[idx].AnalyzeTaskId = analyzeTaskID
	}
	return t.saveJob(job)
}

func (t *ClusteringCompactionManager) submitToCompact(job *ClusteringCompactionJob, idx int, plan *datapb.ClusteringCompactionPlan) error {
	trigger := &compactionSignal{
		id:           job.triggerID,
		collectionID: job.collectionID,
		partitionID:  plan.PartitionID,
	}

	segments := make([]*SegmentInfo, 0)
	for _, segmentID := range plan.InputSegments {
		segments = append(segments, t.meta.GetSegment(segmentID))
	}
	compactionPlan := segmentsToPlan(segments, datapb.CompactionType_ClusteringCompaction, &compactTime{collectionTTL: time.Duration(plan.CollectionTtl)})
	compactionPlan.PlanID = plan.GetPlanID()
	compactionPlan.StartTime = plan.GetStartTime()
	compactionPlan.TimeoutInSeconds = plan.GetTimeoutInSeconds()
	compactionPlan.Timetravel = plan.GetTimetravel()
	compactionPlan.ClusteringKeyId = job.clusteringKeyID
	compactionPlan.MaxSegmentRows = job.maxSegmentRows
	compactionPlan.PreferSegmentRows = job.preferSegmentRows
	compactionPlan.AnalyzeResultPath = path.Join(metautil.JoinIDPath(plan.AnalyzeTaskId, plan.AnalyzeTaskVersion))
	err := t.compactionHandler.execCompactionPlan(trigger, compactionPlan)
	if err != nil {
		log.Warn("failed to execute compaction plan", zap.Int64("planID", plan.GetPlanID()), zap.Error(err))
		return err
	}
	job.subPlans[idx].State = int32(executing)
	log.Info("send compaction plan to execute", zap.Int64("triggerID", job.triggerID),
		zap.Int64("planID", plan.GetPlanID()),
		zap.Int64("partitionID", plan.PartitionID),
		zap.Int64s("inputSegments", plan.InputSegments))
	return t.saveJob(job)
}

//func (t *ClusteringCompactionManager) shouldDoClusteringCompaction(analyzeResult *indexpb.AnalysisResult) (bool, error) {
//	return true, nil
//}

// IsClusteringCompacting get clustering compaction info by collection id
func (t *ClusteringCompactionManager) getClusteringCompactingJob(collectionID UniqueID) []*ClusteringCompactionJob {
	jobs := t.meta.GetClusteringCompactionJobsByID(collectionID)
	executingJobs := lo.Filter(jobs, func(job *ClusteringCompactionJob, _ int) bool {
		state := compactionTaskState(job.state)
		return state == pipelining || state == executing
	})
	return executingJobs
}

func (t *ClusteringCompactionManager) setSegmentsCompacting(plan *datapb.CompactionPlan, compacting bool) {
	for _, segmentBinlogs := range plan.GetSegmentBinlogs() {
		t.meta.SetSegmentCompacting(segmentBinlogs.GetSegmentID(), compacting)
	}
}

func (t *ClusteringCompactionManager) fillClusteringCompactionPlans(segments []*SegmentInfo, clusteringKeyId int64, compactTime *compactTime) []*datapb.CompactionPlan {
	plan := segmentsToPlan(segments, datapb.CompactionType_ClusteringCompaction, compactTime)
	plan.ClusteringKeyId = clusteringKeyId
	clusteringMaxSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionMaxSegmentSize.GetAsSize()
	clusteringPreferSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionPreferSegmentSize.GetAsSize()
	segmentMaxSize := paramtable.Get().DataCoordCfg.SegmentMaxSize.GetAsInt64() * 1024 * 1024
	plan.MaxSegmentRows = segments[0].MaxRowNum * clusteringMaxSegmentSize / segmentMaxSize
	plan.PreferSegmentRows = segments[0].MaxRowNum * clusteringPreferSegmentSize / segmentMaxSize
	return []*datapb.CompactionPlan{
		plan,
	}
}

// GetAllJobs returns cloned ClusteringCompactionJob from local meta cache
func (t *ClusteringCompactionManager) GetAllJobs() []*ClusteringCompactionJob {
	return t.meta.GetClusteringCompactionJobs()
}

// dropJob drop clustering compaction job in meta
func (t *ClusteringCompactionManager) dropJob(job *ClusteringCompactionJob) error {
	return t.meta.DropClusteringCompactionJob(job)
}

// saveJob update clustering compaction job in meta
func (t *ClusteringCompactionManager) saveJob(job *ClusteringCompactionJob) error {
	return t.meta.SaveClusteringCompactionJob(job)
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

func (t *ClusteringCompactionManager) addCompactionPlans(job *ClusteringCompactionJob, plans []*datapb.CompactionPlan, state compactionTaskState) error {
	currentID, _, err := t.allocator.allocN(int64(len(plans)))
	if err != nil {
		return err
	}

	for _, plan := range plans {
		segmentIDs := make([]int64, 0)
		for _, segment := range plan.SegmentBinlogs {
			segmentIDs = append(segmentIDs, segment.GetSegmentID())
		}
		planID := currentID
		currentID++
		subPlan := &datapb.ClusteringCompactionPlan{
			PlanID:           planID,
			State:            int32(state),
			StartTime:        plan.GetStartTime(),
			TimeoutInSeconds: plan.GetTimeoutInSeconds(),
			Timetravel:       plan.GetTimetravel(),
			Channel:          plan.GetChannel(),
			CollectionTtl:    plan.GetCollectionTtl(),
			TotalRows:        plan.GetTotalRows(),
			PartitionID:      plan.SegmentBinlogs[0].PartitionID,
			InputSegments:    segmentIDs,
		}
		job.subPlans = append(job.subPlans, subPlan)
	}
	return nil
}
