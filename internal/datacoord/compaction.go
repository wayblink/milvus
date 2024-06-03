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

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/lock"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

//go:generate mockery --name=compactionPlanContext --structname=MockCompactionPlanContext --output=./  --filename=mock_compaction_plan_context.go --with-expecter --inpackage
type compactionPlanContext interface {
	start()
	stop()
	// enqueueCompaction start to enqueue compaction task and return immediately
	enqueueCompaction(task *datapb.CompactionTask) error
	// isFull return true if the task pool is full
	isFull() bool
	// get compaction tasks by signal id
	getCompactionTasksNumBySignalID(signalID int64) int
	getCompactionInfo(signalID int64) *compactionInfo
	removeTasksByChannel(channel string)
}

var (
	errChannelNotWatched = errors.New("channel is not watched")
	errChannelInBuffer   = errors.New("channel is in buffer")
	errCompactionBusy    = errors.New("compaction task queue is full")
)

var _ compactionPlanContext = (*compactionPlanHandler)(nil)

type compactionInfo struct {
	state        commonpb.CompactionState
	executingCnt int
	completedCnt int
	failedCnt    int
	timeoutCnt   int
	mergeInfos   map[int64]*milvuspb.CompactionMergeInfo
}

type compactionPlanHandler struct {
	mu         lock.RWMutex
	queueTasks map[int64]CompactionTask // planID -> task

	executingMu    lock.RWMutex
	executingTasks map[int64]CompactionTask // planID -> task

	meta      CompactionMeta
	allocator allocator
	chManager ChannelManager
	sessions  SessionManager
	cluster   Cluster

	stopCh   chan struct{}
	stopOnce sync.Once
	stopWg   sync.WaitGroup

	taskNumber *atomic.Int32
}

func (c *compactionPlanHandler) getCompactionInfo(triggerID int64) *compactionInfo {
	var executingCnt int
	var completedCnt int
	var failedCnt int
	var timeoutCnt int
	ret := &compactionInfo{}

	mergeInfos := make(map[int64]*milvuspb.CompactionMergeInfo)
	tasks := c.meta.GetCompactionTasksByTriggerID(triggerID)
	for _, t := range tasks {
		switch t.GetState() {
		case datapb.CompactionTaskState_pipelining, datapb.CompactionTaskState_executing, datapb.CompactionTaskState_meta_saved:
			executingCnt++
		case datapb.CompactionTaskState_completed:
			completedCnt++
		case datapb.CompactionTaskState_failed:
			failedCnt++
		case datapb.CompactionTaskState_timeout:
			timeoutCnt++
		}
		mergeInfos[t.GetPlanID()] = getCompactionMergeInfo(t)
	}

	ret.executingCnt = executingCnt
	ret.completedCnt = completedCnt
	ret.timeoutCnt = timeoutCnt
	ret.failedCnt = failedCnt
	ret.mergeInfos = mergeInfos

	if executingCnt != 0 {
		ret.state = commonpb.CompactionState_Executing
	} else {
		ret.state = commonpb.CompactionState_Completed
	}
	return ret
}

func (c *compactionPlanHandler) getCompactionTasksNumBySignalID(triggerID int64) int {
	cnt := 0
	c.mu.RLock()
	for _, t := range c.queueTasks {
		if t.GetTriggerID() == triggerID {
			cnt += 1
		}
		// if t.GetPlanID()
	}
	cnt += len(c.queueTasks)
	c.mu.RUnlock()
	c.executingMu.RLock()
	for _, t := range c.executingTasks {
		if t.GetTriggerID() == triggerID {
			cnt += 1
		}
	}
	cnt += len(c.queueTasks)
	c.executingMu.RUnlock()
	return cnt
}

func newCompactionPlanHandler(cluster Cluster, sessions SessionManager, cm ChannelManager, meta CompactionMeta, allocator allocator,
) *compactionPlanHandler {
	return &compactionPlanHandler{
		queueTasks:     make(map[int64]CompactionTask),
		chManager:      cm,
		meta:           meta,
		sessions:       sessions,
		allocator:      allocator,
		stopCh:         make(chan struct{}),
		cluster:        cluster,
		executingTasks: make(map[int64]CompactionTask),
		taskNumber:     atomic.NewInt32(0),
	}
}

func (c *compactionPlanHandler) schedule() []CompactionTask {
	c.mu.RLock()
	if len(c.queueTasks) == 0 {
		c.mu.RUnlock()
		return nil
	}
	c.mu.RUnlock()

	l0ChannelExcludes := typeutil.NewSet[string]()
	mixChannelExcludes := typeutil.NewSet[string]()
	// clusterChannelExcludes := typeutil.NewSet[string]()
	mixLabelExcludes := typeutil.NewSet[string]()
	// clusterLabelExcludes := typeutil.NewSet[string]()

	c.executingMu.RLock()
	for _, t := range c.executingTasks {
		switch t.GetPlan().GetType() {
		case datapb.CompactionType_Level0DeleteCompaction:
			l0ChannelExcludes.Insert(t.GetChannel())
		case datapb.CompactionType_MixCompaction:
			mixChannelExcludes.Insert(t.GetChannel())
			mixLabelExcludes.Insert(t.GetLabel())
			// case datapb.CompactionType_ClusteringCompaction:
			//	clusterChannelExcludes.Insert(t.GetChannel())
			//	clusterLabelExcludes.Insert(t.GetLabel())
		}
	}
	c.executingMu.RUnlock()

	var picked []CompactionTask
	c.mu.RLock()
	defer c.mu.RUnlock()
	keys := lo.Keys(c.queueTasks)
	sort.SliceStable(keys, func(i, j int) bool {
		return keys[i] < keys[j]
	})
	for _, planID := range keys {
		t := c.queueTasks[planID]
		switch t.GetType() {
		case datapb.CompactionType_Level0DeleteCompaction:
			if l0ChannelExcludes.Contain(t.GetChannel()) ||
				mixChannelExcludes.Contain(t.GetChannel()) {
				continue
			}
			picked = append(picked, t)
			l0ChannelExcludes.Insert(t.GetChannel())
		case datapb.CompactionType_MixCompaction:
			if l0ChannelExcludes.Contain(t.GetChannel()) {
				continue
			}
			picked = append(picked, t)
			mixChannelExcludes.Insert(t.GetChannel())
			mixLabelExcludes.Insert(t.GetLabel())
			// case datapb.CompactionType_ClusteringCompaction:
			//	if l0ChannelExcludes.Contain(t.GetChannel()) ||
			//		mixLabelExcludes.Contain(t.GetLabel()) ||
			//		clusterLabelExcludes.Contain(t.GetLabel()){
			//		continue
			//	}
			//	picked = append(picked, t)
			//	slot -= 1
			//	clusterChannelExcludes.Insert(t.GetChannel())
			//	clusterLabelExcludes.Insert(t.GetLabel())
		}
	}
	return picked
}

func (c *compactionPlanHandler) start() {
	c.stopWg.Add(3)
	go c.loopSchedule()
	go c.loopCheck()
	go c.loopClean()
}

func (c *compactionPlanHandler) doSchedule() {
	picked := c.schedule()
	if len(picked) > 0 {
		c.executingMu.Lock()
		for _, t := range picked {
			c.executingTasks[t.GetPlanID()] = t
		}
		c.executingMu.Unlock()

		c.mu.Lock()
		for _, t := range picked {
			delete(c.queueTasks, t.GetPlanID())
		}
		c.mu.Unlock()
	}
}

func (c *compactionPlanHandler) loopSchedule() {
	log.Info("compactionPlanHandler start loop schedule")
	defer c.stopWg.Done()

	scheduleTicker := time.NewTicker(3 * time.Second)
	defer scheduleTicker.Stop()
	for {
		select {
		case <-c.stopCh:
			log.Info("compactionPlanHandler quit loop schedule")
			return

		case <-scheduleTicker.C:
			c.doSchedule()
		}
	}
}

func (c *compactionPlanHandler) loopCheck() {
	interval := Params.DataCoordCfg.CompactionCheckIntervalInSeconds.GetAsDuration(time.Second)
	log.Info("compactionPlanHandler start loop check", zap.Any("check result interval", interval))
	defer c.stopWg.Done()
	checkResultTicker := time.NewTicker(interval)
	for {
		select {
		case <-c.stopCh:
			log.Info("compactionPlanHandler quit loop check")
			return

		case <-checkResultTicker.C:
			err := c.checkCompaction()
			if err != nil {
				log.Info("fail to update compaction", zap.Error(err))
			}
		}
	}
}

func (c *compactionPlanHandler) loopClean() {
	defer c.stopWg.Done()
	cleanTicker := time.NewTicker(30 * time.Minute)
	defer cleanTicker.Stop()
	for {
		select {
		case <-c.stopCh:
			log.Info("Compaction handler quit loopClean")
			return
		case <-cleanTicker.C:
			c.Clean()
		}
	}
}

func (c *compactionPlanHandler) Clean() {
	c.cleanCompactionTaskMeta()
}

func (c *compactionPlanHandler) cleanCompactionTaskMeta() {
	// gc clustering compaction tasks
	triggers := c.meta.GetCompactionTasks()
	for _, tasks := range triggers {
		for _, task := range tasks {
			if task.State == datapb.CompactionTaskState_completed || task.State == datapb.CompactionTaskState_cleaned {
				duration := time.Since(time.Unix(task.StartTime, 0)).Seconds()
				if duration > float64(Params.DataCoordCfg.CompactionDropToleranceInSeconds.GetAsDuration(time.Second)) {
					// try best to delete meta
					err := c.meta.DropCompactionTask(task)
					if err != nil {
						log.Warn("fail to drop task", zap.Int64("taskPlanID", task.PlanID), zap.Error(err))
					}
				}
			}
		}
	}
}

func (c *compactionPlanHandler) stop() {
	c.stopOnce.Do(func() {
		close(c.stopCh)
	})
	c.stopWg.Wait()
}

func (c *compactionPlanHandler) removeTasksByChannel(channel string) {
	c.mu.Lock()
	for id, task := range c.queueTasks {
		log.Info("Compaction handler removing tasks by channel",
			zap.String("channel", channel), zap.Any("id", id), zap.Any("task_channel", task.GetChannel()))
		if task.GetChannel() == channel {
			log.Info("Compaction handler removing tasks by channel",
				zap.String("channel", channel),
				zap.Int64("planID", task.GetPlanID()),
				zap.Int64("node", task.GetNodeID()),
			)
			delete(c.queueTasks, id)
			c.taskNumber.Dec()
		}
	}
	c.mu.Unlock()
	c.executingMu.Lock()
	for id, task := range c.executingTasks {
		log.Info("Compaction handler removing tasks by channel",
			zap.String("channel", channel), zap.Any("id", id), zap.Any("task_channel", task.GetChannel()))
		if task.GetChannel() == channel {
			log.Info("Compaction handler removing tasks by channel",
				zap.String("channel", channel),
				zap.Int64("planID", task.GetPlanID()),
				zap.Int64("node", task.GetNodeID()),
			)
			delete(c.executingTasks, id)
			c.taskNumber.Dec()
		}
	}
	c.executingMu.Unlock()
}

func (c *compactionPlanHandler) submitTask(t CompactionTask) {
	c.mu.Lock()
	c.queueTasks[t.GetPlanID()] = t
	c.mu.Unlock()
	c.taskNumber.Add(1)
}

// restoreTask used to restore Task from etcd
func (c *compactionPlanHandler) restoreTask(t CompactionTask) {
	c.executingMu.Lock()
	c.executingTasks[t.GetPlanID()] = t
	c.executingMu.Unlock()
	c.taskNumber.Add(1)
}

// getCompactionTask return compaction
func (c *compactionPlanHandler) getCompactionTask(planID int64) CompactionTask {
	c.mu.RLock()
	t, ok := c.queueTasks[planID]
	if ok {
		c.mu.RUnlock()
		return t
	}
	c.mu.RUnlock()
	c.executingMu.RLock()
	t, ok = c.executingTasks[planID]
	if ok {
		c.executingMu.RUnlock()
		return t
	}
	c.executingMu.RUnlock()
	return t
}

func (c *compactionPlanHandler) enqueueCompaction(task *datapb.CompactionTask) error {
	log := log.With(zap.Int64("planID", task.GetPlanID()), zap.Int64("collectionID", task.GetCollectionID()), zap.String("type", task.GetType().String()))
	if c.isFull() {
		return errCompactionBusy
	}
	// TODO change to set this on scheduling task
	exist, succeed := c.checkAndSetSegmentsCompacting(task)
	if !exist {
		return merr.WrapErrIllegalCompactionPlan("segment not exist")
	}
	if !succeed {
		return merr.WrapErrCompactionPlanConflict("segment is compacting")
	}

	// TODO change to set this on scheduling task
	t := c.createCompactTask(task)
	if t == nil {
		return merr.WrapErrIllegalCompactionPlan("illegal compaction type")
	}
	t.SetTask(t.ShadowClone(setStartTime(time.Now().Unix())))
	err := t.SaveTaskMeta()
	if err != nil {
		return err
	}
	_, span := otel.Tracer(typeutil.DataCoordRole).Start(context.Background(), fmt.Sprintf("Compaction-%s", task.GetType()))
	t.SetSpan(span)

	c.submitTask(t)
	log.Info("Compaction plan submitted")
	return nil
}

// set segments compacting, one segment can only participate one compactionTask
func (c *compactionPlanHandler) createCompactTask(t *datapb.CompactionTask) CompactionTask {
	var task CompactionTask
	switch t.GetType() {
	case datapb.CompactionType_MixCompaction:
		task = &mixCompactionTask{
			CompactionTask: t,
			meta:           c.meta,
			sessions:       c.sessions,
		}
	case datapb.CompactionType_Level0DeleteCompaction:
		task = &l0CompactionTask{
			CompactionTask: t,
			meta:           c.meta,
			sessions:       c.sessions,
		}
	}
	return task
}

// set segments compacting, one segment can only participate one compactionTask
func (c *compactionPlanHandler) setSegmentsCompacting(task CompactionTask, compacting bool) {
	for _, segmentID := range task.GetInputSegments() {
		c.meta.SetSegmentCompacting(segmentID, compacting)
	}
}

func (c *compactionPlanHandler) checkAndSetSegmentsCompacting(task *datapb.CompactionTask) (bool, bool) {
	return c.meta.CheckAndSetSegmentsCompacting(task.GetInputSegments())
}

func (c *compactionPlanHandler) assignNodeIDs(tasks []CompactionTask) {
	slots := c.cluster.QuerySlots()
	if len(slots) == 0 {
		return
	}

	for _, t := range tasks {
		nodeID := c.pickAnyNode(slots)
		if nodeID == NullNodeID {
			log.Info("cannot find datanode for compaction task",
				zap.Int64("planID", t.GetPlanID()), zap.String("vchannel", t.GetChannel()))
			continue
		}
		err := t.SetNodeID(nodeID)
		if err != nil {
			log.Info("compactionHandler assignNodeID failed",
				zap.Int64("planID", t.GetPlanID()), zap.String("vchannel", t.GetChannel()), zap.Error(err))
		} else {
			log.Info("compactionHandler assignNodeID success",
				zap.Int64("planID", t.GetPlanID()), zap.String("vchannel", t.GetChannel()), zap.Any("nodeID", nodeID))
		}
	}
}

func (c *compactionPlanHandler) checkCompaction() error {
	// Get executing executingTasks before GetCompactionState from DataNode to prevent false failure,
	//  for DC might add new task while GetCompactionState.

	var needAssignIDTasks []CompactionTask
	c.executingMu.RLock()
	for _, t := range c.executingTasks {
		if t.NeedReAssignNodeID() {
			needAssignIDTasks = append(needAssignIDTasks, t)
		}
	}
	c.executingMu.RUnlock()
	if len(needAssignIDTasks) > 0 {
		c.assignNodeIDs(needAssignIDTasks)
	}

	var finishedTasks []CompactionTask
	c.executingMu.RLock()
	for _, t := range c.executingTasks {
		finished := t.Process()
		if finished {
			finishedTasks = append(finishedTasks, t)
		}
	}
	c.executingMu.RUnlock()

	// delete all finished
	c.executingMu.Lock()
	for _, t := range finishedTasks {
		delete(c.executingTasks, t.GetPlanID())
	}
	c.executingMu.Unlock()
	c.taskNumber.Add(-int32(len(finishedTasks)))
	return nil
}

func (c *compactionPlanHandler) pickAnyNode(nodeSlots map[int64]int64) int64 {
	var (
		nodeID   int64 = NullNodeID
		maxSlots int64 = -1
	)
	for id, slots := range nodeSlots {
		if slots > 0 && slots > maxSlots {
			nodeID = id
			maxSlots = slots
		}
	}
	return nodeID
}

func (c *compactionPlanHandler) pickShardNode(nodeSlots map[int64]int64, t CompactionTask) int64 {
	nodeID, err := c.chManager.FindWatcher(t.GetChannel())
	if err != nil {
		log.Info("failed to find watcher", zap.Int64("planID", t.GetPlanID()), zap.Error(err))
		return NullNodeID
	}

	if nodeSlots[nodeID] > 0 {
		return nodeID
	}
	return NullNodeID
}

// isFull return true if the task pool is full
func (c *compactionPlanHandler) isFull() bool {
	return c.getTaskCount() >= Params.DataCoordCfg.CompactionMaxParallelTasks.GetAsInt()
}

func (c *compactionPlanHandler) getTaskCount() int {
	return int(c.taskNumber.Load())
}

func (c *compactionPlanHandler) getTasksByState(state datapb.CompactionTaskState) []CompactionTask {
	c.mu.RLock()
	defer c.mu.RUnlock()
	tasks := make([]CompactionTask, 0, len(c.queueTasks))
	for _, t := range c.queueTasks {
		if t.GetState() == state {
			tasks = append(tasks, t)
		}
	}
	return tasks
}

var (
	ioPool         *conc.Pool[any]
	ioPoolInitOnce sync.Once
)

func initIOPool() {
	capacity := Params.DataNodeCfg.IOConcurrency.GetAsInt()
	if capacity > 32 {
		capacity = 32
	}
	// error only happens with negative expiry duration or with negative pre-alloc size.
	ioPool = conc.NewPool[any](capacity)
}

func getOrCreateIOPool() *conc.Pool[any] {
	ioPoolInitOnce.Do(initIOPool)
	return ioPool
}
