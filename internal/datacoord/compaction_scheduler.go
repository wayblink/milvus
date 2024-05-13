package datacoord

import (
	"fmt"

	"github.com/samber/lo"
	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/lock"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type Scheduler interface {
	Submit(t ...CompactionTask)
	Schedule() []CompactionTask
	Finish(nodeID int64, task CompactionTask)
	GetTaskCount() int
	LogStatus()

	// Start()
	// Stop()
	// IsFull() bool
	// GetCompactionTasksBySignalID(signalID int64) []defaultCompactionTask
}

type CompactionScheduler struct {
	taskNumber *atomic.Int32

	queuingTasks  []CompactionTask
	parallelTasks map[int64][]CompactionTask // parallel by nodeID
	taskGuard     lock.RWMutex

	planHandler *compactionPlanHandler
}

var _ Scheduler = (*CompactionScheduler)(nil)

func NewCompactionScheduler() *CompactionScheduler {
	return &CompactionScheduler{
		taskNumber:    atomic.NewInt32(0),
		queuingTasks:  make([]CompactionTask, 0),
		parallelTasks: make(map[int64][]CompactionTask),
	}
}

func (s *CompactionScheduler) Submit(tasks ...CompactionTask) {
	s.taskGuard.Lock()
	s.queuingTasks = append(s.queuingTasks, tasks...)
	s.taskGuard.Unlock()

	s.taskNumber.Add(int32(len(tasks)))
	lo.ForEach(tasks, func(t CompactionTask, _ int) {
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(t.GetNodeID()), t.GetState().String(), metrics.Pending).Inc()
	})
	s.LogStatus()
}

// Schedule pick 1 or 0 tasks for 1 node
func (s *CompactionScheduler) Schedule() []CompactionTask {
	nodeTasks := make(map[int64][]CompactionTask) // nodeID

	s.taskGuard.Lock()
	defer s.taskGuard.Unlock()
	for _, task := range s.queuingTasks {
		if _, ok := nodeTasks[task.GetNodeID()]; !ok {
			nodeTasks[task.GetNodeID()] = make([]CompactionTask, 0)
		}

		nodeTasks[task.GetNodeID()] = append(nodeTasks[task.GetNodeID()], task)
	}

	executable := make(map[int64]CompactionTask)

	pickPriorPolicy := func(tasks []CompactionTask, exclusiveChannels []string, executing []string) CompactionTask {
		for _, task := range tasks {
			if lo.Contains(exclusiveChannels, task.GetChannel()) {
				continue
			}

			if task.GetType() == datapb.CompactionType_Level0DeleteCompaction {
				// Channel of LevelZeroCompaction task with no executing compactions
				if !lo.Contains(executing, task.GetChannel()) {
					return task
				}

				// Don't schedule any tasks for channel with LevelZeroCompaction task
				// when there're executing compactions
				exclusiveChannels = append(exclusiveChannels, task.GetChannel())
				continue
			}

			return task
		}

		return nil
	}

	// pick 1 or 0 task for 1 node
	for node, tasks := range nodeTasks {
		parallel := s.parallelTasks[node]
		if len(parallel) >= calculateParallel() {
			log.Info("Compaction parallel in DataNode reaches the limit", zap.Int64("nodeID", node), zap.Int("parallel", len(parallel)))
			continue
		}

		var (
			executing         = typeutil.NewSet[string]()
			channelsExecPrior = typeutil.NewSet[string]()
		)
		for _, t := range parallel {
			executing.Insert(t.GetChannel())
			if t.GetType() == datapb.CompactionType_Level0DeleteCompaction {
				channelsExecPrior.Insert(t.GetChannel())
			}
		}

		picked := pickPriorPolicy(tasks, channelsExecPrior.Collect(), executing.Collect())
		if picked != nil {
			executable[node] = picked
		}
	}

	var pickPlans []int64
	for node, task := range executable {
		pickPlans = append(pickPlans, task.GetPlanID())
		if _, ok := s.parallelTasks[node]; !ok {
			s.parallelTasks[node] = []CompactionTask{task}
		} else {
			s.parallelTasks[node] = append(s.parallelTasks[node], task)
		}
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(node), task.GetType().String(), metrics.Executing).Inc()
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(node), task.GetType().String(), metrics.Pending).Dec()
	}

	s.queuingTasks = lo.Filter(s.queuingTasks, func(t CompactionTask, _ int) bool {
		return !lo.Contains(pickPlans, t.GetPlanID())
	})

	// clean parallelTasks with nodes of no running tasks
	for node, tasks := range s.parallelTasks {
		if len(tasks) == 0 {
			delete(s.parallelTasks, node)
		}
	}

	return lo.Values(executable)
}

func (s *CompactionScheduler) Finish(nodeID UniqueID, task CompactionTask) {
	planID := task.GetPlanID()
	log := log.With(zap.Int64("planID", planID), zap.Int64("nodeID", nodeID))

	s.taskGuard.Lock()
	if parallel, ok := s.parallelTasks[nodeID]; ok {
		tasks := lo.Filter(parallel, func(t CompactionTask, _ int) bool {
			return t.GetPlanID() != planID
		})
		s.parallelTasks[nodeID] = tasks
		s.taskNumber.Dec()
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(nodeID), task.GetType().String(), metrics.Executing).Dec()
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(nodeID), task.GetType().String(), metrics.Done).Inc()
		log.Info("Compaction scheduler remove task from executing")
	}

	filtered := lo.Filter(s.queuingTasks, func(t CompactionTask, _ int) bool {
		return t.GetPlanID() != planID
	})
	if len(filtered) < len(s.queuingTasks) {
		s.queuingTasks = filtered
		s.taskNumber.Dec()
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(nodeID), task.GetType().String(), metrics.Pending).Dec()
		metrics.DataCoordCompactionTaskNum.
			WithLabelValues(fmt.Sprint(nodeID), task.GetType().String(), metrics.Done).Inc()
		log.Info("Compaction scheduler remove task from queue")
	}

	s.taskGuard.Unlock()
	s.LogStatus()
}

func (s *CompactionScheduler) LogStatus() {
	s.taskGuard.RLock()
	defer s.taskGuard.RUnlock()

	if s.GetTaskCount() > 0 {
		waiting := lo.Map(s.queuingTasks, func(t CompactionTask, _ int) int64 {
			return t.GetPlanID()
		})

		var executing []int64
		for _, tasks := range s.parallelTasks {
			executing = append(executing, lo.Map(tasks, func(t CompactionTask, _ int) int64 {
				return t.GetPlanID()
			})...)
		}

		log.Info("Compaction scheduler status", zap.Int64s("waiting", waiting), zap.Int64s("executing", executing))
	}
}

func (s *CompactionScheduler) GetTaskCount() int {
	return int(s.taskNumber.Load())
}
