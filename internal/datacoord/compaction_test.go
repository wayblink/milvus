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
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

func TestCompactionPlanHandlerSuite(t *testing.T) {
	suite.Run(t, new(CompactionPlanHandlerSuite))
}

type CompactionPlanHandlerSuite struct {
	suite.Suite

	mockMeta    *MockCompactionMeta
	mockAlloc   *NMockAllocator
	mockCm      *MockChannelManager
	mockSessMgr *MockSessionManager
	handler     *compactionPlanHandler
	cluster     Cluster
}

func (s *CompactionPlanHandlerSuite) SetupTest() {
	s.mockMeta = NewMockCompactionMeta(s.T())
	s.mockAlloc = NewNMockAllocator(s.T())
	s.mockCm = NewMockChannelManager(s.T())
	s.mockSessMgr = NewMockSessionManager(s.T())
	s.cluster = NewMockCluster(s.T())
	s.handler = newCompactionPlanHandler(s.cluster, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc)
}

func (s *CompactionPlanHandlerSuite) TestScheduleEmpty() {
	s.SetupTest()
	s.handler.schedule()
	s.Empty(s.handler.executingTasks)
}

func (s *CompactionPlanHandlerSuite) generateInitTasksForSchedule() {
	ret := []CompactionTask{
		&mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  1,
				Type:    datapb.CompactionType_MixCompaction,
				State:   datapb.CompactionTaskState_pipelining,
				Channel: "ch-1",
				NodeID:  100,
			},
			plan:     &datapb.CompactionPlan{PlanID: 1, Channel: "ch-1", Type: datapb.CompactionType_MixCompaction},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		&mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  2,
				Type:    datapb.CompactionType_MixCompaction,
				State:   datapb.CompactionTaskState_pipelining,
				Channel: "ch-1",
				NodeID:  100,
			},
			plan:     &datapb.CompactionPlan{PlanID: 2, Channel: "ch-1", Type: datapb.CompactionType_MixCompaction},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		&mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  3,
				Type:    datapb.CompactionType_MixCompaction,
				State:   datapb.CompactionTaskState_pipelining,
				Channel: "ch-2",
				NodeID:  101,
			},
			plan:     &datapb.CompactionPlan{PlanID: 3, Channel: "ch-2", Type: datapb.CompactionType_MixCompaction},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		&mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  4,
				Type:    datapb.CompactionType_Level0DeleteCompaction,
				State:   datapb.CompactionTaskState_pipelining,
				Channel: "ch-3",
				NodeID:  102,
			},
			plan:     &datapb.CompactionPlan{PlanID: 4, Channel: "ch-3", Type: datapb.CompactionType_Level0DeleteCompaction},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
	}
	for _, t := range ret {
		s.handler.restoreTask(t)
	}
}

func (s *CompactionPlanHandlerSuite) TestScheduleNodeWith1ParallelTask() {
	// dataNode 101's paralleTasks has 1 task running, not L0 task
	tests := []struct {
		description string
		tasks       []CompactionTask
		expectedOut []UniqueID // planID
	}{
		{"with L0 tasks diff channel", []CompactionTask{
			&l0CompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  10,
					Type:    datapb.CompactionType_Level0DeleteCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-10",
					NodeID:  101,
				},
				plan:     &datapb.CompactionPlan{PlanID: 10, Channel: "ch-10", Type: datapb.CompactionType_Level0DeleteCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  11,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  101,
				},
				plan:     &datapb.CompactionPlan{PlanID: 11, Channel: "ch-11", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
		}, []UniqueID{10, 11}},
		{"with L0 tasks same channel", []CompactionTask{
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  11,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  101,
				},
				plan:     &datapb.CompactionPlan{PlanID: 11, Channel: "ch-11", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&l0CompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  10,
					Type:    datapb.CompactionType_Level0DeleteCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  101,
				},
				plan:     &datapb.CompactionPlan{PlanID: 10, Channel: "ch-11", Type: datapb.CompactionType_Level0DeleteCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
		}, []UniqueID{10}},
		{"without L0 tasks", []CompactionTask{
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  14,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-2",
					NodeID:  101,
				},
				plan:     &datapb.CompactionPlan{PlanID: 14, Channel: "ch-2", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  13,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  101,
				},
				plan:     &datapb.CompactionPlan{PlanID: 13, Channel: "ch-11", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
		}, []UniqueID{13, 14}},
		{"empty tasks", []CompactionTask{}, []UniqueID{}},
	}

	for _, test := range tests {
		s.Run(test.description, func() {
			s.SetupTest()
			s.generateInitTasksForSchedule()
			s.Require().Equal(4, s.handler.getTaskCount())
			// submit the testing tasks
			for _, t := range test.tasks {
				s.handler.submitTask(t)
			}
			s.Equal(4+len(test.tasks), s.handler.getTaskCount())

			gotTasks := s.handler.schedule()
			s.Equal(test.expectedOut, lo.Map(gotTasks, func(t CompactionTask, _ int) int64 {
				return t.GetPlanID()
			}))

			s.Equal(4+len(test.tasks), s.handler.getTaskCount())
		})
	}
}

func (s *CompactionPlanHandlerSuite) TestScheduleNodeWithL0Executing() {
	// dataNode 102's paralleTasks has running L0 tasks
	// nothing of the same channel will be able to schedule
	tests := []struct {
		description string
		tasks       []CompactionTask
		expectedOut []UniqueID // planID
	}{
		{"with L0 tasks diff channel", []CompactionTask{
			&l0CompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  10,
					Type:    datapb.CompactionType_Level0DeleteCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-10",
					NodeID:  102,
				},
				// plan:     &datapb.CompactionPlan{PlanID: 10, Channel: "ch-10", Type: datapb.CompactionType_Level0DeleteCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  11,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  102,
				},
				// plan:     &datapb.CompactionPlan{PlanID: 11, Channel: "ch-11", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
		}, []UniqueID{10, 11}},
		{"with L0 tasks same channel", []CompactionTask{
			&l0CompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  10,
					Type:    datapb.CompactionType_Level0DeleteCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  102,
				},
				plan:     &datapb.CompactionPlan{PlanID: 10, Channel: "ch-3", Type: datapb.CompactionType_Level0DeleteCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  11,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-11",
					NodeID:  102,
				},
				plan:     &datapb.CompactionPlan{PlanID: 11, Channel: "ch-11", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  13,
					Type:    datapb.CompactionType_MixCompaction,
					State:   datapb.CompactionTaskState_pipelining,
					Channel: "ch-3",
					NodeID:  102,
				},
				plan:     &datapb.CompactionPlan{PlanID: 13, Channel: "ch-3", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
		}, []UniqueID{10, 13}},
		{"without L0 tasks", []CompactionTask{
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  14,
					Type:    datapb.CompactionType_MixCompaction,
					Channel: "ch-3",
					NodeID:  102,
				},
				plan:     &datapb.CompactionPlan{PlanID: 14, Channel: "ch-3", Type: datapb.CompactionType_MixCompaction},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
			&mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					PlanID:  13,
					Type:    datapb.CompactionType_MixCompaction,
					Channel: "ch-11",
					NodeID:  102,
				},
				sessions: s.mockSessMgr,
				meta:     s.mockMeta,
			},
		}, []UniqueID{13, 14}},
		{"empty tasks", []CompactionTask{}, []UniqueID{}},
	}

	for _, test := range tests {
		s.Run(test.description, func() {
			s.SetupTest()
			s.Require().Equal(0, s.handler.getTaskCount())

			// submit the testing tasks
			for _, t := range test.tasks {
				s.handler.submitTask(t)
			}
			s.Equal(len(test.tasks), s.handler.getTaskCount())

			gotTasks := s.handler.schedule()
			s.Equal(test.expectedOut, lo.Map(gotTasks, func(t CompactionTask, _ int) int64 {
				return t.GetPlanID()
			}))
		})
	}
}

func (s *CompactionPlanHandlerSuite) TestPickAnyNode() {
	s.SetupTest()
	nodeSlots := map[int64]int64{
		100: 2,
		101: 3,
	}
	node := s.handler.pickAnyNode(nodeSlots)
	s.Equal(int64(101), node)

	node = s.handler.pickAnyNode(map[int64]int64{})
	s.Equal(int64(NullNodeID), node)
}

func (s *CompactionPlanHandlerSuite) TestPickShardNode() {
	s.SetupTest()
	nodeSlots := map[int64]int64{
		100: 2,
		101: 6,
	}

	t1 := &mixCompactionTask{
		CompactionTask: &datapb.CompactionTask{
			PlanID:  19530,
			Type:    datapb.CompactionType_MixCompaction,
			Channel: "ch-01",
			NodeID:  1,
		},
		plan: &datapb.CompactionPlan{
			PlanID:  19530,
			Channel: "ch-01",
			Type:    datapb.CompactionType_MixCompaction,
		},
		sessions: s.mockSessMgr,
		meta:     s.mockMeta,
	}
	t2 := &l0CompactionTask{
		CompactionTask: &datapb.CompactionTask{
			PlanID:  19531,
			Type:    datapb.CompactionType_MixCompaction,
			Channel: "ch-02",
			NodeID:  1,
		},
		plan: &datapb.CompactionPlan{
			PlanID:  19531,
			Channel: "ch-02",
			Type:    datapb.CompactionType_Level0DeleteCompaction,
		},
		sessions: s.mockSessMgr,
		meta:     s.mockMeta,
	}

	s.mockCm.EXPECT().FindWatcher(mock.Anything).RunAndReturn(func(channel string) (int64, error) {
		if channel == "ch-01" {
			return 100, nil
		}
		if channel == "ch-02" {
			return 101, nil
		}
		return 1, nil
	}).Twice()

	node := s.handler.pickShardNode(nodeSlots, t1)
	s.Equal(int64(100), node)

	node = s.handler.pickShardNode(nodeSlots, t2)
	s.Equal(int64(101), node)
}

func (s *CompactionPlanHandlerSuite) TestRemoveTasksByChannel() {
<<<<<<< HEAD
	s.SetupTest()
	ch := "ch1"
	t1 := &mixCompactionTask{
		CompactionTask: &datapb.CompactionTask{
			PlanID:  19530,
			Type:    datapb.CompactionType_MixCompaction,
			Channel: ch,
			NodeID:  1,
		},
		plan: &datapb.CompactionPlan{
			PlanID:  19530,
			Channel: ch,
			Type:    datapb.CompactionType_MixCompaction,
		},
		sessions: s.mockSessMgr,
		meta:     s.mockMeta,
	}
	t2 := &mixCompactionTask{
		CompactionTask: &datapb.CompactionTask{
			PlanID:  19531,
			Type:    datapb.CompactionType_MixCompaction,
			Channel: ch,
			NodeID:  1,
		},
		plan: &datapb.CompactionPlan{
			PlanID:  19531,
			Channel: ch,
			Type:    datapb.CompactionType_MixCompaction,
		},
		sessions: s.mockSessMgr,
		meta:     s.mockMeta,
	}
	s.handler.submitTask(t1)
	s.handler.restoreTask(t2)
	s.handler.removeTasksByChannel(ch)
	s.Equal(0, s.handler.getTaskCount())
}

func (s *CompactionPlanHandlerSuite) TestGetCompactionTask() {
	s.SetupTest()
	inTasks := map[int64]CompactionTask{
		1: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				TriggerID: 1,
				PlanID:    1,
				Type:      datapb.CompactionType_MixCompaction,
				Channel:   "ch-01",
				State:     datapb.CompactionTaskState_executing,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  1,
				Type:    datapb.CompactionType_MixCompaction,
				Channel: "ch-01",
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		2: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				TriggerID: 1,
				PlanID:    2,
				Type:      datapb.CompactionType_MixCompaction,
				Channel:   "ch-01",
				State:     datapb.CompactionTaskState_completed,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  2,
				Type:    datapb.CompactionType_MixCompaction,
				Channel: "ch-01",
=======
	// s.mockSch.EXPECT().Finish(mock.Anything, mock.Anything).Return().Once()
	handler := newCompactionPlanHandler(nil, nil, nil, nil, nil, nil, nil)
	handler.scheduler = s.mockSch

	var ch string = "ch1"
	handler.mu.Lock()
	handler.plans[1] = &mixCompactionTask{
		plan:       &datapb.CompactionPlan{PlanID: 19530},
		dataNodeID: 1,
	}
	handler.mu.Unlock()

	handler.removeTasksByChannel(ch)

	handler.mu.Lock()
	s.Equal(1, len(handler.plans))
	handler.mu.Unlock()
}

func (s *CompactionPlanHandlerSuite) TestCheckResult() {
	s.mockSessMgr.EXPECT().GetCompactionPlansResults().Return(map[int64]*typeutil.Pair[int64, *datapb.CompactionPlanResult]{
		1: {A: 100, B: &datapb.CompactionPlanResult{PlanID: 1, State: commonpb.CompactionState_Executing}},
		2: {A: 100, B: &datapb.CompactionPlanResult{PlanID: 2, State: commonpb.CompactionState_Completed, Segments: []*datapb.CompactionSegment{{PlanID: 2}}}},
		3: {A: 100, B: &datapb.CompactionPlanResult{PlanID: 3, State: commonpb.CompactionState_Executing}},
		4: {A: 100, B: &datapb.CompactionPlanResult{PlanID: 4, State: commonpb.CompactionState_Executing}},
	}, nil)

	{
		s.mockAlloc.EXPECT().allocTimestamp(mock.Anything).Return(0, errors.New("mock")).Once()
		handler := newCompactionPlanHandler(nil, s.mockSessMgr, nil, nil, s.mockAlloc, nil, nil)
		handler.checkResult()
	}

	{
		s.mockAlloc.EXPECT().allocTimestamp(mock.Anything).Return(19530, nil).Once()
		handler := newCompactionPlanHandler(nil, s.mockSessMgr, nil, nil, s.mockAlloc, nil, nil)
		handler.checkResult()
	}
}

func (s *CompactionPlanHandlerSuite) TestClean() {
	startTime := tsoutil.ComposeTSByTime(time.Now(), 0)
	cleanTime := tsoutil.ComposeTSByTime(time.Now().Add(-2*time.Hour), 0)
	s.mockMeta.Mock.On("GetCompactionTaskMeta").Return(nil)
	s.mockMeta.Mock.On("GetPartitionStatsMeta").Return(nil)
	c := &compactionPlanHandler{
		allocator: s.mockAlloc,
		meta:      s.mockMeta,
		plans: map[int64]CompactionTask{
			1: &mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					State: datapb.CompactionTaskState_executing,
				},
			},
			2: &mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					State: datapb.CompactionTaskState_pipelining,
				},
			},
			3: &mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					State:     datapb.CompactionTaskState_completed,
					StartTime: startTime,
				},
			},
			4: &mixCompactionTask{
				CompactionTask: &datapb.CompactionTask{
					State:     datapb.CompactionTaskState_completed,
					StartTime: cleanTime,
				},
>>>>>>> c10a400603 (feat: Major compaction)
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		3: &l0CompactionTask{
			CompactionTask: &datapb.CompactionTask{
				TriggerID: 1,
				PlanID:    3,
				Type:      datapb.CompactionType_Level0DeleteCompaction,
				Channel:   "ch-02",
				State:     datapb.CompactionTaskState_failed,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  3,
				Type:    datapb.CompactionType_Level0DeleteCompaction,
				Channel: "ch-02",
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
	}
	s.mockMeta.EXPECT().GetCompactionTasksByTriggerID(mock.Anything).RunAndReturn(func(i int64) []*datapb.CompactionTask {
		var ret []*datapb.CompactionTask
		for _, t := range inTasks {
			if t.GetTriggerID() != i {
				continue
			}
			ret = append(ret, t.ShadowClone())
		}
		return ret
	})

	for _, t := range inTasks {
		s.handler.submitTask(t)
	}

	s.Equal(3, s.handler.getTaskCount())
	s.handler.doSchedule()
	s.Equal(3, s.handler.getTaskCount())

<<<<<<< HEAD
	info := s.handler.getCompactionInfo(1)
	s.Equal(1, info.completedCnt)
	s.Equal(1, info.executingCnt)
	s.Equal(1, info.failedCnt)
}

func (s *CompactionPlanHandlerSuite) TestExecCompactionPlan() {
	s.SetupTest()
	s.mockMeta.EXPECT().CheckAndSetSegmentsCompacting(mock.Anything).Return(true, true).Once()
	s.mockMeta.EXPECT().SaveCompactionTask(mock.Anything).Return(nil).Once()
	handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc)

	task := &datapb.CompactionTask{
		TriggerID: 1,
		PlanID:    1,
		Channel:   "ch-1",
		Type:      datapb.CompactionType_MixCompaction,
	}
	err := handler.enqueueCompaction(task)
	s.NoError(err)
	t := handler.getCompactionTask(1)
	s.NotNil(t)
	s.handler.taskNumber.Add(1000)
	task.PlanID = 2
	err = s.handler.enqueueCompaction(task)
	s.Error(err)
=======
	handler := newCompactionPlanHandler(nil, nil, nil, s.mockMeta, s.mockAlloc, nil, nil)
	err := handler.handleL0CompactionResult(plan, result)
	s.NoError(err)
}

//func (s *CompactionPlanHandlerSuite) TestRefreshL0Plan() {
//	channel := "Ch-1"
//	deltalogs := []*datapb.FieldBinlog{getFieldBinlogIDs(101, 3)}
//	s.mockMeta.EXPECT().SelectSegments(mock.Anything, mock.Anything).Return(
//		[]*SegmentInfo{
//			{SegmentInfo: &datapb.SegmentInfo{
//				ID:            200,
//				Level:         datapb.SegmentLevel_L1,
//				InsertChannel: channel,
//			}},
//			{SegmentInfo: &datapb.SegmentInfo{
//				ID:            201,
//				Level:         datapb.SegmentLevel_L1,
//				InsertChannel: channel,
//			}},
//			{SegmentInfo: &datapb.SegmentInfo{
//				ID:            202,
//				Level:         datapb.SegmentLevel_L1,
//				InsertChannel: channel,
//			}},
//		},
//	)
//
//	s.Run("normal_refresh", func() {
//		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).RunAndReturn(func(segID int64) *SegmentInfo {
//			return &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
//				ID:            segID,
//				Level:         datapb.SegmentLevel_L0,
//				InsertChannel: channel,
//				State:         commonpb.SegmentState_Flushed,
//				Deltalogs:     deltalogs,
//			}}
//		}).Times(2)
//		// 2 l0 segments
//		plan := &datapb.CompactionPlan{
//			PlanID: 1,
//			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
//				{
//					SegmentID:     100,
//					Level:         datapb.SegmentLevel_L0,
//					InsertChannel: channel,
//				},
//				{
//					SegmentID:     101,
//					Level:         datapb.SegmentLevel_L0,
//					InsertChannel: channel,
//				},
//			},
//			Type: datapb.CompactionType_Level0DeleteCompaction,
//		}
//
//		task := &mixCompactionTask{
//			CompactionTask: &datapb.CompactionTask{
//				TriggerID: 19530,
//				CollectionID: 1,
//				PartitionID: 10,
//				State: datapb.CompactionTaskState_executing,
//			},
//			plan:        plan,
//			dataNodeID:  1,
//		}
//
//		//handler := newCompactionPlanHandler(nil, nil, nil, s.mockMeta, s.mockAlloc, nil)
//		//err := handler.RefreshPlan(task)
//		//s.Require().NoError(err)
//
//		s.Equal(5, len(task.plan.GetSegmentBinlogs()))
//		segIDs := lo.Map(task.plan.GetSegmentBinlogs(), func(b *datapb.CompactionSegmentBinlogs, _ int) int64 {
//			return b.GetSegmentID()
//		})
//
//		s.ElementsMatch([]int64{200, 201, 202, 100, 101}, segIDs)
//	})
//
//	s.Run("segment_not_found", func() {
//		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).RunAndReturn(func(segID int64) *SegmentInfo {
//			return nil
//		}).Once()
//		plan := &datapb.CompactionPlan{
//			PlanID: 1,
//			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
//				{
//					SegmentID:     102,
//					Level:         datapb.SegmentLevel_L0,
//					InsertChannel: channel,
//				},
//			},
//			Type: datapb.CompactionType_Level0DeleteCompaction,
//		}
//
//		task := &defaultCompactionTask{
//			triggerInfo: &compactionSignal{id: 19531, collectionID: 1, partitionID: 10},
//			state:       executing,
//			plan:        plan,
//			dataNodeID:  1,
//		}
//
//		handler := newCompactionPlanHandler(nil, nil, nil, s.mockMeta, s.mockAlloc)
//		err := handler.RefreshPlan(task)
//		s.Error(err)
//		s.ErrorIs(err, merr.ErrSegmentNotFound)
//	})
//
//	s.Run("select zero segments", func() {
//		s.SetupTest()
//		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).RunAndReturn(func(segID int64) *SegmentInfo {
//			return &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
//				ID:            segID,
//				Level:         datapb.SegmentLevel_L0,
//				InsertChannel: channel,
//				State:         commonpb.SegmentState_Flushed,
//				Deltalogs:     deltalogs,
//			}}
//		}).Times(2)
//		s.mockMeta.EXPECT().SelectSegments(mock.Anything, mock.Anything).Return(nil).Once()
//
//		// 2 l0 segments
//		plan := &datapb.CompactionPlan{
//			PlanID: 1,
//			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
//				{
//					SegmentID:     100,
//					Level:         datapb.SegmentLevel_L0,
//					InsertChannel: channel,
//				},
//				{
//					SegmentID:     101,
//					Level:         datapb.SegmentLevel_L0,
//					InsertChannel: channel,
//				},
//			},
//			Type: datapb.CompactionType_Level0DeleteCompaction,
//		}
//
//		task := &defaultCompactionTask{
//			triggerInfo: &compactionSignal{id: 19530, collectionID: 1, partitionID: 10},
//			state:       executing,
//			plan:        plan,
//			dataNodeID:  1,
//		}
//
//		handler := newCompactionPlanHandler(nil, nil, nil, s.mockMeta, s.mockAlloc)
//		err := handler.RefreshPlan(task)
//		s.Error(err)
//	})
//}

//func (s *CompactionPlanHandlerSuite) TestRefreshPlanMixCompaction() {
//	channel := "Ch-1"
//	binlogs := []*datapb.FieldBinlog{getFieldBinlogIDs(101, 3)}
//
//	s.Run("normal_refresh", func() {
//		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).RunAndReturn(func(segID int64) *SegmentInfo {
//			return &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
//				ID:            segID,
//				Level:         datapb.SegmentLevel_L1,
//				InsertChannel: channel,
//				State:         commonpb.SegmentState_Flushed,
//				Binlogs:       binlogs,
//			}}
//		}).Times(2)
//		// 2 l0 segments
//		plan := &datapb.CompactionPlan{
//			PlanID: 1,
//			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
//				{
//					SegmentID:     200,
//					Level:         datapb.SegmentLevel_L1,
//					InsertChannel: channel,
//				},
//				{
//					SegmentID:     201,
//					Level:         datapb.SegmentLevel_L1,
//					InsertChannel: channel,
//				},
//			},
//			Type: datapb.CompactionType_MixCompaction,
//		}
//
//		task := &defaultCompactionTask{
//			triggerInfo: &compactionSignal{id: 19530, collectionID: 1, partitionID: 10},
//			state:       executing,
//			plan:        plan,
//			dataNodeID:  1,
//		}
//
//		handler := newCompactionPlanHandler(nil, nil, nil, s.mockMeta, s.mockAlloc)
//		err := handler.RefreshPlan(task)
//		s.Require().NoError(err)
//
//		s.Equal(2, len(task.plan.GetSegmentBinlogs()))
//		segIDs := lo.Map(task.plan.GetSegmentBinlogs(), func(b *datapb.CompactionSegmentBinlogs, _ int) int64 {
//			return b.GetSegmentID()
//		})
//
//		s.ElementsMatch([]int64{200, 201}, segIDs)
//	})
//
//	s.Run("segment_not_found", func() {
//		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).RunAndReturn(func(segID int64) *SegmentInfo {
//			return nil
//		}).Once()
//		// 2 l0 segments
//		plan := &datapb.CompactionPlan{
//			PlanID: 1,
//			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
//				{
//					SegmentID:     200,
//					Level:         datapb.SegmentLevel_L1,
//					InsertChannel: channel,
//				},
//				{
//					SegmentID:     201,
//					Level:         datapb.SegmentLevel_L1,
//					InsertChannel: channel,
//				},
//			},
//			Type: datapb.CompactionType_MixCompaction,
//		}
//
//		task := &defaultCompactionTask{
//			triggerInfo: &compactionSignal{id: 19530, collectionID: 1, partitionID: 10},
//			state:       executing,
//			plan:        plan,
//			dataNodeID:  1,
//		}
//
//		handler := newCompactionPlanHandler(nil, nil, nil, s.mockMeta, s.mockAlloc, nil)
//		err := handler.RefreshPlan(task)
//		s.Error(err)
//		s.ErrorIs(err, merr.ErrSegmentNotFound)
//	})
//}

func (s *CompactionPlanHandlerSuite) TestExecCompactionPlan() {
	s.mockMeta.EXPECT().CheckAndSetSegmentsCompacting(mock.Anything).Return(true, true).Once()
	handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil, nil)
	handler.scheduler = s.mockSch

	task := &mixCompactionTask{
		CompactionTask: &datapb.CompactionTask{
			TriggerID: 1,
			PlanID:    1,
			Channel:   "ch-1",
		},
	}

	handler.enqueueCompaction(task)
	handler.mu.RLock()
	defer handler.mu.RUnlock()
	_, ok := handler.plans[int64(1)]
	s.True(ok)
>>>>>>> c10a400603 (feat: Major compaction)
}

func (s *CompactionPlanHandlerSuite) TestCheckCompaction() {
	s.SetupTest()

	s.mockSessMgr.EXPECT().GetCompactionPlanResult(UniqueID(111), int64(1)).Return(
		&datapb.CompactionPlanResult{PlanID: 1, State: commonpb.CompactionState_Executing}, nil).Once()

	s.mockSessMgr.EXPECT().GetCompactionPlanResult(UniqueID(111), int64(2)).Return(
		&datapb.CompactionPlanResult{
			PlanID:   2,
			State:    commonpb.CompactionState_Completed,
			Segments: []*datapb.CompactionSegment{{PlanID: 2}},
		}, nil).Once()

	s.mockSessMgr.EXPECT().GetCompactionPlanResult(UniqueID(111), int64(6)).Return(
		&datapb.CompactionPlanResult{
			PlanID:   6,
			Channel:  "ch-2",
			State:    commonpb.CompactionState_Completed,
			Segments: []*datapb.CompactionSegment{{PlanID: 6}},
		}, nil).Once()

	inTasks := map[int64]CompactionTask{
		1: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:           1,
				Type:             datapb.CompactionType_MixCompaction,
				TimeoutInSeconds: 1,
				Channel:          "ch-1",
				State:            datapb.CompactionTaskState_executing,
				NodeID:           111,
			},
			plan: &datapb.CompactionPlan{
				PlanID: 1, Channel: "ch-1",
				TimeoutInSeconds: 1,
				Type:             datapb.CompactionType_MixCompaction,
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		2: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  2,
				Type:    datapb.CompactionType_MixCompaction,
				Channel: "ch-1",
				State:   datapb.CompactionTaskState_executing,
				NodeID:  111,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  2,
				Channel: "ch-1",
				Type:    datapb.CompactionType_MixCompaction,
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		3: &l0CompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  3,
				Type:    datapb.CompactionType_MixCompaction,
				Channel: "ch-1",
				State:   datapb.CompactionTaskState_timeout,
				NodeID:  111,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  3,
				Channel: "ch-1",
				Type:    datapb.CompactionType_MixCompaction,
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		4: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  4,
				Type:    datapb.CompactionType_MixCompaction,
				Channel: "ch-1",
				State:   datapb.CompactionTaskState_timeout,
				NodeID:  111,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  4,
				Channel: "ch-1",
				Type:    datapb.CompactionType_MixCompaction,
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
		6: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{
				PlanID:  6,
				Type:    datapb.CompactionType_MixCompaction,
				Channel: "ch-2",
				State:   datapb.CompactionTaskState_executing,
				NodeID:  111,
			},
			plan: &datapb.CompactionPlan{
				PlanID:  6,
				Channel: "ch-2",
				Type:    datapb.CompactionType_MixCompaction,
			},
			sessions: s.mockSessMgr,
			meta:     s.mockMeta,
		},
	}

	// s.mockSessMgr.EXPECT().SyncSegments(int64(111), mock.Anything).Return(nil)
	// s.mockMeta.EXPECT().UpdateSegmentsInfo(mock.Anything).Return(nil)
	s.mockMeta.EXPECT().SaveCompactionTask(mock.Anything).Return(nil)
	s.mockMeta.EXPECT().CompleteCompactionMutation(mock.Anything, mock.Anything).RunAndReturn(
		func(plan *datapb.CompactionPlan, result *datapb.CompactionPlanResult) ([]*SegmentInfo, *segMetricMutation, error) {
			if plan.GetPlanID() == 2 {
				segment := NewSegmentInfo(&datapb.SegmentInfo{ID: 100})
				return []*SegmentInfo{segment}, &segMetricMutation{}, nil
			} else if plan.GetPlanID() == 6 {
				return nil, nil, errors.Errorf("intended error")
			}
			return nil, nil, errors.Errorf("unexpected error")
		}).Twice()

	for _, t := range inTasks {
		s.handler.submitTask(t)
	}

	picked := s.handler.schedule()
	s.NotEmpty(picked)

	s.handler.doSchedule()
	// time.Sleep(2 * time.Second)
	s.handler.checkCompaction()

	t := s.handler.getCompactionTask(1)
	// timeout
	s.Nil(t)

	t = s.handler.getCompactionTask(2)
	// completed
	s.Nil(t)

	t = s.handler.getCompactionTask(3)
	s.Nil(t)

	t = s.handler.getCompactionTask(4)
	s.Nil(t)

	t = s.handler.getCompactionTask(5)
	// not exist
	s.Nil(t)

	t = s.handler.getCompactionTask(6)
	s.Equal(datapb.CompactionTaskState_executing, t.GetState())
}

func (s *CompactionPlanHandlerSuite) TestProcessCompleteCompaction() {
	s.SetupTest()

	// s.mockSessMgr.EXPECT().SyncSegments(mock.Anything, mock.Anything).Return(nil).Once()
	s.mockMeta.EXPECT().SaveCompactionTask(mock.Anything).Return(nil)
	s.mockMeta.EXPECT().SetSegmentCompacting(mock.Anything, mock.Anything).Return().Twice()
	segment := NewSegmentInfo(&datapb.SegmentInfo{ID: 100})
	s.mockMeta.EXPECT().CompleteCompactionMutation(mock.Anything, mock.Anything).Return(
		[]*SegmentInfo{segment},
		&segMetricMutation{}, nil).Once()

	dataNodeID := UniqueID(111)

	seg1 := &datapb.SegmentInfo{
		ID:        1,
		Binlogs:   []*datapb.FieldBinlog{getFieldBinlogIDs(101, 1)},
		Statslogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 2)},
		Deltalogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 3)},
	}

	seg2 := &datapb.SegmentInfo{
		ID:        2,
		Binlogs:   []*datapb.FieldBinlog{getFieldBinlogIDs(101, 4)},
		Statslogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 5)},
		Deltalogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 6)},
	}

	plan := &datapb.CompactionPlan{
		PlanID: 1,
		SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
			{
				SegmentID:           seg1.ID,
				FieldBinlogs:        seg1.GetBinlogs(),
				Field2StatslogPaths: seg1.GetStatslogs(),
				Deltalogs:           seg1.GetDeltalogs(),
			},
			{
				SegmentID:           seg2.ID,
				FieldBinlogs:        seg2.GetBinlogs(),
				Field2StatslogPaths: seg2.GetStatslogs(),
				Deltalogs:           seg2.GetDeltalogs(),
			},
		},
		Type: datapb.CompactionType_MixCompaction,
	}

<<<<<<< HEAD
	task := &mixCompactionTask{
		CompactionTask: &datapb.CompactionTask{
			PlanID:    plan.GetPlanID(),
			TriggerID: 1,
			Type:      plan.GetType(),
			State:     datapb.CompactionTaskState_executing,
			NodeID:    dataNodeID,
		},
		plan:     plan,
		sessions: s.mockSessMgr,
		meta:     s.mockMeta,
	}

	compactionResult := datapb.CompactionPlanResult{
		PlanID: 1,
		State:  commonpb.CompactionState_Completed,
		Segments: []*datapb.CompactionSegment{
			{
				SegmentID:           3,
				NumOfRows:           15,
				InsertLogs:          []*datapb.FieldBinlog{getFieldBinlogIDs(101, 301)},
				Field2StatslogPaths: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 302)},
				Deltalogs:           []*datapb.FieldBinlog{getFieldBinlogIDs(101, 303)},
			},
=======
	s.Run("illegal nil result", func() {
		s.SetupTest()
		handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil, nil)
		err := handler.handleMergeCompactionResult(nil, nil)
		s.Error(err)
	})

	s.Run("not empty compacted to segment info", func() {
		s.SetupTest()
		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).RunAndReturn(
			func(segID int64) *SegmentInfo {
				if segID == 3 {
					return NewSegmentInfo(&datapb.SegmentInfo{ID: 3})
				}
				return nil
			}).Once()
		s.mockSessMgr.EXPECT().SyncSegments(mock.Anything, mock.Anything).Return(nil).Maybe()

		handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil, nil)
		handler.plans[plan.PlanID] = &mixCompactionTask{dataNodeID: 111, plan: plan}

		compactionResult := &datapb.CompactionPlanResult{
			PlanID: plan.PlanID,
			Segments: []*datapb.CompactionSegment{
				{SegmentID: 3, NumOfRows: 15},
			},
		}

		err := handler.handleMergeCompactionResult(plan, compactionResult)
		s.NoError(err)
	})

	s.Run("complete compaction mutation error", func() {
		s.SetupTest()
		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).Return(nil).Once()
		s.mockMeta.EXPECT().CompleteCompactionMutation(mock.Anything, mock.Anything).Return(
			nil, nil, errors.New("mock error")).Once()

		handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil, nil)
		handler.plans[plan.PlanID] = &mixCompactionTask{dataNodeID: 111, plan: plan}
		compactionResult := &datapb.CompactionPlanResult{
			PlanID: plan.PlanID,
			Segments: []*datapb.CompactionSegment{
				{SegmentID: 4, NumOfRows: 15},
			},
		}

		err := handler.handleMergeCompactionResult(plan, compactionResult)
		s.Error(err)
	})

	s.Run("sync segment error", func() {
		s.SetupTest()
		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).Return(nil).Once()
		segment := NewSegmentInfo(&datapb.SegmentInfo{ID: 100})
		s.mockMeta.EXPECT().CompleteCompactionMutation(mock.Anything, mock.Anything).Return(
			[]*SegmentInfo{segment},
			&segMetricMutation{}, nil).Once()

		handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil, nil)
		handler.plans[plan.PlanID] = &mixCompactionTask{dataNodeID: 111, plan: plan}
		compactionResult := &datapb.CompactionPlanResult{
			PlanID: plan.PlanID,
			Segments: []*datapb.CompactionSegment{
				{SegmentID: 4, NumOfRows: 15},
			},
		}

		err := handler.handleMergeCompactionResult(plan, compactionResult)
		s.NoError(err)
	})
}

func (s *CompactionPlanHandlerSuite) TestCompleteCompaction() {
	s.Run("test not exists compaction task", func() {
		handler := newCompactionPlanHandler(nil, nil, nil, nil, nil, nil, nil)
		err := handler.handleMergeCompactionResult(&datapb.CompactionPlan{PlanID: 2}, &datapb.CompactionPlanResult{PlanID: 2})
		s.Error(err)
	})

	s.Run("test completed compaction task", func() {
		c := &compactionPlanHandler{
			plans: map[int64]CompactionTask{
				1: &mixCompactionTask{
					CompactionTask: &datapb.CompactionTask{PlanID: 1},
				},
			},
		}
		err := c.handleMergeCompactionResult(&datapb.CompactionPlan{PlanID: 1}, &datapb.CompactionPlanResult{PlanID: 1})
		s.Error(err)
	})

	s.Run("test complete merge compaction task", func() {
		// mock for handleMergeCompactionResult
		s.mockMeta.EXPECT().GetHealthySegment(mock.Anything).Return(nil).Once()
		segment := NewSegmentInfo(&datapb.SegmentInfo{ID: 100})
		s.mockMeta.EXPECT().CompleteCompactionMutation(mock.Anything, mock.Anything).Return(
			[]*SegmentInfo{segment},
			&segMetricMutation{}, nil).Once()
		// s.mockSch.EXPECT().Finish(mock.Anything, mock.Anything).Return()

		dataNodeID := UniqueID(111)

		seg1 := &datapb.SegmentInfo{
			ID:        1,
			Binlogs:   []*datapb.FieldBinlog{getFieldBinlogIDs(101, 1)},
			Statslogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 2)},
			Deltalogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 3)},
		}

		seg2 := &datapb.SegmentInfo{
			ID:        2,
			Binlogs:   []*datapb.FieldBinlog{getFieldBinlogIDs(101, 4)},
			Statslogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 5)},
			Deltalogs: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 6)},
		}

		plan := &datapb.CompactionPlan{
			PlanID: 1,
			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
				{
					SegmentID:           seg1.ID,
					FieldBinlogs:        seg1.GetBinlogs(),
					Field2StatslogPaths: seg1.GetStatslogs(),
					Deltalogs:           seg1.GetDeltalogs(),
				},
				{
					SegmentID:           seg2.ID,
					FieldBinlogs:        seg2.GetBinlogs(),
					Field2StatslogPaths: seg2.GetStatslogs(),
					Deltalogs:           seg2.GetDeltalogs(),
				},
			},
			Type: datapb.CompactionType_MixCompaction,
		}

		task := &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{TriggerID: 1, PlanID: 1, State: datapb.CompactionTaskState_executing},
			plan:           plan,
			dataNodeID:     dataNodeID,
		}

		plans := map[int64]CompactionTask{1: task}

		compactionResult := datapb.CompactionPlanResult{
			PlanID: 1,
			Segments: []*datapb.CompactionSegment{
				{
					SegmentID:           3,
					NumOfRows:           15,
					InsertLogs:          []*datapb.FieldBinlog{getFieldBinlogIDs(101, 301)},
					Field2StatslogPaths: []*datapb.FieldBinlog{getFieldBinlogIDs(101, 302)},
					Deltalogs:           []*datapb.FieldBinlog{getFieldBinlogIDs(101, 303)},
				},
			},
		}

		c := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil, nil)
		c.scheduler = s.mockSch
		c.plans = plans

		err := c.handleMergeCompactionResult(plan, &compactionResult)
		s.NoError(err)
		// todo wayblink
		// s.Nil(compactionResult.GetSegments()[0].GetInsertLogs())
		// s.Nil(compactionResult.GetSegments()[0].GetField2StatslogPaths())
		// s.Nil(compactionResult.GetSegments()[0].GetDeltalogs())
	})
}

func (s *CompactionPlanHandlerSuite) TestGetCompactionTask() {
	inPlans := map[int64]CompactionTask{
		1: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{TriggerID: 1, PlanID: 1, State: datapb.CompactionTaskState_executing},
		},
		2: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{TriggerID: 1, PlanID: 2, State: datapb.CompactionTaskState_completed},
		},
		3: &mixCompactionTask{
			CompactionTask: &datapb.CompactionTask{TriggerID: 1, PlanID: 3, State: datapb.CompactionTaskState_failed},
>>>>>>> c10a400603 (feat: Major compaction)
		},
	}

	s.mockSessMgr.EXPECT().GetCompactionPlanResult(UniqueID(111), int64(1)).Return(&compactionResult, nil).Once()

<<<<<<< HEAD
	s.handler.submitTask(task)
	s.handler.doSchedule()
	s.Equal(1, s.handler.getTaskCount())
	err := s.handler.checkCompaction()
	s.NoError(err)
	s.Equal(0, len(s.handler.getTasksByState(datapb.CompactionTaskState_completed)))
=======
	handler := &compactionPlanHandler{plans: inPlans}
	got := handler.getCompactionTasksBySignalID(1)
	s.ElementsMatch(expected, got)

	task := handler.getCompaction(1)
	s.NotNil(task)
	s.EqualValues(1, task.GetPlanID())

	task = handler.getCompaction(19530)
	s.Nil(task)
}

func (s *CompactionPlanHandlerSuite) TestUpdateCompaction() {
	//s.mockSessMgr.EXPECT().GetCompactionPlansResults().Return(map[int64]*typeutil.Pair[int64, *datapb.CompactionPlanResult]{
	//	1: {A: 111, B: &datapb.CompactionPlanResult{PlanID: 1, State: commonpb.CompactionState_Executing}},
	//	2: {A: 111, B: &datapb.CompactionPlanResult{PlanID: 2, State: commonpb.CompactionState_Completed, Segments: []*datapb.CompactionSegment{{PlanID: 2}}}},
	//	3: {A: 111, B: &datapb.CompactionPlanResult{PlanID: 3, State: commonpb.CompactionState_Executing}},
	//	5: {A: 222, B: &datapb.CompactionPlanResult{PlanID: 5, State: commonpb.CompactionState_Completed, Segments: []*datapb.CompactionSegment{{PlanID: 5}}}},
	//	6: {A: 111, B: &datapb.CompactionPlanResult{Channel: "ch-2", PlanID: 5, State: commonpb.CompactionState_Completed, Segments: []*datapb.CompactionSegment{{PlanID: 6}}}},
	//}, nil)
	//
	//inPlans := map[int64]CompactionTask{
	//	1: &mixCompactionTask{
	//		CompactionTask: &datapb.CompactionTask{PlanID: 1, State: datapb.CompactionTaskState_executing, Channel: "ch-1"},
	//		dataNodeID:     111,
	//	},
	//	2: &mixCompactionTask{
	//		CompactionTask: &datapb.CompactionTask{PlanID: 2, State: datapb.CompactionTaskState_executing, Channel: "ch-1"},
	//		dataNodeID:     111,
	//	},
	//	3: &mixCompactionTask{
	//		CompactionTask: &datapb.CompactionTask{PlanID: 3, State: datapb.CompactionTaskState_timeout, Channel: "ch-1"},
	//		dataNodeID:     111,
	//	},
	//	4: &mixCompactionTask{
	//		CompactionTask: &datapb.CompactionTask{PlanID: 4, State: datapb.CompactionTaskState_timeout, Channel: "ch-1"},
	//		dataNodeID:     111,
	//	},
	//	6: &mixCompactionTask{
	//		CompactionTask: &datapb.CompactionTask{PlanID: 6, State: datapb.CompactionTaskState_executing, Channel: "ch-2"},
	//		dataNodeID:     111,
	//	},
	//}
	//
	//s.mockCm.EXPECT().Match(int64(111), "ch-1").Return(true)
	//s.mockCm.EXPECT().Match(int64(111), "ch-2").Return(false).Once()
	//
	//handler := newCompactionPlanHandler(nil, s.mockSessMgr, s.mockCm, s.mockMeta, s.mockAlloc, nil)
	//handler.plans = inPlans
	//
	//_, ok := handler.plans[5]
	//s.Require().False(ok)
	//
	//err := handler.updateCompaction(0)
	//s.NoError(err)
	//
	//task := handler.plans[1]
	//s.Equal(datapb.CompactionTaskState_timeout, task.GetState())
	//
	//task = handler.plans[2]
	//s.Equal(datapb.CompactionTaskState_executing, task.GetState())
	//
	//task = handler.plans[3]
	//s.Equal(datapb.CompactionTaskState_timeout, task.GetState())
	//
	//task = handler.plans[4]
	//s.Equal(datapb.CompactionTaskState_failed, task.GetState())
	//
	//task = handler.plans[6]
	//s.Equal(datapb.CompactionTaskState_failed, task.GetState())
>>>>>>> c10a400603 (feat: Major compaction)
}

func getFieldBinlogIDs(fieldID int64, logIDs ...int64) *datapb.FieldBinlog {
	l := &datapb.FieldBinlog{
		FieldID: fieldID,
		Binlogs: make([]*datapb.Binlog, 0, len(logIDs)),
	}
	for _, id := range logIDs {
		l.Binlogs = append(l.Binlogs, &datapb.Binlog{LogID: id})
	}
	return l
}

func getFieldBinlogPaths(fieldID int64, paths ...string) *datapb.FieldBinlog {
	l := &datapb.FieldBinlog{
		FieldID: fieldID,
		Binlogs: make([]*datapb.Binlog, 0, len(paths)),
	}
	for _, path := range paths {
		l.Binlogs = append(l.Binlogs, &datapb.Binlog{LogPath: path})
	}
	return l
}

func getFieldBinlogIDsWithEntry(fieldID int64, entry int64, logIDs ...int64) *datapb.FieldBinlog {
	l := &datapb.FieldBinlog{
		FieldID: fieldID,
		Binlogs: make([]*datapb.Binlog, 0, len(logIDs)),
	}
	for _, id := range logIDs {
		l.Binlogs = append(l.Binlogs, &datapb.Binlog{LogID: id, EntriesNum: entry})
	}
	return l
}

func getInsertLogPath(rootPath string, segmentID typeutil.UniqueID) string {
	return metautil.BuildInsertLogPath(rootPath, 10, 100, segmentID, 1000, 10000)
}

func getStatsLogPath(rootPath string, segmentID typeutil.UniqueID) string {
	return metautil.BuildStatsLogPath(rootPath, 10, 100, segmentID, 1000, 10000)
}

func getDeltaLogPath(rootPath string, segmentID typeutil.UniqueID) string {
	return metautil.BuildDeltaLogPath(rootPath, 10, 100, segmentID, 10000)
}
