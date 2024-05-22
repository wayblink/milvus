// Code generated by mockery v2.32.4. DO NOT EDIT.

package datacoord

import (
	datapb "github.com/milvus-io/milvus/internal/proto/datapb"
	mock "github.com/stretchr/testify/mock"
)

// MockCompactionMeta is an autogenerated mock type for the CompactionMeta type
type MockCompactionMeta struct {
	mock.Mock
}

type MockCompactionMeta_Expecter struct {
	mock *mock.Mock
}

func (_m *MockCompactionMeta) EXPECT() *MockCompactionMeta_Expecter {
	return &MockCompactionMeta_Expecter{mock: &_m.Mock}
}

// CompleteCompactionMutation provides a mock function with given fields: plan, result
func (_m *MockCompactionMeta) CompleteCompactionMutation(plan *datapb.CompactionPlan, result *datapb.CompactionPlanResult) ([]*SegmentInfo, *segMetricMutation, error) {
	ret := _m.Called(plan, result)

	var r0 []*SegmentInfo
	var r1 *segMetricMutation
	var r2 error
	if rf, ok := ret.Get(0).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) ([]*SegmentInfo, *segMetricMutation, error)); ok {
		return rf(plan, result)
	}
	if rf, ok := ret.Get(0).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) []*SegmentInfo); ok {
		r0 = rf(plan, result)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*SegmentInfo)
		}
	}

	if rf, ok := ret.Get(1).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) *segMetricMutation); ok {
		r1 = rf(plan, result)
	} else {
		if ret.Get(1) != nil {
			r1 = ret.Get(1).(*segMetricMutation)
		}
	}

	if rf, ok := ret.Get(2).(func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) error); ok {
		r2 = rf(plan, result)
	} else {
		r2 = ret.Error(2)
	}

	return r0, r1, r2
}

// MockCompactionMeta_CompleteCompactionMutation_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CompleteCompactionMutation'
type MockCompactionMeta_CompleteCompactionMutation_Call struct {
	*mock.Call
}

// CompleteCompactionMutation is a helper method to define mock.On call
//   - plan *datapb.CompactionPlan
//   - result *datapb.CompactionPlanResult
func (_e *MockCompactionMeta_Expecter) CompleteCompactionMutation(plan interface{}, result interface{}) *MockCompactionMeta_CompleteCompactionMutation_Call {
	return &MockCompactionMeta_CompleteCompactionMutation_Call{Call: _e.mock.On("CompleteCompactionMutation", plan, result)}
}

func (_c *MockCompactionMeta_CompleteCompactionMutation_Call) Run(run func(plan *datapb.CompactionPlan, result *datapb.CompactionPlanResult)) *MockCompactionMeta_CompleteCompactionMutation_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*datapb.CompactionPlan), args[1].(*datapb.CompactionPlanResult))
	})
	return _c
}

func (_c *MockCompactionMeta_CompleteCompactionMutation_Call) Return(_a0 []*SegmentInfo, _a1 *segMetricMutation, _a2 error) *MockCompactionMeta_CompleteCompactionMutation_Call {
	_c.Call.Return(_a0, _a1, _a2)
	return _c
}

func (_c *MockCompactionMeta_CompleteCompactionMutation_Call) RunAndReturn(run func(*datapb.CompactionPlan, *datapb.CompactionPlanResult) ([]*SegmentInfo, *segMetricMutation, error)) *MockCompactionMeta_CompleteCompactionMutation_Call {
	_c.Call.Return(run)
	return _c
}

// DropClusteringCompactionTask provides a mock function with given fields: task
func (_m *MockCompactionMeta) DropClusteringCompactionTask(task *datapb.CompactionTask) error {
	ret := _m.Called(task)

	var r0 error
	if rf, ok := ret.Get(0).(func(*datapb.CompactionTask) error); ok {
		r0 = rf(task)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockCompactionMeta_DropClusteringCompactionTask_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'DropClusteringCompactionTask'
type MockCompactionMeta_DropClusteringCompactionTask_Call struct {
	*mock.Call
}

// DropClusteringCompactionTask is a helper method to define mock.On call
//   - task *datapb.CompactionTask
func (_e *MockCompactionMeta_Expecter) DropClusteringCompactionTask(task interface{}) *MockCompactionMeta_DropClusteringCompactionTask_Call {
	return &MockCompactionMeta_DropClusteringCompactionTask_Call{Call: _e.mock.On("DropClusteringCompactionTask", task)}
}

func (_c *MockCompactionMeta_DropClusteringCompactionTask_Call) Run(run func(task *datapb.CompactionTask)) *MockCompactionMeta_DropClusteringCompactionTask_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*datapb.CompactionTask))
	})
	return _c
}

func (_c *MockCompactionMeta_DropClusteringCompactionTask_Call) Return(_a0 error) *MockCompactionMeta_DropClusteringCompactionTask_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_DropClusteringCompactionTask_Call) RunAndReturn(run func(*datapb.CompactionTask) error) *MockCompactionMeta_DropClusteringCompactionTask_Call {
	_c.Call.Return(run)
	return _c
}

// GetClusteringCompactionTasks provides a mock function with given fields:
func (_m *MockCompactionMeta) GetClusteringCompactionTasks() map[int64][]*datapb.CompactionTask {
	ret := _m.Called()

	var r0 map[int64][]*datapb.CompactionTask
	if rf, ok := ret.Get(0).(func() map[int64][]*datapb.CompactionTask); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(map[int64][]*datapb.CompactionTask)
		}
	}

	return r0
}

// MockCompactionMeta_GetClusteringCompactionTasks_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetClusteringCompactionTasks'
type MockCompactionMeta_GetClusteringCompactionTasks_Call struct {
	*mock.Call
}

// GetClusteringCompactionTasks is a helper method to define mock.On call
func (_e *MockCompactionMeta_Expecter) GetClusteringCompactionTasks() *MockCompactionMeta_GetClusteringCompactionTasks_Call {
	return &MockCompactionMeta_GetClusteringCompactionTasks_Call{Call: _e.mock.On("GetClusteringCompactionTasks")}
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasks_Call) Run(run func()) *MockCompactionMeta_GetClusteringCompactionTasks_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasks_Call) Return(_a0 map[int64][]*datapb.CompactionTask) *MockCompactionMeta_GetClusteringCompactionTasks_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasks_Call) RunAndReturn(run func() map[int64][]*datapb.CompactionTask) *MockCompactionMeta_GetClusteringCompactionTasks_Call {
	_c.Call.Return(run)
	return _c
}

// GetClusteringCompactionTasksByCollection provides a mock function with given fields: collectionID
func (_m *MockCompactionMeta) GetClusteringCompactionTasksByCollection(collectionID int64) map[int64][]*datapb.CompactionTask {
	ret := _m.Called(collectionID)

	var r0 map[int64][]*datapb.CompactionTask
	if rf, ok := ret.Get(0).(func(int64) map[int64][]*datapb.CompactionTask); ok {
		r0 = rf(collectionID)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(map[int64][]*datapb.CompactionTask)
		}
	}

	return r0
}

// MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetClusteringCompactionTasksByCollection'
type MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call struct {
	*mock.Call
}

// GetClusteringCompactionTasksByCollection is a helper method to define mock.On call
//   - collectionID int64
func (_e *MockCompactionMeta_Expecter) GetClusteringCompactionTasksByCollection(collectionID interface{}) *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call {
	return &MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call{Call: _e.mock.On("GetClusteringCompactionTasksByCollection", collectionID)}
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call) Run(run func(collectionID int64)) *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call) Return(_a0 map[int64][]*datapb.CompactionTask) *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call) RunAndReturn(run func(int64) map[int64][]*datapb.CompactionTask) *MockCompactionMeta_GetClusteringCompactionTasksByCollection_Call {
	_c.Call.Return(run)
	return _c
}

// GetClusteringCompactionTasksByTriggerID provides a mock function with given fields: triggerID
func (_m *MockCompactionMeta) GetClusteringCompactionTasksByTriggerID(triggerID int64) []*datapb.CompactionTask {
	ret := _m.Called(triggerID)

	var r0 []*datapb.CompactionTask
	if rf, ok := ret.Get(0).(func(int64) []*datapb.CompactionTask); ok {
		r0 = rf(triggerID)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*datapb.CompactionTask)
		}
	}

	return r0
}

// MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetClusteringCompactionTasksByTriggerID'
type MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call struct {
	*mock.Call
}

// GetClusteringCompactionTasksByTriggerID is a helper method to define mock.On call
//   - triggerID int64
func (_e *MockCompactionMeta_Expecter) GetClusteringCompactionTasksByTriggerID(triggerID interface{}) *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call {
	return &MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call{Call: _e.mock.On("GetClusteringCompactionTasksByTriggerID", triggerID)}
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call) Run(run func(triggerID int64)) *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call) Return(_a0 []*datapb.CompactionTask) *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call) RunAndReturn(run func(int64) []*datapb.CompactionTask) *MockCompactionMeta_GetClusteringCompactionTasksByTriggerID_Call {
	_c.Call.Return(run)
	return _c
}

// GetHealthySegment provides a mock function with given fields: segID
func (_m *MockCompactionMeta) GetHealthySegment(segID int64) *SegmentInfo {
	ret := _m.Called(segID)

	var r0 *SegmentInfo
	if rf, ok := ret.Get(0).(func(int64) *SegmentInfo); ok {
		r0 = rf(segID)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*SegmentInfo)
		}
	}

	return r0
}

// MockCompactionMeta_GetHealthySegment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetHealthySegment'
type MockCompactionMeta_GetHealthySegment_Call struct {
	*mock.Call
}

// GetHealthySegment is a helper method to define mock.On call
//   - segID int64
func (_e *MockCompactionMeta_Expecter) GetHealthySegment(segID interface{}) *MockCompactionMeta_GetHealthySegment_Call {
	return &MockCompactionMeta_GetHealthySegment_Call{Call: _e.mock.On("GetHealthySegment", segID)}
}

func (_c *MockCompactionMeta_GetHealthySegment_Call) Run(run func(segID int64)) *MockCompactionMeta_GetHealthySegment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockCompactionMeta_GetHealthySegment_Call) Return(_a0 *SegmentInfo) *MockCompactionMeta_GetHealthySegment_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_GetHealthySegment_Call) RunAndReturn(run func(int64) *SegmentInfo) *MockCompactionMeta_GetHealthySegment_Call {
	_c.Call.Return(run)
	return _c
}

// GetSegment provides a mock function with given fields: segID
func (_m *MockCompactionMeta) GetSegment(segID int64) *SegmentInfo {
	ret := _m.Called(segID)

	var r0 *SegmentInfo
	if rf, ok := ret.Get(0).(func(int64) *SegmentInfo); ok {
		r0 = rf(segID)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*SegmentInfo)
		}
	}

	return r0
}

// MockCompactionMeta_GetSegment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetSegment'
type MockCompactionMeta_GetSegment_Call struct {
	*mock.Call
}

// GetSegment is a helper method to define mock.On call
//   - segID int64
func (_e *MockCompactionMeta_Expecter) GetSegment(segID interface{}) *MockCompactionMeta_GetSegment_Call {
	return &MockCompactionMeta_GetSegment_Call{Call: _e.mock.On("GetSegment", segID)}
}

func (_c *MockCompactionMeta_GetSegment_Call) Run(run func(segID int64)) *MockCompactionMeta_GetSegment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockCompactionMeta_GetSegment_Call) Return(_a0 *SegmentInfo) *MockCompactionMeta_GetSegment_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_GetSegment_Call) RunAndReturn(run func(int64) *SegmentInfo) *MockCompactionMeta_GetSegment_Call {
	_c.Call.Return(run)
	return _c
}

// SaveClusteringCompactionTask provides a mock function with given fields: task
func (_m *MockCompactionMeta) SaveClusteringCompactionTask(task *datapb.CompactionTask) error {
	ret := _m.Called(task)

	var r0 error
	if rf, ok := ret.Get(0).(func(*datapb.CompactionTask) error); ok {
		r0 = rf(task)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockCompactionMeta_SaveClusteringCompactionTask_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SaveClusteringCompactionTask'
type MockCompactionMeta_SaveClusteringCompactionTask_Call struct {
	*mock.Call
}

// SaveClusteringCompactionTask is a helper method to define mock.On call
//   - task *datapb.CompactionTask
func (_e *MockCompactionMeta_Expecter) SaveClusteringCompactionTask(task interface{}) *MockCompactionMeta_SaveClusteringCompactionTask_Call {
	return &MockCompactionMeta_SaveClusteringCompactionTask_Call{Call: _e.mock.On("SaveClusteringCompactionTask", task)}
}

func (_c *MockCompactionMeta_SaveClusteringCompactionTask_Call) Run(run func(task *datapb.CompactionTask)) *MockCompactionMeta_SaveClusteringCompactionTask_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*datapb.CompactionTask))
	})
	return _c
}

func (_c *MockCompactionMeta_SaveClusteringCompactionTask_Call) Return(_a0 error) *MockCompactionMeta_SaveClusteringCompactionTask_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_SaveClusteringCompactionTask_Call) RunAndReturn(run func(*datapb.CompactionTask) error) *MockCompactionMeta_SaveClusteringCompactionTask_Call {
	_c.Call.Return(run)
	return _c
}

// SelectSegments provides a mock function with given fields: filters
func (_m *MockCompactionMeta) SelectSegments(filters ...SegmentFilter) []*SegmentInfo {
	_va := make([]interface{}, len(filters))
	for _i := range filters {
		_va[_i] = filters[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 []*SegmentInfo
	if rf, ok := ret.Get(0).(func(...SegmentFilter) []*SegmentInfo); ok {
		r0 = rf(filters...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*SegmentInfo)
		}
	}

	return r0
}

// MockCompactionMeta_SelectSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SelectSegments'
type MockCompactionMeta_SelectSegments_Call struct {
	*mock.Call
}

// SelectSegments is a helper method to define mock.On call
//   - filters ...SegmentFilter
func (_e *MockCompactionMeta_Expecter) SelectSegments(filters ...interface{}) *MockCompactionMeta_SelectSegments_Call {
	return &MockCompactionMeta_SelectSegments_Call{Call: _e.mock.On("SelectSegments",
		append([]interface{}{}, filters...)...)}
}

func (_c *MockCompactionMeta_SelectSegments_Call) Run(run func(filters ...SegmentFilter)) *MockCompactionMeta_SelectSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]SegmentFilter, len(args)-0)
		for i, a := range args[0:] {
			if a != nil {
				variadicArgs[i] = a.(SegmentFilter)
			}
		}
		run(variadicArgs...)
	})
	return _c
}

func (_c *MockCompactionMeta_SelectSegments_Call) Return(_a0 []*SegmentInfo) *MockCompactionMeta_SelectSegments_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_SelectSegments_Call) RunAndReturn(run func(...SegmentFilter) []*SegmentInfo) *MockCompactionMeta_SelectSegments_Call {
	_c.Call.Return(run)
	return _c
}

// SetSegmentCompacting provides a mock function with given fields: segmentID, compacting
func (_m *MockCompactionMeta) SetSegmentCompacting(segmentID int64, compacting bool) {
	_m.Called(segmentID, compacting)
}

// MockCompactionMeta_SetSegmentCompacting_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetSegmentCompacting'
type MockCompactionMeta_SetSegmentCompacting_Call struct {
	*mock.Call
}

// SetSegmentCompacting is a helper method to define mock.On call
//   - segmentID int64
//   - compacting bool
func (_e *MockCompactionMeta_Expecter) SetSegmentCompacting(segmentID interface{}, compacting interface{}) *MockCompactionMeta_SetSegmentCompacting_Call {
	return &MockCompactionMeta_SetSegmentCompacting_Call{Call: _e.mock.On("SetSegmentCompacting", segmentID, compacting)}
}

func (_c *MockCompactionMeta_SetSegmentCompacting_Call) Run(run func(segmentID int64, compacting bool)) *MockCompactionMeta_SetSegmentCompacting_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64), args[1].(bool))
	})
	return _c
}

func (_c *MockCompactionMeta_SetSegmentCompacting_Call) Return() *MockCompactionMeta_SetSegmentCompacting_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockCompactionMeta_SetSegmentCompacting_Call) RunAndReturn(run func(int64, bool)) *MockCompactionMeta_SetSegmentCompacting_Call {
	_c.Call.Return(run)
	return _c
}

// UpdateSegmentsInfo provides a mock function with given fields: operators
func (_m *MockCompactionMeta) UpdateSegmentsInfo(operators ...UpdateOperator) error {
	_va := make([]interface{}, len(operators))
	for _i := range operators {
		_va[_i] = operators[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 error
	if rf, ok := ret.Get(0).(func(...UpdateOperator) error); ok {
		r0 = rf(operators...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockCompactionMeta_UpdateSegmentsInfo_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UpdateSegmentsInfo'
type MockCompactionMeta_UpdateSegmentsInfo_Call struct {
	*mock.Call
}

// UpdateSegmentsInfo is a helper method to define mock.On call
//   - operators ...UpdateOperator
func (_e *MockCompactionMeta_Expecter) UpdateSegmentsInfo(operators ...interface{}) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	return &MockCompactionMeta_UpdateSegmentsInfo_Call{Call: _e.mock.On("UpdateSegmentsInfo",
		append([]interface{}{}, operators...)...)}
}

func (_c *MockCompactionMeta_UpdateSegmentsInfo_Call) Run(run func(operators ...UpdateOperator)) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]UpdateOperator, len(args)-0)
		for i, a := range args[0:] {
			if a != nil {
				variadicArgs[i] = a.(UpdateOperator)
			}
		}
		run(variadicArgs...)
	})
	return _c
}

func (_c *MockCompactionMeta_UpdateSegmentsInfo_Call) Return(_a0 error) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockCompactionMeta_UpdateSegmentsInfo_Call) RunAndReturn(run func(...UpdateOperator) error) *MockCompactionMeta_UpdateSegmentsInfo_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockCompactionMeta creates a new instance of MockCompactionMeta. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockCompactionMeta(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockCompactionMeta {
	mock := &MockCompactionMeta{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
