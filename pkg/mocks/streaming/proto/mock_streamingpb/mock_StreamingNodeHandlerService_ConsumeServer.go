// Code generated by mockery v2.32.4. DO NOT EDIT.

package mock_streamingpb

import (
	context "context"

	mock "github.com/stretchr/testify/mock"
	metadata "google.golang.org/grpc/metadata"

	streamingpb "github.com/milvus-io/milvus/pkg/streaming/proto/streamingpb"
)

// MockStreamingNodeHandlerService_ConsumeServer is an autogenerated mock type for the StreamingNodeHandlerService_ConsumeServer type
type MockStreamingNodeHandlerService_ConsumeServer struct {
	mock.Mock
}

type MockStreamingNodeHandlerService_ConsumeServer_Expecter struct {
	mock *mock.Mock
}

func (_m *MockStreamingNodeHandlerService_ConsumeServer) EXPECT() *MockStreamingNodeHandlerService_ConsumeServer_Expecter {
	return &MockStreamingNodeHandlerService_ConsumeServer_Expecter{mock: &_m.Mock}
}

// Context provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeServer) Context() context.Context {
	ret := _m.Called()

	var r0 context.Context
	if rf, ok := ret.Get(0).(func() context.Context); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(context.Context)
		}
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeServer_Context_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Context'
type MockStreamingNodeHandlerService_ConsumeServer_Context_Call struct {
	*mock.Call
}

// Context is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) Context() *MockStreamingNodeHandlerService_ConsumeServer_Context_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_Context_Call{Call: _e.mock.On("Context")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Context_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeServer_Context_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Context_Call) Return(_a0 context.Context) *MockStreamingNodeHandlerService_ConsumeServer_Context_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Context_Call) RunAndReturn(run func() context.Context) *MockStreamingNodeHandlerService_ConsumeServer_Context_Call {
	_c.Call.Return(run)
	return _c
}

// Recv provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeServer) Recv() (*streamingpb.ConsumeRequest, error) {
	ret := _m.Called()

	var r0 *streamingpb.ConsumeRequest
	var r1 error
	if rf, ok := ret.Get(0).(func() (*streamingpb.ConsumeRequest, error)); ok {
		return rf()
	}
	if rf, ok := ret.Get(0).(func() *streamingpb.ConsumeRequest); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*streamingpb.ConsumeRequest)
		}
	}

	if rf, ok := ret.Get(1).(func() error); ok {
		r1 = rf()
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockStreamingNodeHandlerService_ConsumeServer_Recv_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Recv'
type MockStreamingNodeHandlerService_ConsumeServer_Recv_Call struct {
	*mock.Call
}

// Recv is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) Recv() *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_Recv_Call{Call: _e.mock.On("Recv")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call) Return(_a0 *streamingpb.ConsumeRequest, _a1 error) *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call) RunAndReturn(run func() (*streamingpb.ConsumeRequest, error)) *MockStreamingNodeHandlerService_ConsumeServer_Recv_Call {
	_c.Call.Return(run)
	return _c
}

// RecvMsg provides a mock function with given fields: m
func (_m *MockStreamingNodeHandlerService_ConsumeServer) RecvMsg(m interface{}) error {
	ret := _m.Called(m)

	var r0 error
	if rf, ok := ret.Get(0).(func(interface{}) error); ok {
		r0 = rf(m)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RecvMsg'
type MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call struct {
	*mock.Call
}

// RecvMsg is a helper method to define mock.On call
//   - m interface{}
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) RecvMsg(m interface{}) *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call{Call: _e.mock.On("RecvMsg", m)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call) Run(run func(m interface{})) *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(interface{}))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call) RunAndReturn(run func(interface{}) error) *MockStreamingNodeHandlerService_ConsumeServer_RecvMsg_Call {
	_c.Call.Return(run)
	return _c
}

// Send provides a mock function with given fields: _a0
func (_m *MockStreamingNodeHandlerService_ConsumeServer) Send(_a0 *streamingpb.ConsumeResponse) error {
	ret := _m.Called(_a0)

	var r0 error
	if rf, ok := ret.Get(0).(func(*streamingpb.ConsumeResponse) error); ok {
		r0 = rf(_a0)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeServer_Send_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Send'
type MockStreamingNodeHandlerService_ConsumeServer_Send_Call struct {
	*mock.Call
}

// Send is a helper method to define mock.On call
//   - _a0 *streamingpb.ConsumeResponse
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) Send(_a0 interface{}) *MockStreamingNodeHandlerService_ConsumeServer_Send_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_Send_Call{Call: _e.mock.On("Send", _a0)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Send_Call) Run(run func(_a0 *streamingpb.ConsumeResponse)) *MockStreamingNodeHandlerService_ConsumeServer_Send_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*streamingpb.ConsumeResponse))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Send_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeServer_Send_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_Send_Call) RunAndReturn(run func(*streamingpb.ConsumeResponse) error) *MockStreamingNodeHandlerService_ConsumeServer_Send_Call {
	_c.Call.Return(run)
	return _c
}

// SendHeader provides a mock function with given fields: _a0
func (_m *MockStreamingNodeHandlerService_ConsumeServer) SendHeader(_a0 metadata.MD) error {
	ret := _m.Called(_a0)

	var r0 error
	if rf, ok := ret.Get(0).(func(metadata.MD) error); ok {
		r0 = rf(_a0)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SendHeader'
type MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call struct {
	*mock.Call
}

// SendHeader is a helper method to define mock.On call
//   - _a0 metadata.MD
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) SendHeader(_a0 interface{}) *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call{Call: _e.mock.On("SendHeader", _a0)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call) Run(run func(_a0 metadata.MD)) *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(metadata.MD))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call) RunAndReturn(run func(metadata.MD) error) *MockStreamingNodeHandlerService_ConsumeServer_SendHeader_Call {
	_c.Call.Return(run)
	return _c
}

// SendMsg provides a mock function with given fields: m
func (_m *MockStreamingNodeHandlerService_ConsumeServer) SendMsg(m interface{}) error {
	ret := _m.Called(m)

	var r0 error
	if rf, ok := ret.Get(0).(func(interface{}) error); ok {
		r0 = rf(m)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SendMsg'
type MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call struct {
	*mock.Call
}

// SendMsg is a helper method to define mock.On call
//   - m interface{}
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) SendMsg(m interface{}) *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call{Call: _e.mock.On("SendMsg", m)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call) Run(run func(m interface{})) *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(interface{}))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call) RunAndReturn(run func(interface{}) error) *MockStreamingNodeHandlerService_ConsumeServer_SendMsg_Call {
	_c.Call.Return(run)
	return _c
}

// SetHeader provides a mock function with given fields: _a0
func (_m *MockStreamingNodeHandlerService_ConsumeServer) SetHeader(_a0 metadata.MD) error {
	ret := _m.Called(_a0)

	var r0 error
	if rf, ok := ret.Get(0).(func(metadata.MD) error); ok {
		r0 = rf(_a0)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetHeader'
type MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call struct {
	*mock.Call
}

// SetHeader is a helper method to define mock.On call
//   - _a0 metadata.MD
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) SetHeader(_a0 interface{}) *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call{Call: _e.mock.On("SetHeader", _a0)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call) Run(run func(_a0 metadata.MD)) *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(metadata.MD))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call) RunAndReturn(run func(metadata.MD) error) *MockStreamingNodeHandlerService_ConsumeServer_SetHeader_Call {
	_c.Call.Return(run)
	return _c
}

// SetTrailer provides a mock function with given fields: _a0
func (_m *MockStreamingNodeHandlerService_ConsumeServer) SetTrailer(_a0 metadata.MD) {
	_m.Called(_a0)
}

// MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetTrailer'
type MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call struct {
	*mock.Call
}

// SetTrailer is a helper method to define mock.On call
//   - _a0 metadata.MD
func (_e *MockStreamingNodeHandlerService_ConsumeServer_Expecter) SetTrailer(_a0 interface{}) *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call {
	return &MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call{Call: _e.mock.On("SetTrailer", _a0)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call) Run(run func(_a0 metadata.MD)) *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(metadata.MD))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call) Return() *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call) RunAndReturn(run func(metadata.MD)) *MockStreamingNodeHandlerService_ConsumeServer_SetTrailer_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockStreamingNodeHandlerService_ConsumeServer creates a new instance of MockStreamingNodeHandlerService_ConsumeServer. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockStreamingNodeHandlerService_ConsumeServer(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockStreamingNodeHandlerService_ConsumeServer {
	mock := &MockStreamingNodeHandlerService_ConsumeServer{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}