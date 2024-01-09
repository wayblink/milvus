package datacoord

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	grpcStatus "google.golang.org/grpc/status"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
)

type ServerSuite struct {
	suite.Suite

	testServer *Server
	mockChMgr  *MockChannelManager
}

func (s *ServerSuite) SetupTest() {
	s.testServer = newTestServer(s.T(), nil)
	if s.testServer.channelManager != nil {
		s.testServer.channelManager.Close()
	}

	s.mockChMgr = NewMockChannelManager(s.T())
	s.testServer.channelManager = s.mockChMgr
	if s.mockChMgr != nil {
		s.mockChMgr.EXPECT().Close().Maybe()
	}
}

func (s *ServerSuite) TearDownTest() {
	if s.testServer != nil {
		log.Info("ServerSuite tears down test", zap.String("name", s.T().Name()))
		closeTestServer(s.T(), s.testServer)
	}
}

func TestServerSuite(t *testing.T) {
	suite.Run(t, new(ServerSuite))
}

func (s *ServerSuite) TestGetFlushState_ByFlushTs() {
	s.mockChMgr.EXPECT().GetChannelsByCollectionID(int64(0)).
		Return([]RWChannel{&channelMeta{Name: "ch1", CollectionID: 0}}).Times(3)

	s.mockChMgr.EXPECT().GetChannelsByCollectionID(int64(1)).Return(nil).Times(1)
	tests := []struct {
		description string
		inTs        Timestamp

		expected bool
	}{
		{"channel cp > flush ts", 11, true},
		{"channel cp = flush ts", 12, true},
		{"channel cp < flush ts", 13, false},
	}

	err := s.testServer.meta.UpdateChannelCheckpoint("ch1", &msgpb.MsgPosition{
		MsgID:     []byte{1},
		Timestamp: 12,
	})
	s.Require().NoError(err)
	for _, test := range tests {
		s.Run(test.description, func() {
			resp, err := s.testServer.GetFlushState(context.TODO(), &datapb.GetFlushStateRequest{FlushTs: test.inTs})
			s.NoError(err)
			s.EqualValues(&milvuspb.GetFlushStateResponse{
				Status:  merr.Success(),
				Flushed: test.expected,
			}, resp)
		})
	}

	resp, err := s.testServer.GetFlushState(context.TODO(), &datapb.GetFlushStateRequest{CollectionID: 1, FlushTs: 13})
	s.NoError(err)
	s.EqualValues(&milvuspb.GetFlushStateResponse{
		Status:  merr.Success(),
		Flushed: true,
	}, resp)
}

func (s *ServerSuite) TestGetFlushState_BySegment() {
	s.mockChMgr.EXPECT().GetChannelsByCollectionID(mock.Anything).
		Return([]RWChannel{&channelMeta{Name: "ch1", CollectionID: 0}}).Times(3)

	tests := []struct {
		description string
		segID       int64
		state       commonpb.SegmentState

		expected bool
	}{
		{"flushed seg1", 1, commonpb.SegmentState_Flushed, true},
		{"flushed seg2", 2, commonpb.SegmentState_Flushed, true},
		{"sealed seg3", 3, commonpb.SegmentState_Sealed, false},
		{"compacted/dropped seg4", 4, commonpb.SegmentState_Dropped, true},
	}

	for _, test := range tests {
		s.Run(test.description, func() {
			err := s.testServer.meta.AddSegment(context.TODO(), &SegmentInfo{
				SegmentInfo: &datapb.SegmentInfo{
					ID:    test.segID,
					State: test.state,
				},
			})

			s.Require().NoError(err)
			err = s.testServer.meta.UpdateChannelCheckpoint("ch1", &msgpb.MsgPosition{
				MsgID:     []byte{1},
				Timestamp: 12,
			})
			s.Require().NoError(err)

			resp, err := s.testServer.GetFlushState(context.TODO(), &datapb.GetFlushStateRequest{SegmentIDs: []int64{test.segID}})
			s.NoError(err)
			s.EqualValues(&milvuspb.GetFlushStateResponse{
				Status:  merr.Success(),
				Flushed: test.expected,
			}, resp)
		})
	}
}

func (s *ServerSuite) TestSaveBinlogPath_ClosedServer() {
	s.TearDownTest()
	resp, err := s.testServer.SaveBinlogPaths(context.Background(), &datapb.SaveBinlogPathsRequest{
		SegmentID: 1,
		Channel:   "test",
	})
	s.NoError(err)
	s.ErrorIs(merr.Error(resp), merr.ErrServiceNotReady)
}

func (s *ServerSuite) TestSaveBinlogPath_ChannelNotMatch() {
	s.mockChMgr.EXPECT().Match(mock.Anything, mock.Anything).Return(false)
	resp, err := s.testServer.SaveBinlogPaths(context.Background(), &datapb.SaveBinlogPathsRequest{
		SegmentID: 1,
		Channel:   "test",
	})
	s.NoError(err)
	s.ErrorIs(merr.Error(resp), merr.ErrChannelNotFound)
}

func (s *ServerSuite) TestSaveBinlogPath_SaveUnhealthySegment() {
	s.mockChMgr.EXPECT().Match(int64(0), "ch1").Return(true)
	s.testServer.meta.AddCollection(&collectionInfo{ID: 0})

	segments := map[int64]commonpb.SegmentState{
		0: commonpb.SegmentState_NotExist,
	}
	for segID, state := range segments {
		info := &datapb.SegmentInfo{
			ID:            segID,
			InsertChannel: "ch1",
			State:         state,
		}
		err := s.testServer.meta.AddSegment(context.TODO(), NewSegmentInfo(info))
		s.Require().NoError(err)
	}

	ctx := context.Background()
	resp, err := s.testServer.SaveBinlogPaths(ctx, &datapb.SaveBinlogPathsRequest{
		Base: &commonpb.MsgBase{
			Timestamp: uint64(time.Now().Unix()),
		},
		SegmentID: 1,
		Channel:   "ch1",
	})
	s.NoError(err)
	s.ErrorIs(merr.Error(resp), merr.ErrSegmentNotFound)

	resp, err = s.testServer.SaveBinlogPaths(ctx, &datapb.SaveBinlogPathsRequest{
		Base: &commonpb.MsgBase{
			Timestamp: uint64(time.Now().Unix()),
		},
		SegmentID: 2,
		Channel:   "ch1",
	})
	s.NoError(err)
	s.ErrorIs(merr.Error(resp), merr.ErrSegmentNotFound)
}

func (s *ServerSuite) TestSaveBinlogPath_SaveDroppedSegment() {
	s.mockChMgr.EXPECT().Match(int64(0), "ch1").Return(true)
	s.testServer.meta.AddCollection(&collectionInfo{ID: 0})

	segments := map[int64]int64{
		0: 0,
		1: 0,
	}
	for segID, collID := range segments {
		info := &datapb.SegmentInfo{
			ID:            segID,
			CollectionID:  collID,
			InsertChannel: "ch1",
			State:         commonpb.SegmentState_Dropped,
		}
		err := s.testServer.meta.AddSegment(context.TODO(), NewSegmentInfo(info))
		s.Require().NoError(err)
	}

	ctx := context.Background()
	resp, err := s.testServer.SaveBinlogPaths(ctx, &datapb.SaveBinlogPathsRequest{
		Base: &commonpb.MsgBase{
			Timestamp: uint64(time.Now().Unix()),
		},
		SegmentID:    1,
		CollectionID: 0,
		Channel:      "ch1",
		Flushed:      false,
	})
	s.NoError(err)
	s.EqualValues(resp.ErrorCode, commonpb.ErrorCode_Success)

	segment := s.testServer.meta.GetSegment(1)
	s.NotNil(segment)
	s.EqualValues(0, len(segment.GetBinlogs()))
	s.EqualValues(segment.NumOfRows, 0)
}

func (s *ServerSuite) TestSaveBinlogPath_L0Segment() {
	s.mockChMgr.EXPECT().Match(int64(0), "ch1").Return(true)
	s.testServer.meta.AddCollection(&collectionInfo{ID: 0})

	segment := s.testServer.meta.GetHealthySegment(1)
	s.Require().Nil(segment)
	ctx := context.Background()
	resp, err := s.testServer.SaveBinlogPaths(ctx, &datapb.SaveBinlogPathsRequest{
		Base: &commonpb.MsgBase{
			Timestamp: uint64(time.Now().Unix()),
		},
		SegmentID:    1,
		PartitionID:  1,
		CollectionID: 0,
		SegLevel:     datapb.SegmentLevel_L0,
		Channel:      "ch1",
		Deltalogs: []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						LogPath:    "/by-dev/test/0/1/1/1/Allo1",
						EntriesNum: 5,
					},
					{
						LogPath:    "/by-dev/test/0/1/1/1/Allo2",
						EntriesNum: 5,
					},
				},
			},
		},
		CheckPoints: []*datapb.CheckPoint{
			{
				SegmentID: 1,
				Position: &msgpb.MsgPosition{
					ChannelName: "ch1",
					MsgID:       []byte{1, 2, 3},
					MsgGroup:    "",
					Timestamp:   0,
				},
				NumOfRows: 12,
			},
		},
		Flushed: true,
	})
	s.NoError(err)
	s.EqualValues(resp.ErrorCode, commonpb.ErrorCode_Success)

	segment = s.testServer.meta.GetHealthySegment(1)
	s.NotNil(segment)
	s.EqualValues(datapb.SegmentLevel_L0, segment.GetLevel())
}

func (s *ServerSuite) TestSaveBinlogPath_NormalCase() {
	s.mockChMgr.EXPECT().Match(int64(0), "ch1").Return(true)
	s.testServer.meta.AddCollection(&collectionInfo{ID: 0})

	segments := map[int64]int64{
		0: 0,
		1: 0,
	}
	for segID, collID := range segments {
		info := &datapb.SegmentInfo{
			ID:            segID,
			CollectionID:  collID,
			InsertChannel: "ch1",
			State:         commonpb.SegmentState_Growing,
		}
		err := s.testServer.meta.AddSegment(context.TODO(), NewSegmentInfo(info))
		s.Require().NoError(err)
	}

	ctx := context.Background()

	resp, err := s.testServer.SaveBinlogPaths(ctx, &datapb.SaveBinlogPathsRequest{
		Base: &commonpb.MsgBase{
			Timestamp: uint64(time.Now().Unix()),
		},
		SegmentID:    1,
		CollectionID: 0,
		Channel:      "ch1",
		Field2BinlogPaths: []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						LogPath:    "/by-dev/test/0/1/1/1/Allo1",
						EntriesNum: 5,
					},
					{
						LogPath:    "/by-dev/test/0/1/1/1/Allo2",
						EntriesNum: 5,
					},
				},
			},
		},
		Field2StatslogPaths: []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						LogPath:    "/by-dev/test_stats/0/1/1/1/Allo1",
						EntriesNum: 5,
					},
					{
						LogPath:    "/by-dev/test_stats/0/1/1/1/Allo2",
						EntriesNum: 5,
					},
				},
			},
		},
		CheckPoints: []*datapb.CheckPoint{
			{
				SegmentID: 1,
				Position: &msgpb.MsgPosition{
					ChannelName: "ch1",
					MsgID:       []byte{1, 2, 3},
					MsgGroup:    "",
					Timestamp:   0,
				},
				NumOfRows: 12,
			},
		},
		Flushed: false,
	})
	s.NoError(err)
	s.EqualValues(resp.ErrorCode, commonpb.ErrorCode_Success)

	segment := s.testServer.meta.GetHealthySegment(1)
	s.NotNil(segment)
	binlogs := segment.GetBinlogs()
	s.EqualValues(1, len(binlogs))
	fieldBinlogs := binlogs[0]
	s.NotNil(fieldBinlogs)
	s.EqualValues(2, len(fieldBinlogs.GetBinlogs()))
	s.EqualValues(1, fieldBinlogs.GetFieldID())
	s.EqualValues("/by-dev/test/0/1/1/1/Allo1", fieldBinlogs.GetBinlogs()[0].GetLogPath())
	s.EqualValues("/by-dev/test/0/1/1/1/Allo2", fieldBinlogs.GetBinlogs()[1].GetLogPath())

	s.EqualValues(segment.DmlPosition.ChannelName, "ch1")
	s.EqualValues(segment.DmlPosition.MsgID, []byte{1, 2, 3})
	s.EqualValues(segment.NumOfRows, 10)
}

func (s *ServerSuite) TestFlush_NormalCase() {
	req := &datapb.FlushRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Flush,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  0,
		},
		DbID:         0,
		CollectionID: 0,
	}

	s.mockChMgr.EXPECT().GetNodeChannelsByCollectionID(mock.Anything).Return(map[int64][]string{
		1: {"channel-1"},
	})

	mockCluster := NewMockCluster(s.T())
	mockCluster.EXPECT().FlushChannels(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(nil)
	mockCluster.EXPECT().Close().Maybe()
	s.testServer.cluster = mockCluster

	schema := newTestSchema()
	s.testServer.meta.AddCollection(&collectionInfo{ID: 0, Schema: schema, Partitions: []int64{}})
	allocations, err := s.testServer.segmentManager.AllocSegment(context.TODO(), 0, 1, "channel-1", 1)
	s.NoError(err)
	s.EqualValues(1, len(allocations))
	expireTs := allocations[0].ExpireTime
	segID := allocations[0].SegmentID

	resp, err := s.testServer.Flush(context.TODO(), req)
	s.NoError(err)
	s.EqualValues(commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())

	s.testServer.meta.SetCurrentRows(segID, 1)
	ids, err := s.testServer.segmentManager.GetFlushableSegments(context.TODO(), "channel-1", expireTs)
	s.NoError(err)
	s.EqualValues(1, len(ids))
	s.EqualValues(segID, ids[0])
}

func (s *ServerSuite) TestFlush_BulkLoadSegment() {
	req := &datapb.FlushRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Flush,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  0,
		},
		DbID:         0,
		CollectionID: 0,
	}
	s.mockChMgr.EXPECT().GetNodeChannelsByCollectionID(mock.Anything).Return(map[int64][]string{
		1: {"channel-1"},
	}).Twice()

	mockCluster := NewMockCluster(s.T())
	mockCluster.EXPECT().FlushChannels(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(nil).Twice()
	mockCluster.EXPECT().Close().Maybe()
	s.testServer.cluster = mockCluster

	schema := newTestSchema()
	s.testServer.meta.AddCollection(&collectionInfo{ID: 0, Schema: schema, Partitions: []int64{}})

	allocations, err := s.testServer.segmentManager.allocSegmentForImport(context.TODO(), 0, 1, "channel-1", 1, 100)
	s.NoError(err)
	expireTs := allocations.ExpireTime
	segID := allocations.SegmentID

	resp, err := s.testServer.Flush(context.TODO(), req)
	s.NoError(err)
	s.EqualValues(commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	s.EqualValues(0, len(resp.SegmentIDs))
	// should not flush anything since this is a normal flush
	s.testServer.meta.SetCurrentRows(segID, 1)
	ids, err := s.testServer.segmentManager.GetFlushableSegments(context.TODO(), "channel-1", expireTs)
	s.NoError(err)
	s.EqualValues(0, len(ids))

	req = &datapb.FlushRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Flush,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  0,
		},
		DbID:         0,
		CollectionID: 0,
		IsImport:     true,
	}

	resp, err = s.testServer.Flush(context.TODO(), req)
	s.NoError(err)
	s.EqualValues(commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	s.EqualValues(1, len(resp.SegmentIDs))

	ids, err = s.testServer.segmentManager.GetFlushableSegments(context.TODO(), "channel-1", expireTs)
	s.NoError(err)
	s.EqualValues(1, len(ids))
	s.EqualValues(segID, ids[0])
}

func (s *ServerSuite) TestFlush_ClosedServer() {
	s.TearDownTest()
	req := &datapb.FlushRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Flush,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  0,
		},
		DbID:         0,
		CollectionID: 0,
	}
	resp, err := s.testServer.Flush(context.Background(), req)
	s.NoError(err)
	s.ErrorIs(merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
}

func (s *ServerSuite) TestFlush_RollingUpgrade() {
	req := &datapb.FlushRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Flush,
			MsgID:     0,
			Timestamp: 0,
			SourceID:  0,
		},
		DbID:         0,
		CollectionID: 0,
	}
	mockCluster := NewMockCluster(s.T())
	mockCluster.EXPECT().FlushChannels(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(merr.WrapErrServiceUnimplemented(grpcStatus.Error(codes.Unimplemented, "mock grpc unimplemented error")))
	mockCluster.EXPECT().Close().Maybe()
	s.testServer.cluster = mockCluster
	s.mockChMgr.EXPECT().GetNodeChannelsByCollectionID(mock.Anything).Return(map[int64][]string{
		1: {"channel-1"},
	}).Once()

	resp, err := s.testServer.Flush(context.TODO(), req)
	s.NoError(err)
	s.EqualValues(commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	s.EqualValues(0, resp.GetFlushTs())
}

func (s *ServerSuite) TestGetSegmentInfoChannel() {
	resp, err := s.testServer.GetSegmentInfoChannel(context.TODO(), nil)
	s.NoError(err)
	s.EqualValues(commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	s.EqualValues(Params.CommonCfg.DataCoordSegmentInfo.GetValue(), resp.Value)
}

func (s *ServerSuite) TestAssignSegmentID() {
	s.TearDownTest()
	const collID = 100
	const collIDInvalid = 101
	const partID = 0
	const channel0 = "channel0"

	s.Run("assign segment normally", func() {
		s.SetupTest()
		defer s.TearDownTest()

		schema := newTestSchema()
		s.testServer.meta.AddCollection(&collectionInfo{
			ID:         collID,
			Schema:     schema,
			Partitions: []int64{},
		})
		req := &datapb.SegmentIDRequest{
			Count:        1000,
			ChannelName:  channel0,
			CollectionID: collID,
			PartitionID:  partID,
		}

		resp, err := s.testServer.AssignSegmentID(context.TODO(), &datapb.AssignSegmentIDRequest{
			NodeID:            0,
			PeerRole:          "",
			SegmentIDRequests: []*datapb.SegmentIDRequest{req},
		})
		s.NoError(err)
		s.EqualValues(1, len(resp.SegIDAssignments))
		assign := resp.SegIDAssignments[0]
		s.EqualValues(commonpb.ErrorCode_Success, assign.GetStatus().GetErrorCode())
		s.EqualValues(collID, assign.CollectionID)
		s.EqualValues(partID, assign.PartitionID)
		s.EqualValues(channel0, assign.ChannelName)
		s.EqualValues(1000, assign.Count)
	})

	s.Run("assign segment for bulkload", func() {
		s.SetupTest()
		defer s.TearDownTest()

		schema := newTestSchema()
		s.testServer.meta.AddCollection(&collectionInfo{
			ID:         collID,
			Schema:     schema,
			Partitions: []int64{},
		})
		req := &datapb.SegmentIDRequest{
			Count:        1000,
			ChannelName:  channel0,
			CollectionID: collID,
			PartitionID:  partID,
			IsImport:     true,
		}

		resp, err := s.testServer.AssignSegmentID(context.TODO(), &datapb.AssignSegmentIDRequest{
			NodeID:            0,
			PeerRole:          "",
			SegmentIDRequests: []*datapb.SegmentIDRequest{req},
		})
		s.NoError(err)
		s.EqualValues(1, len(resp.SegIDAssignments))
		assign := resp.SegIDAssignments[0]
		s.EqualValues(commonpb.ErrorCode_Success, assign.GetStatus().GetErrorCode())
		s.EqualValues(collID, assign.CollectionID)
		s.EqualValues(partID, assign.PartitionID)
		s.EqualValues(channel0, assign.ChannelName)
		s.EqualValues(1000, assign.Count)
	})

	s.Run("with closed server", func() {
		s.SetupTest()
		s.TearDownTest()

		req := &datapb.SegmentIDRequest{
			Count:        100,
			ChannelName:  channel0,
			CollectionID: collID,
			PartitionID:  partID,
		}
		resp, err := s.testServer.AssignSegmentID(context.Background(), &datapb.AssignSegmentIDRequest{
			NodeID:            0,
			PeerRole:          "",
			SegmentIDRequests: []*datapb.SegmentIDRequest{req},
		})
		s.NoError(err)
		s.ErrorIs(merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	s.Run("assign segment with invalid collection", func() {
		s.SetupTest()
		defer s.TearDownTest()

		s.testServer.rootCoordClient = &mockRootCoord{
			RootCoordClient: s.testServer.rootCoordClient,
			collID:          collID,
		}

		schema := newTestSchema()
		s.testServer.meta.AddCollection(&collectionInfo{
			ID:         collID,
			Schema:     schema,
			Partitions: []int64{},
		})
		req := &datapb.SegmentIDRequest{
			Count:        1000,
			ChannelName:  channel0,
			CollectionID: collIDInvalid,
			PartitionID:  partID,
		}

		resp, err := s.testServer.AssignSegmentID(context.TODO(), &datapb.AssignSegmentIDRequest{
			NodeID:            0,
			PeerRole:          "",
			SegmentIDRequests: []*datapb.SegmentIDRequest{req},
		})
		s.NoError(err)
		s.EqualValues(0, len(resp.SegIDAssignments))
	})
}

func TestBroadcastAlteredCollection(t *testing.T) {
	t.Run("test server is closed", func(t *testing.T) {
		s := &Server{}
		s.stateCode.Store(commonpb.StateCode_Initializing)
		ctx := context.Background()
		resp, err := s.BroadcastAlteredCollection(ctx, nil)
		assert.NotNil(t, resp.Reason)
		assert.NoError(t, err)
	})

	t.Run("test meta non exist", func(t *testing.T) {
		s := &Server{meta: &meta{collections: make(map[UniqueID]*collectionInfo, 1)}}
		s.stateCode.Store(commonpb.StateCode_Healthy)
		ctx := context.Background()
		req := &datapb.AlterCollectionRequest{
			CollectionID: 1,
			PartitionIDs: []int64{1},
			Properties:   []*commonpb.KeyValuePair{{Key: "k", Value: "v"}},
		}
		resp, err := s.BroadcastAlteredCollection(ctx, req)
		assert.NotNil(t, resp)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(s.meta.collections))
	})

	t.Run("test update meta", func(t *testing.T) {
		s := &Server{meta: &meta{collections: map[UniqueID]*collectionInfo{
			1: {ID: 1},
		}}}
		s.stateCode.Store(commonpb.StateCode_Healthy)
		ctx := context.Background()
		req := &datapb.AlterCollectionRequest{
			CollectionID: 1,
			PartitionIDs: []int64{1},
			Properties:   []*commonpb.KeyValuePair{{Key: "k", Value: "v"}},
		}

		assert.Nil(t, s.meta.collections[1].Properties)
		resp, err := s.BroadcastAlteredCollection(ctx, req)
		assert.NotNil(t, resp)
		assert.NoError(t, err)
		assert.NotNil(t, s.meta.collections[1].Properties)
	})
}

func TestServer_GcConfirm(t *testing.T) {
	t.Run("closed server", func(t *testing.T) {
		s := &Server{}
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.GcConfirm(context.TODO(), &datapb.GcConfirmRequest{CollectionId: 100, PartitionId: 10000})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		s := &Server{}
		s.stateCode.Store(commonpb.StateCode_Healthy)

		m := &meta{}
		catalog := mocks.NewDataCoordCatalog(t)
		m.catalog = catalog

		catalog.On("GcConfirm",
			mock.Anything,
			mock.AnythingOfType("int64"),
			mock.AnythingOfType("int64")).
			Return(false)

		s.meta = m

		resp, err := s.GcConfirm(context.TODO(), &datapb.GcConfirmRequest{CollectionId: 100, PartitionId: 10000})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.False(t, resp.GetGcFinished())
	})
}

func TestGetRecoveryInfoV2(t *testing.T) {
	t.Run("test get recovery info with no segments", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 0, len(resp.GetChannels()))
	})

	createSegment := func(id, collectionID, partitionID, numOfRows int64, posTs uint64,
		channel string, state commonpb.SegmentState,
	) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID:            id,
			CollectionID:  collectionID,
			PartitionID:   partitionID,
			InsertChannel: channel,
			NumOfRows:     numOfRows,
			State:         state,
			DmlPosition: &msgpb.MsgPosition{
				ChannelName: channel,
				MsgID:       []byte{},
				Timestamp:   posTs,
			},
			StartPosition: &msgpb.MsgPosition{
				ChannelName: "",
				MsgID:       []byte{},
				MsgGroup:    "",
				Timestamp:   0,
			},
		}
	}

	t.Run("test get earliest position of flushed segments as seek position", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   10,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		err = svr.meta.CreateIndex(&model.Index{
			TenantID:     "",
			CollectionID: 0,
			FieldID:      2,
			IndexID:      0,
			IndexName:    "",
		})
		assert.NoError(t, err)

		seg1 := createSegment(0, 0, 0, 100, 10, "vchan1", commonpb.SegmentState_Flushed)
		seg1.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 0, 1, 901),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 0, 1, 902),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 0, 1, 903),
					},
				},
			},
		}
		seg2 := createSegment(1, 0, 0, 100, 20, "vchan1", commonpb.SegmentState_Flushed)
		seg2.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 30,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 1, 1, 801),
					},
					{
						EntriesNum: 70,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 1, 1, 802),
					},
				},
			},
		}
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)
		err = svr.meta.AddSegmentIndex(&model.SegmentIndex{
			SegmentID: seg1.ID,
			BuildID:   seg1.ID,
		})
		assert.NoError(t, err)
		err = svr.meta.FinishTask(&indexpb.IndexTaskInfo{
			BuildID: seg1.ID,
			State:   commonpb.IndexState_Finished,
		})
		assert.NoError(t, err)
		err = svr.meta.AddSegmentIndex(&model.SegmentIndex{
			SegmentID: seg2.ID,
			BuildID:   seg2.ID,
		})
		assert.NoError(t, err)
		err = svr.meta.FinishTask(&indexpb.IndexTaskInfo{
			BuildID: seg2.ID,
			State:   commonpb.IndexState_Finished,
		})
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.EqualValues(t, 0, len(resp.GetChannels()[0].GetUnflushedSegmentIds()))
		assert.ElementsMatch(t, []int64{0, 1}, resp.GetChannels()[0].GetFlushedSegmentIds())
		assert.EqualValues(t, 10, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
		assert.EqualValues(t, 2, len(resp.GetSegments()))
		// Row count corrected from 100 + 100 -> 100 + 60.
		assert.EqualValues(t, 160, resp.GetSegments()[0].GetNumOfRows()+resp.GetSegments()[1].GetNumOfRows())
	})

	t.Run("test get recovery of unflushed segments ", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		seg1 := createSegment(3, 0, 0, 100, 30, "vchan1", commonpb.SegmentState_Growing)
		seg1.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 3, 1, 901),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 3, 1, 902),
					},
					{
						EntriesNum: 20,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 3, 1, 903),
					},
				},
			},
		}
		seg2 := createSegment(4, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Growing)
		seg2.Binlogs = []*datapb.FieldBinlog{
			{
				FieldID: 1,
				Binlogs: []*datapb.Binlog{
					{
						EntriesNum: 30,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 4, 1, 801),
					},
					{
						EntriesNum: 70,
						LogPath:    metautil.BuildInsertLogPath("a", 0, 0, 4, 1, 802),
					},
				},
			},
		}
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
	})

	t.Run("test get binlogs", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.meta.AddCollection(&collectionInfo{
			Schema: newTestSchema(),
		})

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		binlogReq := &datapb.SaveBinlogPathsRequest{
			SegmentID:    0,
			CollectionID: 0,
			Field2BinlogPaths: []*datapb.FieldBinlog{
				{
					FieldID: 1,
					Binlogs: []*datapb.Binlog{
						{
							LogPath: metautil.BuildInsertLogPath("a", 0, 100, 0, 1, 801),
						},
						{
							LogPath: metautil.BuildInsertLogPath("a", 0, 100, 0, 1, 801),
						},
					},
				},
			},
			Field2StatslogPaths: []*datapb.FieldBinlog{
				{
					FieldID: 1,
					Binlogs: []*datapb.Binlog{
						{
							LogPath: metautil.BuildStatsLogPath("a", 0, 100, 0, 1000, 10000),
						},
						{
							LogPath: metautil.BuildStatsLogPath("a", 0, 100, 0, 1000, 10000),
						},
					},
				},
			},
			Deltalogs: []*datapb.FieldBinlog{
				{
					Binlogs: []*datapb.Binlog{
						{
							TimestampFrom: 0,
							TimestampTo:   1,
							LogPath:       metautil.BuildDeltaLogPath("a", 0, 100, 0, 100000),
							LogSize:       1,
						},
					},
				},
			},
		}
		segment := createSegment(0, 0, 1, 100, 10, "vchan1", commonpb.SegmentState_Flushed)
		err := svr.meta.AddSegment(context.TODO(), NewSegmentInfo(segment))
		assert.NoError(t, err)

		err = svr.meta.CreateIndex(&model.Index{
			TenantID:     "",
			CollectionID: 0,
			FieldID:      2,
			IndexID:      0,
			IndexName:    "",
		})
		assert.NoError(t, err)
		err = svr.meta.AddSegmentIndex(&model.SegmentIndex{
			SegmentID: segment.ID,
			BuildID:   segment.ID,
		})
		assert.NoError(t, err)
		err = svr.meta.FinishTask(&indexpb.IndexTaskInfo{
			BuildID: segment.ID,
			State:   commonpb.IndexState_Finished,
		})
		assert.NoError(t, err)

		err = svr.channelManager.AddNode(0)
		assert.NoError(t, err)
		err = svr.channelManager.Watch(context.Background(), &channelMeta{Name: "vchan1", CollectionID: 0})
		assert.NoError(t, err)

		sResp, err := svr.SaveBinlogPaths(context.TODO(), binlogReq)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, sResp.ErrorCode)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
			PartitionIDs: []int64{1},
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.NoError(t, merr.Error(resp.Status))
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 1, len(resp.GetSegments()))
		assert.EqualValues(t, 0, resp.GetSegments()[0].GetID())
		assert.EqualValues(t, 0, len(resp.GetSegments()[0].GetBinlogs()))
	})
	t.Run("with dropped segments", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		seg1 := createSegment(7, 0, 0, 100, 30, "vchan1", commonpb.SegmentState_Growing)
		seg2 := createSegment(8, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Dropped)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
		assert.Len(t, resp.GetChannels()[0].GetDroppedSegmentIds(), 1)
		assert.Equal(t, UniqueID(8), resp.GetChannels()[0].GetDroppedSegmentIds()[0])
	})

	t.Run("with fake segments", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		require.NoError(t, err)

		seg1 := createSegment(7, 0, 0, 100, 30, "vchan1", commonpb.SegmentState_Growing)
		seg2 := createSegment(8, 0, 0, 100, 40, "vchan1", commonpb.SegmentState_Flushed)
		seg2.IsFake = true
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.EqualValues(t, 0, len(resp.GetSegments()))
		assert.EqualValues(t, 1, len(resp.GetChannels()))
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
	})

	t.Run("with continuous compaction", func(t *testing.T) {
		svr := newTestServer(t, nil)
		defer closeTestServer(t, svr)

		svr.rootCoordClientCreator = func(ctx context.Context) (types.RootCoordClient, error) {
			return newMockRootCoordClient(), nil
		}

		svr.meta.AddCollection(&collectionInfo{
			ID:     0,
			Schema: newTestSchema(),
		})

		err := svr.meta.UpdateChannelCheckpoint("vchan1", &msgpb.MsgPosition{
			ChannelName: "vchan1",
			Timestamp:   0,
			MsgID:       []byte{0, 0, 0, 0, 0, 0, 0, 0},
		})
		assert.NoError(t, err)

		seg1 := createSegment(9, 0, 0, 2048, 30, "vchan1", commonpb.SegmentState_Dropped)
		seg2 := createSegment(10, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Dropped)
		seg3 := createSegment(11, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Dropped)
		seg3.CompactionFrom = []int64{9, 10}
		seg4 := createSegment(12, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Dropped)
		seg5 := createSegment(13, 0, 0, 2048, 40, "vchan1", commonpb.SegmentState_Flushed)
		seg5.CompactionFrom = []int64{11, 12}
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg3))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg4))
		assert.NoError(t, err)
		err = svr.meta.AddSegment(context.TODO(), NewSegmentInfo(seg5))
		assert.NoError(t, err)
		err = svr.meta.CreateIndex(&model.Index{
			TenantID:        "",
			CollectionID:    0,
			FieldID:         2,
			IndexID:         0,
			IndexName:       "_default_idx_2",
			IsDeleted:       false,
			CreateTime:      0,
			TypeParams:      nil,
			IndexParams:     nil,
			IsAutoIndex:     false,
			UserIndexParams: nil,
		})
		assert.NoError(t, err)
		svr.meta.segments.SetSegmentIndex(seg4.ID, &model.SegmentIndex{
			SegmentID:     seg4.ID,
			CollectionID:  0,
			PartitionID:   0,
			NumRows:       100,
			IndexID:       0,
			BuildID:       0,
			NodeID:        0,
			IndexVersion:  1,
			IndexState:    commonpb.IndexState_Finished,
			FailReason:    "",
			IsDeleted:     false,
			CreateTime:    0,
			IndexFileKeys: nil,
			IndexSize:     0,
		})

		ch := &channelMeta{Name: "vchan1", CollectionID: 0}
		svr.channelManager.AddNode(0)
		svr.channelManager.Watch(context.Background(), ch)

		req := &datapb.GetRecoveryInfoRequestV2{
			CollectionID: 0,
		}
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), req)
		assert.NoError(t, err)
		assert.EqualValues(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.NotNil(t, resp.GetChannels()[0].SeekPosition)
		assert.NotEqual(t, 0, resp.GetChannels()[0].GetSeekPosition().GetTimestamp())
		assert.Len(t, resp.GetChannels()[0].GetDroppedSegmentIds(), 0)
		assert.ElementsMatch(t, []UniqueID{}, resp.GetChannels()[0].GetUnflushedSegmentIds())
		assert.ElementsMatch(t, []UniqueID{9, 10, 12}, resp.GetChannels()[0].GetFlushedSegmentIds())
	})

	t.Run("with closed server", func(t *testing.T) {
		svr := newTestServer(t, nil)
		closeTestServer(t, svr)
		resp, err := svr.GetRecoveryInfoV2(context.TODO(), &datapb.GetRecoveryInfoRequestV2{})
		assert.NoError(t, err)
		err = merr.Error(resp.GetStatus())
		assert.ErrorIs(t, err, merr.ErrServiceNotReady)
	})
}

type GcControlServiceSuite struct {
	suite.Suite

	server *Server
}

func (s *GcControlServiceSuite) SetupTest() {
	s.server = newTestServer(s.T(), nil)
}

func (s *GcControlServiceSuite) TearDownTest() {
	if s.server != nil {
		closeTestServer(s.T(), s.server)
	}
}

func (s *GcControlServiceSuite) TestClosedServer() {
	closeTestServer(s.T(), s.server)
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{})
	s.NoError(err)
	s.False(merr.Ok(resp))
	s.server = nil
}

func (s *GcControlServiceSuite) TestUnknownCmd() {
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: 0,
	})
	s.NoError(err)
	s.False(merr.Ok(resp))
}

func (s *GcControlServiceSuite) TestPause() {
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
	})
	s.Nil(err)
	s.False(merr.Ok(resp))

	resp, err = s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
		Params: []*commonpb.KeyValuePair{
			{Key: "duration", Value: "not_int"},
		},
	})
	s.Nil(err)
	s.False(merr.Ok(resp))

	resp, err = s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
		Params: []*commonpb.KeyValuePair{
			{Key: "duration", Value: "60"},
		},
	})
	s.Nil(err)
	s.True(merr.Ok(resp))
}

func (s *GcControlServiceSuite) TestResume() {
	resp, err := s.server.GcControl(context.TODO(), &datapb.GcControlRequest{
		Command: datapb.GcCommand_Resume,
	})
	s.Nil(err)
	s.True(merr.Ok(resp))
}

func (s *GcControlServiceSuite) TestTimeoutCtx() {
	s.server.garbageCollector.close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	resp, err := s.server.GcControl(ctx, &datapb.GcControlRequest{
		Command: datapb.GcCommand_Resume,
	})
	s.Nil(err)
	s.False(merr.Ok(resp))

	resp, err = s.server.GcControl(ctx, &datapb.GcControlRequest{
		Command: datapb.GcCommand_Pause,
		Params: []*commonpb.KeyValuePair{
			{Key: "duration", Value: "60"},
		},
	})
	s.Nil(err)
	s.False(merr.Ok(resp))
}

func TestGcControlService(t *testing.T) {
	suite.Run(t, new(GcControlServiceSuite))
}
