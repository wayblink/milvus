package rootcoord

import (
	"context"
	"errors"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"

	"github.com/milvus-io/milvus/internal/util/funcutil"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/allocator"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/api/milvuspb"
)

func TestRootCoord_CreateCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DropCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DescribeCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_HasCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.HasCollection(ctx, &milvuspb.HasCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_ShowCollections(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_CreatePartition(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.CreatePartition(ctx, &milvuspb.CreatePartitionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DropPartition(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.DropPartition(ctx, &milvuspb.DropPartitionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_HasPartition(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())

		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())

		ctx := context.Background()
		resp, err := c.HasPartition(ctx, &milvuspb.HasPartitionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_ShowPartitions(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.ShowPartitions(ctx, &milvuspb.ShowPartitionsRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestRootCoord_CreateAlias(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.CreateAlias(ctx, &milvuspb.CreateAliasRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_DropAlias(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.DropAlias(ctx, &milvuspb.DropAliasRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_AlterAlias(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withTaskFailScheduler())
		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withValidScheduler())
		ctx := context.Background()
		resp, err := c.AlterAlias(ctx, &milvuspb.AlterAliasRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_AllocTimestamp(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.AllocTimestamp(ctx, &rootcoordpb.AllocTimestampRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to allocate ts", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withInvalidTsoAllocator())
		resp, err := c.AllocTimestamp(ctx, &rootcoordpb.AllocTimestampRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		alloc := newMockTsoAllocator()
		count := uint32(10)
		ts := Timestamp(100)
		alloc.GenerateTSOF = func(count uint32) (uint64, error) {
			// end ts
			return ts, nil
		}
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withTsoAllocator(alloc))
		resp, err := c.AllocTimestamp(ctx, &rootcoordpb.AllocTimestampRequest{Count: count})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		// begin ts
		assert.Equal(t, ts-uint64(count)+1, resp.GetTimestamp())
		assert.Equal(t, count, resp.GetCount())
	})
}

func TestRootCoord_AllocID(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.AllocID(ctx, &rootcoordpb.AllocIDRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to allocate id", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withInvalidIDAllocator())
		resp, err := c.AllocID(ctx, &rootcoordpb.AllocIDRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		alloc := newMockIDAllocator()
		id := UniqueID(100)
		alloc.AllocF = func(count uint32) (allocator.UniqueID, allocator.UniqueID, error) {
			return id, id + int64(count), nil
		}
		count := uint32(10)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withIDAllocator(alloc))
		resp, err := c.AllocID(ctx, &rootcoordpb.AllocIDRequest{Count: count})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, id, resp.GetID())
		assert.Equal(t, count, resp.GetCount())
	})
}

func TestRootCoord_UpdateChannelTimeTick(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("invalid msg type", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{Base: &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection}})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("invalid msg", func(t *testing.T) {
		defer cleanTestEnv()

		ticker := newRocksMqTtSynchronizer()

		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withTtSynchronizer(ticker))

		// the length of channel names & timestamps mismatch.
		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_TimeTick,
			},
			ChannelNames: []string{funcutil.GenRandomStr()},
			Timestamps:   []uint64{},
		})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		defer cleanTestEnv()

		source := int64(20220824)
		ts := Timestamp(100)
		defaultTs := Timestamp(101)

		ticker := newRocksMqTtSynchronizer()
		ticker.addSession(&sessionutil.Session{ServerID: source})

		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withTtSynchronizer(ticker))

		resp, err := c.UpdateChannelTimeTick(ctx, &internalpb.ChannelTimeTickMsg{
			Base: &commonpb.MsgBase{
				SourceID: source,
				MsgType:  commonpb.MsgType_TimeTick,
			},
			ChannelNames:     []string{funcutil.GenRandomStr()},
			Timestamps:       []uint64{ts},
			DefaultTimestamp: defaultTs,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_InvalidateCollectionMetaCache(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.InvalidateCollectionMetaCache(ctx, &proxypb.InvalidateCollMetaCacheRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to invalidate cache", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withInvalidProxyManager())
		resp, err := c.InvalidateCollectionMetaCache(ctx, &proxypb.InvalidateCollMetaCacheRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withValidProxyManager())
		resp, err := c.InvalidateCollectionMetaCache(ctx, &proxypb.InvalidateCollMetaCacheRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestRootCoord_ShowConfigurations(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.ShowConfigurations(ctx, &internalpb.ShowConfigurationsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		Params.InitOnce()

		pattern := "Port"
		req := &internalpb.ShowConfigurationsRequest{
			Base: &commonpb.MsgBase{
				MsgID: rand.Int63(),
			},
			Pattern: pattern,
		}

		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.ShowConfigurations(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, 1, len(resp.GetConfiguations()))
		assert.Equal(t, "rootcoord.port", resp.GetConfiguations()[0].Key)
	})
}

func TestRootCoord_GetMetrics(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.GetMetrics(ctx, &milvuspb.GetMetricsRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("failed to parse metric type", func(t *testing.T) {
		req := &milvuspb.GetMetricsRequest{
			Request: "invalid request",
		}
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("unsupported metric type", func(t *testing.T) {
		// unsupported metric type
		unsupportedMetricType := "unsupported"
		req, err := metricsinfo.ConstructRequestByMetricType(unsupportedMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("get system info metrics from cache", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		c.metricsCacheManager.UpdateSystemInfoMetrics(&milvuspb.GetMetricsResponse{
			Status:        succStatus(),
			Response:      "cached response",
			ComponentName: "cached component",
		})
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("get system info metrics, cache miss", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		c.metricsCacheManager.InvalidateSystemInfoMetrics()
		resp, err := c.GetMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("get system info metrics", func(t *testing.T) {
		systemInfoMetricType := metricsinfo.SystemInfoMetrics
		req, err := metricsinfo.ConstructRequestByMetricType(systemInfoMetricType)
		assert.NoError(t, err)
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMetricsCacheManager())
		resp, err := c.getSystemInfoMetrics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})
}

func TestCore_Import(t *testing.T) {
	meta := newMockMetaTable()
	meta.AddCollectionFunc = func(ctx context.Context, coll *model.Collection) error {
		return nil
	}
	meta.ChangeCollectionStateFunc = func(ctx context.Context, collectionID UniqueID, state etcdpb.CollectionState, ts Timestamp) error {
		return nil
	}

	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.Import(ctx, &milvuspb.ImportRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("bad collection name", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMeta(meta))
		meta.GetCollectionIDByNameFunc = func(name string) (UniqueID, error) {
			return 0, errors.New("error mock GetCollectionIDByName")
		}
		_, err := c.Import(ctx, &milvuspb.ImportRequest{
			CollectionName: "a-bad-name",
		})
		assert.Error(t, err)
	})

	t.Run("bad partition name", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMeta(meta))
		meta.GetCollectionIDByNameFunc = func(name string) (UniqueID, error) {
			return 100, nil
		}
		meta.GetCollectionVirtualChannelsFunc = func(colID int64) []string {
			return []string{"ch-1", "ch-2"}
		}
		meta.GetPartitionByNameFunc = func(collID UniqueID, partitionName string, ts Timestamp) (UniqueID, error) {
			return 0, errors.New("mock GetPartitionByNameFunc error")
		}
		_, err := c.Import(ctx, &milvuspb.ImportRequest{
			CollectionName: "a-good-name",
		})
		assert.Error(t, err)
	})

	t.Run("normal case", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode(),
			withMeta(meta))
		meta.GetCollectionIDByNameFunc = func(name string) (UniqueID, error) {
			return 100, nil
		}
		meta.GetCollectionVirtualChannelsFunc = func(colID int64) []string {
			return []string{"ch-1", "ch-2"}
		}
		meta.GetPartitionByNameFunc = func(collID UniqueID, partitionName string, ts Timestamp) (UniqueID, error) {
			return 101, nil
		}
		_, err := c.Import(ctx, &milvuspb.ImportRequest{
			CollectionName: "a-good-name",
		})
		assert.NoError(t, err)
	})
}

func TestCore_GetImportState(t *testing.T) {
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id: 200,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(1), "value")
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))

	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.GetImportState(ctx, &milvuspb.GetImportStateRequest{
			Task: 100,
		})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, nil, nil, nil, nil)
		resp, err := c.GetImportState(ctx, &milvuspb.GetImportStateRequest{
			Task: 100,
		})
		assert.NoError(t, err)
		assert.Equal(t, int64(100), resp.GetId())
		assert.NotEqual(t, 0, resp.GetCreateTs())
		assert.Equal(t, commonpb.ImportState_ImportPending, resp.GetState())
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestCore_ListImportTasks(t *testing.T) {
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id:             100,
		CollectionName: "collection-A",
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id:             200,
		CollectionName: "collection-A",
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti3 := &datapb.ImportTaskInfo{
		Id:             300,
		CollectionName: "collection-B",
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	taskInfo3, err := proto.Marshal(ti3)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(1), "value")
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))
	mockKv.Save(BuildImportTaskKey(300), string(taskInfo3))

	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.ListImportTasks(ctx, &milvuspb.ListImportTasksRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("normal case", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, nil, nil, nil, nil)
		resp, err := c.ListImportTasks(ctx, &milvuspb.ListImportTasksRequest{})
		assert.NoError(t, err)
		assert.Equal(t, 3, len(resp.GetTasks()))
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestCore_ReportImport(t *testing.T) {
	Params.RootCoordCfg.ImportTaskSubPath = "importtask"
	var countLock sync.RWMutex
	var globalCount = typeutil.UniqueID(0)
	var idAlloc = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		countLock.Lock()
		defer countLock.Unlock()
		globalCount++
		return globalCount, 0, nil
	}
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id: 100,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id: 200,
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(1), "value")
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))

	ticker := newRocksMqTtSynchronizer()
	meta := newMockMetaTable()
	meta.GetCollectionByNameFunc = func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
		return nil, errors.New("error mock GetCollectionByName")
	}
	meta.AddCollectionFunc = func(ctx context.Context, coll *model.Collection) error {
		return nil
	}
	meta.ChangeCollectionStateFunc = func(ctx context.Context, collectionID UniqueID, state etcdpb.CollectionState, ts Timestamp) error {
		return nil
	}

	dc := newMockDataCoord()
	dc.GetComponentStatesFunc = func(ctx context.Context) (*internalpb.ComponentStates, error) {
		return &internalpb.ComponentStates{
			State: &internalpb.ComponentInfo{
				NodeID:    TestRootCoordID,
				StateCode: internalpb.StateCode_Healthy,
			},
			SubcomponentStates: nil,
			Status:             succStatus(),
		}, nil
	}
	dc.WatchChannelsFunc = func(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error) {
		return &datapb.WatchChannelsResponse{Status: succStatus()}, nil
	}
	dc.FlushFunc = func(ctx context.Context, req *datapb.FlushRequest) (*datapb.FlushResponse, error) {
		return &datapb.FlushResponse{Status: succStatus()}, nil
	}

	mockCallImportServiceErr := false
	callImportServiceFn := func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		if mockCallImportServiceErr {
			return &datapb.ImportTaskResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
			}, errors.New("mock err")
		}
		return &datapb.ImportTaskResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		}, nil
	}
	callMarkSegmentsDropped := func(ctx context.Context, segIDs []typeutil.UniqueID) (*commonpb.Status, error) {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	t.Run("not healthy", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withAbnormalCode())
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("report complete import", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 100,
			State:  commonpb.ImportState_ImportCompleted,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
		// Change the state back.
		err = c.importManager.setImportTaskState(100, commonpb.ImportState_ImportPending)
		assert.NoError(t, err)
	})

	t.Run("report complete import with task not found", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 101,
			State:  commonpb.ImportState_ImportCompleted,
		})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("report import started state", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(withHealthyCode())
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		c.importManager.loadFromTaskStore(true)
		c.importManager.sendOutTasks(ctx)
		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 100,
			State:  commonpb.ImportState_ImportStarted,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
		// Change the state back.
		err = c.importManager.setImportTaskState(100, commonpb.ImportState_ImportPending)
		assert.NoError(t, err)
	})

	t.Run("report persisted import", func(t *testing.T) {
		ctx := context.Background()
		c := newTestCore(
			withHealthyCode(),
			withValidIDAllocator(),
			withMeta(meta),
			withTtSynchronizer(ticker),
			withDataCoord(dc))
		c.broker = newServerBroker(c)
		c.importManager = newImportManager(ctx, mockKv, idAlloc, callImportServiceFn, callMarkSegmentsDropped, nil)
		c.importManager.loadFromTaskStore(true)
		c.importManager.sendOutTasks(ctx)

		resp, err := c.ReportImport(ctx, &rootcoordpb.ImportResult{
			TaskId: 100,
			State:  commonpb.ImportState_ImportPersisted,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
		// Change the state back.
		err = c.importManager.setImportTaskState(100, commonpb.ImportState_ImportPending)
		assert.NoError(t, err)
	})
}

func TestCore_completeImportAsync(t *testing.T) {
	mockKv := &kv.MockMetaKV{}
	mockKv.InMemKv = sync.Map{}
	ti1 := &datapb.ImportTaskInfo{
		Id:             100,
		CollectionName: "collection-A",
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPending,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti2 := &datapb.ImportTaskInfo{
		Id:             200,
		CollectionName: "collection-A",
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	ti3 := &datapb.ImportTaskInfo{
		Id:             300,
		CollectionName: "collection-B",
		State: &datapb.ImportTaskState{
			StateCode: commonpb.ImportState_ImportPersisted,
		},
		CreateTs: time.Now().Unix() - 100,
	}
	taskInfo1, err := proto.Marshal(ti1)
	assert.NoError(t, err)
	taskInfo2, err := proto.Marshal(ti2)
	assert.NoError(t, err)
	taskInfo3, err := proto.Marshal(ti3)
	assert.NoError(t, err)
	mockKv.Save(BuildImportTaskKey(100), string(taskInfo1))
	mockKv.Save(BuildImportTaskKey(200), string(taskInfo2))
	mockKv.Save(BuildImportTaskKey(300), string(taskInfo3))

	t.Run("wait persisted timeout", func(t *testing.T) {
		CheckTaskPersistedInterval = 50 * time.Millisecond
		CheckTaskPersistedWaitLimit = 300 * time.Millisecond
		c := newTestCore(withHealthyCode())
		c.ctx = context.Background()
		c.importManager = newImportManager(c.ctx, mockKv, nil, nil, nil, nil)
		c.importManager.loadFromTaskStore(true)
		c.completeImportAsync(200)
	})

	t.Run("wait persisted context done", func(t *testing.T) {
		CheckTaskPersistedInterval = 50 * time.Millisecond
		CheckTaskPersistedWaitLimit = 300 * time.Millisecond
		c := newTestCore(withHealthyCode())
		var cancel func()
		c.ctx, cancel = context.WithCancel(context.Background())
		cancel()
		c.importManager = newImportManager(c.ctx, mockKv, nil, nil, nil, nil)
		c.completeImportAsync(200)
	})

	t.Run("normal case", func(t *testing.T) {
		meta := newMockMetaTable()
		Params.RootCoordCfg.ImportIndexCheckInterval = 0.1
		Params.RootCoordCfg.ImportIndexWaitLimit = 0.5
		ctx := context.Background()

		dc := newMockDataCoord()
		dc.UnsetIsImportingStateFunc = func(ctx context.Context, req *datapb.UnsetIsImportingStateRequest) (*commonpb.Status, error) {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			}, nil
		}
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return []*indexpb.SegmentIndexState{
				{
					SegmentID: 200,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 201,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 202,
					State:     commonpb.IndexState_Finished,
				},
			}, nil
		}
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{
					{},
				},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker),
			withDataCoord(dc))
		c.ctx = ctx
		c.importManager = newImportManager(c.ctx, mockKv, nil, nil, nil, nil)
		c.importManager.loadFromTaskStore(true)
		ti3 = &datapb.ImportTaskInfo{
			Id:             300,
			CollectionName: "collection-B",
			State: &datapb.ImportTaskState{
				StateCode: commonpb.ImportState_ImportPersisted,
			},
			CreateTs: time.Now().Unix() - 100,
		}
		taskInfo3, err = proto.Marshal(ti3)
		assert.NoError(t, err)
		mockKv.Save(BuildImportTaskKey(300), string(taskInfo3))
		c.completeImportAsync(300)
	})

	t.Run("unsetting fail", func(t *testing.T) {
		meta := newMockMetaTable()
		Params.RootCoordCfg.ImportIndexCheckInterval = 0.1
		Params.RootCoordCfg.ImportIndexWaitLimit = 0.5
		ctx := context.Background()

		dc := newMockDataCoord()
		dc.UnsetIsImportingStateFunc = func(ctx context.Context, req *datapb.UnsetIsImportingStateRequest) (*commonpb.Status, error) {
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
			}, errors.New("mock error")
		}
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return []*indexpb.SegmentIndexState{
				{
					SegmentID: 200,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 201,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 202,
					State:     commonpb.IndexState_Finished,
				},
			}, nil
		}
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{
					{},
				},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker),
			withDataCoord(dc))
		c.ctx = ctx
		c.importManager = newImportManager(c.ctx, mockKv, nil, nil, nil, nil)
		c.importManager.loadFromTaskStore(true)
		ti3 = &datapb.ImportTaskInfo{
			Id:             300,
			CollectionName: "collection-B",
			State: &datapb.ImportTaskState{
				StateCode: commonpb.ImportState_ImportPersisted,
			},
			CreateTs: time.Now().Unix() - 100,
		}
		taskInfo3, err = proto.Marshal(ti3)
		assert.NoError(t, err)
		mockKv.Save(BuildImportTaskKey(300), string(taskInfo3))
		c.completeImportAsync(300)
	})
}

func TestCore_checkSegmentIndexReady(t *testing.T) {
	meta := newMockMetaTable()
	t.Run("failed to get collection by ID", func(t *testing.T) {
		ctx := context.Background()
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return nil, errors.New("mock GetCollectionByID error")
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta))
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_CollectionNameNotFound, status.GetErrorCode())
	})
	t.Run("failed to describe index", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return nil, errors.New("mock DescribeIndex error")
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker))
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.GetErrorCode())
	})
	t.Run("index not exist", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_IndexNotExist,
				},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker))
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())
	})
	t.Run("zero index info", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker))
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())
	})
	t.Run("context done", func(t *testing.T) {
		Params.RootCoordCfg.ImportIndexCheckInterval = 0.1
		Params.RootCoordCfg.ImportIndexWaitLimit = 0.5
		ctx, cancel := context.WithCancel(context.Background())
		broker := newMockBroker()
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{
					{},
				},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker))
		c.ctx = ctx
		cancel()
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.GetErrorCode())
	})
	t.Run("max wait expire", func(t *testing.T) {
		Params.RootCoordCfg.ImportIndexCheckInterval = 0.1
		Params.RootCoordCfg.ImportIndexWaitLimit = 0.5
		ctx := context.Background()
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return nil, errors.New("mock GetSegmentIndexState error")
		}
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{
					{},
				},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker))
		c.ctx = ctx
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.GetErrorCode())
	})
	t.Run("normal case", func(t *testing.T) {
		Params.RootCoordCfg.ImportIndexCheckInterval = 0.1
		Params.RootCoordCfg.ImportIndexWaitLimit = 0.5
		ctx := context.Background()
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return []*indexpb.SegmentIndexState{
				{
					SegmentID: 200,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 201,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 202,
					State:     commonpb.IndexState_Finished,
				},
			}, nil
		}
		broker.DescribeIndexFunc = func(ctx context.Context, colID UniqueID) (*indexpb.DescribeIndexResponse, error) {
			return &indexpb.DescribeIndexResponse{
				Status: &commonpb.Status{
					ErrorCode: commonpb.ErrorCode_Success,
				},
				IndexInfos: []*indexpb.IndexInfo{
					{},
				},
			}, nil
		}
		meta.GetCollectionByIDFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
			return &model.Collection{
				CollectionID: 100,
				Name:         "collection-A",
			}, nil
		}
		c := newTestCore(withHealthyCode(),
			withMeta(meta),
			withBroker(broker))
		c.ctx = ctx
		status, err := c.checkSegmentIndexReady(ctx, 100, 100, []int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())
	})
}

func TestCore_countCompleteIndex(t *testing.T) {
	t.Run("get segment index state failure", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return nil, errors.New("mock GetSegmentIndexState error")
		}
		c := newTestCore(withHealthyCode(), withBroker(broker))
		done, err := c.countCompleteIndex(ctx, "collection-A", 100,
			[]*indexpb.IndexInfo{
				{
					CollectionID: 100,
					FieldID:      500,
					IndexName:    "_idx_1",
					IndexID:      1001,
				},
				{
					CollectionID: 100,
					FieldID:      600,
					IndexName:    "_idx_2",
					IndexID:      1002,
				},
				{
					CollectionID: 100,
					FieldID:      700,
					IndexName:    "_idx_3",
					IndexID:      1003,
				},
			},
			[]int64{})
		assert.Error(t, err)
		assert.Equal(t, false, done)
	})

	t.Run("index partly done", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return []*indexpb.SegmentIndexState{
				{
					SegmentID: 200,
					State:     commonpb.IndexState_InProgress,
				},
				{
					SegmentID: 201,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 202,
					State:     commonpb.IndexState_Finished,
				},
			}, nil
		}
		c := newTestCore(withHealthyCode(), withBroker(broker))
		done, err := c.countCompleteIndex(ctx, "collection-A", 100,
			[]*indexpb.IndexInfo{
				{
					CollectionID: 100,
					FieldID:      500,
					IndexName:    "_idx_1",
					IndexID:      1001,
				},
				{
					CollectionID: 100,
					FieldID:      600,
					IndexName:    "_idx_2",
					IndexID:      1002,
				},
				{
					CollectionID: 100,
					FieldID:      700,
					IndexName:    "_idx_3",
					IndexID:      1003,
				},
			},
			[]int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, false, done)
	})

	t.Run("checking index with missing segments", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return []*indexpb.SegmentIndexState{
				{
					SegmentID: 200,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 201,
					State:     commonpb.IndexState_Finished,
				},
			}, nil
		}
		c := newTestCore(withHealthyCode(), withBroker(broker))
		done, err := c.countCompleteIndex(ctx, "collection-A", 100,
			[]*indexpb.IndexInfo{
				{
					CollectionID: 100,
					FieldID:      500,
					IndexName:    "_idx_1",
					IndexID:      1001,
				},
				{
					CollectionID: 100,
					FieldID:      600,
					IndexName:    "_idx_2",
					IndexID:      1002,
				},
				{
					CollectionID: 100,
					FieldID:      700,
					IndexName:    "_idx_3",
					IndexID:      1003,
				},
			},
			[]int64{200, 201, 999})
		assert.NoError(t, err)
		assert.Equal(t, false, done)
	})

	t.Run("normal case", func(t *testing.T) {
		ctx := context.Background()
		broker := newMockBroker()
		broker.GetSegmentIndexStateFunc = func(ctx context.Context, collID UniqueID, indexName string, segIDs []UniqueID) ([]*indexpb.SegmentIndexState, error) {
			return []*indexpb.SegmentIndexState{
				{
					SegmentID: 200,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 201,
					State:     commonpb.IndexState_Finished,
				},
				{
					SegmentID: 202,
					State:     commonpb.IndexState_Finished,
				},
			}, nil
		}
		c := newTestCore(withHealthyCode(), withBroker(broker))
		done, err := c.countCompleteIndex(ctx, "collection-A", 100,
			[]*indexpb.IndexInfo{
				{
					CollectionID: 100,
					FieldID:      500,
					IndexName:    "_idx_1",
					IndexID:      1001,
				},
				{
					CollectionID: 100,
					FieldID:      600,
					IndexName:    "_idx_2",
					IndexID:      1002,
				},
				{
					CollectionID: 100,
					FieldID:      700,
					IndexName:    "_idx_3",
					IndexID:      1003,
				},
			},
			[]int64{200, 201, 202})
		assert.NoError(t, err)
		assert.Equal(t, true, done)
	})
}

func TestCore_Rbac(t *testing.T) {
	ctx := context.Background()
	c := &Core{
		ctx: ctx,
	}

	// not healthy.
	c.stateCode.Store(internalpb.StateCode_Abnormal)

	{
		resp, err := c.CreateRole(ctx, &milvuspb.CreateRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.DropRole(ctx, &milvuspb.DropRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.OperateUserRole(ctx, &milvuspb.OperateUserRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.SelectRole(ctx, &milvuspb.SelectRoleRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}

	{
		resp, err := c.SelectUser(ctx, &milvuspb.SelectUserRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}

	{
		resp, err := c.OperatePrivilege(ctx, &milvuspb.OperatePrivilegeRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	}

	{
		resp, err := c.SelectGrant(ctx, &milvuspb.SelectGrantRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}

	{
		resp, err := c.ListPolicy(ctx, &internalpb.ListPolicyRequest{})
		assert.NotNil(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	}
}

func TestCore_sendMinDdlTsAsTt(t *testing.T) {
	ticker := newRocksMqTtSynchronizer()
	ddlManager := newMockDdlTsLockManager()
	ddlManager.GetMinDdlTsFunc = func() Timestamp {
		return 100
	}
	c := newTestCore(
		withTtSynchronizer(ticker),
		withDdlTsLockManager(ddlManager))
	c.sendMinDdlTsAsTt() // no session.
	ticker.addSession(&sessionutil.Session{ServerID: TestRootCoordID})
	c.sendMinDdlTsAsTt()
}

func TestCore_startTimeTickLoop(t *testing.T) {
	ticker := newRocksMqTtSynchronizer()
	ticker.addSession(&sessionutil.Session{ServerID: TestRootCoordID})
	ddlManager := newMockDdlTsLockManager()
	ddlManager.GetMinDdlTsFunc = func() Timestamp {
		return 100
	}
	c := newTestCore(
		withTtSynchronizer(ticker),
		withDdlTsLockManager(ddlManager))
	ctx, cancel := context.WithCancel(context.Background())
	c.ctx = ctx
	Params.ProxyCfg.TimeTickInterval = time.Millisecond
	c.wg.Add(1)
	go c.startTimeTickLoop()

	time.Sleep(time.Millisecond * 4)
	cancel()
	c.wg.Wait()
}
