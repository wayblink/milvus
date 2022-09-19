package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"go.uber.org/zap"
)

type GetCollectionNameFunc func(collID, partitionID UniqueID) (string, string, error)
type IDAllocator func(count uint32) (UniqueID, UniqueID, error)
type ImportFunc func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error)
type UnsetImportingFunc func(ctx context.Context, segIDs []int64) (*commonpb.Status, error)
type MarkSegmentsDroppedFunc func(ctx context.Context, segIDs []int64) (*commonpb.Status, error)

type ImportFactory interface {
	NewGetCollectionNameFunc() GetCollectionNameFunc
	NewIDAllocator() IDAllocator
	NewImportFunc() ImportFunc
	NewUnsetImportingFunc() UnsetImportingFunc
	NewMarkSegmentsDroppedFunc() MarkSegmentsDroppedFunc
}

type ImportFactoryImpl struct {
	c *Core
}

func (f ImportFactoryImpl) NewGetCollectionNameFunc() GetCollectionNameFunc {
	return GetCollectionNameWithCore(f.c)
}

func (f ImportFactoryImpl) NewIDAllocator() IDAllocator {
	return IDAllocatorWithCore(f.c)
}

func (f ImportFactoryImpl) NewImportFunc() ImportFunc {
	return ImportFuncWithCore(f.c)
}

func (f ImportFactoryImpl) NewUnsetImportingFunc() UnsetImportingFunc {
	return UnsetImportingFuncWithCore(f.c)
}

func (f ImportFactoryImpl) NewMarkSegmentsDroppedFunc() MarkSegmentsDroppedFunc {
	return MarkSegmentsDroppedWithCore(f.c)
}

func NewImportFactory(c *Core) ImportFactory {
	return &ImportFactoryImpl{c: c}
}

func GetCollectionNameWithCore(c *Core) GetCollectionNameFunc {
	return func(collID, partitionID UniqueID) (string, string, error) {
		colName, err := c.meta.GetCollectionNameByID(collID)
		if err != nil {
			log.Error("Core failed to get collection name by id", zap.Int64("ID", collID), zap.Error(err))
			return "", "", err
		}

		partName, err := c.meta.GetPartitionNameByID(collID, partitionID, 0)
		if err != nil {
			log.Error("Core failed to get partition name by id", zap.Int64("ID", partitionID), zap.Error(err))
			return colName, "", err
		}

		return colName, partName, nil
	}
}

func IDAllocatorWithCore(c *Core) IDAllocator {
	return func(count uint32) (UniqueID, UniqueID, error) {
		return c.idAllocator.Alloc(count)
	}
}

func ImportFuncWithCore(c *Core) ImportFunc {
	return func(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
		return c.broker.Import(ctx, req)
	}
}

func UnsetImportingFuncWithCore(c *Core) UnsetImportingFunc {
	return func(ctx context.Context, segIDs []int64) (*commonpb.Status, error) {
		return c.broker.UnsetIsImportingState(ctx, &datapb.UnsetIsImportingStateRequest{
			SegmentIds: segIDs,
		})
	}
}

func MarkSegmentsDroppedWithCore(c *Core) MarkSegmentsDroppedFunc {
	return func(ctx context.Context, segIDs []int64) (*commonpb.Status, error) {
		return c.broker.MarkSegmentsDropped(ctx, &datapb.MarkSegmentsDroppedRequest{
			SegmentIds: segIDs,
		})
	}
}
