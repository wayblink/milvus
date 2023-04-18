package memorypool

import (
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_MemoryPool(t *testing.T) {
	Init()
	memPool := Get()
	assert.Equal(t, 0, memPool.totalUsage)
	assert.Equal(t, int64(0), memPool.TotalUsage())
	log.Info("wayblink")
}

func Test_MemoryPool_Aquire(t *testing.T) {
	Init()
	memPool := Get()
	_, err := memPool.Acquire(1000, MemoryCategory_Compact)
	assert.NoError(t, err)
}
