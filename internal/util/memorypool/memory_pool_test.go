package memorypool

import (
	"testing"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/stretchr/testify/assert"
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
	id, err := memPool.Acquire(1000, MemoryCategory_Compact)
	id2, err := memPool.Acquire(1000, MemoryCategory_Compact)
	assert.Equal(t, true, id > 0)
	assert.Equal(t, true, id2 > 0)
	assert.NoError(t, err)
}
