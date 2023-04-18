package memorypool

import (
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/hardware"
)

const (
	MemoryCategory_Compact = "compact"
	MemoryCategory_Insert  = "insert"
)

// MemoryUnit declare memory usage of certain category
type MemoryUnit struct {
	// id is unqiue id of MemoryUnit
	id int64
	// category is type of the memory-consuming behaviour, such as insert, compact, buildindex, etc
	category string
	// memory usage in bytes
	memorySize int64
}

type MemoryPool interface {
	// Acquire memory use by size and category, return success, memoryUnitID, error
	Acquire(memorySize int64, category string, option ...AcquireOption) (int64, error)
	// Release memory usage by memoryUnitID, memoryUnitID is given by Acquire
	Release(id int64) (bool, error)
	// UsageByCategory query the memory usage of one category
	UsageByCategory(category string) (int64, error)
	// UsageAllCategory query the memory usage of all category
	UsageAllCategory() (map[string]int64, error)
	// TotalUsage query the total memory usage
	TotalUsage() (int64, error)
}

// Use memory pool as a singleton
var memoryPool GlobalMemoryPool

func Init() {
	totalLimit := int64(float64(hardware.GetMemoryCount()) * 0.6)
	categoryQuota := make(map[string]int64)
	categoryQuota[MemoryCategory_Compact] = int64(0.5 * float64(totalLimit))
	memoryPool.once.Do(func() {
		memoryPool.init(totalLimit, categoryQuota)
	})
}

func Get() *GlobalMemoryPool {
	return &memoryPool
}

type GlobalMemoryPool struct {
	memory        map[int64]MemoryUnit
	totalQuota    int64
	catagoryQuota map[string]int64
	totalUsage    int64
	catagoryUsage map[string]int64

	nextId atomic.Int64
	once   sync.Once
	mu     sync.Mutex
}

func (g *GlobalMemoryPool) init(totalQuota int64, catagoryQuota map[string]int64) {
	g.catagoryQuota = catagoryQuota
	g.totalQuota = totalQuota
	g.totalUsage = 0
	g.memory = make(map[int64]MemoryUnit, 0)
	g.catagoryUsage = make(map[string]int64, 0)
	log.Info("Initial Memory Pool", zap.Int64("totalQuota", g.totalQuota), zap.Any("catagoryQuota", g.catagoryQuota))
}

func (g *GlobalMemoryPool) Acquire(memorySize int64, category string, opts ...AcquireOption) (int64, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	c := &acquireConfig{}
	c.apply(opts...)

	if g.totalUsage+memorySize <= g.totalQuota {
		if catagoryLimit, exists := g.catagoryQuota[category]; exists &&
			g.catagoryUsage[category]+memorySize <= catagoryLimit {
			id := g.nextId.Add(1)
			g.memory[id] = MemoryUnit{
				id:         id,
				category:   category,
				memorySize: memorySize,
			}
			g.catagoryUsage[category] = g.catagoryUsage[category] + memorySize
			g.totalUsage = g.totalUsage + memorySize
			log.Debug("Acquire",
				zap.Int64("id", id),
				zap.String("category", category),
				zap.Int64("size", memorySize),
				zap.Int64("category usage", g.catagoryUsage[category]))
			return id, nil
		}
	}

	return 0, errors.New("memory quota is used up")
}

func (g *GlobalMemoryPool) Release(id int64) (bool, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, ok := g.memory[id]; ok {
		category := g.memory[id].category
		memorySize := g.memory[id].memorySize
		delete(g.memory, id)
		g.catagoryUsage[category] = g.catagoryUsage[category] - memorySize
		g.totalUsage = g.totalUsage - memorySize
		log.Debug("Release",
			zap.Int64("id", id),
			zap.String("category", category),
			zap.Int64("size", memorySize),
			zap.Int64("category usage", g.catagoryUsage[category]))
	}
	return true, nil
}

func (g *GlobalMemoryPool) UsageByCategory(category string) (int64, error) {
	return g.catagoryUsage[category], nil
}

func (g *GlobalMemoryPool) UsageAllCategory() (map[string]int64, error) {
	return g.catagoryUsage, nil
}

func (g *GlobalMemoryPool) TotalUsage() int64 {
	return g.totalUsage
}

func (g *GlobalMemoryPool) TotalQuota() int64 {
	return g.totalQuota
}

type acquireConfig struct {
	timeout          time.Duration
	waitUntilSuccess bool
}

func (c *acquireConfig) apply(opts ...AcquireOption) {
	for _, opt := range opts {
		opt(c)
	}
}

type AcquireOption func(c *acquireConfig)

func WithTimeout(timeout time.Duration) AcquireOption {
	return func(c *acquireConfig) {
		c.timeout = timeout
	}
}

func WithWaitUntilSuccess() AcquireOption {
	return func(c *acquireConfig) {
		c.waitUntilSuccess = true
	}
}
