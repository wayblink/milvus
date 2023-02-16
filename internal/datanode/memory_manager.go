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

package datanode

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/util/commonpbutil"
	"github.com/milvus-io/milvus/internal/util/hardware"
	"go.uber.org/zap"
)

type memoryManager struct {
	mu       sync.RWMutex
	datanode *DataNode
	closeCh  chan struct{}
}

func newMemoryManager(dataNode *DataNode) *memoryManager {
	return &memoryManager{
		datanode: dataNode,
		closeCh:  make(chan struct{}),
	}
}

func (mm *memoryManager) start(ctx context.Context) {
	log.Info("Datanode check start")
	ticker := time.NewTicker(time.Duration(Params.DataNodeCfg.MemoryControlIntervalSeconds) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Warn("memoryManager context done!")
			return
		case <-mm.closeCh:
			log.Warn("memoryManager close!")
			return
		case <-ticker.C:
			mm.mu.Lock()
			memoryUsageRatio := hardware.GetMemoryUseRatio()
			log.Info("Datanode memory usage",
				zap.Float64("ratio", memoryUsageRatio),
				zap.Float64("threshold", Params.DataNodeCfg.MemoryControlForceFlushThreshold),
				zap.Bool("MemoryControlForceFlushEnable", Params.DataNodeCfg.MemoryControlForceFlushEnable))
			if Params.DataNodeCfg.MemoryControlForceFlushEnable &&
				memoryUsageRatio > Params.DataNodeCfg.MemoryControlForceFlushThreshold {
				go mm.forceFlush(ctx)
			}
			mm.mu.Unlock()
		}
	}
}

func (mm *memoryManager) forceFlush(ctx context.Context) {
	log.Info("try to force flush due to memory too large")

	allSegments := make([]*Segment, 0)
	mm.datanode.flowgraphManager.flowgraphs.Range(func(key, value interface{}) bool {
		fg := value.(*dataSyncService)
		_, segments := fg.channel.listNotFlushedSegmentIDs()
		for _, seg := range segments {
			allSegments = append(allSegments, seg)
		}
		return true
	})
	log.Info("force flush candidates", zap.Int("length", len(allSegments)))

	toFlushSegmentNum := int(math.Max(float64(len(allSegments))*Params.DataNodeCfg.MemoryControlForceFlushSegmentRatio, 1.0))
	log.Info("to force flush segment", zap.Int("length", toFlushSegmentNum))

	if len(allSegments) > 0 {
		sort.Slice(allSegments, func(i, j int) bool {
			return allSegments[i].memorySize > allSegments[j].memorySize
		})
	}

	toFlushSegments := allSegments[:toFlushSegmentNum]
	for _, seg := range toFlushSegments {
		log.Info("force flush due to memory is too high", zap.Int64("collID", seg.collectionID), zap.Int64("segmentID", seg.segmentID))
		mm.datanode.FlushSegments(ctx, &datapb.FlushSegmentsRequest{
			Base: commonpbutil.NewMsgBase(
				commonpbutil.WithMsgType(commonpb.MsgType_Flush),
				commonpbutil.WithSourceID(mm.datanode.session.ServerID),
				commonpbutil.WithTargetID(mm.datanode.session.ServerID),
			),
			CollectionID: seg.collectionID,
			SegmentIDs:   []int64{seg.segmentID},
		})
	}
}

func (mm *memoryManager) Stop() {
	close(mm.closeCh)
}
