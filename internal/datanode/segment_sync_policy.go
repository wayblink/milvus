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
	"github.com/samber/lo"
	"math"
	"sort"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
)

const minSyncSize = 0.5 * 1024 * 1024

// segmentsSyncPolicy sync policy applies to segments
type segmentSyncPolicy func(segments []*Segment, c Channel, ts Timestamp) []UniqueID

// syncPeriodically get segmentSyncPolicy with segments sync periodically.
func syncPeriodically() segmentSyncPolicy {
	return func(segments []*Segment, c Channel, ts Timestamp) []UniqueID {
		segsToSync := make([]UniqueID, 0)
		for _, seg := range segments {
			endTime := tsoutil.PhysicalTime(ts)
			lastSyncTime := tsoutil.PhysicalTime(seg.lastSyncTs)
			shouldSync := endTime.Sub(lastSyncTime) >= Params.DataNodeCfg.SyncPeriod && !seg.isBufferEmpty()
			if shouldSync {
				segsToSync = append(segsToSync, seg.segmentID)
			}
		}
		if len(segsToSync) > 0 {
			log.Info("sync segment periodically", zap.Int64s("segmentIDs", segsToSync))
		}
		return segsToSync
	}
}

// syncMemoryTooHigh force sync the largest segment.
func syncMemoryTooHigh() segmentSyncPolicy {
	return func(segments []*Segment, c Channel, _ Timestamp) []UniqueID {
		if len(segments) == 0 || !c.getIsHighMemory() {
			return nil
		}
		sort.Slice(segments, func(i, j int) bool {
			return segments[i].memorySize > segments[j].memorySize
		})
		syncSegments := make([]UniqueID, 0)
		syncSegmentsNum := math.Min(float64(Params.DataNodeCfg.MemoryForceSyncSegmentNum), float64(len(segments)))
		for i := 0; i < int(syncSegmentsNum); i++ {
			if segments[i].memorySize < minSyncSize { // prevent generating too many small binlogs
				break
			}
			syncSegments = append(syncSegments, segments[i].segmentID)
			log.Info("sync segment due to memory usage is too high",
				zap.Int64("segmentID", segments[i].segmentID),
				zap.Int64("memorySize", segments[i].memorySize))
		}
		return syncSegments
	}
}

// syncCPLagTooBehind force sync the segments lagging too behind the channel checkPoint
func syncCPLagTooBehind() segmentSyncPolicy {
	segmentMinTs := func(segment *Segment) uint64 {
		var minTs uint64 = math.MaxUint64
		if segment.curInsertBuf != nil && segment.curInsertBuf.startPos != nil && segment.curInsertBuf.startPos.Timestamp < minTs {
			minTs = segment.curInsertBuf.startPos.Timestamp
		}
		if segment.curDeleteBuf != nil && segment.curDeleteBuf.startPos != nil && segment.curDeleteBuf.startPos.Timestamp < minTs {
			minTs = segment.curDeleteBuf.startPos.Timestamp
		}
		for _, ib := range segment.historyInsertBuf {
			if ib != nil && ib.startPos != nil && ib.startPos.Timestamp < minTs {
				minTs = ib.startPos.Timestamp
			}
		}
		for _, db := range segment.historyDeleteBuf {
			if db != nil && db.startPos != nil && db.startPos.Timestamp < minTs {
				minTs = db.startPos.Timestamp
			}
		}
		return minTs
	}

	return func(segments []*Segment, c Channel, ts Timestamp) []UniqueID {
		segmentsToSync := make([]UniqueID, 0)
		for _, segment := range segments {
			segmentMinTs := segmentMinTs(segment)
			segmentStartTime := tsoutil.PhysicalTime(segmentMinTs)
			cpLagDuration := tsoutil.PhysicalTime(ts).Sub(segmentStartTime)
			shouldSync := cpLagDuration > Params.DataNodeCfg.CpLagPeriod && !segment.isBufferEmpty()
			if shouldSync {
				segmentsToSync = append(segmentsToSync, segment.segmentID)
			}
		}
		if len(segmentsToSync) > 0 {
			log.Info("sync segment for cp lag behind too much",
				zap.Int64s("segmentID", segmentsToSync))
		}
		return segmentsToSync
	}
}

// syncSegmentsAtTs returns a new segmentSyncPolicy, sync segments when ts exceeds ChannelMeta.flushTs
func syncSegmentsAtTs() segmentSyncPolicy {
	return func(segments []*Segment, c Channel, ts Timestamp) []UniqueID {
		flushTs := c.getFlushTs()
		if flushTs != 0 && ts >= flushTs {
			segmentsWithBuffer := lo.Filter(segments, func(segment *Segment, _ int) bool {
				return !segment.isBufferEmpty()
			})
			segmentIDs := lo.Map(segmentsWithBuffer, func(segment *Segment, _ int) UniqueID {
				return segment.segmentID
			})
			log.Info("sync segment at ts", zap.Int64s("segmentIDs", segmentIDs),
				zap.Time("ts", tsoutil.PhysicalTime(ts)), zap.Time("flushTs", tsoutil.PhysicalTime(flushTs)))
			return segmentIDs
		}
		return nil
	}
}
