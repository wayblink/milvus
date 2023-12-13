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
	"math"
	"sort"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/util/tsoutil"

	"github.com/samber/lo"
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
		segmentsSyncPairs := make([][2]int64, 0)
		for _, segment := range segments {
			if segment == nil || segment.sType.Load() == nil || segment.getType() != datapb.SegmentType_Flushed {
				continue //cp behind check policy only towards flushed segments generated by compaction
			}
			segmentStartTime := tsoutil.PhysicalTime(segmentMinTs(segment))
			cpLagDuration := tsoutil.PhysicalTime(ts).Sub(segmentStartTime)
			shouldSync := cpLagDuration > Params.DataNodeCfg.CpLagPeriod && !segment.isBufferEmpty()
			lagInfo := [2]int64{segment.segmentID, cpLagDuration.Nanoseconds()}
			if shouldSync {
				segmentsSyncPairs = append(segmentsSyncPairs, lagInfo)
			}
		}
		segmentsIDsToSync := make([]UniqueID, 0)
		if len(segmentsSyncPairs) > 0 {
			if uint16(len(segmentsSyncPairs)) > Params.DataNodeCfg.CpLagSyncLimit {
				//sort all segments according to the length of lag duration
				sort.Slice(segmentsSyncPairs, func(i, j int) bool {
					return segmentsSyncPairs[i][1] > segmentsSyncPairs[j][1]
				})
				segmentsSyncPairs = segmentsSyncPairs[:Params.DataNodeCfg.CpLagSyncLimit]
			}
			segmentsIDsToSync = lo.Map(segmentsSyncPairs, func(t [2]int64, _ int) int64 {
				return t[0]
			})
			log.Info("sync segment for cp lag behind too much", zap.Int("segmentCount", len(segmentsSyncPairs)),
				zap.Int64s("segmentIDs", segmentsIDsToSync))
		}
		return segmentsIDsToSync
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
