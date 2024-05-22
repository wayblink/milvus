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

package datacoord

import (
	"context"
	"sort"
	"time"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
)

const TaskMaxRetryTimes int32 = 3

type ClusteringCompactionSummary struct {
	state         commonpb.CompactionState
	executingCnt  int
	pipeliningCnt int
	completedCnt  int
	failedCnt     int
	timeoutCnt    int
	initCnt       int
	analyzingCnt  int
	analyzedCnt   int
	indexingCnt   int
	indexedCnt    int
	cleanedCnt    int
}

func summaryClusteringCompactionState(compactionTasks []CompactionTask) ClusteringCompactionSummary {
	var state commonpb.CompactionState
	var executingCnt, pipeliningCnt, completedCnt, failedCnt, timeoutCnt, initCnt, analyzingCnt, analyzedCnt, indexingCnt, indexedCnt, cleanedCnt int
	for _, task := range compactionTasks {
		if task == nil {
			continue
		}
		switch task.GetState() {
		case datapb.CompactionTaskState_executing:
			executingCnt++
		case datapb.CompactionTaskState_pipelining:
			pipeliningCnt++
		case datapb.CompactionTaskState_completed:
			completedCnt++
		case datapb.CompactionTaskState_failed:
			failedCnt++
		case datapb.CompactionTaskState_timeout:
			timeoutCnt++
		case datapb.CompactionTaskState_init:
			initCnt++
		case datapb.CompactionTaskState_analyzing:
			analyzingCnt++
		case datapb.CompactionTaskState_analyzed:
			analyzedCnt++
		case datapb.CompactionTaskState_indexing:
			indexingCnt++
		case datapb.CompactionTaskState_indexed:
			indexedCnt++
		case datapb.CompactionTaskState_cleaned:
			cleanedCnt++
		default:
		}
	}

	// fail and timeout task must be cleaned first before mark the job complete
	if executingCnt+pipeliningCnt+completedCnt+initCnt+analyzingCnt+analyzedCnt+indexingCnt+failedCnt+timeoutCnt != 0 {
		state = commonpb.CompactionState_Executing
	} else {
		state = commonpb.CompactionState_Completed
	}

	log.Debug("compaction states",
		zap.Int64("triggerID", compactionTasks[0].GetTriggerID()),
		zap.String("state", state.String()),
		zap.Int("executingCnt", executingCnt),
		zap.Int("pipeliningCnt", pipeliningCnt),
		zap.Int("completedCnt", completedCnt),
		zap.Int("failedCnt", failedCnt),
		zap.Int("timeoutCnt", timeoutCnt),
		zap.Int("initCnt", initCnt),
		zap.Int("analyzingCnt", analyzingCnt),
		zap.Int("analyzedCnt", analyzedCnt),
		zap.Int("indexingCnt", indexingCnt),
		zap.Int("indexedCnt", indexedCnt),
		zap.Int("cleanedCnt", cleanedCnt))
	return ClusteringCompactionSummary{
		state:         state,
		executingCnt:  executingCnt,
		pipeliningCnt: pipeliningCnt,
		completedCnt:  completedCnt,
		failedCnt:     failedCnt,
		timeoutCnt:    timeoutCnt,
		initCnt:       initCnt,
		analyzingCnt:  analyzingCnt,
		analyzedCnt:   analyzedCnt,
		indexingCnt:   indexingCnt,
		indexedCnt:    indexedCnt,
		cleanedCnt:    cleanedCnt,
	}
}

func fillClusteringCompactionTask(segments []*SegmentInfo) (segmentIDs []int64, totalRows, maxSegmentRows, preferSegmentRows int64) {
	for _, s := range segments {
		totalRows += s.GetNumOfRows()
		segmentIDs = append(segmentIDs, s.GetID())
	}
	clusteringMaxSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionMaxSegmentSize.GetAsSize()
	clusteringPreferSegmentSize := paramtable.Get().DataCoordCfg.ClusteringCompactionPreferSegmentSize.GetAsSize()
	segmentMaxSize := paramtable.Get().DataCoordCfg.SegmentMaxSize.GetAsInt64() * 1024 * 1024
	maxSegmentRows = segments[0].MaxRowNum * clusteringMaxSegmentSize / segmentMaxSize
	preferSegmentRows = segments[0].MaxRowNum * clusteringPreferSegmentSize / segmentMaxSize
	return
}

func triggerClusteringCompactionPolicy(ctx context.Context, meta *meta, collectionID int64, partitionID int64, channel string, segments []*SegmentInfo) (bool, error) {
	log := log.With(zap.Int64("collectionID", collectionID), zap.Int64("partitionID", partitionID))
	partitionStatsInfos := meta.ListPartitionStatsInfos(collectionID, partitionID, channel)
	sort.Slice(partitionStatsInfos, func(i, j int) bool {
		return partitionStatsInfos[i].Version > partitionStatsInfos[j].Version
	})

	if len(partitionStatsInfos) == 0 {
		var newDataSize int64 = 0
		for _, seg := range segments {
			newDataSize += seg.getSegmentSize()
		}
		if newDataSize > Params.DataCoordCfg.ClusteringCompactionNewDataSizeThreshold.GetAsSize() {
			log.Info("New data is larger than threshold, do compaction", zap.Int64("newDataSize", newDataSize))
			return true, nil
		}
		log.Info("No partition stats and no enough new data, skip compaction")
		return false, nil
	}

	partitionStats := partitionStatsInfos[0]
	version := partitionStats.Version
	pTime, _ := tsoutil.ParseTS(uint64(version))
	if time.Since(pTime) < Params.DataCoordCfg.ClusteringCompactionMinInterval.GetAsDuration(time.Second) {
		log.Info("Too short time before last clustering compaction, skip compaction")
		return false, nil
	}
	if time.Since(pTime) > Params.DataCoordCfg.ClusteringCompactionMaxInterval.GetAsDuration(time.Second) {
		log.Info("It is a long time after last clustering compaction, do compaction")
		return true, nil
	}

	var compactedSegmentSize int64 = 0
	var uncompactedSegmentSize int64 = 0
	for _, seg := range segments {
		if lo.Contains(partitionStats.SegmentIDs, seg.ID) {
			compactedSegmentSize += seg.getSegmentSize()
		} else {
			uncompactedSegmentSize += seg.getSegmentSize()
		}
	}

	// size based
	if uncompactedSegmentSize > Params.DataCoordCfg.ClusteringCompactionNewDataSizeThreshold.GetAsSize() {
		log.Info("New data is larger than threshold, do compaction", zap.Int64("newDataSize", uncompactedSegmentSize))
		return true, nil
	}
	log.Info("New data is smaller than threshold, skip compaction", zap.Int64("newDataSize", uncompactedSegmentSize))
	return false, nil
}
