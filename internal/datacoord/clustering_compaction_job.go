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
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
)

type clusteringCompactionTaskState int

const (
	clustering_pipelining clusteringCompactionTaskState = iota + 1
	clustering_executing
	clustering_completed
	clustering_failed
	clustering_timeout
)

type ClusteringCompactionJob struct {
	triggerID         UniqueID
	collectionID      UniqueID
	clusteringKeyID   UniqueID
	clusteringKeyName string
	clusteringKeyType schemapb.DataType
	// ClusteringCompactionJob life cycle:
	//   trigger -> pipelining:
	//              executing:
	//              completed or failed or timeout
	state             clusteringCompactionTaskState
	startTime         uint64
	endTime           uint64
	maxSegmentRows    int64
	preferSegmentRows int64
	subPlans          []*datapb.ClusteringCompactionPlan
	//persistSubPlans   []*datapb.ClusteringCompactionPlan
}

func convertToClusteringCompactionJob(info *datapb.ClusteringCompactionInfo) *ClusteringCompactionJob {
	job := &ClusteringCompactionJob{
		triggerID:         info.GetTriggerID(),
		collectionID:      info.GetCollectionID(),
		clusteringKeyID:   info.GetClusteringKeyID(),
		clusteringKeyName: info.GetClusteringKeyName(),
		clusteringKeyType: info.GetClusteringKeyType(),
		state:             clusteringCompactionTaskState(info.GetState()),
		startTime:         info.GetStartTime(),
		endTime:           info.GetEndTime(),
		maxSegmentRows:    info.GetMaxSegmentRows(),
		preferSegmentRows: info.GetPreferSegmentRows(),
		subPlans:          info.GetSubPlans(),
	}
	return job
}

func convertFromClusteringCompactionJob(job *ClusteringCompactionJob) *datapb.ClusteringCompactionInfo {
	info := &datapb.ClusteringCompactionInfo{
		TriggerID:         job.triggerID,
		CollectionID:      job.collectionID,
		ClusteringKeyID:   job.clusteringKeyID,
		ClusteringKeyName: job.clusteringKeyName,
		ClusteringKeyType: job.clusteringKeyType,
		State:             int32(job.state),
		StartTime:         job.startTime,
		EndTime:           job.endTime,
		MaxSegmentRows:    job.maxSegmentRows,
		PreferSegmentRows: job.preferSegmentRows,
		SubPlans:          job.subPlans,
	}
	return info
}
