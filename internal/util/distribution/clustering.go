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

package distribution

import (
	"encoding/json"
	"strconv"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
)

const (
	ClusteringCentroid    = "clustering.centroid"
	ClusteringSize        = "clustering.size"
	ClusteringId          = "clustering.id"
	ClusteringOperationId = "clustering.operation_id"
)

func ClusteringInfoFromKV(kv []*commonpb.KeyValuePair) (*internalpb.VectorClusteringInfo, error) {
	kvMap := funcutil.KeyValuePair2Map(kv)
	if v, ok := kvMap[ClusteringCentroid]; ok {
		var floatSlice []float32
		err := json.Unmarshal([]byte(v), &floatSlice)
		if err != nil {
			log.Error("Failed to parse cluster center value:", zap.String("value", v), zap.Error(err))
			return nil, err
		}
		clusterInfo := &internalpb.VectorClusteringInfo{
			Centroid: floatSlice,
		}
		if sizeStr, ok := kvMap[ClusteringSize]; ok {
			size, err := strconv.ParseInt(sizeStr, 10, 64)
			if err != nil {
				log.Error("Failed to parse cluster size value:", zap.String("value", sizeStr), zap.Error(err))
				return nil, err
			}
			clusterInfo.Size = size
		}
		if clusterIDStr, ok := kvMap[ClusteringId]; ok {
			clusterID, err := strconv.ParseInt(clusterIDStr, 10, 64)
			if err != nil {
				log.Error("Failed to parse cluster id value:", zap.String("value", clusterIDStr), zap.Error(err))
				return nil, err
			}
			clusterInfo.ClusterId = clusterID
		}
		if operationIDStr, ok := kvMap[ClusteringOperationId]; ok {
			operationID, err := strconv.ParseInt(operationIDStr, 10, 64)
			if err != nil {
				log.Error("Failed to parse cluster group id value:", zap.String("value", operationIDStr), zap.Error(err))
				return nil, err
			}
			clusterInfo.OperationId = operationID
		}
		return clusterInfo, nil
	}
	return nil, nil
}
