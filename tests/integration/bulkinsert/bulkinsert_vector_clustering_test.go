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

package bulkinsert

import (
	"context"
	"fmt"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/suite"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/util/distribution"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/tests/integration"
)

type BulkInsertClusteringSuite struct {
	integration.MiniClusterSuite
}

// test bulk insert with clustering info
// 1, create collection with a vector column and a varchar column
// 2, generate numpy files
// 3, import
// 4, check segment clustering info
func (s *BulkInsertClusteringSuite) TestBulkInsertClustering() {
	c := s.Cluster
	ctx, cancel := context.WithCancel(c.GetContext())
	defer cancel()

	prefix := "BulkInsertClusteringSuite"
	dbName := ""
	collectionName := prefix + funcutil.GenRandomStr()
	dim := DIM128

	pkFieldSchema := &schemapb.FieldSchema{Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true, AutoID: true}
	varcharFieldSchema := &schemapb.FieldSchema{Name: "image_path", DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: common.MaxLengthKey, Value: "65535"}}}
	vecFieldSchema := &schemapb.FieldSchema{Name: "embeddings", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: fmt.Sprint(dim)}}}

	schema := integration.ConstructSchema(collectionName, dim, true,
		pkFieldSchema,
		varcharFieldSchema,
		vecFieldSchema,
	)
	marshaledSchema, err := proto.Marshal(schema)
	s.NoError(err)

	createCollectionStatus, err := c.Proxy.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{
		DbName:         dbName,
		CollectionName: collectionName,
		Schema:         marshaledSchema,
		ShardsNum:      common.DefaultShardsNum,
	})
	s.NoError(err)
	if createCollectionStatus.GetErrorCode() != commonpb.ErrorCode_Success {
		log.Warn("createCollectionStatus fail reason", zap.String("reason", createCollectionStatus.GetReason()))
		s.FailNow("failed to create collection")
	}
	s.Equal(createCollectionStatus.GetErrorCode(), commonpb.ErrorCode_Success)

	log.Info("CreateCollection result", zap.Any("createCollectionStatus", createCollectionStatus))
	showCollectionsResp, err := c.Proxy.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
	s.NoError(err)
	s.Equal(showCollectionsResp.GetStatus().GetErrorCode(), commonpb.ErrorCode_Success)
	log.Info("ShowCollections result", zap.Any("showCollectionsResp", showCollectionsResp))

	err = GenerateNumpyFile(c.ChunkManager.RootPath()+"/"+"embeddings.npy", 100, vecFieldSchema)
	s.NoError(err)
	err = GenerateNumpyFile(c.ChunkManager.RootPath()+"/"+"image_path.npy", 100, varcharFieldSchema)
	s.NoError(err)

	bulkInsertFiles := []string{
		c.ChunkManager.RootPath() + "/" + "embeddings.npy",
		c.ChunkManager.RootPath() + "/" + "image_path.npy",
	}

	allocTimestampResp, err := c.Proxy.AllocTimestamp(ctx, &milvuspb.AllocTimestampRequest{})
	s.NoError(err)
	clusteringGroupId := allocTimestampResp.Timestamp

	err = BulkInsertSync(ctx, c, collectionName, bulkInsertFiles, []*commonpb.KeyValuePair{
		{Key: distribution.ClusteringCentroid, Value: "[0.0,0.0,0.0,0.0]"},
		{Key: distribution.ClusteringSize, Value: "100"},
		{Key: distribution.ClusteringOperationId, Value: fmt.Sprint(clusteringGroupId)},
	})
	s.NoError(err)

	health2, err := c.DataCoord.CheckHealth(ctx, &milvuspb.CheckHealthRequest{})
	s.NoError(err)
	log.Info("dataCoord health", zap.Any("health2", health2))

	segments, err := c.MetaWatcher.ShowSegments()
	s.NoError(err)
	s.NotEmpty(segments)
	for _, segment := range segments {
		log.Info("ShowSegments result", zap.String("segment", segment.String()))
		// check clustering info is inserted
		s.True(len(segment.GetDistributionInfo().GetVectorClusteringInfos()) > 0)
		s.True(segment.GetDistributionInfo().GetVectorClusteringInfos()[0].GetOperationId() == int64(clusteringGroupId))
		s.True(segment.GetDistributionInfo().GetVectorClusteringInfos()[0].GetSize() == int64(100))
		s.True(segment.GetDistributionInfo().GetVectorClusteringInfos()[0].GetClusterId() != 0)
		s.True(segment.GetDistributionInfo().GetVectorClusteringInfos()[0].GetCentroid() != nil)
	}

	log.Info("======================")
	log.Info("======================")
	log.Info("BulkInsertClusteringSuite succeed")
	log.Info("======================")
	log.Info("======================")
}

func TestBulkInsertClustering(t *testing.T) {
	t.Skip("Skip integration test, need to refactor integration test framework")
	suite.Run(t, new(BulkInsertClusteringSuite))
}
