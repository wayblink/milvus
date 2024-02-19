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

package importv2

import (
	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type HashedData [][]*storage.InsertData // [vchannelIndex][partitionIndex]*storage.InsertData

func newHashedData(schema *schemapb.CollectionSchema, channelNum, partitionNum int) (HashedData, error) {
	var err error
	res := make(HashedData, channelNum)
	for i := 0; i < channelNum; i++ {
		res[i] = make([]*storage.InsertData, partitionNum)
		for j := 0; j < partitionNum; j++ {
			res[i][j], err = storage.NewInsertData(schema)
			if err != nil {
				return nil, err
			}
		}
	}
	return res, nil
}

func HashData(task Task, rows *storage.InsertData) (HashedData, error) {
	var (
		schema       = typeutil.AppendSystemFields(task.GetSchema())
		channelNum   = len(task.GetVchannels())
		partitionNum = len(task.GetPartitionIDs())
	)

	pkField, err := typeutil.GetPrimaryFieldSchema(schema)
	if err != nil {
		return nil, err
	}
	partKeyField, _ := typeutil.GetPartitionKeyFieldSchema(schema)

	f1 := hashByVChannel(int64(channelNum), pkField)
	f2 := hashByPartition(int64(partitionNum), partKeyField)

	res, err := newHashedData(schema, channelNum, partitionNum)
	if err != nil {
		return nil, err
	}

	for i := 0; i < rows.GetRowNum(); i++ {
		row := rows.GetRow(i)
		p1, p2 := f1(row), f2(row)
		err = res[p1][p2].Append(row)
		if err != nil {
			return nil, err
		}
	}
	return res, nil
}

func GetRowsStats(task Task, rows *storage.InsertData) (map[string]*datapb.PartitionRows, error) {
	var (
		schema       = task.GetSchema()
		channelNum   = len(task.GetVchannels())
		partitionNum = len(task.GetPartitionIDs())
	)

	pkField, err := typeutil.GetPrimaryFieldSchema(schema)
	if err != nil {
		return nil, err
	}
	partKeyField, _ := typeutil.GetPartitionKeyFieldSchema(schema)

	hashRowsCount := make([][]int, channelNum)
	for i := 0; i < channelNum; i++ {
		hashRowsCount[i] = make([]int, partitionNum)
	}

	rowNum := GetInsertDataRowCount(rows, schema)
	if pkField.GetAutoID() {
		id := int64(0)
		num := int64(channelNum)
		fn1 := hashByID()
		fn2 := hashByPartition(int64(partitionNum), partKeyField)
		rows.Data = lo.PickBy(rows.Data, func(fieldID int64, _ storage.FieldData) bool {
			return fieldID != pkField.GetFieldID()
		})
		for i := 0; i < rowNum; i++ {
			p1, p2 := fn1(id, num), fn2(rows.GetRow(i))
			hashRowsCount[p1][p2]++
			id++
		}
	} else {
		f1 := hashByVChannel(int64(channelNum), pkField)
		f2 := hashByPartition(int64(partitionNum), partKeyField)
		for i := 0; i < rowNum; i++ {
			row := rows.GetRow(i)
			p1, p2 := f1(row), f2(row)
			hashRowsCount[p1][p2]++
		}
	}

	res := make(map[string]*datapb.PartitionRows)
	for _, channel := range task.GetVchannels() {
		res[channel] = &datapb.PartitionRows{
			PartitionRows: make(map[int64]int64),
		}
	}
	for i, partitionRows := range hashRowsCount {
		channel := task.GetVchannels()[i]
		for j, n := range partitionRows {
			partition := task.GetPartitionIDs()[j]
			res[channel].PartitionRows[partition] = int64(n)
		}
	}
	return res, nil
}

func hashByVChannel(channelNum int64, pkField *schemapb.FieldSchema) func(row map[int64]interface{}) int64 {
	if channelNum == 1 || pkField == nil {
		return func(_ map[int64]interface{}) int64 {
			return 0
		}
	}
	switch pkField.GetDataType() {
	case schemapb.DataType_Int64:
		return func(row map[int64]interface{}) int64 {
			pk := row[pkField.GetFieldID()]
			hash, _ := typeutil.Hash32Int64(pk.(int64))
			return int64(hash) % channelNum
		}
	case schemapb.DataType_VarChar:
		return func(row map[int64]interface{}) int64 {
			pk := row[pkField.GetFieldID()]
			hash := typeutil.HashString2Uint32(pk.(string))
			return int64(hash) % channelNum
		}
	default:
		return nil
	}
}

func hashByPartition(partitionNum int64, partField *schemapb.FieldSchema) func(row map[int64]interface{}) int64 {
	if partitionNum == 1 {
		return func(_ map[int64]interface{}) int64 {
			return 0
		}
	}
	switch partField.GetDataType() {
	case schemapb.DataType_Int64:
		return func(row map[int64]interface{}) int64 {
			data := row[partField.GetFieldID()]
			hash, _ := typeutil.Hash32Int64(data.(int64))
			return int64(hash) % partitionNum
		}
	case schemapb.DataType_VarChar:
		return func(row map[int64]interface{}) int64 {
			data := row[partField.GetFieldID()]
			hash := typeutil.HashString2Uint32(data.(string))
			return int64(hash) % partitionNum
		}
	default:
		return nil
	}
}

func hashByID() func(id int64, shardNum int64) int64 {
	return func(id int64, shardNum int64) int64 {
		hash, _ := typeutil.Hash32Int64(id)
		return int64(hash) % shardNum
	}
}

func MergeHashedRowsCount(src, dst map[string]*datapb.PartitionRows) {
	for channel, partitionRows := range src {
		for partitionID, rowCount := range partitionRows.GetPartitionRows() {
			if dst[channel] == nil {
				dst[channel] = &datapb.PartitionRows{
					PartitionRows: make(map[int64]int64),
				}
			}
			dst[channel].PartitionRows[partitionID] += rowCount
		}
	}
}
