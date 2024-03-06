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

package storage

import (
	"encoding/json"
	"testing"

	"github.com/bits-and-blooms/bloom/v3"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

func TestFieldStatsUpdate(t *testing.T) {
	fieldStat1, err := NewFieldStats(1, schemapb.DataType_Int64, 2)
	assert.NoError(t, err)
	fieldStat1.Update(NewInt64FieldValue(99))
	fieldStat1.Update(NewInt64FieldValue(201))
	assert.Equal(t, int64(201), fieldStat1.Max.GetValue())
	assert.Equal(t, int64(99), fieldStat1.Min.GetValue())

	fieldStat2, err := NewFieldStats(2, schemapb.DataType_VarChar, 2)
	assert.NoError(t, err)
	fieldStat2.Update(NewVarCharFieldValue("a"))
	fieldStat2.Update(NewVarCharFieldValue("z"))
	assert.Equal(t, "z", fieldStat2.Max.GetValue())
	assert.Equal(t, "a", fieldStat2.Min.GetValue())
}

func TestFieldStatsWriter_Int64FieldValue(t *testing.T) {
	data := &Int64FieldData{
		Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	sw := &FieldStatsWriter{}
	err := sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, data)
	assert.NoError(t, err)
	b := sw.GetBuffer()

	sr := &FieldStatsReader{}
	sr.SetBuffer(b)
	statsList, err := sr.GetFieldStatsList()
	assert.NoError(t, err)
	stats := statsList[0]
	maxPk := NewInt64FieldValue(9)
	minPk := NewInt64FieldValue(1)
	assert.Equal(t, true, stats.Max.EQ(maxPk))
	assert.Equal(t, true, stats.Min.EQ(minPk))
	buffer := make([]byte, 8)
	for _, id := range data.Data {
		common.Endian.PutUint64(buffer, uint64(id))
		assert.True(t, stats.BF.Test(buffer))
	}

	msgs := &Int64FieldData{
		Data: []int64{},
	}
	err = sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, msgs)
	assert.NoError(t, err)
}

func TestFieldStatsWriter_VarCharFieldValue(t *testing.T) {
	data := &StringFieldData{
		Data: []string{"bc", "ac", "abd", "cd", "milvus"},
	}
	sw := &FieldStatsWriter{}
	err := sw.GenerateByData(common.RowIDField, schemapb.DataType_VarChar, data)
	assert.NoError(t, err)
	b := sw.GetBuffer()
	t.Log(string(b))

	sr := &FieldStatsReader{}
	sr.SetBuffer(b)
	statsList, err := sr.GetFieldStatsList()
	assert.NoError(t, err)
	stats := statsList[0]
	maxPk := NewVarCharFieldValue("milvus")
	minPk := NewVarCharFieldValue("abd")
	assert.Equal(t, true, stats.Max.EQ(maxPk))
	assert.Equal(t, true, stats.Min.EQ(minPk))
	for _, id := range data.Data {
		assert.True(t, stats.BF.TestString(id))
	}

	msgs := &Int64FieldData{
		Data: []int64{},
	}
	err = sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, msgs)
	assert.NoError(t, err)
}

func TestFieldStatsWriter_BF(t *testing.T) {
	value := make([]int64, 1000000)
	for i := 0; i < 1000000; i++ {
		value[i] = int64(i)
	}
	data := &Int64FieldData{
		Data: value,
	}
	t.Log(data.RowNum())
	sw := &FieldStatsWriter{}
	err := sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, data)
	assert.NoError(t, err)

	sr := &FieldStatsReader{}
	sr.SetBuffer(sw.GetBuffer())
	statsList, err := sr.GetFieldStatsList()
	assert.NoError(t, err)
	stats := statsList[0]
	buf := make([]byte, 8)

	for i := 0; i < 1000000; i++ {
		common.Endian.PutUint64(buf, uint64(i))
		assert.True(t, stats.BF.Test(buf))
	}

	common.Endian.PutUint64(buf, uint64(1000001))
	assert.False(t, stats.BF.Test(buf))

	assert.True(t, stats.Min.EQ(NewInt64FieldValue(0)))
	assert.True(t, stats.Max.EQ(NewInt64FieldValue(999999)))
}

func TestFieldStatsWriter_UpgradePrimaryKey(t *testing.T) {
	data := &Int64FieldData{
		Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}

	stats := &PrimaryKeyStats{
		FieldID: common.RowIDField,
		Min:     1,
		Max:     9,
		BF:      bloom.NewWithEstimates(100000, 0.05),
	}

	b := make([]byte, 8)
	for _, int64Value := range data.Data {
		common.Endian.PutUint64(b, uint64(int64Value))
		stats.BF.Add(b)
	}
	blob, err := json.Marshal(stats)
	assert.NoError(t, err)
	sr := &FieldStatsReader{}
	sr.SetBuffer(blob)
	unmarshalledStats, err := sr.GetFieldStats()
	assert.NoError(t, err)
	maxPk := &Int64FieldValue{
		Value: 9,
	}
	minPk := &Int64FieldValue{
		Value: 1,
	}
	assert.Equal(t, true, unmarshalledStats.Max.EQ(maxPk))
	assert.Equal(t, true, unmarshalledStats.Min.EQ(minPk))
	buffer := make([]byte, 8)
	for _, id := range data.Data {
		common.Endian.PutUint64(buffer, uint64(id))
		assert.True(t, unmarshalledStats.BF.Test(buffer))
	}
}

func TestDeserializeFieldStatsFailed(t *testing.T) {
	t.Run("empty field stats", func(t *testing.T) {
		blob := &Blob{
			Value: []byte{},
		}

		_, err := DeserializeFieldStatsList(blob)
		assert.NoError(t, err)
	})

	t.Run("invalid field stats", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("abc"),
		}

		_, err := DeserializeFieldStatsList(blob)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	})

	t.Run("valid field stats", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("[{\"fieldID\":1,\"max\":10, \"min\":1}]"),
		}
		_, err := DeserializeFieldStatsList(blob)
		assert.NoError(t, err)
	})
}

func TestDeserializeFieldStats(t *testing.T) {
	t.Run("empty field stats", func(t *testing.T) {
		blob := &Blob{
			Value: []byte{},
		}

		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.NoError(t, err)
	})

	t.Run("invalid field stats, not valid json", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("abc"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, no fieldID", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"field\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid fieldID", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid type", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"type\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid type", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"type\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid max int64", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"max\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid min int64", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"min\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid max varchar", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"type\":21,\"max\":2}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("invalid field stats, invalid min varchar", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"type\":21,\"min\":1}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.Error(t, err)
	})

	t.Run("valid int64 field stats", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"max\":10, \"min\":1}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.NoError(t, err)
	})

	t.Run("valid varchar field stats", func(t *testing.T) {
		blob := &Blob{
			Value: []byte("{\"fieldID\":1,\"type\":21,\"max\":\"z\", \"min\":\"a\"}"),
		}
		_, err := DeserializeFieldStats([]*Blob{blob})
		assert.NoError(t, err)
	})
}

func TestCompatible_ReadPrimaryKeyStatsWithFieldStatsReader(t *testing.T) {
	data := &Int64FieldData{
		Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	sw := &StatsWriter{}
	err := sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, data)
	assert.NoError(t, err)
	b := sw.GetBuffer()

	sr := &FieldStatsReader{}
	sr.SetBuffer(b)
	stats, err := sr.GetFieldStats()
	assert.NoError(t, err)
	maxPk := &Int64FieldValue{
		Value: 9,
	}
	minPk := &Int64FieldValue{
		Value: 1,
	}
	assert.Equal(t, true, stats.Max.EQ(maxPk))
	assert.Equal(t, true, stats.Min.EQ(minPk))
	assert.Equal(t, schemapb.DataType_Int64.String(), stats.Type.String())
	buffer := make([]byte, 8)
	for _, id := range data.Data {
		common.Endian.PutUint64(buffer, uint64(id))
		assert.True(t, stats.BF.Test(buffer))
	}

	msgs := &Int64FieldData{
		Data: []int64{},
	}
	err = sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, msgs)
	assert.NoError(t, err)
}

func TestFieldStatsMarshal(t *testing.T) {
	stats, err := NewFieldStats(1, schemapb.DataType_Int64, 1)
	assert.NoError(t, err)
	err = stats.UnmarshalJSON([]byte("{\"fieldID\":1,\"max\":10, }"))
	assert.Error(t, err)
	err = stats.UnmarshalJSON([]byte("{\"fieldID\":1,\"max\":10, \"maxPk\":\"A\"}"))
	assert.Error(t, err)
	err = stats.UnmarshalJSON([]byte("{\"fieldID\":1,\"max\":10, \"maxPk\":10, \"minPk\": \"b\"}"))
	assert.Error(t, err)
	err = stats.UnmarshalJSON([]byte("{\"fieldID\":1,\"max\":10, \"maxPk\":10, \"minPk\": 1, \"bf\": \"2\"}"))
	assert.Error(t, err)
}

func TestCompatible_ReadFieldStatsWithPrimaryKeyStatsReader(t *testing.T) {
	data := &Int64FieldData{
		Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	sw := &FieldStatsWriter{}
	err := sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, data)
	assert.NoError(t, err)
	b := sw.GetBuffer()

	sr := &StatsReader{}
	sr.SetBuffer(b)
	statsList, err := sr.GetPrimaryKeyStatsList()
	assert.NoError(t, err)
	stats := statsList[0]
	maxPk := &Int64PrimaryKey{
		Value: 9,
	}
	minPk := &Int64PrimaryKey{
		Value: 1,
	}
	assert.Equal(t, true, stats.MaxPk.EQ(maxPk))
	assert.Equal(t, true, stats.MinPk.EQ(minPk))
	buffer := make([]byte, 8)
	for _, id := range data.Data {
		common.Endian.PutUint64(buffer, uint64(id))
		assert.True(t, stats.BF.Test(buffer))
	}

	msgs := &Int64FieldData{
		Data: []int64{},
	}
	err = sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, msgs)
	assert.NoError(t, err)
}

func TestMultiFieldStats(t *testing.T) {
	pkData := &Int64FieldData{
		Data: []int64{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	partitionKeyData := &Int64FieldData{
		Data: []int64{1, 10, 21, 31, 41, 51, 61, 71, 81},
	}

	sw := &FieldStatsWriter{}
	err := sw.GenerateByData(common.RowIDField, schemapb.DataType_Int64, pkData, partitionKeyData)
	assert.NoError(t, err)
	b := sw.GetBuffer()

	sr := &FieldStatsReader{}
	sr.SetBuffer(b)
	statsList, err := sr.GetFieldStatsList()
	assert.Equal(t, 2, len(statsList))
	assert.NoError(t, err)

	pkStats := statsList[0]
	maxPk := NewInt64FieldValue(9)
	minPk := NewInt64FieldValue(1)
	assert.Equal(t, true, pkStats.Max.EQ(maxPk))
	assert.Equal(t, true, pkStats.Min.EQ(minPk))

	partitionKeyStats := statsList[1]
	maxPk2 := NewInt64FieldValue(81)
	minPk2 := NewInt64FieldValue(1)
	assert.Equal(t, true, partitionKeyStats.Max.EQ(maxPk2))
	assert.Equal(t, true, partitionKeyStats.Min.EQ(minPk2))
}
