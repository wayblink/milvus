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

package numpy

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

type Reader struct {
	schema *schemapb.CollectionSchema
	count  int64
	frs    map[int64]*FieldReader // fieldID -> FieldReader
}

func NewReader(ctx context.Context, schema *schemapb.CollectionSchema, paths []string, cm storage.ChunkManager, bufferSize int) (*Reader, error) {
	fields := lo.KeyBy(schema.GetFields(), func(field *schemapb.FieldSchema) int64 {
		return field.GetFieldID()
	})
	count, err := calcRowCount(bufferSize, schema)
	if err != nil {
		return nil, err
	}
	crs := make(map[int64]*FieldReader)
	readers, err := CreateReaders(ctx, paths, cm, schema)
	if err != nil {
		return nil, err
	}
	for fieldID, r := range readers {
		cr, err := NewFieldReader(r, fields[fieldID])
		if err != nil {
			return nil, err
		}
		crs[fieldID] = cr
	}
	return &Reader{
		schema: schema,
		count:  count,
		frs:    crs,
	}, nil
}

func (r *Reader) Read() (*storage.InsertData, error) {
	insertData, err := storage.NewInsertData(r.schema)
	if err != nil {
		return nil, err
	}
	for fieldID, cr := range r.frs {
		var data any
		data, err = cr.Next(r.count)
		if err != nil {
			return nil, err
		}
		if data == nil {
			return nil, io.EOF
		}
		err = insertData.Data[fieldID].AppendRows(data)
		if err != nil {
			return nil, err
		}
	}
	err = fillDynamicData(insertData, r.schema)
	if err != nil {
		return nil, err
	}
	return insertData, nil
}

func (r *Reader) Close() {
	for _, cr := range r.frs {
		cr.Close()
	}
}

func CreateReaders(ctx context.Context, paths []string, cm storage.ChunkManager, schema *schemapb.CollectionSchema) (map[int64]io.Reader, error) {
	readers := make(map[int64]io.Reader)
	nameToPath := lo.SliceToMap(paths, func(path string) (string, string) {
		nameWithExt := filepath.Base(path)
		name := strings.TrimSuffix(nameWithExt, filepath.Ext(nameWithExt))
		return name, path
	})
	for _, field := range schema.GetFields() {
		if field.GetIsPrimaryKey() && field.GetAutoID() {
			continue
		}
		if _, ok := nameToPath[field.GetName()]; !ok {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("no file for field: %s, files: %v", field.GetName(), lo.Values(nameToPath)))
		}
		reader, err := cm.Reader(ctx, nameToPath[field.GetName()])
		if err != nil {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("failed to read the file '%s', error: %s", nameToPath[field.GetName()], err.Error()))
		}
		readers[field.GetFieldID()] = reader
	}
	return readers, nil
}
