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

package parquet

import (
	"fmt"

	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/apache/arrow/go/v12/parquet"
	"github.com/apache/arrow/go/v12/parquet/file"
	"github.com/apache/arrow/go/v12/parquet/pqarrow"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

type Reader struct {
	reader *file.Reader

	bufferSize int

	schema *schemapb.CollectionSchema
	frs    map[int64]*FieldReader // fieldID -> FieldReader
}

func NewReader(schema *schemapb.CollectionSchema, cmReader storage.FileReader, bufferSize int) (*Reader, error) {
	const pqBufSize = 32 * 1024 * 1024 // TODO: dyh, make if configurable
	size := calcBufferSize(pqBufSize, schema)
	reader, err := file.NewParquetReader(cmReader, file.WithReadProps(&parquet.ReaderProperties{
		BufferSize:            int64(size),
		BufferedStreamEnabled: true,
	}))
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("new parquet reader failed, err=%v", err))
	}
	log.Info("create parquet reader done", zap.Int("row group num", reader.NumRowGroups()),
		zap.Int64("num rows", reader.NumRows()))

	fileReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("new parquet file reader failed, err=%v", err))
	}

	crs, err := CreateFieldReaders(fileReader, schema)
	if err != nil {
		return nil, err
	}
	return &Reader{
		reader:     reader,
		bufferSize: bufferSize,
		schema:     schema,
		frs:        crs,
	}, nil
}

func (r *Reader) Read() (*storage.InsertData, error) {
	insertData, err := storage.NewInsertData(r.schema)
	if err != nil {
		return nil, err
	}
OUTER:
	for {
		for fieldID, cr := range r.frs {
			data, err := cr.Next(1)
			if err != nil {
				return nil, err
			}
			if data == nil {
				break OUTER
			}
			err = insertData.Data[fieldID].AppendRows(data)
			if err != nil {
				return nil, err
			}
		}
		if insertData.GetMemorySize() >= r.bufferSize {
			break
		}
	}
	for fieldID := range r.frs {
		if insertData.Data[fieldID].RowNum() == 0 {
			return nil, nil
		}
	}
	return insertData, nil
}

func (r *Reader) Close() {
	for _, cr := range r.frs {
		cr.Close()
	}
	err := r.reader.Close()
	if err != nil {
		log.Warn("close parquet reader failed", zap.Error(err))
	}
}
