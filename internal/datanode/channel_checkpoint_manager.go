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
	"context"
	"sync"

	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/lock"
)

type channelCPManager struct {
	dn *DataNode

	workerPool  *conc.Pool[any]
	channelLock *lock.KeyLock[string]

	closeChan chan struct{}
	closeOnce sync.Once
}

func newChannelCPManager(dn *DataNode) *channelCPManager {
	return &channelCPManager{
		dn:          dn,
		workerPool:  conc.NewPool[any](500, conc.WithPreAlloc(true)),
		channelLock: lock.NewKeyLock[string](),
		closeChan:   make(chan struct{}),
	}
}

var (
	singletonChannelCPManager *channelCPManager
	channelCPManagerInitOnce  sync.Once
)

func getOrCreateChannelCPManager(dn *DataNode) *channelCPManager {
	channelCPManagerInitOnce.Do(func() {
		singletonChannelCPManager = newChannelCPManager(dn)
	})
	return singletonChannelCPManager
}

func (cm *channelCPManager) close() {
	cm.closeOnce.Do(func() {
		cm.workerPool.Release()
	})
}

func (cm *channelCPManager) Submit(channelPos *msgpb.MsgPosition, callback func() error) *conc.Future[any] {
	cm.channelLock.Lock(channelPos.GetChannelName())
	return cm.workerPool.Submit(func() (any, error) {
		defer cm.channelLock.Unlock(channelPos.GetChannelName())
		// TODO, change to ETCD operation, avoid datacoord operation
		ctx, cancel := context.WithTimeout(context.Background(), updateChanCPTimeout)
		defer cancel()
		err := cm.dn.broker.UpdateChannelCheckpoint(ctx, channelPos.GetChannelName(), channelPos)
		if err != nil {
			return nil, err
		}
		err = callback()
		return nil, err
	})
}
