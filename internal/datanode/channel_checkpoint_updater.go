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
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/lock"
)

const (
	updateChanCPInterval    = 1 * time.Minute
	updateChanCPTimeout     = 10 * time.Second
	updateChanCPMaxParallel = 1000 // todo configurable
)

type channelCheckpointUpdater struct {
	dn          *DataNode
	workerPool  *conc.Pool[any]
	channelLock *lock.KeyLock[string]
	closeOnce   sync.Once
}

func newChannelCheckpointUpdater(dn *DataNode) *channelCheckpointUpdater {
	return &channelCheckpointUpdater{
		dn:          dn,
		workerPool:  conc.NewPool[any](updateChanCPMaxParallel, conc.WithPreAlloc(true)),
		channelLock: lock.NewKeyLock[string](),
	}
}

var (
	singletonCpUpdater *channelCheckpointUpdater
	cpUpdaterInitOnce  sync.Once
)

func getOrCreateChannelCheckPointUpdater(dn *DataNode) *channelCheckpointUpdater {
	cpUpdaterInitOnce.Do(func() {
		singletonCpUpdater = newChannelCheckpointUpdater(dn)
	})
	return singletonCpUpdater
}

func (cm *channelCheckpointUpdater) close() {
	cm.closeOnce.Do(func() {
		cm.workerPool.Release()
	})
}

func (cm *channelCheckpointUpdater) updateChannelCP(channelPos *msgpb.MsgPosition, callback func() error) error {
	cm.channelLock.Lock(channelPos.GetChannelName())
	defer cm.channelLock.Unlock(channelPos.GetChannelName())

	cm.workerPool.Submit(func() (any, error) {
		ctx, cancel := context.WithTimeout(context.Background(), updateChanCPTimeout)
		defer cancel()
		err := cm.dn.broker.UpdateChannelCheckpoint(ctx, channelPos.GetChannelName(), channelPos)
		if err != nil {
			return nil, err
		}
		err = callback()
		return nil, err
	})
	return nil
}
