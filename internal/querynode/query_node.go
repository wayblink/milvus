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

package querynode

/*

#cgo CFLAGS: -I${SRCDIR}/../core/output/include

#cgo darwin LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath,"${SRCDIR}/../core/output/lib"
#cgo linux LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath=${SRCDIR}/../core/output/lib
#cgo windows LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath=${SRCDIR}/../core/output/lib

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"
#include "segcore/segcore_init_c.h"

*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/golang/protobuf/proto"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

// make sure QueryNode implements types.QueryNode
var _ types.QueryNode = (*QueryNode)(nil)

// make sure QueryNode implements types.QueryNodeComponent
var _ types.QueryNodeComponent = (*QueryNode)(nil)

var Params paramtable.ComponentParam

// QueryNode communicates with outside services and union all
// services in querynode package.
//
// QueryNode implements `types.Component`, `types.QueryNode` interfaces.
//  `rootCoord` is a grpc client of root coordinator.
//  `indexCoord` is a grpc client of index coordinator.
//  `stateCode` is current statement of this query node, indicating whether it's healthy.
type QueryNode struct {
	queryNodeLoopCtx    context.Context
	queryNodeLoopCancel context.CancelFunc

	wg sync.WaitGroup

	stateCode atomic.Value

	//call once
	initOnce sync.Once

	// internal components
	historical *historical
	streaming  *streaming

	// tSafeReplica
	tSafeReplica TSafeReplicaInterface

	// dataSyncService
	dataSyncService *dataSyncService

	// internal services
	//queryService *queryService
	statsService *statsService

	// segment loader
	loader *segmentLoader

	// etcd client
	etcdCli *clientv3.Client

	factory   dependency.Factory
	scheduler *taskScheduler

	session        *sessionutil.Session
	eventCh        <-chan *sessionutil.SessionEvent
	sessionManager *SessionManager

	vectorStorage storage.ChunkManager
	cacheStorage  storage.ChunkManager
	etcdKV        *etcdkv.EtcdKV

	// shard cluster service, handle shard leader functions
	ShardClusterService *ShardClusterService
	//shard query service, handles shard-level query & search
	queryShardService *queryShardService
}

// NewQueryNode will return a QueryNode with abnormal state.
func NewQueryNode(ctx context.Context, factory dependency.Factory) *QueryNode {
	ctx1, cancel := context.WithCancel(ctx)
	node := &QueryNode{
		queryNodeLoopCtx:    ctx1,
		queryNodeLoopCancel: cancel,
		factory:             factory,
	}

	node.scheduler = newTaskScheduler(ctx1)
	node.UpdateStateCode(internalpb.StateCode_Abnormal)

	return node
}

func (node *QueryNode) initSession() error {
	node.session = sessionutil.NewSession(node.queryNodeLoopCtx, Params.EtcdCfg.MetaRootPath, node.etcdCli)
	if node.session == nil {
		return fmt.Errorf("session is nil, the etcd client connection may have failed")
	}
	node.session.Init(typeutil.QueryNodeRole, Params.QueryNodeCfg.QueryNodeIP+":"+strconv.FormatInt(Params.QueryNodeCfg.QueryNodePort, 10), false, true)
	Params.QueryNodeCfg.SetNodeID(node.session.ServerID)
	Params.SetLogger(Params.QueryNodeCfg.GetNodeID())
	log.Info("QueryNode init session", zap.Int64("nodeID", Params.QueryNodeCfg.GetNodeID()), zap.String("node address", node.session.Address))
	return nil
}

// Register register query node at etcd
func (node *QueryNode) Register() error {
	node.session.Register(func() {
		log.Error("Query Node disconnected from etcd, process will exit", zap.Int64("Server Id", node.session.ServerID))
		if err := node.Stop(); err != nil {
			log.Fatal("failed to stop server", zap.Error(err))
		}
		// manually send signal to starter goroutine
		if node.session.TriggerKill {
			if p, err := os.FindProcess(os.Getpid()); err == nil {
				p.Signal(syscall.SIGINT)
			}
		}
	})

	//TODO Reset the logger
	//Params.initLogCfg()
	return nil
}

// InitSegcore set init params of segCore, such as chunckRows, SIMD type...
func (node *QueryNode) InitSegcore() {
	cEasyloggingYaml := C.CString(path.Join(Params.BaseTable.GetConfigDir(), paramtable.DefaultEasyloggingYaml))
	C.SegcoreInit(cEasyloggingYaml)
	C.free(unsafe.Pointer(cEasyloggingYaml))

	// override segcore chunk size
	cChunkRows := C.int64_t(Params.QueryNodeCfg.ChunkRows)
	C.SegcoreSetChunkRows(cChunkRows)

	nlist := C.int64_t(Params.QueryNodeCfg.SmallIndexNlist)
	C.SegcoreSetNlist(nlist)

	nprobe := C.int64_t(Params.QueryNodeCfg.SmallIndexNProbe)
	C.SegcoreSetNprobe(nprobe)

	// override segcore SIMD type
	cSimdType := C.CString(Params.CommonCfg.SimdType)
	cRealSimdType := C.SegcoreSetSimdType(cSimdType)
	Params.CommonCfg.SimdType = C.GoString(cRealSimdType)
	C.free(unsafe.Pointer(cRealSimdType))
	C.free(unsafe.Pointer(cSimdType))

	// override segcore index slice size
	cIndexSliceSize := C.int64_t(Params.CommonCfg.IndexSliceSize)
	C.SegcoreSetIndexSliceSize(cIndexSliceSize)
}

func (node *QueryNode) initServiceDiscovery() error {
	if node.session == nil {
		return errors.New("session is nil")
	}

	sessions, rev, err := node.session.GetSessions(typeutil.ProxyRole)
	if err != nil {
		log.Warn("QueryNode failed to init service discovery", zap.Error(err))
		return err
	}
	log.Info("QueryNode success to get Proxy sessions", zap.Any("sessions", sessions))

	nodes := make([]*NodeInfo, 0, len(sessions))
	for _, session := range sessions {
		info := &NodeInfo{
			NodeID:  session.ServerID,
			Address: session.Address,
		}
		nodes = append(nodes, info)
	}

	node.sessionManager.Startup(nodes)

	node.eventCh = node.session.WatchServices(typeutil.ProxyRole, rev+1, nil)
	return nil
}

func (node *QueryNode) watchService(ctx context.Context) {
	defer node.wg.Done()
	for {
		select {
		case <-ctx.Done():
			log.Info("watch service shutdown")
			return
		case event, ok := <-node.eventCh:
			if !ok {
				// ErrCompacted is handled inside SessionWatcher
				log.Error("Session Watcher channel closed", zap.Int64("server id", node.session.ServerID))
				// need to call stop in separate goroutine
				go node.Stop()
				if node.session.TriggerKill {
					if p, err := os.FindProcess(os.Getpid()); err == nil {
						p.Signal(syscall.SIGINT)
					}
				}
				return
			}
			node.handleSessionEvent(ctx, event)
		}
	}
}

func (node *QueryNode) handleSessionEvent(ctx context.Context, event *sessionutil.SessionEvent) {
	info := &NodeInfo{
		NodeID:  event.Session.ServerID,
		Address: event.Session.Address,
	}
	switch event.EventType {
	case sessionutil.SessionAddEvent:
		node.sessionManager.AddSession(info)
	case sessionutil.SessionDelEvent:
		node.sessionManager.DeleteSession(info)
	default:
		log.Warn("receive unknown service event type",
			zap.Any("type", event.EventType))
	}
}

// Init function init historical and streaming module to manage segments
func (node *QueryNode) Init() error {
	var initError error = nil
	node.initOnce.Do(func() {
		//ctx := context.Background()
		log.Info("QueryNode session info", zap.String("metaPath", Params.EtcdCfg.MetaRootPath))
		err := node.initSession()
		if err != nil {
			log.Error("QueryNode init session failed", zap.Error(err))
			initError = err
			return
		}

		node.factory.Init(&Params)

		node.vectorStorage, err = node.factory.NewVectorStorageChunkManager(node.queryNodeLoopCtx)
		if err != nil {
			log.Error("QueryNode init vector storage failed", zap.Error(err))
			initError = err
			return
		}

		node.cacheStorage, err = node.factory.NewCacheStorageChunkManager(node.queryNodeLoopCtx)
		if err != nil {
			log.Error("QueryNode init cache storage failed", zap.Error(err))
			initError = err
			return
		}

		node.etcdKV = etcdkv.NewEtcdKV(node.etcdCli, Params.EtcdCfg.MetaRootPath)
		log.Info("queryNode try to connect etcd success", zap.Any("MetaRootPath", Params.EtcdCfg.MetaRootPath))
		node.tSafeReplica = newTSafeReplica()

		streamingReplica := newCollectionReplica(node.etcdKV)
		historicalReplica := newCollectionReplica(node.etcdKV)

		node.historical = newHistorical(node.queryNodeLoopCtx,
			historicalReplica,
			node.tSafeReplica,
		)
		node.streaming = newStreaming(node.queryNodeLoopCtx,
			streamingReplica,
			node.factory,
			node.etcdKV,
			node.tSafeReplica,
		)

		node.loader = newSegmentLoader(
			node.historical.replica,
			node.streaming.replica,
			node.etcdKV,
			node.vectorStorage,
			node.factory)

		// node.statsService = newStatsService(node.queryNodeLoopCtx, node.historical.replica, node.factory)
		node.dataSyncService = newDataSyncService(node.queryNodeLoopCtx, streamingReplica, historicalReplica, node.tSafeReplica, node.factory)

		node.InitSegcore()

		// TODO: add session creator to node
		node.sessionManager = NewSessionManager(withSessionCreator(defaultSessionCreator()))

		// init services and manager
		// TODO: pass node.streaming.replica to search service
		// node.queryService = newQueryService(node.queryNodeLoopCtx,
		// 	node.historical,
		// 	node.streaming,
		// 	node.vectorStorage,
		// 	node.cacheStorage,
		// 	node.factory,
		// 	qsOptWithSessionManager(node.sessionManager))

		log.Info("query node init successfully",
			zap.Any("queryNodeID", Params.QueryNodeCfg.GetNodeID()),
			zap.Any("IP", Params.QueryNodeCfg.QueryNodeIP),
			zap.Any("Port", Params.QueryNodeCfg.QueryNodePort),
		)
	})

	return initError
}

// Start mainly start QueryNode's query service.
func (node *QueryNode) Start() error {
	// start task scheduler
	go node.scheduler.Start()

	// start services
	go node.watchChangeInfo()
	//go node.statsService.start()

	// watch proxy
	if err := node.initServiceDiscovery(); err != nil {
		return err
	}

	node.wg.Add(1)
	go node.watchService(node.queryNodeLoopCtx)

	// create shardClusterService for shardLeader functions.
	node.ShardClusterService = newShardClusterService(node.etcdCli, node.session, node)
	// create shard-level query service
	node.queryShardService = newQueryShardService(node.queryNodeLoopCtx, node.historical, node.streaming, node.ShardClusterService, node.factory)

	Params.QueryNodeCfg.CreatedTime = time.Now()
	Params.QueryNodeCfg.UpdatedTime = time.Now()

	node.UpdateStateCode(internalpb.StateCode_Healthy)
	log.Info("query node start successfully",
		zap.Any("queryNodeID", Params.QueryNodeCfg.GetNodeID()),
		zap.Any("IP", Params.QueryNodeCfg.QueryNodeIP),
		zap.Any("Port", Params.QueryNodeCfg.QueryNodePort),
	)
	return nil
}

// Stop mainly stop QueryNode's query service, historical loop and streaming loop.
func (node *QueryNode) Stop() error {
	log.Warn("Query node stop..")
	node.UpdateStateCode(internalpb.StateCode_Abnormal)
	node.queryNodeLoopCancel()

	// close services
	if node.dataSyncService != nil {
		node.dataSyncService.close()
	}
	if node.historical != nil {
		node.historical.close()
	}
	if node.streaming != nil {
		node.streaming.close()
	}
	/*
		if node.queryService != nil {
			node.queryService.close()
		}*/

	if node.queryShardService != nil {
		node.queryShardService.close()
	}
	//if node.statsService != nil {
	//	node.statsService.close()
	//}
	node.session.Revoke(time.Second)
	node.wg.Wait()
	return nil
}

// UpdateStateCode updata the state of query node, which can be initializing, healthy, and abnormal
func (node *QueryNode) UpdateStateCode(code internalpb.StateCode) {
	node.stateCode.Store(code)
}

// SetEtcdClient assigns parameter client to its member etcdCli
func (node *QueryNode) SetEtcdClient(client *clientv3.Client) {
	node.etcdCli = client
}

func (node *QueryNode) watchChangeInfo() {
	log.Info("query node watchChangeInfo start")
	watchChan := node.etcdKV.WatchWithPrefix(util.ChangeInfoMetaPrefix)
	for {
		select {
		case <-node.queryNodeLoopCtx.Done():
			log.Info("query node watchChangeInfo close")
			return
		case resp := <-watchChan:
			for _, event := range resp.Events {
				switch event.Type {
				case mvccpb.PUT:
					infoID, err := strconv.ParseInt(filepath.Base(string(event.Kv.Key)), 10, 64)
					if err != nil {
						log.Warn("Parse SealedSegmentsChangeInfo id failed", zap.Any("error", err.Error()))
						continue
					}
					log.Info("get SealedSegmentsChangeInfo from etcd",
						zap.Any("infoID", infoID),
					)
					info := &querypb.SealedSegmentsChangeInfo{}
					err = proto.Unmarshal(event.Kv.Value, info)
					if err != nil {
						log.Warn("Unmarshal SealedSegmentsChangeInfo failed", zap.Any("error", err.Error()))
						continue
					}
					go node.handleSealedSegmentsChangeInfo(info)
				default:
					// do nothing
				}
			}
		}
	}
}

func (node *QueryNode) handleSealedSegmentsChangeInfo(info *querypb.SealedSegmentsChangeInfo) {
	for _, line := range info.GetInfos() {
		vchannel, err := validateChangeChannel(line)
		if err != nil {
			log.Warn("failed to validate vchannel for SegmentChangeInfo", zap.Error(err))
			continue
		}

		node.ShardClusterService.HandoffVChannelSegments(vchannel, line)
	}
}

func validateChangeChannel(info *querypb.SegmentChangeInfo) (string, error) {
	if len(info.GetOnlineSegments()) == 0 && len(info.GetOfflineSegments()) == 0 {
		return "", errors.New("SegmentChangeInfo with no segments info")
	}

	var channelName string

	for _, segment := range info.GetOnlineSegments() {
		if channelName == "" {
			channelName = segment.GetDmChannel()
		}
		if segment.GetDmChannel() != channelName {
			return "", fmt.Errorf("found multilple channel name in one SegmentChangeInfo, channel1: %s, channel 2:%s", channelName, segment.GetDmChannel())
		}
	}
	for _, segment := range info.GetOfflineSegments() {
		if channelName == "" {
			channelName = segment.GetDmChannel()
		}
		if segment.GetDmChannel() != channelName {
			return "", fmt.Errorf("found multilple channel name in one SegmentChangeInfo, channel1: %s, channel 2:%s", channelName, segment.GetDmChannel())
		}
	}

	return channelName, nil
}

// remove the segments since it's already compacted or balanced to other QueryNodes
func (node *QueryNode) removeSegments(segmentChangeInfos *querypb.SealedSegmentsChangeInfo) error {

	node.streaming.replica.queryLock()
	node.historical.replica.queryLock()
	defer node.streaming.replica.queryUnlock()
	defer node.historical.replica.queryUnlock()
	for _, info := range segmentChangeInfos.Infos {
		// For online segments:
		for _, segmentInfo := range info.OnlineSegments {
			// delete growing segment because these segments are loaded in historical.
			hasGrowingSegment := node.streaming.replica.hasSegment(segmentInfo.SegmentID)
			if hasGrowingSegment {
				err := node.streaming.replica.removeSegment(segmentInfo.SegmentID)
				if err != nil {
					return err
				}
				log.Info("remove growing segment in removeSegments",
					zap.Any("collectionID", segmentInfo.CollectionID),
					zap.Any("segmentID", segmentInfo.SegmentID),
					zap.Any("infoID", segmentChangeInfos.Base.GetMsgID()),
				)
			}
		}

		// For offline segments:
		for _, segmentInfo := range info.OfflineSegments {
			// load balance or compaction, remove old sealed segments.
			if info.OfflineNodeID == Params.QueryNodeCfg.GetNodeID() {
				err := node.historical.replica.removeSegment(segmentInfo.SegmentID)
				if err != nil {
					return err
				}
				log.Info("remove sealed segment", zap.Any("collectionID", segmentInfo.CollectionID),
					zap.Any("segmentID", segmentInfo.SegmentID),
					zap.Any("infoID", segmentChangeInfos.Base.GetMsgID()),
				)
			}
		}
	}
	return nil
}
