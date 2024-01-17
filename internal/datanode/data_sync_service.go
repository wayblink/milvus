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
	"errors"
	"fmt"
	"sync"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/mq/msgstream/mqwrapper"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/commonpbutil"
	"github.com/milvus-io/milvus/internal/util/concurrency"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/retry"
)

// dataSyncService controls a flowgraph for a specific collection
type dataSyncService struct {
	ctx          context.Context
	cancelFn     context.CancelFunc
	fg           *flowgraph.TimeTickedFlowGraph // internal flowgraph processes insert/delta messages
	flushCh      chan flushMsg
	resendTTCh   chan resendTTMsg   // chan to ask for resending DataNode time tick message.
	channel      Channel            // channel stores meta of channel
	idAllocator  allocatorInterface // id/timestamp allocator
	msFactory    msgstream.Factory
	collectionID UniqueID // collection id of vchan for which this data sync service serves
	vchannelName string
	dataCoord    types.DataCoord // DataCoord instance to interact with
	clearSignal  chan<- string   // signal channel to notify flowgraph close for collection/partition drop msg consumed

	delBufferManager *DeltaBufferManager
	flushingSegCache *Cache       // a guarding cache stores currently flushing segment ids
	flushManager     flushManager // flush manager handles flush process
	chunkManager     storage.ChunkManager
	compactor        *compactionExecutor // reference to compaction executor

	serverID       int64
	stopOnce       sync.Once
	flushListener  chan *segmentFlushPack // chan to listen flush event
	timetickSender *timeTickSender        // reference to timeTickSender
}

func newDataSyncService(ctx context.Context,
	flushCh chan flushMsg,
	resendTTCh chan resendTTMsg,
	channel Channel,
	alloc allocatorInterface,
	factory msgstream.Factory,
	vchan *datapb.VchannelInfo,
	clearSignal chan<- string,
	dataCoord types.DataCoord,
	flushingSegCache *Cache,
	chunkManager storage.ChunkManager,
	compactor *compactionExecutor,
	tickler *tickler,
	serverID int64,
	timetickSender *timeTickSender,
) (*dataSyncService, error) {

	if channel == nil {
		return nil, errors.New("Nil input")
	}

	ctx1, cancel := context.WithCancel(ctx)

	delBufferManager := &DeltaBufferManager{
		channel:    channel,
		delBufHeap: &PriorityQueue{},
	}

	service := &dataSyncService{
		ctx:              ctx1,
		cancelFn:         cancel,
		fg:               nil,
		flushCh:          flushCh,
		resendTTCh:       resendTTCh,
		channel:          channel,
		idAllocator:      alloc,
		msFactory:        factory,
		collectionID:     vchan.GetCollectionID(),
		vchannelName:     vchan.GetChannelName(),
		dataCoord:        dataCoord,
		clearSignal:      clearSignal,
		delBufferManager: delBufferManager,
		flushingSegCache: flushingSegCache,
		chunkManager:     chunkManager,
		compactor:        compactor,
		serverID:         serverID,
		timetickSender:   timetickSender,
	}

	if err := service.initNodes(vchan, tickler); err != nil {
		return nil, err
	}
	if tickler.isWatchFailed.Load() {
		return nil, errors.New("tickler watch failed")
	}
	return service, nil
}

type parallelConfig struct {
	maxQueueLength int32
	maxParallelism int32
}

type nodeConfig struct {
	msFactory    msgstream.Factory // msgStream factory
	collectionID UniqueID
	vChannelName string
	channel      Channel // Channel info
	allocator    allocatorInterface

	// defaults
	parallelConfig
}

func newParallelConfig() parallelConfig {
	return parallelConfig{Params.DataNodeCfg.FlowGraphMaxQueueLength, Params.DataNodeCfg.FlowGraphMaxParallelism}
}

// start the flow graph in datasyncservice
func (dsService *dataSyncService) start() {
	if dsService.fg != nil {
		log.Info("dataSyncService starting flow graph", zap.Int64("collectionID", dsService.collectionID),
			zap.String("vChanName", dsService.vchannelName))
		dsService.fg.Start()
	} else {
		log.Warn("dataSyncService starting flow graph is nil", zap.Int64("collectionID", dsService.collectionID),
			zap.String("vChanName", dsService.vchannelName))
	}
}

func (dsService *dataSyncService) close() {
	dsService.stopOnce.Do(func() {
		if dsService.fg != nil {
			log.Info("dataSyncService closing flowgraph", zap.Int64("collectionID", dsService.collectionID),
				zap.String("vChanName", dsService.vchannelName))
			dsService.fg.Close()
			metrics.DataNodeNumConsumers.WithLabelValues(fmt.Sprint(Params.DataNodeCfg.GetNodeID())).Dec()
			metrics.DataNodeNumProducers.WithLabelValues(fmt.Sprint(Params.DataNodeCfg.GetNodeID())).Sub(2) // timeTickChannel + deltaChannel
		}

		dsService.clearGlobalFlushingCache()
		close(dsService.flushCh)
		dsService.flushManager.close()
		dsService.cancelFn()
		dsService.channel.close()
	})
}

func (dsService *dataSyncService) clearGlobalFlushingCache() {
	segments := dsService.channel.listAllSegmentIDs()
	dsService.flushingSegCache.Remove(segments...)
}

// initNodes inits a TimetickedFlowGraph
func (dsService *dataSyncService) initNodes(vchanInfo *datapb.VchannelInfo, tickler *tickler) error {
	dsService.fg = flowgraph.NewTimeTickedFlowGraph(dsService.ctx)
	// initialize flush manager for DataSync Service
	dsService.flushManager = NewRendezvousFlushManager(dsService.idAllocator, dsService.chunkManager, dsService.channel,
		flushNotifyFunc(dsService, retry.Attempts(50)), dropVirtualChannelFunc(dsService))

	log.Info("begin to init data sync service", zap.Int64("collection", vchanInfo.CollectionID),
		zap.String("Chan", vchanInfo.ChannelName),
		zap.Int64s("unflushed", vchanInfo.GetUnflushedSegmentIds()),
		zap.Int64s("flushed", vchanInfo.GetFlushedSegmentIds()),
	)
	var err error
	// recover segment checkpoints
	unflushedSegmentInfos, err := dsService.getSegmentInfos(vchanInfo.GetUnflushedSegmentIds())
	if err != nil {
		return err
	}
	flushedSegmentInfos, err := dsService.getSegmentInfos(vchanInfo.GetFlushedSegmentIds())
	if err != nil {
		return err
	}
	log.Info("Finish get segmentInfos", zap.Int64("collection", vchanInfo.CollectionID),
		zap.String("Chan", vchanInfo.ChannelName),
		zap.Int("unflushed", len(vchanInfo.GetUnflushedSegmentIds())),
		zap.Int("flushed", len(vchanInfo.GetFlushedSegmentIds())),
	)

	//tickler will update addSegment progress to watchInfo
	tickler.watch()
	defer tickler.stop()
	futures := make([]*concurrency.Future, 0, len(unflushedSegmentInfos)+len(flushedSegmentInfos))

	for _, us := range unflushedSegmentInfos {
		if us.CollectionID != dsService.collectionID ||
			us.GetInsertChannel() != vchanInfo.ChannelName {
			log.Warn("Collection ID or ChannelName not match",
				zap.Int64("Wanted ID", dsService.collectionID),
				zap.Int64("Actual ID", us.CollectionID),
				zap.String("Wanted Channel Name", vchanInfo.ChannelName),
				zap.String("Actual Channel Name", us.GetInsertChannel()),
			)
			continue
		}

		log.Info("recover growing segments form checkpoints",
			zap.String("vChannelName", us.GetInsertChannel()),
			zap.Int64("segmentID", us.GetID()),
			zap.Int64("numRows", us.GetNumOfRows()),
		)

		// avoid closure capture iteration variable
		segment := us
		future := getOrCreateIOPool().Submit(func() (interface{}, error) {
			if err := dsService.channel.addSegment(addSegmentReq{
				segType:      datapb.SegmentType_Normal,
				segID:        segment.GetID(),
				collID:       segment.CollectionID,
				partitionID:  segment.PartitionID,
				numOfRows:    segment.GetNumOfRows(),
				statsBinLogs: segment.Statslogs,
				endPos:       segment.GetDmlPosition(),
				recoverTs:    vchanInfo.GetSeekPosition().GetTimestamp()}); err != nil {
				return nil, err
			}
			tickler.inc()
			return nil, nil
		})
		futures = append(futures, future)
	}

	for _, fs := range flushedSegmentInfos {
		log.Info("recover flushedSegmentInfos ",
			zap.Int64("Wanted ID", dsService.collectionID),
			zap.Int64("Actual ID", fs.GetCollectionID()),
			zap.String("Wanted Channel Name", vchanInfo.ChannelName),
			zap.String("Actual Channel Name", fs.GetInsertChannel()),
		)
		if fs.GetCollectionID() != dsService.collectionID ||
			fs.GetInsertChannel() != vchanInfo.ChannelName {
			log.Warn("Collection ID or ChannelName not match",
				zap.Int64("Wanted ID", dsService.collectionID),
				zap.Int64("Actual ID", fs.GetCollectionID()),
				zap.String("Wanted Channel Name", vchanInfo.ChannelName),
				zap.String("Actual Channel Name", fs.GetInsertChannel()),
			)
			continue
		}
		log.Info("recover sealed segments form checkpoints",
			zap.String("vChannelName", fs.GetInsertChannel()),
			zap.Int64("segmentID", fs.GetID()),
			zap.Int64("numRows", fs.GetNumOfRows()),
		)
		// avoid closure capture iteration variable
		segment := fs
		future := getOrCreateIOPool().Submit(func() (interface{}, error) {
			if err := dsService.channel.addSegment(addSegmentReq{
				segType:      datapb.SegmentType_Flushed,
				segID:        segment.GetID(),
				collID:       segment.CollectionID,
				partitionID:  segment.PartitionID,
				numOfRows:    segment.GetNumOfRows(),
				statsBinLogs: segment.Statslogs,
				recoverTs:    vchanInfo.GetSeekPosition().GetTimestamp(),
			}); err != nil {
				return nil, err
			}
			tickler.inc()
			return nil, nil
		})
		futures = append(futures, future)
	}

	err = concurrency.AwaitAll(futures...)
	if err != nil {
		return err
	}

	c := &nodeConfig{
		msFactory:    dsService.msFactory,
		collectionID: vchanInfo.GetCollectionID(),
		vChannelName: vchanInfo.GetChannelName(),
		channel:      dsService.channel,
		allocator:    dsService.idAllocator,

		parallelConfig: newParallelConfig(),
	}

	var dmStreamNode Node
	dmStreamNode, err = newDmInputNode(dsService.ctx, vchanInfo.GetSeekPosition(), c)
	if err != nil {
		return err
	}

	var ddNode Node
	ddNode, err = newDDNode(
		dsService.ctx,
		dsService.collectionID,
		vchanInfo.GetChannelName(),
		vchanInfo.GetDroppedSegmentIds(),
		flushedSegmentInfos,
		unflushedSegmentInfos,
		dsService.msFactory,
		dsService.compactor)
	if err != nil {
		return err
	}

	var insertBufferNode Node
	insertBufferNode, err = newInsertBufferNode(
		dsService.ctx,
		dsService.collectionID,
		dsService.delBufferManager,
		dsService.flushCh,
		dsService.resendTTCh,
		dsService.flushManager,
		dsService.flushingSegCache,
		c,
		dsService.timetickSender,
	)
	if err != nil {
		return err
	}

	var deleteNode Node
	deleteNode, err = newDeleteNode(dsService.ctx, dsService.flushManager, dsService.delBufferManager, dsService.clearSignal, c)
	if err != nil {
		return err
	}

	var ttNode Node
	ttNode, err = newTTNode(c, dsService.dataCoord)
	if err != nil {
		return err
	}

	dsService.fg.AddNode(dmStreamNode)
	dsService.fg.AddNode(ddNode)
	dsService.fg.AddNode(insertBufferNode)
	dsService.fg.AddNode(deleteNode)
	dsService.fg.AddNode(ttNode)

	// ddStreamNode
	err = dsService.fg.SetEdges(dmStreamNode.Name(),
		[]string{ddNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node", zap.String("name", dmStreamNode.Name()), zap.Error(err))
		return err
	}

	// ddNode
	err = dsService.fg.SetEdges(ddNode.Name(),
		[]string{insertBufferNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node", zap.String("name", ddNode.Name()), zap.Error(err))
		return err
	}

	// insertBufferNode
	err = dsService.fg.SetEdges(insertBufferNode.Name(),
		[]string{deleteNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node", zap.String("name", insertBufferNode.Name()), zap.Error(err))
		return err
	}

	//deleteNode
	err = dsService.fg.SetEdges(deleteNode.Name(),
		[]string{ttNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node", zap.String("name", deleteNode.Name()), zap.Error(err))
		return err
	}

	// ttNode
	err = dsService.fg.SetEdges(ttNode.Name(),
		[]string{},
	)
	if err != nil {
		log.Error("set edges failed in node", zap.String("name", ttNode.Name()), zap.Error(err))
		return err
	}
	log.Info("Finish init data sync service", zap.Int64("collection", vchanInfo.CollectionID),
		zap.String("Chan", vchanInfo.ChannelName),
	)
	return nil
}

// getSegmentInfos return the SegmentInfo details according to the given ids through RPC to datacoord
func (dsService *dataSyncService) getSegmentInfos(segmentIDs []int64) ([]*datapb.SegmentInfo, error) {
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}
	res := make([]*datapb.SegmentInfo, 0)
	getSegmentInfosFunc := func(segmentIDs []int64) error {
		log.Info("dataSyncService call GetSegmentInfo", zap.Int("length", len(segmentIDs)), zap.Int64s("ids", segmentIDs))
		infoResp, err := dsService.dataCoord.GetSegmentInfo(dsService.ctx, &datapb.GetSegmentInfoRequest{
			Base: commonpbutil.NewMsgBase(
				commonpbutil.WithMsgType(commonpb.MsgType_SegmentInfo),
				commonpbutil.WithMsgID(0),
				commonpbutil.WithTimeStamp(0),
				commonpbutil.WithSourceID(Params.ProxyCfg.GetNodeID()),
			),
			SegmentIDs:       segmentIDs,
			IncludeUnHealthy: true,
		})
		if err != nil {
			log.Error("Fail to get datapb.SegmentInfo by ids from datacoord", zap.Error(err))
			return err
		}
		if infoResp.GetStatus().ErrorCode != commonpb.ErrorCode_Success {
			err = errors.New(infoResp.GetStatus().Reason)
			log.Error("Fail to get datapb.SegmentInfo by ids from datacoord", zap.Error(err))
			return err
		}
		mu.Lock()
		res = append(res, infoResp.Infos...)
		mu.Unlock()
		return nil
	}
	var buckets [][]int64
	chunkSize := 500
	length := len(segmentIDs)
	for i := 0; i < length; i += chunkSize {
		end := i + chunkSize
		if end > length {
			end = length
		}
		buckets = append(buckets, segmentIDs[i:end])
	}
	var finalErr error
	for _, bucket := range buckets {
		bucketClone := bucket
		wg.Add(1)
		go func() {
			err := getSegmentInfosFunc(bucketClone)
			if err != nil {
				finalErr = err
			}
			wg.Done()
		}()
	}
	wg.Wait()
	return res, finalErr
}

func (dsService *dataSyncService) getChannelLatestMsgID(ctx context.Context, channelName string, segmentID int64) ([]byte, error) {
	pChannelName := funcutil.ToPhysicalChannel(channelName)
	dmlStream, err := dsService.msFactory.NewMsgStream(ctx)
	if err != nil {
		return nil, err
	}
	defer dmlStream.Close()

	subName := fmt.Sprintf("datanode-%d-%s-%d", Params.DataNodeCfg.GetNodeID(), channelName, segmentID)
	log.Debug("dataSyncService register consumer for getChannelLatestMsgID",
		zap.String("pChannelName", pChannelName),
		zap.String("subscription", subName),
	)
	dmlStream.AsConsumer([]string{pChannelName}, subName, mqwrapper.SubscriptionPositionUnknown)
	id, err := dmlStream.GetLatestMsgID(pChannelName)
	if err != nil {
		log.Error("fail to GetLatestMsgID", zap.String("pChannel", pChannelName), zap.Error(err))
		return nil, err
	}
	return id.Serialize(), nil
}
