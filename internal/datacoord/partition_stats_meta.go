package datacoord

import (
	"context"
	"fmt"
	"sync"

	"github.com/milvus-io/milvus/pkg/util/merr"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
)

type partitionStatsMeta struct {
	sync.RWMutex
	ctx                 context.Context
	catalog             metastore.DataCoordCatalog
	partitionStatsInfos map[string]map[int64]*partitionStatsInfo // channel -> partition -> PartitionStatsInfo
}

type partitionStatsInfo struct {
	currentPlanID int64
	infos         map[int64]*datapb.PartitionStatsInfo
}

func newPartitionStatsMeta(ctx context.Context, catalog metastore.DataCoordCatalog) (*partitionStatsMeta, error) {
	psm := &partitionStatsMeta{
		RWMutex:             sync.RWMutex{},
		ctx:                 ctx,
		catalog:             catalog,
		partitionStatsInfos: make(map[string]map[int64]*partitionStatsInfo),
	}
	if err := psm.reloadFromKV(); err != nil {
		return nil, err
	}
	return psm, nil
}

func (psm *partitionStatsMeta) reloadFromKV() error {
	record := timerecord.NewTimeRecorder("partitionStatsMeta-reloadFromKV")

	partitionStatsInfos, err := psm.catalog.ListPartitionStatsInfos(psm.ctx)
	if err != nil {
		return err
	}
	for _, info := range partitionStatsInfos {
		if _, ok := psm.partitionStatsInfos[info.GetVChannel()]; !ok {
			psm.partitionStatsInfos[info.GetVChannel()] = make(map[int64]*partitionStatsInfo)
		}
		if _, ok := psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()]; !ok {
			currentPlanID, err := psm.catalog.GetPartitionStatsCurrentPlanID(psm.ctx, info.GetCollectionID(), info.GetPartitionID(), info.GetVChannel())
			if err != nil {
				return err
			}
			psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()] = &partitionStatsInfo{
				currentPlanID: currentPlanID,
				infos:         make(map[int64]*datapb.PartitionStatsInfo),
			}
		}
		psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()].infos[info.GetVersion()] = info
	}
	log.Info("DataCoord partitionStatsMeta reloadFromKV done", zap.Duration("duration", record.ElapseSpan()))
	return nil
}

func (psm *partitionStatsMeta) ListAllPartitionStatsInfos() []*datapb.PartitionStatsInfo {
	psm.RLock()
	defer psm.RUnlock()
	res := make([]*datapb.PartitionStatsInfo, 0)
	for _, partitionStats := range psm.partitionStatsInfos {
		for _, infos := range partitionStats {
			for _, info := range infos.infos {
				res = append(res, info)
			}
		}
	}
	return res
}

func (psm *partitionStatsMeta) ListPartitionStatsInfos(collectionID int64, partitionID int64, vchannel string, filters ...func([]*datapb.PartitionStatsInfo) []*datapb.PartitionStatsInfo) []*datapb.PartitionStatsInfo {
	psm.RLock()
	defer psm.RUnlock()
	res := make([]*datapb.PartitionStatsInfo, 0)
	partitionStats, ok := psm.partitionStatsInfos[vchannel]
	if !ok {
		return res
	}
	infos, ok := partitionStats[partitionID]
	if !ok {
		return res
	}
	for _, info := range infos.infos {
		res = append(res, info)
	}

	for _, filter := range filters {
		res = filter(res)
	}
	return res
}

func (psm *partitionStatsMeta) SavePartitionStatsInfo(info *datapb.PartitionStatsInfo) error {
	psm.Lock()
	defer psm.Unlock()
	if err := psm.catalog.SavePartitionStatsInfo(psm.ctx, info); err != nil {
		log.Error("meta update: update PartitionStatsInfo info fail", zap.Error(err))
		return err
	}
	if _, ok := psm.partitionStatsInfos[info.GetVChannel()]; !ok {
		psm.partitionStatsInfos[info.GetVChannel()] = make(map[int64]*partitionStatsInfo)
	}
	if _, ok := psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()]; !ok {
		psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()] = &partitionStatsInfo{
			infos: make(map[int64]*datapb.PartitionStatsInfo),
		}
	}

	psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()].infos[info.GetVersion()] = info
	return nil
}

func (psm *partitionStatsMeta) DropPartitionStatsInfo(info *datapb.PartitionStatsInfo) error {
	psm.Lock()
	defer psm.Unlock()
	if err := psm.catalog.DropPartitionStatsInfo(psm.ctx, info); err != nil {
		log.Error("meta update: drop PartitionStatsInfo info fail",
			zap.Int64("collectionID", info.GetCollectionID()),
			zap.Int64("partitionID", info.GetPartitionID()),
			zap.String("vchannel", info.GetVChannel()),
			zap.Int64("version", info.GetVersion()),
			zap.Error(err))
		return err
	}
	if _, ok := psm.partitionStatsInfos[info.GetVChannel()]; !ok {
		return nil
	}
	if _, ok := psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()]; !ok {
		return nil
	}
	delete(psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()].infos, info.GetVersion())
	if len(psm.partitionStatsInfos[info.GetVChannel()][info.GetPartitionID()].infos) == 0 {
		delete(psm.partitionStatsInfos[info.GetVChannel()], info.GetPartitionID())
	}
	if len(psm.partitionStatsInfos[info.GetVChannel()]) == 0 {
		delete(psm.partitionStatsInfos, info.GetVChannel())
	}
	return nil
}

func (psm *partitionStatsMeta) SaveCurrentPlanID(collectionID, partitionID int64, vChannel string, currentPlanID int64) error {
	psm.Lock()
	defer psm.Unlock()

	log.Info("update current planID", zap.Int64("collectionID", collectionID),
		zap.Int64("partitionID", partitionID),
		zap.String("vChannel", vChannel), zap.Int64("currentPlanID", currentPlanID))

	if _, ok := psm.partitionStatsInfos[vChannel]; !ok {
		return merr.WrapErrClusteringCompactionMetaError("SaveCurrentPlanID",
			fmt.Errorf("update current planID failed, there is no partition info exists with collID: %d, partID: %d, vChannel: %s", collectionID, partitionID, vChannel))
	}
	if _, ok := psm.partitionStatsInfos[vChannel][partitionID]; !ok {
		return merr.WrapErrClusteringCompactionMetaError("SaveCurrentPlanID",
			fmt.Errorf("update current planID failed, there is no partition info exists with collID: %d, partID: %d, vChannel: %s", collectionID, partitionID, vChannel))
	}

	if err := psm.catalog.SavePartitionStatsCurrentPlanID(psm.ctx, collectionID, partitionID, vChannel, currentPlanID); err != nil {
		return err
	}

	psm.partitionStatsInfos[vChannel][partitionID].currentPlanID = currentPlanID
	return nil
}

func (psm *partitionStatsMeta) GetCurrentPlanID(collectionID, partitionID int64, vChannel string) int64 {
	psm.RLock()
	defer psm.RUnlock()

	if _, ok := psm.partitionStatsInfos[vChannel]; !ok {
		return 0
	}
	if _, ok := psm.partitionStatsInfos[vChannel][partitionID]; !ok {
		return 0
	}
	return psm.partitionStatsInfos[vChannel][partitionID].currentPlanID
}
