package sessionutil

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/retry"
	"go.etcd.io/etcd/api/v3/mvccpb"
	v3rpc "go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"
	"path"
	"strconv"
	"sync"
	"time"
)

//var _ SessionManager = (*EtcdSessionManager)(nil)

type EtcdSessionManager struct {
	ctx               context.Context
	etcdCli           *clientv3.Client
	sessionTTL        int64
	sessionRetryTimes int64

	metaRoot    string
	serverIDMu  sync.Mutex
	reuseNodeID bool

	registerSessions   map[*Session]*EtcdSessionRegister
	preemptiveSessions map[*Session]*EtcdSessionRegister
	//competeRegisterSessions map[string]Session
	watchSessions map[string]*EtcdSessionWatcher
}

func NewEtcdSessionManager(
	ctx context.Context,
	etcdCli *clientv3.Client,
	metaRoot string,
	//sessionTTL int64,
	//sessionRetryTimes int64,
	opts ...SessionOption) *EtcdSessionManager {

	session := &EtcdSessionManager{
		ctx:                ctx,
		metaRoot:           metaRoot,
		sessionTTL:         60,
		sessionRetryTimes:  30,
		registerSessions:   make(map[*Session]*EtcdSessionRegister, 0),
		preemptiveSessions: make(map[*Session]*EtcdSessionRegister, 0),
		watchSessions:      make(map[string]*EtcdSessionWatcher, 0),
	}
	// integration test create cluster with different nodeId in one process
	//if paramtable.Get().IntegrationTestCfg.IntegrationMode.GetAsBool() {
	//	session.reuseNodeID = false
	//}

	//session.apply(opts...)

	connectEtcdFn := func() error {
		log.Debug("Session try to connect to etcd")
		ctx2, cancel2 := context.WithTimeout(session.ctx, 5*time.Second)
		defer cancel2()
		if _, err := etcdCli.Get(ctx2, "health"); err != nil {
			return err
		}
		session.etcdCli = etcdCli
		return nil
	}
	err := retry.Do(ctx, connectEtcdFn, retry.Attempts(100))
	if err != nil {
		log.Warn("failed to initialize session",
			zap.Error(err))
		return nil
	}
	log.Debug("Session connect to etcd success")
	return session
}

type EtcdSessionRegister struct {
	Key string

	leaseID          *clientv3.LeaseID
	leaseKeepAliveCh <-chan *clientv3.LeaseKeepAliveResponse

	preemptive bool
	// combine
	keepAliveCtx *context.Context
	CancelFunc   *context.CancelFunc

	EventCh chan SessionEEvent

	//mockCh chan bool
}

func (e *EtcdSessionManager) NewSession(serverName, address string, exclusive bool, triggerKill bool, enableActiveStandBy bool) (*Session, error) {
	s := &Session{
		ServerName:          serverName,
		Address:             address,
		Exclusive:           exclusive,
		TriggerKill:         triggerKill,
		enableActiveStandBy: enableActiveStandBy,
	}
	e.checkIDExist()
	serverID, err := e.getServerID()
	if err != nil {
		return nil, err
	}
	s.ServerID = serverID
	log.Debug("create new session", zap.String("name", serverName), zap.String("address", address), zap.Int64("id", s.ServerID))
	return s, nil
}

// Register 会注册session到etcd中，返回一个EtcdSessionRegister，用于向外通知seesion注册状况。
// 设计这是一个幂等操作，多次注册会更新内部的leaseID和keepAliveChannel，但不会修改eventChan
// 要保证和Watch，UnRegister操作没用同步问题
func (e *EtcdSessionManager) Register(session *Session) (*EtcdSessionRegister, error) {

	isRetry := false
	// if not exist create one registry, if exists, use the old one, this means re-register happens
	if _, exist := e.registerSessions[session]; exist {
		isRetry = true
	} else {
		e.registerSessions[session] = &EtcdSessionRegister{
			Key:     e.getSessionKey(session),
			EventCh: make(chan SessionEEvent, 1),
			//mockCh:    make(chan bool, 1),
		}
	}
	register := e.registerSessions[session]

	registerFn := func() error {
		resp, err := e.etcdCli.Grant(e.ctx, e.sessionTTL)
		if err != nil {
			log.Error("register service", zap.Error(err))
			return err
		}
		register.leaseID = &resp.ID

		sessionJSON, err := json.Marshal(session)
		if err != nil {
			return err
		}

		if isRetry {
			txnResp, err := e.etcdCli.Txn(e.ctx).
				Then(clientv3.OpPut(register.Key, string(sessionJSON), clientv3.WithLease(resp.ID))).Commit()
			if err != nil {
				log.Warn("retry register session error", zap.String("Prefix", register.Key), zap.Error(err))
				return err
			}
			if !txnResp.Succeeded {
				return fmt.Errorf("retry register session error not succeed: %s", register.Key)
			}
		} else {
			txnResp, err := e.etcdCli.Txn(e.ctx).If(
				clientv3.Compare(
					clientv3.Version(register.Key),
					"=",
					0)).
				Then(clientv3.OpPut(register.Key, string(sessionJSON), clientv3.WithLease(resp.ID))).Commit()

			if err != nil {
				log.Warn("compare and swap error, maybe the Prefix has already been registered", zap.Error(err))
				return err
			}

			if !txnResp.Succeeded {
				return fmt.Errorf("function CompareAndSwap error for compare is false for Prefix: %s", session.ServerName)
			}
		}

		if isRetry {
			register.EventCh <- SessionEEvent{
				EventType: SessionEvent_Reregister,
			}
		} else {
			register.EventCh <- SessionEEvent{
				EventType: SessionEvent_ADD,
			}
		}
		log.Debug("Successfully put session Prefix into etcd",
			zap.String("Prefix", register.Key),
			zap.String("value", string(sessionJSON)),
			zap.Int64("leaseID", int64(*register.leaseID)))

		keepAliveCtx, cancelFunc := context.WithCancel(e.ctx)
		register.keepAliveCtx = &keepAliveCtx
		register.CancelFunc = &cancelFunc
		register.leaseKeepAliveCh, err = e.etcdCli.KeepAlive(keepAliveCtx, *register.leaseID)
		if err != nil {
			log.Warn("go error during keeping alive with etcd", zap.Error(err))
			return err
		}
		return nil
	}

	err := retry.Do(e.ctx, registerFn, retry.Attempts(uint(e.sessionRetryTimes)))
	if err != nil {
		log.Debug("Session registered failed", zap.Error(err))
		return nil, err
	}

	go func() {
		for {
			select {
			case <-e.ctx.Done():
				register.EventCh <- SessionEEvent{
					EventType: SessionEvent_Lost,
				}
				return
			case <-(*register.keepAliveCtx).Done():
				log.Warn("session keepalive ctx done, supposed to be killed by milvus inside logic")
				register.EventCh <- SessionEEvent{
					EventType: SessionEvent_CtxDone,
				}
				return
			case resp, ok := <-register.leaseKeepAliveCh:
				if !ok || resp == nil {
					if !ok {
						log.Warn("session keepalive channel closed")
					} else {
						log.Warn("session keepalive response failed")
					}
					register.EventCh <- SessionEEvent{
						EventType: SessionEvent_TemporaryLost,
					}
					// re-register
					_, err := e.Register(session)
					if err != nil {
						log.Error("re-register after keepalive channel close failed", zap.Error(err))
						register.EventCh <- SessionEEvent{
							EventType: SessionEvent_Lost,
						}
						return
					}
				}
				//case v := <-register.mockCh:
				//	if v {
				//		register.EventCh <- SessionEEvent{
				//			EventType: SessionEvent_TemporaryLost,
				//		}
				//		// re-register
				//		log.Info("start re-register the session")
				//		_, err := e.Register(session)
				//		if err != nil {
				//			log.Error("re-register after keepalive channel close failed", zap.Error(err))
				//			register.EventCh <- SessionEEvent{
				//				EventType: SessionEvent_Lost,
				//			}
				//		}
				//		return
				//	}
			}
		}
	}()

	log.Info("Session registered successfully",
		zap.String("key", register.Key),
		zap.String("ServerName", session.ServerName),
		zap.Int64("serverID", session.ServerID))
	return register, nil
}

func (e *EtcdSessionManager) Unregister(session *Session) (bool, error) {
	if register, exist := e.registerSessions[session]; exist {
		(*register.CancelFunc)()
		e.etcdCli.Revoke(e.ctx, *register.leaseID)
		_, _ = e.etcdCli.Delete(e.ctx, register.Key)
		delete(e.registerSessions, session)
		return true, nil
	}

	return false, errors.New(fmt.Sprintf("Registered session not found: %s", session.String()))
}

func (e EtcdSessionManager) Get(key string, isPrefix bool) (map[string]*Session, error) {
	res, _, err := e.getSessions(key, isPrefix)
	return res, err
}

func (e EtcdSessionManager) getSessions(key string, isPrefix bool) (map[string]*Session, int64, error) {
	res := make(map[string]*Session)
	completeKey := path.Join(e.metaRoot, DefaultServiceRoot, key)
	var resp *clientv3.GetResponse
	var err error
	if isPrefix {
		resp, err = e.etcdCli.Get(e.ctx, completeKey, clientv3.WithPrefix(),
			clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend))
	} else {
		resp, err = e.etcdCli.Get(e.ctx, completeKey,
			clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend))
	}
	if err != nil {
		return nil, 0, err
	}
	for _, kv := range resp.Kvs {
		session := &Session{}
		err = json.Unmarshal(kv.Value, session)
		if err != nil {
			return nil, 0, err
		}
		_, mapKey := path.Split(string(kv.Key))
		log.Debug("SessionUtil GetSessions ", zap.Any("Prefix", completeKey),
			zap.String("Prefix", mapKey),
			zap.Any("value", kv.Value))
		res[mapKey] = session
	}
	return res, resp.Header.Revision, nil
}

func (e *EtcdSessionManager) Watch(prefix string) (*EtcdSessionWatcher, error) {

	// if not exist create one registry, if exists, use the old one, this means re-register happens
	if _, exist := e.watchSessions[prefix]; !exist {
		e.watchSessions[prefix] = &EtcdSessionWatcher{
			Prefix:  prefix,
			EventCh: make(chan SessionEEvent, 1),
			//mockCh:    make(chan bool, 1),
		}
	}
	watcher := e.watchSessions[prefix]

	watchCtx, cancelFunc := context.WithCancel(e.ctx)
	watcher.cancelFunc = &cancelFunc
	watcher.watchCtx = &watchCtx
	watchCh := e.etcdCli.Watch(watchCtx, prefix, clientv3.WithPrefix(), clientv3.WithPrevKV())
	watcher.watchCh = watchCh

	go func() {
		for {
			select {
			case <-e.ctx.Done():
				watcher.EventCh <- SessionEEvent{
					EventType: SessionEvent_CtxDone,
				}
				return
			case <-(*watcher.watchCtx).Done():
				log.Warn("watcher ctx done, supposed to be killed by milvus inside logic")
				watcher.EventCh <- SessionEEvent{
					EventType: SessionEvent_CtxDone,
				}
				return
			case wresp, ok := <-watcher.watchCh:
				log.Debug("receive watch event")
				if !ok {
					log.Warn("session watch channel closed")
					return
				}
				e.handleWatchResponse(watcher, wresp)
			}
		}
	}()

	log.Info("Start watcher to monitor", zap.String("prefix", prefix))
	return watcher, nil
}

func (e *EtcdSessionManager) handleWatchResponse(watcher *EtcdSessionWatcher, wresp clientv3.WatchResponse) {
	if wresp.Err() != nil {
		err := e.handleWatchErr(watcher, wresp.Err())
		if err != nil {
			log.Error("failed to handle watch session response", zap.Error(err))
			panic(err)
		}
		return
	}
	for _, ev := range wresp.Events {
		session := &Session{}
		var eventType SessionEEventType
		switch ev.Type {
		case mvccpb.PUT:
			log.Debug("watch services",
				zap.Any("add kv", ev.Kv))
			err := json.Unmarshal(ev.Kv.Value, session)
			if err != nil {
				log.Error("watch services", zap.Error(err))
				continue
			}
			if session.Stopping {
				eventType = SessionEvent_Update
			} else {
				eventType = SessionEvent_ADD
			}
		case mvccpb.DELETE:
			log.Debug("watch services",
				zap.Any("delete kv", ev.PrevKv))
			err := json.Unmarshal(ev.PrevKv.Value, session)
			if err != nil {
				log.Error("watch services", zap.Error(err))
				continue
			}
			eventType = SessionEvent_Del
		}
		log.Debug("WatchService", zap.String("key", watcher.Prefix), zap.Any("event type", eventType))
		watcher.EventCh <- SessionEEvent{
			EventType: eventType,
		}
	}
}

func (e *EtcdSessionManager) handleWatchErr(watcher *EtcdSessionWatcher, err error) error {
	// if not ErrCompacted, just close the channel
	if err != v3rpc.ErrCompacted {
		//close event channel
		log.Warn("Watch service found error", zap.Error(err))
		close(watcher.EventCh)
		return err
	}

	_, revision, err := e.getSessions(watcher.Prefix, true)
	if err != nil {
		log.Warn("GetSession before rewatch failed", zap.String("Prefix", watcher.Prefix), zap.Error(err))
		close(watcher.EventCh)
		return err
	}

	watcher.watchCh = e.etcdCli.Watch(e.ctx, path.Join(e.metaRoot, DefaultServiceRoot, watcher.Prefix), clientv3.WithPrefix(), clientv3.WithPrevKV(), clientv3.WithRev(revision))
	return nil
}

type EtcdSessionWatcher struct {
	Prefix  string
	watchCh clientv3.WatchChan

	watchCtx   *context.Context
	cancelFunc *context.CancelFunc

	EventCh chan SessionEEvent
}

func (e *EtcdSessionManager) getServerID() (int64, error) {
	return e.getServerIDWithKey(DefaultIDKey)
}

func (s *EtcdSessionManager) checkIDExist() {
	s.etcdCli.Txn(s.ctx).If(
		clientv3.Compare(
			clientv3.Version(path.Join(s.metaRoot, DefaultServiceRoot, DefaultIDKey)),
			"=",
			0)).
		Then(clientv3.OpPut(path.Join(s.metaRoot, DefaultServiceRoot, DefaultIDKey), "1")).Commit()
}

func (e *EtcdSessionManager) getServerIDWithKey(key string) (int64, error) {
	for {
		getResp, err := e.etcdCli.Get(e.ctx, path.Join(e.metaRoot, DefaultServiceRoot, key))
		if err != nil {
			log.Warn("Session get etcd Key error", zap.String("Key", key), zap.Error(err))
			return -1, err
		}
		if getResp.Count <= 0 {
			log.Warn("Session there is no value", zap.String("Key", key))
			continue
		}
		value := string(getResp.Kvs[0].Value)
		valueInt, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			log.Warn("Session ParseInt error", zap.String("value", value), zap.Error(err))
			continue
		}
		txnResp, err := e.etcdCli.Txn(e.ctx).If(
			clientv3.Compare(
				clientv3.Value(path.Join(e.metaRoot, DefaultServiceRoot, key)),
				"=",
				value)).
			Then(clientv3.OpPut(path.Join(e.metaRoot, DefaultServiceRoot, key), strconv.FormatInt(valueInt+1, 10))).Commit()
		if err != nil {
			log.Warn("Session Txn failed", zap.String("Key", key), zap.Error(err))
			return -1, err
		}

		if !txnResp.Succeeded {
			log.Warn("Session Txn unsuccessful", zap.String("Key", key))
			continue
		}
		log.Debug("Session get serverID success", zap.String("Key", key), zap.Int64("ServerId", valueInt))
		return valueInt, nil
	}
}

// todo split session and etcd operations
func (e EtcdSessionManager) RegisterPreemptive(session *Session) error {
	watcher, err := e.Watch(e.getRoleKey(session))
	if err != nil {
		log.Error("fail to watch session in RegisterPreemptive")
		return err
	}

	_, err = e.Register(session)
	if err == nil {
		return nil
	}

	ticker := time.NewTicker(time.Second * 3)
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			ticker.Stop()
		case event, ok := <-watcher.EventCh:
			if !ok {
				log.Warn("session watch channel closed")
			}
			switch event.EventType {
			case SessionEvent_Lost:
			case SessionEvent_Del:
				_, err = e.Register(session)
				if err == nil {
					return nil
				}
			default:
				continue
			}
		case <-ticker.C:
			log.Debug("key is standing by", zap.String("key", watcher.Prefix))
		}
	}

	return nil
}
