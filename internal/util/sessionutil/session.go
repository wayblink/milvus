package sessionutil

//type Session struct {
//	ServerID    int64  `json:"ServerID,omitempty"`
//	ServerName  string `json:"ServerName,omitempty"`
//	Address     string `json:"Address,omitempty"`
//	Exclusive   bool   `json:"Exclusive,omitempty"`
//	Stopping    bool   `json:"Stopping,omitempty"`
//	TriggerKill bool
//	Version     semver.Version `json:"Version,omitempty"`
//
//	enableActiveStandBy bool
//}

// String makes Session struct able to be logged by zap
//func (s *Session) String() string {
//	return fmt.Sprintf("Session:<ServerID: %d, ServerName: %s, Version: %s>", s.ServerID, s.ServerName, s.Version.String())
//}

type SessionEEvent struct {
	EventType SessionEventType
	Session   *Session
}

type SessionEEventType string

const (
	SessionEvent_ADD           SessionEEventType = "SessionEvent_ADD"
	SessionEvent_TemporaryLost SessionEEventType = "SessionEvent_TemporaryLost"
	SessionEvent_Lost          SessionEEventType = "SessionEvent_Lost"
	SessionEvent_Reregister    SessionEEventType = "SessionEvent_Reregister"
	SessionEvent_CtxDone       SessionEEventType = "SessionEvent_CtxDone"
	SessionEvent_Del           SessionEEventType = "SessionEvent_Del"
	// SessionUpdateEvent event type for a Session stopping
	SessionEvent_Update SessionEEventType = "SessionEvent_Update"
)

//
//type SessionManager interface {
//	NewSession(serverName, address string, exclusive bool, triggerKill bool, enableActiveStandBy bool) (*Session, error)
//
//	// Register session to service discovery, this register will keepalive.
//	// return:
//	// 		chan to inform caller the updates of this session,
//	// 		cancel function to stop the keepalive,
//	//      error if happens
//	Register(session *Session) (*EtcdSessionRegister, error)
//
//	// Unregister session from service discovery.
//	// return:
//	// 		bool show if this call succeed
//	//      error if happens
//	Unregister(session *Session) (bool, error)
//
//	// CompeteRegister is used when more than one session competes to register the same name.
//	// return:
//	// 		chan to inform caller the updates of this session,
//	// 		cancel function to stop the keepalive,
//	//      error if happens
//	CompeteRegister(session *Session) (*EtcdSessionRegister, error)
//
//	// Get registered services from service discovery
//	Get(Key string, isPrefix bool) (map[string]*Session, error)
//
//	// Watch the updates of a certain session
//	// return:
//	// 	 	chan to inform caller the updates of this session,
//	//		cancel function to stop the keepalive,
//	//      error if happens
//	Watch(Key string) (*EtcdSessionWatcher, error)
//}
