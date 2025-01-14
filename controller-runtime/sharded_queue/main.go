package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"hash/fnv"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/go-logr/logr"

	corev1 "k8s.io/api/core/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	controllerruntime "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

type CacheKey[T reconcile.Request] func(T reconcile.Request, shardCount int) int

type TypedShardedQueueConfig[T reconcile.Request] struct {
	// Name for the queue. If unnamed, the metrics will not be registered.
	Name string
	// shards is queue arrau
	shards []workqueue.TypedRateLimitingInterface[reconcile.Request]
	// CacheKey is get key hash value
	HashFn CacheKey[reconcile.Request]
	// CurrentShard is record current goroutineID assign shard
	CurrentShard int
	// GoroutineSharding is record goroutineID map to shard index
	GoroutineSharding map[int64]int
	//
	shardCount int
	// mu is lock
	mu sync.RWMutex
}

func NewShardedQueue(controllerName string, rateLimiter workqueue.TypedRateLimiter[reconcile.Request], shardCount int, fn CacheKey[reconcile.Request]) workqueue.TypedRateLimitingInterface[reconcile.Request] {
	shards := make([]workqueue.TypedRateLimitingInterface[reconcile.Request], shardCount)
	for i := 0; i < shardCount; i++ {
		rateLimitingQueue := workqueue.NewTypedRateLimitingQueueWithConfig(rateLimiter, workqueue.TypedRateLimitingQueueConfig[reconcile.Request]{
			Name: fmt.Sprintf("%s-shard-%d", controllerName, i),
		})
		shards[i] = rateLimitingQueue
	}
	return &TypedShardedQueueConfig[reconcile.Request]{
		Name:              controllerName,
		shards:            shards,
		shardCount:        shardCount,
		HashFn:            fn,
		GoroutineSharding: make(map[int64]int),
		mu:                sync.RWMutex{},
	}
}

func (sq *TypedShardedQueueConfig[T]) Len() int {
	goroutineID := GetGoroutineID()
	klog.V(5).InfoS("Len", "goroutineID", goroutineID)
	if shard, ok := sq.GoroutineSharding[goroutineID]; ok {
		return sq.shards[shard].Len()
	}

	var length int
	for _, shard := range sq.shards {
		length += shard.Len()
	}
	return length
}

func (sq *TypedShardedQueueConfig[T]) ShutDownWithDrain() {
	goroutineID := GetGoroutineID()
	klog.V(5).InfoS("ShutDownWithDrain", "goroutineID", goroutineID)
	if shard, ok := sq.GoroutineSharding[goroutineID]; ok {
		sq.shards[shard].ShutDownWithDrain()
		return
	}

	for _, shard := range sq.shards {
		shard.ShutDownWithDrain()
	}
}

func (sq *TypedShardedQueueConfig[T]) ShuttingDown() bool {
	goroutineID := GetGoroutineID()
	klog.V(5).InfoS("ShuttingDown", "goroutineID", goroutineID)
	if shard, ok := sq.GoroutineSharding[goroutineID]; ok {
		return sq.shards[shard].ShuttingDown()
	}

	for _, shard := range sq.shards {
		if !shard.ShuttingDown() {
			return false
		}
	}
	return true
}

func (sq *TypedShardedQueueConfig[T]) AddAfter(item T, duration time.Duration) {
	hashKey := sq.HashFn(reconcile.Request(item), sq.shardCount)
	klog.V(5).InfoS("AddAfter", "hashKey", hashKey)
	sq.shards[hashKey].AddAfter(reconcile.Request(item), duration)
}

func (sq *TypedShardedQueueConfig[T]) AddRateLimited(item T) {
	hashKey := sq.HashFn(reconcile.Request(item), sq.shardCount)
	klog.V(5).InfoS("AddRateLimited", "hashKey", hashKey)
	sq.shards[hashKey].AddRateLimited(reconcile.Request(item))
}

func (sq *TypedShardedQueueConfig[T]) Forget(item T) {
	hashKey := sq.HashFn(reconcile.Request(item), sq.shardCount)
	klog.V(5).InfoS("Forget", "hashKey", hashKey)
	sq.shards[hashKey].Forget(reconcile.Request(item))
}

func (sq *TypedShardedQueueConfig[T]) NumRequeues(item T) int {
	hashKey := sq.HashFn(reconcile.Request(item), sq.shardCount)
	klog.V(5).InfoS("NumRequeues", "hashKey", hashKey)
	return sq.shards[hashKey].NumRequeues(reconcile.Request(item))
}

func (sq *TypedShardedQueueConfig[T]) Add(item T) {
	hashKey := sq.HashFn(reconcile.Request(item), sq.shardCount)
	klog.V(5).InfoS("Add", "hashKey", hashKey)
	sq.shards[hashKey].Add(reconcile.Request(item))
}

func GetGoroutineID() int64 {
	// 获取当前 goroutine 的栈信息
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	stack := string(buf[:n])

	// 解析 goroutine ID
	// 栈信息的第一行格式为 "goroutine <id> [running]:"
	fields := bytes.Fields([]byte(stack))
	if len(fields) >= 2 {
		if id, err := strconv.ParseInt(string(fields[1]), 10, 64); err == nil {
			return id
		}
	}
	return -1
}

func (sq *TypedShardedQueueConfig[T]) Get() (T, bool) {
	goroutineID := GetGoroutineID()
	defer func() {
		klog.V(5).InfoS("Get", "goroutineID", goroutineID, "GoroutineSharding", sq.GoroutineSharding)
	}()
	if goroutineID == -1 {
		panic("goroutine ID is invelida")
	}
	var zero T
	sq.mu.RLock()
	if shads, ok := sq.GoroutineSharding[goroutineID]; ok {
		sq.mu.RUnlock()
		if item, shutdown := sq.shards[shads].Get(); !shutdown {
			return T(item), shutdown
		}
		return zero, true
	}
	sq.mu.RUnlock()

	sq.mu.Lock()
	sq.GoroutineSharding[goroutineID] = sq.CurrentShard
	sq.CurrentShard += 1
	sq.mu.Unlock()

	sq.mu.RLock()
	if shads, ok := sq.GoroutineSharding[goroutineID]; ok {
		sq.mu.RUnlock()
		if item, shutdown := sq.shards[shads].Get(); !shutdown {
			return T(item), shutdown
		}
		return zero, true
	}
	return zero, true
}

func (sq *TypedShardedQueueConfig[T]) Done(item T) {
	hashKey := sq.HashFn(reconcile.Request(item), sq.shardCount)
	klog.V(5).InfoS("Done", "hashKey", hashKey)
	sq.shards[hashKey].Done(reconcile.Request(item))
}

func (sq *TypedShardedQueueConfig[T]) ShutDown() {
	klog.V(5).InfoS("ShutDown")
	for _, shard := range sq.shards {
		shard.ShutDown()
	}
}

const (
	ControllerName       = "custom-controller"
	ConcurrentReconciles = 10
)

type CustomController struct {
	client.Client
	Log logr.Logger
}

func (r *CustomController) Reconcile(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
	klog.V(5).InfoS("Reconciling", "namespace", req.Namespace, "name", req.Name)

	return reconcile.Result{}, nil
}

func (r *CustomController) SetupWithManager(mgr manager.Manager) error {
	return controllerruntime.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		WithOptions(controller.TypedOptions[reconcile.Request]{MaxConcurrentReconciles: ConcurrentReconciles, NewQueue: r.QueueNew}).
		Named(ControllerName).
		Complete(r)
}

// NewController 创建一个控制器
func NewController(c client.Client) CustomController {
	return CustomController{
		Client: c,
	}
}

func (r *CustomController) Hash(value reconcile.Request, shardCount int) int {
	var pod corev1.Pod
	if err := r.Client.Get(context.Background(), client.ObjectKey{Namespace: value.Namespace, Name: value.Name}, &pod); err != nil {
		return 0
	}
	if pod.Spec.NodeName == "" {
		return 0
	}
	h := fnv.New32a()
	h.Write([]byte(pod.Spec.NodeName))
	if shardCount > 0 {
		return int(h.Sum32() % uint32(shardCount))
	}
	return 0
}

func (r *CustomController) QueueNew(controllerName string, rateLimiter workqueue.TypedRateLimiter[reconcile.Request]) workqueue.TypedRateLimitingInterface[reconcile.Request] {
	return NewShardedQueue(controllerName, rateLimiter, ConcurrentReconciles, r.Hash)
}

func main() {
	klog.InitFlags(flag.CommandLine)
	flag.Parse()
	log.SetLogger(zap.New(zap.UseDevMode(true)))

	aggregatedScheme := k8sruntime.NewScheme()
	utilruntime.Must(scheme.AddToScheme(aggregatedScheme))
	config := controllerruntime.GetConfigOrDie()
	controllerOptions := controllerruntime.Options{
		Scheme:         aggregatedScheme,
		LeaderElection: false, // opts.LeaderElection.LeaderElect,
	}
	controllerManager, err := controllerruntime.NewManager(config, controllerOptions)
	if err != nil {
		klog.ErrorS(err, "Failed to build controller manager")
		panic(err)
	}
	c := NewController(controllerManager.GetClient())
	if err = c.SetupWithManager(controllerManager); err != nil {
		panic(err)
	}
	// blocks until the context is done.
	if err = controllerManager.Start(context.Background()); err != nil {
		klog.ErrorS(err, "Starting zestu subcluster controller manager exits unexpectedly")
		panic(err)
	}
}
