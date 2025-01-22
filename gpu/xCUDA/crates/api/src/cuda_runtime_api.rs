// include!(concat!(env!("OUT_DIR"), "/cuda_runtime_types.rs"));

use ::std::os::raw::*;
use code_genearte::cuda_runtime_bindings_11_08 as cuda_runtime;
use code_genearte::cuda_runtime_bindings_11_08::cudaDeviceAttr;
use code_genearte::cuda_runtime_bindings_11_08::cudaError_t;
use code_genearte::cuda_runtime_bindings_11_08::cudaMemcpyKind;
use code_genearte::cuda_runtime_bindings_11_08::{
    cudaArrayMemoryRequirements, cudaArraySparseProperties, cudaArray_const_t, cudaArray_t,
    cudaChannelFormatDesc, cudaChannelFormatKind, cudaDeviceP2PAttr, cudaDeviceProp, cudaEvent_t,
    cudaExtent, cudaExternalMemoryBufferDesc, cudaExternalMemoryHandleDesc,
    cudaExternalMemoryMipmappedArrayDesc, cudaExternalMemory_t, cudaExternalSemaphoreHandleDesc,
    cudaExternalSemaphoreSignalNodeParams, cudaExternalSemaphoreSignalParams,
    cudaExternalSemaphoreWaitNodeParams, cudaExternalSemaphoreWaitParams, cudaExternalSemaphore_t,
    cudaFlushGPUDirectRDMAWritesScope, cudaFlushGPUDirectRDMAWritesTarget, cudaFuncAttribute,
    cudaFuncAttributes, cudaFuncCache, cudaFunction_t, cudaGraphExecUpdateResult, cudaGraphExec_t,
    cudaGraphMemAttributeType, cudaGraphNodeType, cudaGraphNode_t, cudaGraph_t,
    cudaGraphicsResource_t, cudaHostFn_t, cudaHostNodeParams, cudaIpcEventHandle_t,
    cudaIpcMemHandle_t, cudaKernelNodeParams, cudaLaunchAttributeID, cudaLaunchAttributeValue,
    cudaLaunchConfig_t, cudaLaunchParams, cudaLimit, cudaMemAccessDesc, cudaMemAccessFlags,
    cudaMemAllocNodeParams, cudaMemAllocationHandleType, cudaMemLocation, cudaMemPoolAttr,
    cudaMemPoolProps, cudaMemPoolPtrExportData, cudaMemPool_t, cudaMemRangeAttribute,
    cudaMemcpy3DParms, cudaMemcpy3DPeerParms, cudaMemoryAdvise, cudaMemsetParams,
    cudaMipmappedArray_const_t, cudaMipmappedArray_t, cudaPitchedPtr, cudaPointerAttributes,
    cudaResourceDesc, cudaResourceViewDesc, cudaSharedMemConfig, cudaStreamCallback_t,
    cudaStreamCaptureMode, cudaStreamCaptureStatus, cudaStream_t, cudaSurfaceObject_t,
    cudaTextureDesc, cudaTextureDesc_v2, cudaTextureObject_t, cudaUUID_t, cudaUserObject_t, dim3,
    surfaceReference, textureReference,
};

#[tarpc::service]
pub trait CudaRuntime {
    async fn cudaDeviceReset() -> cudaError_t;
    async fn cudaDeviceSynchronize() -> cudaError_t;
    async fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;
    async fn cudaDeviceGetLimit(limit: cudaLimit) -> (usize, cudaError_t);
    async fn cudaDeviceGetTexture1DLinearMaxWidth(
        fmtDesc: *const cudaChannelFormatDesc,
        device: ::std::os::raw::c_int,
    ) -> (usize, cudaError_t);

    // async fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
    // async fn cudaDeviceGetStreamPriorityRange(
    //     leastPriority: *mut ::std::os::raw::c_int,
    //     greatestPriority: *mut ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
    // async fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;
    //
    // async fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;
    //
    // async fn cudaDeviceGetByPCIBusId(
    //     device: *mut ::std::os::raw::c_int,
    //     pciBusId: *const ::std::os::raw::c_char,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGetPCIBusId(
    //     pciBusId: *mut ::std::os::raw::c_char,
    //     len: ::std::os::raw::c_int,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaIpcGetEventHandle(
    //     handle: *mut cudaIpcEventHandle_t,
    //     event: cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaIpcOpenEventHandle(
    //     event: *mut cudaEvent_t,
    //     handle: cudaIpcEventHandle_t,
    // ) -> cudaError_t;
    //
    // async fn cudaIpcGetMemHandle(
    //     handle: *mut cudaIpcMemHandle_t,
    //     devPtr: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaIpcOpenMemHandle(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     handle: cudaIpcMemHandle_t,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaIpcCloseMemHandle(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;
    //
    // async fn cudaDeviceFlushGPUDirectRDMAWrites(
    //     target: cudaFlushGPUDirectRDMAWritesTarget,
    //     scope: cudaFlushGPUDirectRDMAWritesScope,
    // ) -> cudaError_t;
    //
    // async fn cudaThreadExit() -> cudaError_t;
    //
    // async fn cudaThreadSynchronize() -> cudaError_t;
    //
    // async fn cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;
    //
    // async fn cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
    //
    // async fn cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
    //
    // async fn cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
    //
    // async fn cudaGetLastError() -> cudaError_t;
    //
    // async fn cudaPeekAtLastError() -> cudaError_t;
    //
    // async fn cudaGetErrorName(error: cudaError_t) -> *const ::std::os::raw::c_char;
    //
    // async fn cudaGetErrorString(error: cudaError_t) -> *const ::std::os::raw::c_char;
    //
    // async fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaGetDeviceProperties(
    //     prop: *mut cudaDeviceProp,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGetAttribute(
    //     value: *mut ::std::os::raw::c_int,
    //     attr: cudaDeviceAttr,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGetDefaultMemPool(
    //     memPool: *mut cudaMemPool_t,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceSetMemPool(
    //     device: ::std::os::raw::c_int,
    //     memPool: cudaMemPool_t,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGetMemPool(
    //     memPool: *mut cudaMemPool_t,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGetNvSciSyncAttributes(
    //     nvSciSyncAttrList: *mut ::std::os::raw::c_void,
    //     device: ::std::os::raw::c_int,
    //     flags: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGetP2PAttribute(
    //     value: *mut ::std::os::raw::c_int,
    //     attr: cudaDeviceP2PAttr,
    //     srcDevice: ::std::os::raw::c_int,
    //     dstDevice: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaChooseDevice(
    //     device: *mut ::std::os::raw::c_int,
    //     prop: *const cudaDeviceProp,
    // ) -> cudaError_t;
    //
    // async fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaSetValidDevices(
    //     device_arr: *mut ::std::os::raw::c_int,
    //     len: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaSetDeviceFlags(flags: ::std::os::raw::c_uint) -> cudaError_t;
    //
    // async fn cudaGetDeviceFlags(flags: *mut ::std::os::raw::c_uint) -> cudaError_t;
    //
    // async fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    //
    // async fn cudaStreamCreateWithFlags(
    //     pStream: *mut cudaStream_t,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamCreateWithPriority(
    //     pStream: *mut cudaStream_t,
    //     flags: ::std::os::raw::c_uint,
    //     priority: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamGetPriority(
    //     hStream: cudaStream_t,
    //     priority: *mut ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamGetFlags(
    //     hStream: cudaStream_t,
    //     flags: *mut ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaCtxResetPersistingL2Cache() -> cudaError_t;
    //
    // async fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaStreamGetAttribute(
    //     hStream: cudaStream_t,
    //     attr: cudaLaunchAttributeID,
    //     value_out: *mut cudaLaunchAttributeValue,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamSetAttribute(
    //     hStream: cudaStream_t,
    //     attr: cudaLaunchAttributeID,
    //     value: *const cudaLaunchAttributeValue,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaStreamWaitEvent(
    //     stream: cudaStream_t,
    //     event: cudaEvent_t,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamAddCallback(
    //     stream: cudaStream_t,
    //     callback: cudaStreamCallback_t,
    //     userData: *mut ::std::os::raw::c_void,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaStreamAttachMemAsync(
    //     stream: cudaStream_t,
    //     devPtr: *mut ::std::os::raw::c_void,
    //     length: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode)
    //                                 -> cudaError_t;
    //
    // async fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t;
    //
    // async fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
    //
    // async fn cudaStreamIsCapturing(
    //     stream: cudaStream_t,
    //     pCaptureStatus: *mut cudaStreamCaptureStatus,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamGetCaptureInfo(
    //     stream: cudaStream_t,
    //     pCaptureStatus: *mut cudaStreamCaptureStatus,
    //     pId: *mut ::std::os::raw::c_ulonglong,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamGetCaptureInfo_v2(
    //     stream: cudaStream_t,
    //     captureStatus_out: *mut cudaStreamCaptureStatus,
    //     id_out: *mut ::std::os::raw::c_ulonglong,
    //     graph_out: *mut cudaGraph_t,
    //     dependencies_out: *mut *const cudaGraphNode_t,
    //     numDependencies_out: *mut usize,
    // ) -> cudaError_t;
    //
    // async fn cudaStreamUpdateCaptureDependencies(
    //     stream: cudaStream_t,
    //     dependencies: *mut cudaGraphNode_t,
    //     numDependencies: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    //
    // async fn cudaEventCreateWithFlags(
    //     event: *mut cudaEvent_t,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaEventRecordWithFlags(
    //     event: cudaEvent_t,
    //     stream: cudaStream_t,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
    //
    // async fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    //
    // async fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    //
    // async fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
    //
    // async fn cudaImportExternalMemory(
    //     extMem_out: *mut cudaExternalMemory_t,
    //     memHandleDesc: *const cudaExternalMemoryHandleDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaExternalMemoryGetMappedBuffer(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     extMem: cudaExternalMemory_t,
    //     bufferDesc: *const cudaExternalMemoryBufferDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaExternalMemoryGetMappedMipmappedArray(
    //     mipmap: *mut cudaMipmappedArray_t,
    //     extMem: cudaExternalMemory_t,
    //     mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t;
    //
    // async fn cudaImportExternalSemaphore(
    //     extSem_out: *mut cudaExternalSemaphore_t,
    //     semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaSignalExternalSemaphoresAsync_v2(
    //     extSemArray: *const cudaExternalSemaphore_t,
    //     paramsArray: *const cudaExternalSemaphoreSignalParams,
    //     numExtSems: ::std::os::raw::c_uint,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaWaitExternalSemaphoresAsync_v2(
    //     extSemArray: *const cudaExternalSemaphore_t,
    //     paramsArray: *const cudaExternalSemaphoreWaitParams,
    //     numExtSems: ::std::os::raw::c_uint,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t;
    //
    // async fn cudaLaunchKernel(
    //     func: *const ::std::os::raw::c_void,
    //     gridDim: dim3,
    //     blockDim: dim3,
    //     args: *mut *mut ::std::os::raw::c_void,
    //     sharedMem: usize,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaLaunchKernelExC(
    //     config: *const cudaLaunchConfig_t,
    //     func: *const ::std::os::raw::c_void,
    //     args: *mut *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaLaunchCooperativeKernel(
    //     func: *const ::std::os::raw::c_void,
    //     gridDim: dim3,
    //     blockDim: dim3,
    //     args: *mut *mut ::std::os::raw::c_void,
    //     sharedMem: usize,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaLaunchCooperativeKernelMultiDevice(
    //     launchParamsList: *mut cudaLaunchParams,
    //     numDevices: ::std::os::raw::c_uint,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaFuncSetCacheConfig(
    //     func: *const ::std::os::raw::c_void,
    //     cacheConfig: cudaFuncCache,
    // ) -> cudaError_t;
    //
    // async fn cudaFuncSetSharedMemConfig(
    //     func: *const ::std::os::raw::c_void,
    //     config: cudaSharedMemConfig,
    // ) -> cudaError_t;
    //
    // async fn cudaFuncGetAttributes(
    //     attr: *mut cudaFuncAttributes,
    //     func: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaFuncSetAttribute(
    //     func: *const ::std::os::raw::c_void,
    //     attr: cudaFuncAttribute,
    //     value: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t;
    //
    // async fn cudaSetDoubleForHost(d: *mut f64) -> cudaError_t;
    //
    // async fn cudaLaunchHostFunc(
    //     stream: cudaStream_t,
    //     fn_: cudaHostFn_t,
    //     userData: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     numBlocks: *mut ::std::os::raw::c_int,
    //     func: *const ::std::os::raw::c_void,
    //     blockSize: ::std::os::raw::c_int,
    //     dynamicSMemSize: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaOccupancyAvailableDynamicSMemPerBlock(
    //     dynamicSmemSize: *mut usize,
    //     func: *const ::std::os::raw::c_void,
    //     numBlocks: ::std::os::raw::c_int,
    //     blockSize: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    //     numBlocks: *mut ::std::os::raw::c_int,
    //     func: *const ::std::os::raw::c_void,
    //     blockSize: ::std::os::raw::c_int,
    //     dynamicSMemSize: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaOccupancyMaxPotentialClusterSize(
    //     clusterSize: *mut ::std::os::raw::c_int,
    //     func: *const ::std::os::raw::c_void,
    //     launchConfig: *const cudaLaunchConfig_t,
    // ) -> cudaError_t;
    //
    // async fn cudaOccupancyMaxActiveClusters(
    //     numClusters: *mut ::std::os::raw::c_int,
    //     func: *const ::std::os::raw::c_void,
    //     launchConfig: *const cudaLaunchConfig_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMallocManaged(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     size: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaMalloc(devPtr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;
    //
    // async fn cudaMallocHost(ptr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;
    //
    // async fn cudaMallocPitch(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     pitch: *mut usize,
    //     width: usize,
    //     height: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMallocArray(
    //     array: *mut cudaArray_t,
    //     desc: *const cudaChannelFormatDesc,
    //     width: usize,
    //     height: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaFree(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;
    //
    // async fn cudaFreeHost(ptr: *mut ::std::os::raw::c_void) -> cudaError_t;
    //
    //
    // async fn cudaFreeArray(array: cudaArray_t) -> cudaError_t;
    //
    // async fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;
    //
    // async fn cudaHostAlloc(
    //     pHost: *mut *mut ::std::os::raw::c_void,
    //     size: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaHostRegister(
    //     ptr: *mut ::std::os::raw::c_void,
    //     size: usize,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaHostUnregister(ptr: *mut ::std::os::raw::c_void) -> cudaError_t;
    //
    // async fn cudaHostGetDevicePointer(
    //     pDevice: *mut *mut ::std::os::raw::c_void,
    //     pHost: *mut ::std::os::raw::c_void,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaHostGetFlags(
    //     pFlags: *mut ::std::os::raw::c_uint,
    //     pHost: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;
    //
    // async fn cudaMalloc3DArray(
    //     array: *mut cudaArray_t,
    //     desc: *const cudaChannelFormatDesc,
    //     extent: cudaExtent,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaMallocMipmappedArray(
    //     mipmappedArray: *mut cudaMipmappedArray_t,
    //     desc: *const cudaChannelFormatDesc,
    //     extent: cudaExtent,
    //     numLevels: ::std::os::raw::c_uint,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGetMipmappedArrayLevel(
    //     levelArray: *mut cudaArray_t,
    //     mipmappedArray: cudaMipmappedArray_const_t,
    //     level: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t;
    //
    // async fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t;
    //
    // async fn cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaMemcpy3DPeerAsync(
    //     p: *const cudaMemcpy3DPeerParms,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    //
    // async fn cudaArrayGetInfo(
    //     desc: *mut cudaChannelFormatDesc,
    //     extent: *mut cudaExtent,
    //     flags: *mut ::std::os::raw::c_uint,
    //     array: cudaArray_t,
    // ) -> cudaError_t;
    //
    // async fn cudaArrayGetPlane(
    //     pPlaneArray: *mut cudaArray_t,
    //     hArray: cudaArray_t,
    //     planeIdx: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaArrayGetMemoryRequirements(
    //     memoryRequirements: *mut cudaArrayMemoryRequirements,
    //     array: cudaArray_t,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaMipmappedArrayGetMemoryRequirements(
    //     memoryRequirements: *mut cudaArrayMemoryRequirements,
    //     mipmap: cudaMipmappedArray_t,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaArrayGetSparseProperties(
    //     sparseProperties: *mut cudaArraySparseProperties,
    //     array: cudaArray_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMipmappedArrayGetSparseProperties(
    //     sparseProperties: *mut cudaArraySparseProperties,
    //     mipmap: cudaMipmappedArray_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy(
    //     dst: *mut ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyPeer(
    //     dst: *mut ::std::os::raw::c_void,
    //     dstDevice: ::std::os::raw::c_int,
    //     src: *const ::std::os::raw::c_void,
    //     srcDevice: ::std::os::raw::c_int,
    //     count: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2D(
    //     dst: *mut ::std::os::raw::c_void,
    //     dpitch: usize,
    //     src: *const ::std::os::raw::c_void,
    //     spitch: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2DToArray(
    //     dst: cudaArray_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     src: *const ::std::os::raw::c_void,
    //     spitch: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2DFromArray(
    //     dst: *mut ::std::os::raw::c_void,
    //     dpitch: usize,
    //     src: cudaArray_const_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2DArrayToArray(
    //     dst: cudaArray_t,
    //     wOffsetDst: usize,
    //     hOffsetDst: usize,
    //     src: cudaArray_const_t,
    //     wOffsetSrc: usize,
    //     hOffsetSrc: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyToSymbol(
    //     symbol: *const ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyFromSymbol(
    //     dst: *mut ::std::os::raw::c_void,
    //     symbol: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyAsync(
    //     dst: *mut ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyPeerAsync(
    //     dst: *mut ::std::os::raw::c_void,
    //     dstDevice: ::std::os::raw::c_int,
    //     src: *const ::std::os::raw::c_void,
    //     srcDevice: ::std::os::raw::c_int,
    //     count: usize,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2DAsync(
    //     dst: *mut ::std::os::raw::c_void,
    //     dpitch: usize,
    //     src: *const ::std::os::raw::c_void,
    //     spitch: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2DToArrayAsync(
    //     dst: cudaArray_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     src: *const ::std::os::raw::c_void,
    //     spitch: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpy2DFromArrayAsync(
    //     dst: *mut ::std::os::raw::c_void,
    //     dpitch: usize,
    //     src: cudaArray_const_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     width: usize,
    //     height: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyToSymbolAsync(
    //     symbol: *const ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyFromSymbolAsync(
    //     dst: *mut ::std::os::raw::c_void,
    //     symbol: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemset(
    //     devPtr: *mut ::std::os::raw::c_void,
    //     value: ::std::os::raw::c_int,
    //     count: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMemset2D(
    //     devPtr: *mut ::std::os::raw::c_void,
    //     pitch: usize,
    //     value: ::std::os::raw::c_int,
    //     width: usize,
    //     height: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMemset3D(
    //     pitchedDevPtr: cudaPitchedPtr,
    //     value: ::std::os::raw::c_int,
    //     extent: cudaExtent,
    // ) -> cudaError_t;
    //
    // async fn cudaMemsetAsync(
    //     devPtr: *mut ::std::os::raw::c_void,
    //     value: ::std::os::raw::c_int,
    //     count: usize,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemset2DAsync(
    //     devPtr: *mut ::std::os::raw::c_void,
    //     pitch: usize,
    //     value: ::std::os::raw::c_int,
    //     width: usize,
    //     height: usize,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemset3DAsync(
    //     pitchedDevPtr: cudaPitchedPtr,
    //     value: ::std::os::raw::c_int,
    //     extent: cudaExtent,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGetSymbolAddress(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     symbol: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaGetSymbolSize(
    //     size: *mut usize,
    //     symbol: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPrefetchAsync(
    //     devPtr: *const ::std::os::raw::c_void,
    //     count: usize,
    //     dstDevice: ::std::os::raw::c_int,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemAdvise(
    //     devPtr: *const ::std::os::raw::c_void,
    //     count: usize,
    //     advice: cudaMemoryAdvise,
    //     device: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    //
    // async fn cudaMemRangeGetAttribute(
    //     data: *mut ::std::os::raw::c_void,
    //     dataSize: usize,
    //     attribute: cudaMemRangeAttribute,
    //     devPtr: *const ::std::os::raw::c_void,
    //     count: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMemRangeGetAttributes(
    //     data: *mut *mut ::std::os::raw::c_void,
    //     dataSizes: *mut usize,
    //     attributes: *mut cudaMemRangeAttribute,
    //     numAttributes: usize,
    //     devPtr: *const ::std::os::raw::c_void,
    //     count: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyToArray(
    //     dst: cudaArray_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyFromArray(
    //     dst: *mut ::std::os::raw::c_void,
    //     src: cudaArray_const_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyArrayToArray(
    //     dst: cudaArray_t,
    //     wOffsetDst: usize,
    //     hOffsetDst: usize,
    //     src: cudaArray_const_t,
    //     wOffsetSrc: usize,
    //     hOffsetSrc: usize,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyToArrayAsync(
    //     dst: cudaArray_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemcpyFromArrayAsync(
    //     dst: *mut ::std::os::raw::c_void,
    //     src: cudaArray_const_t,
    //     wOffset: usize,
    //     hOffset: usize,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMallocAsync(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     size: usize,
    //     hStream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaFreeAsync(devPtr: *mut ::std::os::raw::c_void, hStream: cudaStream_t)
    //                        -> cudaError_t;
    //
    // async fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t;
    //
    //
    // async fn cudaMemPoolSetAttribute(
    //     memPool: cudaMemPool_t,
    //     attr: cudaMemPoolAttr,
    //     value: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolGetAttribute(
    //     memPool: cudaMemPool_t,
    //     attr: cudaMemPoolAttr,
    //     value: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolSetAccess(
    //     memPool: cudaMemPool_t,
    //     descList: *const cudaMemAccessDesc,
    //     count: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolGetAccess(
    //     flags: *mut cudaMemAccessFlags,
    //     memPool: cudaMemPool_t,
    //     location: *mut cudaMemLocation,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolCreate(
    //     memPool: *mut cudaMemPool_t,
    //     poolProps: *const cudaMemPoolProps,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t;
    //
    // async fn cudaMallocFromPoolAsync(
    //     ptr: *mut *mut ::std::os::raw::c_void,
    //     size: usize,
    //     memPool: cudaMemPool_t,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolExportToShareableHandle(
    //     shareableHandle: *mut ::std::os::raw::c_void,
    //     memPool: cudaMemPool_t,
    //     handleType: cudaMemAllocationHandleType,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolImportFromShareableHandle(
    //     memPool: *mut cudaMemPool_t,
    //     shareableHandle: *mut ::std::os::raw::c_void,
    //     handleType: cudaMemAllocationHandleType,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolExportPointer(
    //     exportData: *mut cudaMemPoolPtrExportData,
    //     ptr: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaMemPoolImportPointer(
    //     ptr: *mut *mut ::std::os::raw::c_void,
    //     memPool: cudaMemPool_t,
    //     exportData: *mut cudaMemPoolPtrExportData,
    // ) -> cudaError_t;
    //
    // async fn cudaPointerGetAttributes(
    //     attributes: *mut cudaPointerAttributes,
    //     ptr: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceCanAccessPeer(
    //     canAccessPeer: *mut ::std::os::raw::c_int,
    //     device: ::std::os::raw::c_int,
    //     peerDevice: ::std::os::raw::c_int,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceEnablePeerAccess(
    //     peerDevice: ::std::os::raw::c_int,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceDisablePeerAccess(peerDevice: ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t;
    //
    // async fn cudaGraphicsResourceSetMapFlags(
    //     resource: cudaGraphicsResource_t,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphicsMapResources(
    //     count: ::std::os::raw::c_int,
    //     resources: *mut cudaGraphicsResource_t,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphicsUnmapResources(
    //     count: ::std::os::raw::c_int,
    //     resources: *mut cudaGraphicsResource_t,
    //     stream: cudaStream_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphicsResourceGetMappedPointer(
    //     devPtr: *mut *mut ::std::os::raw::c_void,
    //     size: *mut usize,
    //     resource: cudaGraphicsResource_t,
    // ) -> cudaError_t;
    //
    //
    // async fn cudaGraphicsSubResourceGetMappedArray(
    //     array: *mut cudaArray_t,
    //     resource: cudaGraphicsResource_t,
    //     arrayIndex: ::std::os::raw::c_uint,
    //     mipLevel: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphicsResourceGetMappedMipmappedArray(
    //     mipmappedArray: *mut cudaMipmappedArray_t,
    //     resource: cudaGraphicsResource_t,
    // ) -> cudaError_t;
    //
    // async fn cudaBindTexture(
    //     offset: *mut usize,
    //     texref: *const textureReference,
    //     devPtr: *const ::std::os::raw::c_void,
    //     desc: *const cudaChannelFormatDesc,
    //     size: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaBindTexture2D(
    //     offset: *mut usize,
    //     texref: *const textureReference,
    //     devPtr: *const ::std::os::raw::c_void,
    //     desc: *const cudaChannelFormatDesc,
    //     width: usize,
    //     height: usize,
    //     pitch: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaBindTextureToArray(
    //     texref: *const textureReference,
    //     array: cudaArray_const_t,
    //     desc: *const cudaChannelFormatDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaBindTextureToMipmappedArray(
    //     texref: *const textureReference,
    //     mipmappedArray: cudaMipmappedArray_const_t,
    //     desc: *const cudaChannelFormatDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaUnbindTexture(texref: *const textureReference) -> cudaError_t;
    //
    // async fn cudaGetTextureAlignmentOffset(
    //     offset: *mut usize,
    //     texref: *const textureReference,
    // ) -> cudaError_t;
    //
    // async fn cudaGetTextureReference(
    //     texref: *mut *const textureReference,
    //     symbol: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaBindSurfaceToArray(
    //     surfref: *const surfaceReference,
    //     array: cudaArray_const_t,
    //     desc: *const cudaChannelFormatDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaGetSurfaceReference(
    //     surfref: *mut *const surfaceReference,
    //     symbol: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaGetChannelDesc(
    //     desc: *mut cudaChannelFormatDesc,
    //     array: cudaArray_const_t,
    // ) -> cudaError_t;
    //
    //
    // async fn cudaCreateChannelDesc(
    //     x: ::std::os::raw::c_int,
    //     y: ::std::os::raw::c_int,
    //     z: ::std::os::raw::c_int,
    //     w: ::std::os::raw::c_int,
    //     f: cudaChannelFormatKind,
    // ) -> cudaChannelFormatDesc;
    //
    // async fn cudaCreateTextureObject(
    //     pTexObject: *mut cudaTextureObject_t,
    //     pResDesc: *const cudaResourceDesc,
    //     pTexDesc: *const cudaTextureDesc,
    //     pResViewDesc: *const cudaResourceViewDesc,
    // ) -> cudaError_t;
    //
    //
    // async fn cudaCreateTextureObject_v2(
    //     pTexObject: *mut cudaTextureObject_t,
    //     pResDesc: *const cudaResourceDesc,
    //     pTexDesc: *const cudaTextureDesc_v2,
    //     pResViewDesc: *const cudaResourceViewDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t;
    //
    //
    // async fn cudaGetTextureObjectResourceDesc(
    //     pResDesc: *mut cudaResourceDesc,
    //     texObject: cudaTextureObject_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGetTextureObjectTextureDesc(
    //     pTexDesc: *mut cudaTextureDesc,
    //     texObject: cudaTextureObject_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGetTextureObjectTextureDesc_v2(
    //     pTexDesc: *mut cudaTextureDesc_v2,
    //     texObject: cudaTextureObject_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGetTextureObjectResourceViewDesc(
    //     pResViewDesc: *mut cudaResourceViewDesc,
    //     texObject: cudaTextureObject_t,
    // ) -> cudaError_t;
    //
    // async fn cudaCreateSurfaceObject(
    //     pSurfObject: *mut cudaSurfaceObject_t,
    //     pResDesc: *const cudaResourceDesc,
    // ) -> cudaError_t;
    //
    // async fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t;
    //
    // async fn cudaGetSurfaceObjectResourceDesc(
    //     pResDesc: *mut cudaResourceDesc,
    //     surfObject: cudaSurfaceObject_t,
    // ) -> cudaError_t;
    //
    // async fn cudaDriverGetVersion(driverVersion: *mut ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaRuntimeGetVersion(runtimeVersion: *mut ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: ::std::os::raw::c_uint) -> cudaError_t;
    //
    // async fn cudaGraphAddKernelNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     pNodeParams: *const cudaKernelNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphKernelNodeGetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *mut cudaKernelNodeParams,
    // ) -> cudaError_t;
    //
    //
    // async fn cudaGraphKernelNodeSetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaKernelNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphKernelNodeCopyAttributes(
    //     hSrc: cudaGraphNode_t,
    //     hDst: cudaGraphNode_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphKernelNodeGetAttribute(
    //     hNode: cudaGraphNode_t,
    //     attr: cudaLaunchAttributeID,
    //     value_out: *mut cudaLaunchAttributeValue,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphKernelNodeSetAttribute(
    //     hNode: cudaGraphNode_t,
    //     attr: cudaLaunchAttributeID,
    //     value: *const cudaLaunchAttributeValue,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddMemcpyNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     pCopyParams: *const cudaMemcpy3DParms,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddMemcpyNodeToSymbol(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     symbol: *const ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddMemcpyNodeFromSymbol(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     dst: *mut ::std::os::raw::c_void,
    //     symbol: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddMemcpyNode1D(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     dst: *mut ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemcpyNodeGetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *mut cudaMemcpy3DParms,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemcpyNodeSetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaMemcpy3DParms,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemcpyNodeSetParamsToSymbol(
    //     node: cudaGraphNode_t,
    //     symbol: *const ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemcpyNodeSetParamsFromSymbol(
    //     node: cudaGraphNode_t,
    //     dst: *mut ::std::os::raw::c_void,
    //     symbol: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemcpyNodeSetParams1D(
    //     node: cudaGraphNode_t,
    //     dst: *mut ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async  fn cudaGraphAddMemsetNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     pMemsetParams: *const cudaMemsetParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemsetNodeGetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *mut cudaMemsetParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemsetNodeSetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaMemsetParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddHostNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     pNodeParams: *const cudaHostNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphHostNodeGetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *mut cudaHostNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphHostNodeSetParams(
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaHostNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddChildGraphNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     childGraph: cudaGraph_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphChildGraphNodeGetGraph(
    //     node: cudaGraphNode_t,
    //     pGraph: *mut cudaGraph_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddEmptyNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddEventRecordNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     event: cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphEventRecordNodeGetEvent(
    //     node: cudaGraphNode_t,
    //     event_out: *mut cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphEventRecordNodeSetEvent(
    //     node: cudaGraphNode_t,
    //     event: cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddEventWaitNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     event: cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphEventWaitNodeGetEvent(
    //     node: cudaGraphNode_t,
    //     event_out: *mut cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t)
    //                                         -> cudaError_t;
    //
    // async fn cudaGraphAddExternalSemaphoresSignalNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExternalSemaphoresSignalNodeGetParams(
    //     hNode: cudaGraphNode_t,
    //     params_out: *mut cudaExternalSemaphoreSignalNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExternalSemaphoresSignalNodeSetParams(
    //     hNode: cudaGraphNode_t,
    //     nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddExternalSemaphoresWaitNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExternalSemaphoresWaitNodeGetParams(
    //     hNode: cudaGraphNode_t,
    //     params_out: *mut cudaExternalSemaphoreWaitNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExternalSemaphoresWaitNodeSetParams(
    //     hNode: cudaGraphNode_t,
    //     nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddMemAllocNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     nodeParams: *mut cudaMemAllocNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemAllocNodeGetParams(
    //     node: cudaGraphNode_t,
    //     params_out: *mut cudaMemAllocNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddMemFreeNode(
    //     pGraphNode: *mut cudaGraphNode_t,
    //     graph: cudaGraph_t,
    //     pDependencies: *const cudaGraphNode_t,
    //     numDependencies: usize,
    //     dptr: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphMemFreeNodeGetParams(
    //     node: cudaGraphNode_t,
    //     dptr_out: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceGraphMemTrim(device: ::std::os::raw::c_int) -> cudaError_t;
    //
    // async fn cudaDeviceGetGraphMemAttribute(
    //     device: ::std::os::raw::c_int,
    //     attr: cudaGraphMemAttributeType,
    //     value: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaDeviceSetGraphMemAttribute(
    //     device: ::std::os::raw::c_int,
    //     attr: cudaGraphMemAttributeType,
    //     value: *mut ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphClone(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t)
    //                         -> cudaError_t;
    //
    // async fn cudaGraphNodeFindInClone(
    //     pNode: *mut cudaGraphNode_t,
    //     originalNode: cudaGraphNode_t,
    //     clonedGraph: cudaGraph_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphNodeGetType(
    //     node: cudaGraphNode_t,
    //     pType: *mut cudaGraphNodeType,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphGetNodes(
    //     graph: cudaGraph_t,
    //     nodes: *mut cudaGraphNode_t,
    //     numNodes: *mut usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphGetRootNodes(
    //     graph: cudaGraph_t,
    //     pRootNodes: *mut cudaGraphNode_t,
    //     pNumRootNodes: *mut usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphGetEdges(
    //     graph: cudaGraph_t,
    //     from: *mut cudaGraphNode_t,
    //     to: *mut cudaGraphNode_t,
    //     numEdges: *mut usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphNodeGetDependencies(
    //     node: cudaGraphNode_t,
    //     pDependencies: *mut cudaGraphNode_t,
    //     pNumDependencies: *mut usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphNodeGetDependentNodes(
    //     node: cudaGraphNode_t,
    //     pDependentNodes: *mut cudaGraphNode_t,
    //     pNumDependentNodes: *mut usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphAddDependencies(
    //     graph: cudaGraph_t,
    //     from: *const cudaGraphNode_t,
    //     to: *const cudaGraphNode_t,
    //     numDependencies: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphRemoveDependencies(
    //     graph: cudaGraph_t,
    //     from: *const cudaGraphNode_t,
    //     to: *const cudaGraphNode_t,
    //     numDependencies: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t;
    //
    // async fn cudaGraphInstantiate(
    //     pGraphExec: *mut cudaGraphExec_t,
    //     graph: cudaGraph_t,
    //     pErrorNode: *mut cudaGraphNode_t,
    //     pLogBuffer: *mut ::std::os::raw::c_char,
    //     bufferSize: usize,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphInstantiateWithFlags(
    //     pGraphExec: *mut cudaGraphExec_t,
    //     graph: cudaGraph_t,
    //     flags: ::std::os::raw::c_ulonglong,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecKernelNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaKernelNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecMemcpyNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaMemcpy3DParms,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecMemcpyNodeSetParamsToSymbol(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     symbol: *const ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     dst: *mut ::std::os::raw::c_void,
    //     symbol: *const ::std::os::raw::c_void,
    //     count: usize,
    //     offset: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecMemcpyNodeSetParams1D(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     dst: *mut ::std::os::raw::c_void,
    //     src: *const ::std::os::raw::c_void,
    //     count: usize,
    //     kind: cudaMemcpyKind,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecMemsetNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaMemsetParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecHostNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     pNodeParams: *const cudaHostNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecChildGraphNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     node: cudaGraphNode_t,
    //     childGraph: cudaGraph_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecEventRecordNodeSetEvent(
    //     hGraphExec: cudaGraphExec_t,
    //     hNode: cudaGraphNode_t,
    //     event: cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecEventWaitNodeSetEvent(
    //     hGraphExec: cudaGraphExec_t,
    //     hNode: cudaGraphNode_t,
    //     event: cudaEvent_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     hNode: cudaGraphNode_t,
    //     nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(
    //     hGraphExec: cudaGraphExec_t,
    //     hNode: cudaGraphNode_t,
    //     nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphNodeSetEnabled(
    //     hGraphExec: cudaGraphExec_t,
    //     hNode: cudaGraphNode_t,
    //     isEnabled: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphNodeGetEnabled(
    //     hGraphExec: cudaGraphExec_t,
    //     hNode: cudaGraphNode_t,
    //     isEnabled: *mut ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphExecUpdate(
    //     hGraphExec: cudaGraphExec_t,
    //     hGraph: cudaGraph_t,
    //     hErrorNode_out: *mut cudaGraphNode_t,
    //     updateResult_out: *mut cudaGraphExecUpdateResult,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    //
    // async fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;
    //
    // async fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    //
    // async fn cudaGraphDebugDotPrint(
    //     graph: cudaGraph_t,
    //     path: *const ::std::os::raw::c_char,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaUserObjectCreate(
    //     object_out: *mut cudaUserObject_t,
    //     ptr: *mut ::std::os::raw::c_void,
    //     destroy: cudaHostFn_t,
    //     initialRefcount: ::std::os::raw::c_uint,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaUserObjectRetain(
    //     object: cudaUserObject_t,
    //     count: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaUserObjectRelease(
    //     object: cudaUserObject_t,
    //     count: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphRetainUserObject(
    //     graph: cudaGraph_t,
    //     object: cudaUserObject_t,
    //     count: ::std::os::raw::c_uint,
    //     flags: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGraphReleaseUserObject(
    //     graph: cudaGraph_t,
    //     object: cudaUserObject_t,
    //     count: ::std::os::raw::c_uint,
    // ) -> cudaError_t;
    //
    // async fn cudaGetDriverEntryPoint(
    //     symbol: *const ::std::os::raw::c_char,
    //     funcPtr: *mut *mut ::std::os::raw::c_void,
    //     flags: ::std::os::raw::c_ulonglong,
    // ) -> cudaError_t;
    //
    // async fn cudaGetExportTable(
    //     ppExportTable: *mut *const ::std::os::raw::c_void,
    //     pExportTableId: *const cudaUUID_t,
    // ) -> cudaError_t;
    //
    // async fn cudaGetFuncBySymbol(
    //     functionPtr: *mut cudaFunction_t,
    //     symbolPtr: *const ::std::os::raw::c_void,
    // ) -> cudaError_t;
    //
    // async fn cudaGetLastError() -> cudaError_t;
    // async fn cudaPeekAtLastError() -> cudaError_t;
    // async fn cudaGetErrorName(error: cudaError_t) -> String;
    // async fn cudaGetErrorString(error: cudaError_t) -> String;
    // async fn cudaGetDeviceCount() -> (i32, cudaError_t);
    // async fn cudaSetDevice(device: c_int) -> cudaError_t;
    // async fn cudaDeviceGetAttribute(attr: cudaDeviceAttr, device: c_int) -> (c_int, cudaError_t);
    // async fn cudaMalloc(size: usize) -> (usize, cudaError_t);
    // async fn cudaFree(devPtr: usize) -> cudaError_t;
    // async fn cudaMemset(devPtr: usize, value: c_int, count: usize) -> cudaError_t;
    // async fn cudaMemcpy(dst: usize, src: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
}
