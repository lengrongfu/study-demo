use code_genearte::cuda_driver_bindings_11_08 as cuda_driver;

/// This is the service definition. It looks a lot like a trait definition.
#[tarpc::service]
pub trait Cuda {
    async fn cuInit(flags: u32) -> u32;
    async fn cuDeviceGet(ordinal: i32) -> (i32, u32);
    async fn cuDeviceGetCount() -> (i32, u32);
    async fn cuDeviceGetName(maxLen: i32, dev: i32) -> (String, u32);
    async fn cuGetErrorName(error: cuda_driver::CUresult) -> (String, cuda_driver::CUresult);
    async fn cuGetErrorString(error: cuda_driver::CUresult) -> (String, cuda_driver::CUresult);
    async fn cuDeviceTotalMem_v2(dev: cuda_driver::CUdevice) -> (usize, cuda_driver::CUresult);
    async fn cuMemAlloc_v2(size: usize) -> (cuda_driver::CUdeviceptr, cuda_driver::CUresult);
    async fn cuMemFree_v2(devPtr: cuda_driver::CUdeviceptr) -> cuda_driver::CUresult;
    async fn cuMemcpyDtoH_v2(
        src: cuda_driver::CUdeviceptr,
        size: usize,
    ) -> (Vec<u8>, cuda_driver::CUresult);
    async fn cuMemcpyHtoD_v2(
        dst: cuda_driver::CUdeviceptr,
        data: Vec<u8>,
        size: usize,
    ) -> cuda_driver::CUresult;
    // async fn cuModuleLoadData(data: Vec::<u8>) -> (usize, cuda_driver::cudaError_t);
    // async fn cuModuleLoadFatBinary(data: Vec::<u8>) -> (usize, cuda_driver::cudaError_t);
    // async fn cuModuleUnload(hmod: usize) -> cuda_driver::cudaError_t;
}
