use api::cuda_runtime_api::CudaRuntime;
use code_genearte::cuda_runtime_bindings_11_08 as cuda_runtime;
use code_genearte::cuda_runtime_bindings_11_08::{cudaChannelFormatDesc, cudaError_t, cudaLimit};
use libc::res_init;
use tarpc::context;

include!(concat!(env!("OUT_DIR"), "/cuda_runtime_bindings.rs"));

#[derive(Clone)]
pub struct CudaRuntimeService;

#[tarpc::server]
impl CudaRuntime for CudaRuntimeService {
    async fn cudaDeviceReset(self, _: context::Context) -> cudaError_t {
        println!("call server cudaDeviceReset");
        unsafe { cuda_runtime::cudaDeviceReset() }
    }
    async fn cudaDeviceSynchronize(self, _: context::Context) -> cudaError_t {
        println!("call server cudaDeviceSynchronize");
        unsafe { cuda_runtime::cudaDeviceSynchronize() }
    }
    async fn cudaDeviceSetLimit(
        self,
        _: context::Context,
        limit: cudaLimit,
        value: usize,
    ) -> cudaError_t {
        println!("call server cudaDeviceSetLimit");
        unsafe { cuda_runtime::cudaDeviceSetLimit(limit, value) }
    }
    async fn cudaDeviceGetLimit(
        self,
        _: context::Context,
        limit: cudaLimit,
    ) -> (usize, cudaError_t) {
        println!("call server cudaDeviceGetLimit");
        let mut vaule: usize = 0;
        let pValue: *mut usize = &mut vaule;
        unsafe {
            let res = cuda_runtime::cudaDeviceGetLimit(pValue, limit);
            (*pValue, res)
        }
    }

    async fn cudaDeviceGetTexture1DLinearMaxWidth(
        self,
        _: context::Context,
        fmtDesc: *const cudaChannelFormatDesc,
        device: ::std::os::raw::c_int,
    ) -> (usize, cudaError_t) {
        println!("call server cudaDecudaDeviceGetTexture1DLinearMaxWidth");
        let mut maxWidthInElements: usize = 0;
        let maxWidthInElementsPtr: *mut usize = &mut maxWidthInElements;
        unsafe {
            let res = cuda_runtime::cudaDeviceGetTexture1DLinearMaxWidth(
                maxWidthInElementsPtr,
                fmtDesc,
                device,
            );
            (*maxWidthInElementsPtr, res)
        }
    }
}
