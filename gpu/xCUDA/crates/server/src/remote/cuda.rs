include!(concat!(env!("OUT_DIR"), "/cuda_driver_bindings.rs"));

use api::cuda_api;
use api::cuda_api::ServeCuda;
use cuda_api::Cuda as CudaService;
use std::ffi::CString;
use tarpc::context;

// This is the type that implements the generated World trait. It is the business logic
// and is used to start the server.
#[derive(Clone)]
pub struct RemoteCuda;

pub fn NewRemoteCudaServer() -> ServeCuda<RemoteCuda> {
    RemoteCuda.serve()
}

#[tarpc::server]
impl CudaService for RemoteCuda {
    async fn cuInit(self, _: context::Context, flags: u32) -> u32 {
        unsafe { cuInit(flags) }
    }
    async fn cuDeviceGet(self, _: context::Context, ordinal: i32) -> (i32, u32) {
        let mut dev: CUdevice = 0;
        let devPtr: *mut CUdevice = &mut dev;

        unsafe {
            let res = cuDeviceGet(devPtr, ordinal);
            (*devPtr, res)
        }
    }
    async fn cuDeviceGetCount(self, _: context::Context) -> (i32, u32) {
        let mut count: i32 = 0;
        let countPtr: *mut i32 = &mut count;

        unsafe {
            let res = cuDeviceGetCount(countPtr);

            (*countPtr, res)
        }
    }
    async fn cuDeviceGetName(self, _: context::Context, max_len: i32, dev: i32) -> (String, u32) {
        let mut name = [0; 1024];
        let mut len = max_len;
        if len > 1024 {
            len = 1024
        }

        unsafe {
            let res = cuDeviceGetName(name.as_mut_ptr(), len, dev);

            let strName = CString::from_raw(name.as_mut_ptr())
                .into_string()
                .expect("failed to convert name");

            (strName, res)
        }
    }
}
