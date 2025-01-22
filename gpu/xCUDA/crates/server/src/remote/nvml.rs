use code_genearte::nvml_driver_bindings_11_08 as nvml;

use api::nvm_api::Nvml as NvmlService;
use api::nvm_api::ServeNvml;
use tarpc::context;
// This is the type that implements the generated World trait. It is the business logic
// and is used to start the server.
#[derive(Clone)]
pub struct RemoteNvml;

pub fn NewRemoteNvmlServer() -> ServeNvml<RemoteNvml> {
    RemoteNvml.serve()
}

#[tarpc::server]
impl NvmlService for RemoteNvml {
    async fn nvmlInitV2(self, _: context::Context) -> u32 {
        println!("call server nvmlInitV2");
        unsafe { nvml::nvmlInit_v2() }
    }

    async fn nvmlDeviceGetCountV2(self, _: context::Context) -> (u32, u32) {
        println!("call server nvmlDeviceGetCountV2");
        let mut deviceCount = 0;
        let deviceCountPtr: *mut ::std::os::raw::c_uint = &mut deviceCount;
        unsafe {
            let res = nvml::nvmlDeviceGetCount_v2(deviceCountPtr);
            (*deviceCountPtr, res)
        }
    }

    async fn nvmlShutdown(self, _: context::Context) -> u32 {
        println!("call server nvmlShutdown");
        unsafe { nvml::nvmlShutdown() }
    }
}
