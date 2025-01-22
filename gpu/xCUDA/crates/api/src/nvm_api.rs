use code_genearte::nvml_driver_bindings_11_08 as nvml;
/// This is the service definition. It looks a lot like a trait definition.
#[tarpc::service]
pub trait Nvml {
    async fn nvmlInitV2() -> u32;
    async fn nvmlDeviceGetCountV2() -> (u32, u32);
    async fn nvmlShutdown() -> u32;
}
