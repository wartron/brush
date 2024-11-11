use burn_wgpu::{AutoGraphicsApi, RuntimeOptions};
use std::path::PathBuf;
use anyhow::Result;


pub mod eval;
pub mod ssim;
pub mod train;

pub mod image;
pub mod scene;

pub async fn create_wgpu_setup() -> burn_wgpu::WgpuSetup {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(
        &burn_wgpu::WgpuDevice::DefaultDevice,
        RuntimeOptions {
            tasks_max: 64,
            memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        },
    )
    .await
}


#[derive(Debug)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    // Add other configuration options as needed
}

pub async fn train(
    input_dir: PathBuf,
    output_dir: PathBuf,
    config: TrainingConfig,
) -> Result<()> {
    // Implement your training logic here
    tracing::info!("Starting training with config: {:?}", config);
    tracing::info!("Input directory: {:?}", input_dir);
    tracing::info!("Output directory: {:?}", output_dir);

    // Your training implementation goes here
    
    Ok(())
}