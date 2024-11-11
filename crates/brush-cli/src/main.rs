use clap::Parser;
use anyhow::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input directory containing training data
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for trained model
    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    /// Number of epochs to train
    #[arg(short, long, default_value_t = 100)]
    epochs: u32,

    /// Batch size
    #[arg(short, long, default_value_t = 32)]
    batch_size: u32,

    /// Learning rate
    #[arg(short, long, default_value_t = 0.001)]
    learning_rate: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let cli = Cli::parse();

    // Validate input directory exists
    if !cli.input.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", cli.input);
    }

    // Create output directory if it doesn't exist
    if !cli.output.exists() {
        std::fs::create_dir_all(&cli.output)?;
    }

    // Initialize training configuration
    let config = brush_train::TrainingConfig {
        epochs: cli.epochs,
        batch_size: cli.batch_size,
        learning_rate: cli.learning_rate,
        // ... add other configuration options as needed
    };

    // Run training
    brush_train::train(
        cli.input,
        cli.output,
        config,
    ).await?;
    
    Ok(())
}