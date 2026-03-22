//! Async inference example using MNN with tokio.
//!
//! This example demonstrates async inference with MNN.
//!
//! Usage:
//! ```bash
//! cargo run --example async_inference --features async -- /path/to/model.mnn
//! ```

#[cfg(feature = "async")]
use mnn::{prelude::*, AsyncInterpreter, ScheduleConfig};

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<(), MnnError> {
    // Get model path from command line
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: async_inference <model.mnn>");

    println!("Loading model asynchronously from: {}", model_path);

    // Create async interpreter
    let interpreter = AsyncInterpreter::from_file(&model_path).await?;
    println!("Model loaded successfully");

    // Create a session
    let config = ScheduleConfig::default();
    let mut session = interpreter.create_session(config).await?;
    println!("Session created");

    // Run inference asynchronously
    println!("Running async inference...");
    session.run_async().await?;
    println!("Inference completed");

    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires the 'async' feature.");
    eprintln!("Run with: cargo run --example async_inference --features async");
}