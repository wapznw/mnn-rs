//! Basic inference example using MNN.
//!
//! This example demonstrates the basic workflow for running inference
//! with an MNN model.
//!
//! Usage:
//! ```bash
//! cargo run --example basic_inference -- /path/to/model.mnn
//! ```

use mnn::{prelude::*, BackendType, ScheduleConfig};

fn main() -> Result<(), MnnError> {
    // Get model path from command line
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: basic_inference <model.mnn>");

    println!("Loading model from: {}", model_path);

    // Create interpreter from file
    let interpreter = Interpreter::from_file(&model_path)?;
    println!("Model loaded successfully");

    // Print model info
    let biz_code = interpreter.business_code();
    if !biz_code.is_empty() {
        println!("Business code: {}", biz_code);
    }

    let uuid = interpreter.uuid();
    if !uuid.is_empty() {
        println!("UUID: {}", uuid);
    }

    // Create a session with CPU backend
    let config = ScheduleConfig::new()
        .backend(BackendType::CPU)
        .num_threads(4);

    let mut session = interpreter.create_session(config)?;
    println!("Session created");

    // Get the first input tensor
    let input = session.get_input(None)?;
    println!(
        "Input shape: {:?}, dtype: {}, format: {:?}",
        input.shape(),
        input.dtype(),
        input.format()
    );

    // Run inference
    println!("Running inference...");
    session.run()?;
    println!("Inference completed");

    // Get output
    let output = session.get_output(None)?;
    println!(
        "Output shape: {:?}, dtype: {}, format: {:?}",
        output.shape(),
        output.dtype(),
        output.format()
    );

    // Print session stats
    println!("Memory usage: {} bytes", session.memory_usage());
    println!("FLOPS: {} M", session.flops());

    Ok(())
}