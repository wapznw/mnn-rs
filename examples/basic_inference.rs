//! Basic inference example using MNN.
//!
//! This example demonstrates the basic workflow for running inference
//! with an MNN model, including:
//! - Loading a model
//! - Creating a session
//! - Preparing input data
//! - Running inference
//! - Reading output results
//!
//! Usage:
//! ```bash
//! cargo run --example basic_inference -- /path/to/model.mnn
//! ```

use mnn_rs::{prelude::*, BackendType, ScheduleConfig};

fn main() -> Result<(), MnnError> {
    // Get model path from command line
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: basic_inference <model.mnn>");

    println!("=== MNN Rust Basic Inference Example ===\n");

    // ===== Step 1: Load Model =====
    println!("[Step 1] Loading model from: {}", model_path);
    let interpreter = Interpreter::from_file(&model_path)?;
    println!("  Model loaded successfully");

    // Print model info
    let biz_code = interpreter.business_code();
    if !biz_code.is_empty() {
        println!("  Business code: {}", biz_code);
    }
    let uuid = interpreter.uuid();
    if !uuid.is_empty() {
        println!("  UUID: {}", uuid);
    }

    // ===== Step 2: Create Session =====
    println!("\n[Step 2] Creating session");
    let config = ScheduleConfig::new()
        .backend(BackendType::CPU)
        .num_threads(4);

    let mut session = interpreter.create_session(config)?;
    println!("  Session created with CPU backend, 4 threads");

    // ===== Step 3: Get Input Tensor =====
    println!("\n[Step 3] Getting input tensor");
    let input = session.get_input(None)?;
    let input_shape = input.shape();
    println!("  Input shape: {:?}", input_shape);
    println!("  Input dtype: {}", input.dtype());
    println!("  Input format: {:?}", input.format());
    println!("  Input element count: {}", input.element_count());

    // ===== Step 4: Prepare Input Data =====
    println!("\n[Step 4] Preparing input data");

    // Calculate input size
    let input_size = input.element_count() as usize;
    println!("  Input size: {} elements", input_size);

    // Create input data (all zeros for simplicity, or random values)
    // Using a pattern that's easy to verify: incremental values normalized to [0, 1]
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| (i % 256) as f32 / 255.0)
        .collect();

    println!("  Generated {} input values", input_data.len());
    println!("  Sample values: {:?}", &input_data[..10.min(input_data.len())]);

    // Write data to input tensor
    input.write(&input_data)?;
    println!("  Input data written to tensor");

    // ===== Step 5: Run Inference =====
    println!("\n[Step 5] Running inference");
    let start = std::time::Instant::now();
    session.run()?;
    let duration = start.elapsed();
    println!("  Inference completed in {:?}", duration);

    // ===== Step 6: Get Output Tensor =====
    println!("\n[Step 6] Getting output tensor");
    let output = session.get_output(None)?;
    let output_shape = output.shape();
    println!("  Output shape: {:?}", output_shape);
    println!("  Output dtype: {}", output.dtype());
    println!("  Output format: {:?}", output.format());
    println!("  Output element count: {}", output.element_count());

    // ===== Step 7: Read Output Data =====
    println!("\n[Step 7] Reading output data");
    let output_data: Vec<f32> = output.read()?;
    println!("  Read {} output values", output_data.len());

    // Print output statistics
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = output_data.iter().sum();
    let mean = sum / output_data.len() as f32;

    println!("\n  Output statistics:");
    println!("    Min: {:.6}", min_val);
    println!("    Max: {:.6}", max_val);
    println!("    Mean: {:.6}", mean);
    println!("    Sum: {:.6}", sum);

    // Print top-k values (for classification models)
    println!("\n  Top 10 output values:");
    let mut indexed: Vec<(usize, f32)> = output_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
        println!("    #{}: index={}, value={:.6}", i + 1, idx, val);
    }

    // Print first and last few values
    let preview_count = 5;
    println!("\n  First {} values: {:?}", preview_count, &output_data[..preview_count.min(output_data.len())]);
    if output_data.len() > preview_count {
        println!("  Last {} values: {:?}", preview_count, &output_data[output_data.len() - preview_count..]);
    }

    // ===== Step 8: Session Statistics =====
    println!("\n[Step 8] Session statistics");
    println!("  Memory usage: {} bytes ({:.2} MB)", session.memory_usage(), session.memory_usage() as f64 / 1024.0 / 1024.0);
    println!("  FLOPS: {:.2} M", session.flops());

    println!("\n=== Inference completed successfully! ===");

    Ok(())
}