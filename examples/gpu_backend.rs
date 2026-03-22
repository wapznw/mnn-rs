//! GPU backend inference example using MNN.
//!
//! This example demonstrates using GPU backends (CUDA, OpenCL, Vulkan, Metal)
//! for inference.
//!
//! Usage:
//! ```bash
//! cargo run --example gpu_backend --features cuda -- /path/to/model.mnn
//! ```

use mnn::{prelude::*, BackendType, MemoryMode, PrecisionMode, ScheduleConfig};

fn main() -> Result<(), MnnError> {
    // Get model path from command line
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: gpu_backend <model.mnn>");

    println!("Loading model from: {}", model_path);

    // Check available backends
    println!("\nAvailable backends:");
    let backends = BackendType::available_backends();
    for backend in &backends {
        println!("  - {}", backend);
    }

    // Select backend based on feature flags
    #[cfg(feature = "cuda")]
    let backend_type = BackendType::Cuda;

    #[cfg(all(not(feature = "cuda"), feature = "opencl"))]
    let backend_type = BackendType::OpenCL;

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "opencl"),
        feature = "vulkan"
    ))]
    let backend_type = BackendType::Vulkan;

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "opencl"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    let backend_type = BackendType::Metal;

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "opencl"),
        not(feature = "vulkan"),
        not(feature = "metal")
    ))]
    let backend_type = BackendType::CPU;

    println!("\nUsing backend: {}", backend_type);

    // Check if backend is available
    if !backend_type.is_available() {
        eprintln!("Warning: Backend {} is not available", backend_type);
        println!("Falling back to CPU");
    }

    // Create interpreter
    let interpreter = Interpreter::from_file(&model_path)?;

    // Create session with GPU configuration
    let config = ScheduleConfig::new()
        .backend(backend_type)
        .memory_mode(MemoryMode::Normal)
        .precision_mode(PrecisionMode::Normal);

    let mut session = interpreter.create_session(config)?;
    println!("Session created with {} backend", backend_type);

    // Get input info
    let input_names = session.get_input_names()?;
    if let Some(first_input) = input_names.first() {
        let input = session.get_input(Some(first_input))?;
        println!("Input '{}' shape: {:?}", first_input, input.shape());
    }

    // Run inference
    println!("\nRunning inference...");
    let start = std::time::Instant::now();
    session.run()?;
    let elapsed = start.elapsed();
    println!("Inference completed in {:?}", elapsed);

    // Get output
    let output_names = session.get_output_names()?;
    if let Some(first_output) = output_names.first() {
        let output = session.get_output(Some(first_output))?;
        println!("Output '{}' shape: {:?}", first_output, output.shape());
    }

    // Print stats
    println!("\nSession stats:");
    println!("  Memory: {} bytes", session.memory_usage());
    println!("  FLOPS: {}", session.flops());

    Ok(())
}