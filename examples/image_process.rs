//! Image preprocessing example using MNN ImageProcess module.
//!
//! This example demonstrates:
//! - Image format conversion (RGB to BGR)
//! - Image resizing using Matrix transformation
//! - Optional: imread/imwrite/resize using MNN CV (requires special build)
//!
//! Usage:
//! ```bash
//! # Basic usage with generated test pattern
//! cargo run --example image_process --features image-process
//!
//! # With custom image (requires MNN with IMGCODECS)
//! cargo run --example image_process -- input.jpg output.png
//! ```

#[cfg(feature = "image-process")]
use mnn_rs::{
    imread, imwrite,
    ImageConfig, ImageFormat, ImageProcess, Matrix, Filter, Wrap,
    ImreadFlags,
};

#[cfg(feature = "image-process")]
fn main() -> Result<(), mnn_rs::MnnError> {
    use std::env;
    use std::path::Path;

    println!("=== MNN Rust Image Process Example ===\n");

    // Parse command line arguments
    let input_path = env::args().nth(1).unwrap_or_else(|| String::new());
    let output_path = env::args().nth(2).unwrap_or_else(|| "output.png".to_string());

    // Check if input path is provided and exists
    let use_test_image = input_path.is_empty() || !Path::new(&input_path).exists();

    if use_test_image {
        println!("[Step 1] Using generated test pattern");
        println!("  Note: Provide an image path to use imread:");
        println!("  cargo run --example image_process -- input.jpg output.png\n");

        // Create a colorful test pattern
        let (width, height) = (256i32, 256i32);
        println!("  Creating test pattern: {}x{}", width, height);

        let image_data: Vec<u8> = (0..(width * height * 3) as usize)
            .map(|i| {
                let x = (i / 3) % width as usize;
                let y = (i / 3) / width as usize;
                let r = ((x * 255 / width as usize) as u8).wrapping_add((y * 100 / height as usize) as u8);
                let g = ((x * 100 / width as usize) as u8).wrapping_add((y * 255 / height as usize) as u8);
                let b = ((x * 128 / width as usize) as u8).wrapping_add((y * 128 / height as usize) as u8);
                [r, g, b][i % 3]
            })
            .collect();

        println!("  Generated {} bytes", image_data.len());

        process_image(&image_data, width, height, 224, 224, &output_path, false)
    } else {
        // Use MNN CV imread to load image
        println!("[Step 1] Loading image using MNN CV::imread");
        println!("  Input: {}", input_path);

        match imread(&input_path, ImreadFlags::Color) {
            Ok(img) => {
                let shape = img.shape();
                let (height, width, channels) = if shape.len() >= 4 {
                    (shape[1], shape[2], shape[3])
                } else {
                    (shape[0], shape[1], shape[2])
                };

                println!("  Loaded: {}x{}x{}", width, height, channels);
                println!("  Dtype: {}", img.dtype());

                let image_data: Vec<u8> = img.read()?;
                println!("  Read {} bytes", image_data.len());

                process_image(&image_data, width, height, 224, 224, &output_path, true)
            }
            Err(e) => {
                eprintln!("  imread failed: {}", e);
                eprintln!("  MNN CV imread requires: -DMNN_IMGCODECS=ON");
                eprintln!("  Falling back to test pattern...");

                let (width, height) = (256i32, 256i32);
                let image_data: Vec<u8> = (0..(width * height * 3) as usize)
                    .map(|i| ((i * 7) % 256) as u8)
                    .collect();
                process_image(&image_data, width, height, 224, 224, &output_path, false)
            }
        }
    };

    println!("\n=== Processing completed! ===");
    Ok(())
}

#[cfg(feature = "image-process")]
fn process_image(
    image_data: &[u8],
    src_width: i32,
    src_height: i32,
    target_width: i32,
    target_height: i32,
    output_path: &str,
    loaded_from_file: bool,
) -> Result<(), mnn_rs::MnnError> {
    println!("\n[Step 2] Creating ImageProcess configuration");

    // Create config: RGB to BGR conversion with bilinear filtering
    let config = ImageConfig {
        source_format: ImageFormat::Rgb,
        dest_format: ImageFormat::Bgr,
        filter: Filter::Bilinear,
        mean: [0.0; 4],
        normal: [1.0; 4],
        wrap: Wrap::ClampToEdge,
    };

    let mut image_process = ImageProcess::new(&config)?;
    println!("  Config: RGB → BGR, Bilinear filter");

    println!("\n[Step 3] Setting up transformation matrix");

    // Calculate scale for resizing
    let scale_x = target_width as f32 / src_width as f32;
    let scale_y = target_height as f32 / src_height as f32;
    println!("  Scale: {:.3} x {:.3}", scale_x, scale_y);

    // Create and set scale matrix
    let matrix = Matrix::scale(scale_x, scale_y);
    image_process.set_matrix(&matrix);
    println!("  Matrix set");

    println!("\n[Step 4] Creating output tensor");

    // Create output tensor (224x224x3 BGR)
    let mut output_tensor = ImageProcess::create_image_tensor(
        target_width, target_height, 3, None
    )?;
    println!("  Tensor shape: {:?}", output_tensor.shape());
    println!("  Tensor dtype: {}", output_tensor.dtype());

    println!("\n[Step 5] Converting image (resize + RGB→BGR)");

    // Perform conversion with resizing
    image_process.convert(
        image_data,
        src_width,
        src_height,
        0,  // stride (auto = width * channels)
        &mut output_tensor,
    )?;
    println!("  Conversion completed");

    println!("\n[Step 6] Analyzing output");

    // Read output data
    let output_data: Vec<u8> = output_tensor.read()?;
    println!("  Output: {} bytes", output_data.len());

    // Statistics
    let min_val = output_data.iter().cloned().min().unwrap_or(0);
    let max_val = output_data.iter().cloned().max().unwrap_or(255);
    let sum: u32 = output_data.iter().map(|&x| x as u32).sum();
    let mean = sum as f32 / output_data.len() as f32;

    println!("  Min: {}, Max: {}, Mean: {:.2}", min_val, max_val, mean);

    // Sample pixels
    println!("\n  Sample pixels (BGR order):");
    for i in 0..3.min(output_data.len() / 3) {
        let idx = i * 3;
        println!("    [{}]: B={}, G={}, R={}", i,
                 output_data[idx], output_data[idx + 1], output_data[idx + 2]);
    }

    println!("\n[Step 7] Saving result");

    // Create tensor for saving
    let save_tensor = ImageProcess::create_image_tensor(
        target_width, target_height, 3, Some(&output_data),
    )?;

    // Try to save using MNN CV imwrite
    match imwrite(output_path, &save_tensor) {
        Ok(()) => {
            println!("  Saved to: {}", output_path);
            println!("  Output format inferred from extension");
        }
        Err(e) => {
            println!("  imwrite not available: {}", e);
            println!("  Enable with: -DMNN_IMGCODECS=ON");
        }
    }

    if loaded_from_file {
        println!("\nWorkflow (with MNN CV):");
        println!("  imread() → ImageProcess (convert+resize) → imwrite()");
    } else {
        println!("\nWorkflow:");
        println!("  Raw data → ImageProcess (convert+resize) → output tensor");
    }

    println!("\nNote: For neural network inference with normalization:");
    println!("  - Set mean=[103.94, 116.78, 123.68, 0]");
    println!("  - Set normal=[0.017, 0.017, 0.017, 1.0]");
    println!("  - Output tensor must be float32 type (use MNN Express)");

    Ok(())
}

#[cfg(not(feature = "image-process"))]
fn main() {
    eprintln!("Error: This example requires the 'image-process' feature.");
    eprintln!("Run with: cargo run --example image_process --features image-process");
    std::process::exit(1);
}
