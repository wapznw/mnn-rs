//! Image preprocessing example using MNN ImageProcess module.
//!
//! This example demonstrates image preprocessing operations:
//! - Converting image formats (RGB/BGR/RGBA/etc.)
//! - Applying normalization (mean/std)
//! - Resizing with bilinear interpolation
//! - Converting to tensor format
//!
//! Note: This example uses raw RGB pixel data. In real applications,
//! you would load images from files using a library like `image`.
//!
//! Usage:
//! ```bash
//! cargo run --example image_process --features image-process
//! ```

#[cfg(feature = "image-process")]
use mnn_rs::{
    ImageConfig, ImageFormat, ImageProcess, Matrix, Filter, Wrap, Tensor, MnnError,
};

#[cfg(feature = "image-process")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    println!("=== MNN Rust Image Process Example ===\n");

    // Parse command line arguments
    let width: i32 = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(224);
    let height: i32 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(224);

    println!("[Step 1] Creating sample image data ({}x{})", width, height);

    // Create sample RGB image data (simulating a loaded image)
    // In real applications, you would load from file using image crate:
    // ```rust
    // let img = image::open("photo.jpg")?;
    // let rgb_img = img.to_rgb8();
    // let image_data = rgb_img.as_raw();
    // ```
    let image_data: Vec<u8> = (0..(width * height * 3))
        .map(|i| ((i as u32 * 7 + 123) % 256) as u8)  // Pattern data
        .collect();
    println!("  Created {} bytes of RGB data", image_data.len());

    // Show sample pixels
    println!("  Sample pixels (first 3):");
    for i in 0..3 {
        let idx = i * 3;
        println!("    Pixel {}: R={}, G={}, B={}", i,
                 image_data[idx], image_data[idx + 1], image_data[idx + 2]);
    }

    println!("\n[Step 2] Creating ImageProcess configuration");

    // Create image process config
    // This config converts RGB to BGR (common for CV models)
    // Using no normalization for this simple demo
    let config = ImageConfig {
        source_format: ImageFormat::Rgb,
        dest_format: ImageFormat::Bgr,
        filter: Filter::Bilinear,
        mean: [0.0, 0.0, 0.0, 0.0],  // No mean subtraction
        normal: [1.0, 1.0, 1.0, 1.0],  // No normalization
        wrap: Wrap::ClampToEdge,
    };

    println!("  Source format: {:?}", config.source_format);
    println!("  Dest format: {:?}", config.dest_format);
    println!("  Filter: {:?}", config.filter);
    println!("  Mean: {:?}", config.mean);
    println!("  Normal: {:?}", config.normal);

    println!("\n[Step 3] Creating ImageProcess instance");
    let mut image_process = ImageProcess::new(&config)?;
    println!("  ImageProcess created successfully");

    println!("\n[Step 4] Creating transformation matrix");

    // Create identity matrix (no transformation)
    let mut matrix = Matrix::identity();
    println!("  Identity matrix created");

    // Optionally apply scaling
    let scale = 224.0 / width.max(height) as f32;
    if scale < 1.0 {
        println!("  Applying scale: {:.4}", scale);
        matrix = Matrix::scale(scale, scale);
    }

    image_process.set_matrix(&matrix);
    println!("  Matrix applied to ImageProcess");

    println!("\n[Step 5] Creating output tensor using ImageProcess");

    // Create output tensor using ImageProcess helper
    // This creates a tensor suitable for image operations (224x224x3 = BGR)
    let output_width = 224;
    let output_height = 224;
    let mut output_tensor = ImageProcess::create_image_tensor(
        output_width,
        output_height,
        3,  // 3 channels (BGR)
        None,  // No initial data
    )?;

    println!("  Output tensor shape: {:?}", output_tensor.shape());
    println!("  Output tensor dtype: {}", output_tensor.dtype());

    println!("\n[Step 6] Converting image to tensor");

    // Perform the image conversion
    // stride = 0 means width * channels
    image_process.convert(
        &image_data,
        width as i32,
        height as i32,
        0,  // stride (0 = auto)
        &mut output_tensor,
    )?;

    println!("  Image conversion completed");

    println!("\n[Step 7] Reading and analyzing output");

    // Read output data as u8 (image tensor is u8 type)
    let output_data: Vec<u8> = output_tensor.read()?;
    println!("  Output data: {} elements", output_data.len());

    // Calculate statistics (convert to f32 for calculation)
    let min_val = output_data.iter().cloned().map(|x| x as f32).fold(f32::INFINITY, f32::min);
    let max_val = output_data.iter().cloned().map(|x| x as f32).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = output_data.iter().map(|&x| x as f32).sum();
    let mean = sum / output_data.len() as f32;

    println!("\n  Output statistics:");
    println!("    Min: {:.6}", min_val);
    println!("    Max: {:.6}", max_val);
    println!("    Mean: {:.6}", mean);

    // Print sample values from each channel
    println!("\n  Sample values (first 3x3 pixels, BGR order):");
    for row in 0..3 {
        for col in 0..3 {
            let idx = (row * output_width * 3 + col * 3) as usize;
            if idx + 2 < output_data.len() {
                println!("    [{},{}]: B={}, G={}, R={}",
                         row, col,
                         output_data[idx],
                         output_data[idx + 1],
                         output_data[idx + 2]);
            }
        }
    }

    // Show first few raw bytes
    println!("\n  First 12 bytes (4 pixels): {:?}", &output_data[..12]);

    println!("\n[Step 8] Demonstrate other image formats");

    // Demonstrate creating tensors with different formats
    let formats = [
        (ImageFormat::Rgba, "RGBA"),
        (ImageFormat::Gray, "Grayscale"),
        (ImageFormat::Bgra, "BGRA"),
        (ImageFormat::Yuv, "YUV"),
    ];

    for (fmt, name) in formats.iter() {
        let cfg = ImageConfig {
            source_format: *fmt,
            dest_format: ImageFormat::Rgb,
            filter: Filter::Nearest,
            mean: [0.0; 4],
            normal: [1.0; 4],
            wrap: Wrap::ClampToEdge,
        };
        let ip = ImageProcess::new(&cfg);
        println!("  {} format supported: {}", name, ip.is_ok());
    }

    println!("\n[Step 9] Demonstrate Matrix operations");

    // Demonstrate different matrix transformations
    let matrices = [
        ("Identity", Matrix::identity()),
        ("Scale 2x", Matrix::scale(2.0, 2.0)),
        ("Scale 0.5x", Matrix::scale(0.5, 0.5)),
        ("Translate (10, 20)", Matrix::translate(10.0, 20.0)),
        ("Rotate 90°", Matrix::rotate(90.0)),
        ("Rotate 180°", Matrix::rotate(180.0)),
    ];

    for (name, _matrix) in matrices.iter() {
        println!("  {} - created", name);
    }

    println!("\n=== Image processing completed successfully! ===");
    println!("Note: To save the processed image, you would need to:");
    println!("  1. Convert CHW format back to HWC");
    println!("  2. Denormalize the values");
    println!("  3. Convert f32 to u8");
    println!("  4. Use image crate to save");

    Ok(())
}

#[cfg(not(feature = "image-process"))]
fn main() {
    eprintln!("Error: This example requires the 'image-process' feature.");
    eprintln!("Run with: cargo run --example image_process --features image-process");
    std::process::exit(1);
}
