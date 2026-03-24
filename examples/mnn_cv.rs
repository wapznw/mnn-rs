//! MNN CV Image Codecs example.
//!
//! This example demonstrates MNN's image loading/saving capabilities
//! using MNN::CV::imread and MNN::CV::imwrite.
//!
//! Requires MNN built with: -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON
//!
//! Usage:
//! ```bash
//! cargo run --example mnn_cv --features "image-process,build-from-source" --no-default-features -- input.jpg output.png
//! ```

#[cfg(feature = "image-process")]
use mnn_rs::{
    imread, imwrite, resize,
    ImreadFlags, ResizeFilter,
};

#[cfg(feature = "image-process")]
fn main() -> Result<(), mnn_rs::MnnError> {
    use std::env;

    println!("=== MNN CV Image Codecs Example ===\n");

    // Get paths from command line
    let input_path = env::args().nth(1)
        .unwrap_or_else(|| "input.jpg".to_string());
    let output_path = env::args().nth(2)
        .unwrap_or_else(|| "output_resized.png".to_string());

    println!("[Step 1] Reading image using MNN CV::imread");
    println!("  Input path: {}", input_path);

    // Read image using MNN CV (supports JPG, PNG, BMP, etc.)
    let img = imread(&input_path, ImreadFlags::Color)?;
    println!("  Image loaded successfully");

    // Get image info
    let shape = img.shape();
    println!("  Shape: {:?}", shape);
    println!("  Dtype: {}", img.dtype());
    println!("  Element count: {}", img.element_count());

    // Try reading as u8 (imread returns uint8 tensor)
    let data: Vec<u8> = img.read()?;
    println!("  Read {} bytes", data.len());
    println!("  First 10 bytes: {:?}", &data[0..10.min(data.len())]);

    // Calculate dimensions from tensor shape
    // For NHWC: [height, width, channels] or [1, height, width, channels]
    let height = if shape.len() >= 4 { shape[1] } else { shape[0] };
    let width = if shape.len() >= 4 { shape[2] } else { shape[1] };
    let channels = if shape.len() >= 4 { shape[3] } else { shape[2] };
    println!("  Dimensions: {}x{}x{}", height, width, channels);

    println!("\n[Step 2] Resizing image using MNN CV::resize");
    let new_width = 224;
    let new_height = 224;
    println!("  Target size: {}x{}", new_width, new_height);

    // Resize using bilinear interpolation
    let resized = resize(&img, new_width, new_height, ResizeFilter::Bilinear)?;
    println!("  Image resized successfully");

    let resized_shape = resized.shape();
    println!("  New shape: {:?}", resized_shape);

    println!("\n[Step 3] Saving image using MNN CV::imwrite");
    println!("  Output path: {}", output_path);

    // Save resized image
    imwrite(&output_path, &resized)?;
    println!("  Image saved successfully");

    println!("\n[Step 4] Demonstrating read flags");

    // Read as grayscale
    let gray = imread(&input_path, ImreadFlags::Grayscale)?;
    let gray_shape = gray.shape();
    println!("  Grayscale shape: {:?}", gray_shape);

    // Read unchanged
    let unchanged = imread(&input_path, ImreadFlags::Unchanged)?;
    let unchanged_shape = unchanged.shape();
    println!("  Unchanged shape: {:?}", unchanged_shape);

    println!("\n=== MNN CV operations completed successfully! ===");
    println!("\nNote: This requires MNN built with:");
    println!("  -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON");
    println!("\nSupported formats: JPG, PNG, BMP, PNM, etc.");

    Ok(())
}

#[cfg(not(feature = "image-process"))]
fn main() {
    eprintln!("Error: This example requires the 'image-process' feature.");
    eprintln!("Run with: cargo run --example mnn_cv --features image-process");
    std::process::exit(1);
}
