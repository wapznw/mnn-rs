# mnn-rs

[![Crates.io](https://img.shields.io/crates/v/mnn-rs.svg)](https://crates.io/crates/mnn-rs)
[![Documentation](https://docs.rs/mnn-rs/badge.svg)](https://docs.rs/mnn-rs)
[![License](https://img.shields.io/crates/l/mnn-rs.svg)](https://github.com/wapznw/mnn-rs#license)

Rust bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network), Alibaba's efficient and lightweight deep learning inference framework.

## Features

- **Safe Rust API**: All MNN operations are wrapped in safe Rust types with proper error handling
- **Cross-platform**: Supports Windows, Linux, macOS, Android, and iOS
- **Multiple Backends**: CPU, CUDA, OpenCL, Vulkan, and Metal
- **Static/Dynamic Linking**: Choose between static or dynamic linking
- **Async Support**: Optional async API with Tokio integration
- **Build from Source**: Automatically clone and build MNN from GitHub

## Quick Start

### Default Build (Recommended)

By default, the crate will automatically clone and build MNN from GitHub:

```bash
cargo build
```

This requires:
- Git (for cloning MNN source)
- CMake (for building MNN)
- C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)

### Using Pre-built MNN

If you have a pre-built MNN library, you can disable the auto-build:

```bash
# Set MNN library paths
export MNN_LIB_DIR=/path/to/mnn/lib
export MNN_INCLUDE_DIR=/path/to/mnn/include

# Build without build-from-source feature
cargo build --no-default-features --features cpu,static
```

## Usage

### Basic Inference

```rust
use mnn_rs::{prelude::*, BackendType, ScheduleConfig};

fn main() -> Result<(), MnnError> {
    // Load model
    let interpreter = Interpreter::from_file("model.mnn")?;

    // Create session
    let config = ScheduleConfig::new()
        .backend(BackendType::CPU)
        .num_threads(4);
    let mut session = interpreter.create_session(config)?;

    // Get input tensor and write data
    let input = session.get_input(None)?;
    let input_data: Vec<f32> = vec![0.0; input.element_count() as usize];
    input.write(&input_data)?;

    // Run inference
    session.run()?;

    // Read output
    let output = session.get_output(None)?;
    let output_data: Vec<f32> = output.read()?;

    println!("Output: {:?}", &output_data[..10]);
    Ok(())
}
```

### Async Inference (with Tokio)

```rust
use mnn_rs::{prelude::*, BackendType, ScheduleConfig};

#[tokio::main]
async fn main() -> Result<(), MnnError> {
    let interpreter = Interpreter::from_file("model.mnn")?;
    let config = ScheduleConfig::new().backend(BackendType::CPU);
    let mut session = interpreter.create_session(config)?;

    // Async inference
    session.run_async().await?;

    Ok(())
}
```

## Features

### Linking Mode

| Feature | Description |
|---------|-------------|
| `static` (default) | Static link MNN library |
| `dynamic` | Dynamic link MNN library |

### Backend Support

| Feature | Description | Platform |
|---------|-------------|----------|
| `cpu` (default) | CPU backend | All |
| `cuda` | NVIDIA GPU backend | Windows, Linux |
| `opencl` | OpenCL GPU backend | Windows, Linux, macOS, Android |
| `vulkan` | Vulkan GPU backend | Windows, Linux, Android |
| `metal` | Metal GPU backend | macOS, iOS |

### Precision Support

| Feature | Description |
|---------|-------------|
| `fp16` | FP16 precision support |
| `int8` | INT8 precision support |
| `quantization` | Quantization support |

### x86 SIMD Optimizations

| Feature | Description | Default (x86_64) | Default (x86) |
|---------|-------------|------------------|---------------|
| `sse` | SSE instructions | ON | OFF |
| `avx2` | AVX2 instructions | OFF | OFF |
| `avx512` | AVX512 instructions | OFF | OFF |

### Build Options

| Feature | Description |
|---------|-------------|
| `build-from-source` | Automatically clone and build MNN from GitHub |
| `system-mnn` | Use system-installed MNN |
| `generate-bindings` | Generate FFI bindings using bindgen |

### Async Support

| Feature | Description |
|---------|-------------|
| `async` | Enable async API with Tokio |

## Examples

See the `examples/` directory for more usage examples:

- `basic_inference.rs` - Basic inference workflow
- `async_inference.rs` - Async inference with Tokio
- `gpu_backend.rs` - Using GPU backends

```bash
# Run basic inference example
cargo run --example basic_inference -- /path/to/model.mnn

# Run with build-from-source
cargo run --features build-from-source --example basic_inference -- /path/to/model.mnn
```

## Cross-Compilation

### Android

```bash
# Install Android target
rustup target add aarch64-linux-android

# Set Android NDK path (required)
export ANDROID_NDK_HOME=/path/to/android-ndk

# Build for Android (requires Ninja)
cargo build --target aarch64-linux-android
```

**Android Requirements:**
- Android NDK (set `ANDROID_NDK_HOME` or `NDK_HOME` environment variable)
- Ninja build system (`choco install ninja` on Windows, `brew install ninja` on macOS)

**Supported Android targets:**
- `aarch64-linux-android` (arm64-v8a)
- `armv7-linux-androideabi` (armeabi-v7a)
- `x86_64-linux-android`
- `i686-linux-android`

### iOS

```bash
# Install iOS target
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim

# Build for iOS device
cargo build --target aarch64-apple-ios

# Build for iOS simulator (Apple Silicon Macs)
cargo build --target aarch64-apple-ios-sim

# Build for iOS simulator (Intel Macs)
cargo build --target x86_64-apple-ios
```

**iOS Requirements:**
- Xcode with iOS SDK
- Ninja build system (recommended: `brew install ninja`)

**Supported iOS targets:**
- `aarch64-apple-ios` (iOS device)
- `aarch64-apple-ios-sim` (iOS simulator on Apple Silicon)
- `x86_64-apple-ios` (iOS simulator on Intel Macs)

### Linux

```bash
# Standard build
cargo build --target x86_64-unknown-linux-gnu

# Cross-compile for ARM
rustup target add aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu
```

### macOS

```bash
# Intel Macs
cargo build --target x86_64-apple-darwin

# Apple Silicon Macs
cargo build --target aarch64-apple-darwin
```

### Windows

```bash
# MSVC (recommended)
cargo build --target x86_64-pc-windows-msvc

# MinGW
rustup target add x86_64-pc-windows-gnu
cargo build --target x86_64-pc-windows-gnu
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MNN_SOURCE_PATH` | Path to MNN source directory |
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_DEBUG_BUILD` | Print debug information during build |
| `CUDA_PATH` | CUDA installation path |
| `ANDROID_NDK_HOME` | Android NDK installation path |

## API Documentation

See [https://docs.rs/mnn-rs](https://docs.rs/mnn-rs) for full API documentation.

## MNN Version

This crate is compatible with MNN 2.9.5+.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [MNN](https://github.com/alibaba/MNN) - Alibaba's Mobile Neural Network inference engine
- All contributors who helped with this project