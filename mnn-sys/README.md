# mnn-rs-sys

[![Crates.io](https://img.shields.io/crates/v/mnn-rs-sys.svg)](https://crates.io/crates/mnn-rs-sys)
[![Documentation](https://docs.rs/mnn-rs-sys/badge.svg)](https://docs.rs/mnn-rs-sys)

Raw FFI bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network) inference engine.

This crate provides low-level unsafe bindings to MNN's C API. For a safe, idiomatic Rust API, see the [`mnn-rs`](https://crates.io/crates/mnn-rs) crate.

## Features

- **Build from Source**: Automatically clone and compile MNN from GitHub
- **Multiple Backends**: CPU, CUDA, OpenCL, Vulkan, Metal
- **Static/Dynamic Linking**: Flexible linking options
- **Cross-Platform**: Windows, Linux, macOS, Android, iOS
- **SIMD Optimizations**: SSE, AVX2, AVX512 support for x86

## Usage

### Default Build (Recommended)

By default, the crate will automatically clone and build MNN from GitHub:

```bash
cargo build
```

This requires Git, CMake, and a C++ compiler.

### Using Pre-built MNN

If you have a pre-built MNN library:

```bash
# Set environment variables
export MNN_LIB_DIR=/path/to/mnn/lib
export MNN_INCLUDE_DIR=/path/to/mnn/include

# Build without auto-build
cargo build --no-default-features --features cpu,static
```

## Features

| Feature | Description |
|---------|-------------|
| `static` (default) | Static link MNN |
| `dynamic` | Dynamic link MNN |
| `build-from-source` | Clone and build MNN from GitHub |
| `cuda` | Enable CUDA backend |
| `opencl` | Enable OpenCL backend |
| `vulkan` | Enable Vulkan backend |
| `metal` | Enable Metal backend |
| `sse` | Enable SSE instructions |
| `avx2` | Enable AVX2 instructions |
| `avx512` | Enable AVX512 instructions |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MNN_SOURCE_PATH` | Path to MNN source directory |
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_DEBUG_BUILD` | Print debug info during build |

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.