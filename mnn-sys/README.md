# mnn-sys

[![Crates.io](https://img.shields.io/crates/v/mnn-sys.svg)](https://crates.io/crates/mnn-sys)
[![Documentation](https://docs.rs/mnn-sys/badge.svg)](https://docs.rs/mnn-sys)

Raw FFI bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network) inference engine.

This crate provides low-level unsafe bindings to MNN's C API. For a safe, idiomatic Rust API, see the [`mnn`](https://crates.io/crates/mnn) crate.

## Features

- **Build from Source**: Automatically clone and compile MNN from GitHub
- **Multiple Backends**: CPU, CUDA, OpenCL, Vulkan, Metal
- **Static/Dynamic Linking**: Flexible linking options
- **Cross-Platform**: Windows, Linux, macOS, Android, iOS
- **SIMD Optimizations**: SSE, AVX2, AVX512 support for x86

## Usage

### Building from Source (Recommended)

```toml
# Cargo.toml
[dependencies]
mnn-sys = { version = "0.1", features = ["build-from-source"] }
```

The build script will automatically clone and compile MNN from GitHub.

### Using Pre-built MNN

```bash
export MNN_LIB_DIR=/path/to/mnn/lib
export MNN_INCLUDE_DIR=/path/to/mnn/include
cargo build
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