//! Build script for mnn-sys
//!
//! This script handles linking to the MNN library, supporting:
//! - Static and dynamic linking
//! - Multiple backends (CPU, CUDA, OpenCL, Vulkan, Metal)
//! - Build from source or use pre-installed library
//! - Cross-platform builds
//!
//! # Building from Source
//!
//! ## Source Code Location
//!
//! MNN source code is stored in a shared location to avoid duplicate clones:
//! - `MNN_SOURCE_PATH` environment variable (highest priority)
//! - `<project-root>/target/mnn-source` (default for project-level caching)
//!
//! ## Build Artifacts Location
//!
//! Compiled MNN libraries are cached per target to support cross-compilation:
//! - `<project-root>/target/mnn-build/<target-triple>/<profile>/`
//!
//! ## Usage
//!
//! ```bash
//! # Let the script manage everything (recommended)
//! cargo build --features build-from-source
//!
//! # Use existing MNN source
//! MNN_SOURCE_PATH=/path/to/mnn cargo build --features build-from-source
//!
//! # Use pre-built MNN
//! MNN_LIB_DIR=/path/to/mnn/lib MNN_INCLUDE_DIR=/path/to/mnn/include cargo build
//! ```

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Check if we should print debug info
    let debug_build = env::var("MNN_DEBUG_BUILD").is_ok();

    if debug_build {
        println!("cargo:warning=mnn-sys: Starting build script");
    }

    // Read environment variables
    let mnn_source_path = env::var("MNN_SOURCE_PATH").ok();
    let mnn_lib_dir = env::var("MNN_LIB_DIR").ok();
    let mnn_include_dir = env::var("MNN_INCLUDE_DIR").ok();
    let _mnn_use_system = env::var("MNN_USE_SYSTEM").is_ok() || cfg!(feature = "system-mnn");
    let build_from_source = cfg!(feature = "build-from-source");

    // Determine linking mode
    let static_link = cfg!(feature = "static");
    let dynamic_link = cfg!(feature = "dynamic");

    if static_link && dynamic_link {
        panic!("Cannot enable both 'static' and 'dynamic' features simultaneously");
    }

    // If no link mode specified, default to static
    let use_static = !dynamic_link;

    // Build from source if requested
    let (lib_dir, include_dir) = if build_from_source {
        // Get the project's target directory (shared across all builds)
        let target_dir = get_target_directory();

        // Source path: use MNN_SOURCE_PATH or shared location
        let source_path = match mnn_source_path {
            Some(path) => PathBuf::from(path),
            None => {
                // Use shared source directory in project's target folder
                // This is shared across all build targets
                let shared_source = target_dir.join("mnn-source");
                if !shared_source.exists() {
                    clone_mnn_from_github(&shared_source, debug_build);
                } else if debug_build {
                    println!("cargo:warning=mnn-sys: Using cached MNN source at {:?}", shared_source);
                }
                shared_source
            }
        };

        // Build directory: per-target to support cross-compilation
        // Format: target/mnn-build/<target-triple>/<profile>/
        let target_triple = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
        let build_dir = target_dir
            .join("mnn-build")
            .join(&target_triple);

        let (lib_dir, include_dir) = build_mnn_from_source(
            &source_path,
            &build_dir,
            use_static,
            debug_build,
        );
        (Some(lib_dir), Some(include_dir))
    } else {
        (mnn_lib_dir.map(PathBuf::from), mnn_include_dir.map(PathBuf::from))
    };

    // Compile the C++ wrapper
    let wrapper_dir = PathBuf::from("wrapper");
    let wrapper_cpp = wrapper_dir.join("mnn_wrapper.cpp");

    // Get include directory for MNN headers
    let mnn_include = include_dir.clone().unwrap_or_else(|| PathBuf::new());

    // Build the wrapper library
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&wrapper_cpp)
        .include(&wrapper_dir)
        .include(&mnn_include);

    // Set C++ standard
    if cfg!(target_env = "msvc") {
        build.flag("/std:c++14");
        // Match runtime library with MNN's build (always /MT for static builds)
        // MNN is always built with Release configuration, so use /MT
        build.flag("/MT");
    } else {
        build.flag_if_supported("-std=c++14");
    }

    // Print compiler info in debug mode
    if debug_build {
        if let Ok(tool) = build.try_get_compiler() {
            println!("cargo:warning=mnn-sys: C++ compiler: {:?}", tool.path());
        }
    }

    build.compile("mnn_wrapper");

    // Configure include path
    if let Some(ref inc_dir) = include_dir {
        println!("cargo:include={}", inc_dir.display());
    }

    // Configure linking
    if let Some(ref lib_path) = lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    } else if !build_from_source {
        // Try common library paths
        search_common_paths(use_static);
    }

    // Link the library
    let link_type = if use_static { "static" } else { "dylib" };
    println!("cargo:rustc-link-lib={}=mnn", link_type);

    // Print help message if library not found and not building from source
    if lib_dir.is_none() && !build_from_source {
        println!("cargo:warning=mnn-sys: MNN library not found in MNN_LIB_DIR or common paths");
        println!("cargo:warning=mnn-sys: Please do one of the following:");
        println!("cargo:warning=  1. Set MNN_LIB_DIR to the directory containing libmnn.a or mnn.lib");
        println!("cargo:warning=  2. Set MNN_SOURCE_PATH and use 'build-from-source' feature");
        println!("cargo:warning=  3. Use --features build-from-source to auto-clone and build MNN");
        println!("cargo:warning=See https://github.com/alibaba/MNN for MNN installation instructions");
    }

    // Configure platform-specific settings
    configure_platform();

    // Configure backend-specific linking
    configure_backends();

    // Tell Cargo to re-run if relevant env vars change
    println!("cargo:rerun-if-env-changed=MNN_SOURCE_PATH");
    println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MNN_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MNN_USE_SYSTEM");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");

    // Re-run if wrapper changes
    println!("cargo:rerun-if-changed=wrapper/mnn_wrapper.h");
    println!("cargo:rerun-if-changed=wrapper/mnn_wrapper.cpp");

    if debug_build {
        println!("cargo:warning=mnn-sys: Build script completed successfully");
    }
}

/// Get the project's target directory
///
/// This is the directory where Cargo stores build artifacts.
/// It's shared across all build targets (debug/release, different architectures).
fn get_target_directory() -> PathBuf {
    // CARGO_TARGET_DIR is set when building with --target
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(target_dir);
    }

    // Otherwise, find the target directory from OUT_DIR
    // OUT_DIR is typically: <project>/target/<profile>/build/<crate>/out
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Walk up to find the target directory (4 levels up from OUT_DIR)
    let mut current = out_dir.as_path();
    for _ in 0..4 {
        if let Some(parent) = current.parent() {
            current = parent;
        }
    }

    // Verify this is the target directory
    if current.file_name().map(|n| n == "target").unwrap_or(false) {
        return current.to_path_buf();
    }

    // Fallback: use the target directory in the manifest directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    manifest_dir.join("target")
}

/// Clone MNN from GitHub
fn clone_mnn_from_github(dest: &std::path::Path, debug: bool) {
    if debug {
        println!("cargo:warning=mnn-sys: Cloning MNN from GitHub to {:?}", dest);
    }

    let repo_url = "https://github.com/alibaba/MNN.git";

    // Try to clone using git
    let status = Command::new("git")
        .args(["clone", "--depth", "1", repo_url, dest.to_str().unwrap()])
        .status()
        .expect("Failed to run git clone. Please install git or set MNN_SOURCE_PATH to an existing MNN source directory.");

    if !status.success() {
        panic!("Failed to clone MNN from GitHub. Please clone manually or set MNN_SOURCE_PATH.");
    }

    if debug {
        println!("cargo:warning=mnn-sys: MNN cloned successfully");
    }
}

/// Build MNN from source using CMake
#[cfg(feature = "build-from-source")]
fn build_mnn_from_source(
    source_path: &std::path::Path,
    build_dir: &std::path::Path,
    _static: bool,
    debug: bool,
) -> (PathBuf, PathBuf) {
    // Install directory is inside build_dir
    let install_dir = build_dir.join("install");

    if debug {
        println!("cargo:warning=mnn-sys: Building MNN from source");
        println!("cargo:warning=mnn-sys: Source: {:?}", source_path);
        println!("cargo:warning=mnn-sys: Build dir: {:?}", build_dir);
        println!("cargo:warning=mnn-sys: Install dir: {:?}", install_dir);
    }

    // Check if already built
    let lib_path = install_dir.join("lib").join(if cfg!(target_os = "windows") { "MNN.lib" } else { "libmnn.a" });
    if lib_path.exists() {
        if debug {
            println!("cargo:warning=mnn-sys: Using cached MNN build at {:?}", install_dir);
        }
        return (install_dir.join("lib"), install_dir.join("include"));
    }

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Determine generator and architecture based on target
    let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-pc-windows-msvc".to_string());

    let (generator, extra_args) = if cfg!(target_os = "windows") {
        // Use Visual Studio generator on Windows
        // Detect architecture from target
        let arch = if target.contains("x86_64") || target.contains("x64") {
            "x64"
        } else if target.contains("i686") || target.contains("x86") {
            "Win32"
        } else if target.contains("aarch64") || target.contains("arm64") {
            "ARM64"
        } else {
            "x64" // default
        };
        (Some("Visual Studio 17 2022"), vec!["-A", arch])
    } else {
        (None, vec![])
    };

    // Configure CMake options based on features
    let mut cmake_args = vec![
        format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()),
    ];

    // On Windows, use static runtime to match Rust
    if cfg!(target_os = "windows") {
        cmake_args.push("-DMNN_WIN_RUNTIME_MT=ON".to_string());
    }

    // Build static or shared library based on feature
    if _static {
        cmake_args.push("-DMNN_BUILD_SHARED_LIBS=OFF".to_string());
    } else {
        cmake_args.push("-DMNN_BUILD_SHARED_LIBS=ON".to_string());
    }

    // Backend options
    if cfg!(feature = "cuda") {
        cmake_args.push("-DMNN_CUDA=ON".to_string());
    }
    if cfg!(feature = "opencl") {
        cmake_args.push("-DMNN_OPENCL=ON".to_string());
    }
    if cfg!(feature = "vulkan") {
        cmake_args.push("-DMNN_VULKAN=ON".to_string());
    }
    if cfg!(feature = "metal") {
        cmake_args.push("-DMNN_METAL=ON".to_string());
    }

    // Precision options
    if cfg!(feature = "fp16") {
        cmake_args.push("-DMNN_SUPPORT_FP16=ON".to_string());
    }
    if cfg!(feature = "int8") {
        cmake_args.push("-DMNN_SUPPORT_INT8=ON".to_string());
    }
    if cfg!(feature = "quantization") {
        cmake_args.push("-DMNN_BUILD_QUANT=ON".to_string());
    }

    // Build type - always use Release for MNN to avoid debug CRT issues
    cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string());

    if debug {
        println!("cargo:warning=mnn-sys: CMake args: {:?}", cmake_args);
    }

    // Run CMake configure
    let mut cmd = Command::new("cmake");
    cmd.args(&cmake_args);

    if let Some(gen) = generator {
        cmd.args(["-G", gen]);
    }
    cmd.args(&extra_args);

    // Set output directory
    cmd.args(["-B", build_dir.to_str().unwrap()]);
    cmd.args(["-S", source_path.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to run CMake configure. Make sure cmake is installed.");

    if !status.success() {
        panic!("CMake configure failed");
    }

    // Run CMake build with parallel jobs
    let num_jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            // Default to number of CPU cores
            num_cpus::get()
        });

    let mut build_cmd = Command::new("cmake");
    build_cmd
        .args(["--build", build_dir.to_str().unwrap()])
        .args(["--config", "Release"])
        .args(["--target", "install"]);

    // Use parallel build on Windows with MSVC
    if cfg!(target_os = "windows") {
        // /m flag tells MSBuild to use parallel builds
        build_cmd.args(["--", "/m", "/p:CL_MPCount=".to_string() + &num_jobs.to_string()]);
    } else {
        // Use -j for Ninja/Make
        build_cmd.args(["--", "-j", &num_jobs.to_string()]);
    }

    if debug {
        println!("cargo:warning=mnn-sys: Build command: {:?}", build_cmd);
    }

    let build_status = build_cmd
        .status()
        .expect("Failed to run CMake build");

    if !build_status.success() {
        panic!("CMake build failed");
    }

    let lib_dir = install_dir.join("lib");
    let include_dir = install_dir.join("include");

    if debug {
        println!("cargo:warning=mnn-sys: MNN built successfully");
        println!("cargo:warning=mnn-sys: Library dir: {:?}", lib_dir);
        println!("cargo:warning=mnn-sys: Include dir: {:?}", include_dir);
    }

    (lib_dir, include_dir)
}

/// Build from source stub when cmake feature is not enabled
#[cfg(not(feature = "build-from-source"))]
fn build_mnn_from_source(
    _source_path: &std::path::Path,
    _build_dir: &std::path::Path,
    _static: bool,
    _debug: bool,
) -> (PathBuf, PathBuf) {
    panic!("'build-from-source' feature is not enabled. Enable it with --features build-from-source");
}

/// Search common library paths for MNN
fn search_common_paths(use_static: bool) {
    let lib_name = if use_static { "libmnn.a" } else { "libmnn.so" };

    // Common paths to search
    let common_paths = if cfg!(target_os = "windows") {
        vec![
            "C:\\Program Files\\MNN\\lib",
            "C:\\MNN\\lib",
            "C:\\msys64\\mingw64\\lib",
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "/usr/local/lib",
            "/usr/lib",
            "/opt/homebrew/lib",
            "/opt/local/lib",
        ]
    } else {
        vec![
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/mnn/lib",
        ]
    };

    for path in common_paths {
        let full_path = std::path::Path::new(path).join(lib_name);
        if full_path.exists() || std::path::Path::new(path).join("mnn.lib").exists() {
            println!("cargo:rustc-link-search=native={}", path);
            return;
        }
    }
}

/// Configure platform-specific settings
fn configure_platform() {
    // Windows-specific
    if cfg!(target_os = "windows") {
        // Link against required Windows libraries
        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=cudart");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }

    // Linux-specific
    if cfg!(target_os = "linux") {
        // Link against required system libraries
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");

        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=cudart");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    // macOS-specific
    if cfg!(target_os = "macos") {
        // Link against required macOS frameworks
        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        }
    }

    // Android-specific
    if cfg!(feature = "android") {
        // Android NDK provides necessary libraries
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    // iOS-specific
    if cfg!(feature = "ios") {
        // iOS frameworks
        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }
}

/// Configure backend-specific linking
fn configure_backends() {
    // CUDA backend
    if cfg!(feature = "cuda") {
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let cuda_lib = if cfg!(target_os = "windows") {
                format!("{}/lib/x64", cuda_path)
            } else {
                format!("{}/lib64", cuda_path)
            };
            println!("cargo:rustc-link-search=native={}", cuda_lib);
        }
    }
}