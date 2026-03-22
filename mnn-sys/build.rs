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
//! - `<project-root>/target/mnn-build/<target-triple>/`
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

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Check if we should print debug info
    let debug_build = env::var("MNN_DEBUG_BUILD").is_ok();

    if debug_build {
        println!("cargo:warning=mnn-sys: Starting build script");
    }

    // Get target information
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let target_os = get_target_os(&target);
    let target_env = get_target_env(&target);
    let target_arch = get_target_arch(&target);

    if debug_build {
        println!("cargo:warning=mnn-sys: Target: {}", target);
        println!("cargo:warning=mnn-sys: Target OS: {}", target_os);
        println!("cargo:warning=mnn-sys: Target Env: {}", target_env);
        println!("cargo:warning=mnn-sys: Target Arch: {}", target_arch);
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
        // Format: target/mnn-build/<target-triple>/
        let build_dir = target_dir
            .join("mnn-build")
            .join(&target);

        let (lib_dir, include_dir) = build_mnn_from_source(
            &source_path,
            &build_dir,
            use_static,
            debug_build,
            &target,
            &target_os,
            &target_env,
            &target_arch,
        );
        (Some(lib_dir), Some(include_dir))
    } else {
        (mnn_lib_dir.map(PathBuf::from), mnn_include_dir.map(PathBuf::from))
    };

    // Compile the C++ wrapper
    let wrapper_dir = PathBuf::from("wrapper");
    let wrapper_cpp = wrapper_dir.join("mnn_wrapper.cpp");

    // Build the wrapper library
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&wrapper_cpp)
        .include(&wrapper_dir);

    // Add MNN include directory if available
    if let Some(ref inc_dir) = include_dir {
        if !inc_dir.as_os_str().is_empty() {
            build.include(inc_dir);
        }
    }

    // Set C++ standard and compiler flags based on TARGET (not host)
    if target_env == "msvc" {
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
        search_common_paths(use_static, &target_os);
    }

    // Link the library
    let link_type = if use_static { "static" } else { "dylib" };
    // For MinGW on Windows, the library is named libMNN.a (capital MNN)
    // For MSVC it's mnn.lib, for Unix it's libmnn.a
    if target_os == "windows" && target_env == "gnu" {
        println!("cargo:rustc-link-lib={}=MNN", link_type);
    } else {
        println!("cargo:rustc-link-lib={}=mnn", link_type);
    }

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
    configure_platform(&target_os, &target_env);

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

/// Parse target OS from target triple
fn get_target_os(target: &str) -> String {
    let parts: Vec<&str> = target.split('-').collect();
    for part in &parts {
        match *part {
            "windows" => return "windows".to_string(),
            "linux" => return "linux".to_string(),
            "macos" | "darwin" => return "macos".to_string(),
            "android" => return "android".to_string(),
            "ios" => return "ios".to_string(),
            _ => {}
        }
    }
    "unknown".to_string()
}

/// Parse target environment from target triple
fn get_target_env(target: &str) -> String {
    let parts: Vec<&str> = target.split('-').collect();
    if let Some(env) = parts.last() {
        match *env {
            "gnu" | "gnueabi" | "gnueabihf" => return "gnu".to_string(),
            "msvc" => return "msvc".to_string(),
            "android" => return "android".to_string(),
            "androideabi" => return "android".to_string(),
            _ => {}
        }
    }
    if target.contains("windows") {
        "msvc".to_string()
    } else {
        "gnu".to_string()
    }
}

/// Parse target architecture from target triple
fn get_target_arch(target: &str) -> String {
    if let Some(first) = target.split('-').next() {
        match first {
            "x86_64" | "x86-64" | "amd64" => return "x86_64".to_string(),
            "i686" | "i586" | "i386" | "x86" => return "x86".to_string(),
            "aarch64" | "arm64" => return "aarch64".to_string(),
            "arm" | "armv7" | "thumbv7" => return "arm".to_string(),
            _ => return first.to_string(),
        }
    }
    "unknown".to_string()
}

/// Get the project's target directory
fn get_target_directory() -> PathBuf {
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(target_dir);
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let mut current = out_dir.as_path();
    for _ in 0..4 {
        if let Some(parent) = current.parent() {
            current = parent;
        }
    }

    if current.file_name().map(|n| n == "target").unwrap_or(false) {
        return current.to_path_buf();
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    manifest_dir.join("target")
}

/// Clone MNN from GitHub
fn clone_mnn_from_github(dest: &std::path::Path, debug: bool) {
    if debug {
        println!("cargo:warning=mnn-sys: Cloning MNN from GitHub to {:?}", dest);
    }

    let repo_url = "https://github.com/alibaba/MNN.git";

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

/// Compute configuration hash for cache invalidation
fn compute_config_hash(
    target: &str,
    target_os: &str,
    target_env: &str,
    target_arch: &str,
    use_static: bool,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash target info
    target.hash(&mut hasher);
    target_os.hash(&mut hasher);
    target_env.hash(&mut hasher);
    target_arch.hash(&mut hasher);
    use_static.hash(&mut hasher);

    // Hash features
    cfg!(feature = "cuda").hash(&mut hasher);
    cfg!(feature = "opencl").hash(&mut hasher);
    cfg!(feature = "vulkan").hash(&mut hasher);
    cfg!(feature = "metal").hash(&mut hasher);
    cfg!(feature = "fp16").hash(&mut hasher);
    cfg!(feature = "int8").hash(&mut hasher);
    cfg!(feature = "quantization").hash(&mut hasher);
    cfg!(feature = "sse").hash(&mut hasher);
    cfg!(feature = "avx2").hash(&mut hasher);
    cfg!(feature = "avx512").hash(&mut hasher);

    hasher.finish()
}

/// Build MNN from source using CMake
#[cfg(feature = "build-from-source")]
fn build_mnn_from_source(
    source_path: &std::path::Path,
    build_dir: &std::path::Path,
    _static: bool,
    debug: bool,
    target: &str,
    target_os: &str,
    target_env: &str,
    target_arch: &str,
) -> (PathBuf, PathBuf) {
    // Install directory is inside build_dir
    let install_dir = build_dir.join("install");

    if debug {
        println!("cargo:warning=mnn-sys: Building MNN from source");
        println!("cargo:warning=mnn-sys: Source: {:?}", source_path);
        println!("cargo:warning=mnn-sys: Build dir: {:?}", build_dir);
        println!("cargo:warning=mnn-sys: Install dir: {:?}", install_dir);
    }

    // Compute configuration hash
    let config_hash = compute_config_hash(target, target_os, target_env, target_arch, _static);
    let config_file = build_dir.join(".mnn-build-config");

    // Check if we need to rebuild based on configuration changes
    let need_rebuild = if config_file.exists() {
        if let Ok(stored_hash) = std::fs::read_to_string(&config_file) {
            let stored: u64 = stored_hash.trim().parse().unwrap_or(0);
            stored != config_hash
        } else {
            true
        }
    } else {
        true
    };

    // Check if already built with same configuration
    let lib_name = if target_os == "windows" {
        if target_env == "gnu" {
            "libMNN.a"
        } else {
            "mnn.lib"
        }
    } else {
        "libmnn.a"
    };
    let lib_path = install_dir.join("lib").join(lib_name);

    if lib_path.exists() && !need_rebuild {
        if debug {
            println!("cargo:warning=mnn-sys: Using cached MNN build at {:?}", install_dir);
        }
        return (install_dir.join("lib"), install_dir.join("include"));
    }

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Store new configuration hash
    std::fs::write(&config_file, config_hash.to_string()).expect("Failed to write config file");

    // Determine generator and architecture based on target
    let (generator, extra_args) = if target_os == "windows" {
        if target_env == "msvc" {
            let arch = match target_arch {
                "x86_64" => "x64",
                "x86" => "Win32",
                "aarch64" => "ARM64",
                _ => "x64",
            };
            (Some("Visual Studio 17 2022"), vec!["-A", arch])
        } else {
            (Some("MinGW Makefiles"), vec![])
        }
    } else if target_os == "linux" {
        (None, vec![])
    } else if target_os == "macos" {
        (Some("Xcode"), vec![])
    } else {
        (None, vec![])
    };

    // Configure CMake options based on features
    let mut cmake_args = vec![
        format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()),
    ];

    // On Windows MSVC, use static runtime to match Rust
    if target_os == "windows" && target_env == "msvc" {
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

    // x86 SIMD options (controlled by features)
    // Default: enabled for x86_64, disabled for x86 (32-bit)
    let use_sse = cfg!(feature = "sse") || (target_arch == "x86_64" && !cfg!(feature = "sse") && !cfg!(feature = "avx2"));
    let use_avx2 = cfg!(feature = "avx2");
    let use_avx512 = cfg!(feature = "avx512");

    // For 32-bit x86, disable all SIMD by default unless explicitly enabled
    if target_arch == "x86" {
        cmake_args.push(format!("-DMNN_USE_SSE={}", if cfg!(feature = "sse") { "ON" } else { "OFF" }));
        cmake_args.push(format!("-DMNN_AVX2={}", if cfg!(feature = "avx2") { "ON" } else { "OFF" }));
        cmake_args.push(format!("-DMNN_AVX512={}", if cfg!(feature = "avx512") { "ON" } else { "OFF" }));
    } else if target_arch == "x86_64" {
        cmake_args.push(format!("-DMNN_USE_SSE={}", if use_sse || use_avx2 || use_avx512 { "ON" } else { "OFF" }));
        cmake_args.push(format!("-DMNN_AVX2={}", if use_avx2 || use_avx512 { "ON" } else { "OFF" }));
        cmake_args.push(format!("-DMNN_AVX512={}", if use_avx512 { "ON" } else { "OFF" }));
    }

    // Build type
    cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string());

    // Cross-compilation support
    if target_os == "windows" && target_env == "gnu" {
        if target_arch == "x86" {
            cmake_args.push(String::from("-DCMAKE_C_COMPILER=i686-w64-mingw32-gcc.exe"));
            cmake_args.push(String::from("-DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++.exe"));
            cmake_args.push(String::from("-DCMAKE_SYSTEM_NAME=Windows"));
        } else if target_arch == "x86_64" {
            cmake_args.push(String::from("-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc.exe"));
            cmake_args.push(String::from("-DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++.exe"));
            cmake_args.push(String::from("-DCMAKE_SYSTEM_NAME=Windows"));
        }
    }

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
        .unwrap_or_else(num_cpus::get);

    let mut build_cmd = Command::new("cmake");
    build_cmd
        .args(["--build", build_dir.to_str().unwrap()])
        .args(["--config", "Release"])
        .args(["--target", "install"]);

    if target_os == "windows" && target_env == "msvc" {
        build_cmd.args(["--", "/m", &format!("/p:CL_MPCount={}", num_jobs)]);
    } else {
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
    _target: &str,
    _target_os: &str,
    _target_env: &str,
    _target_arch: &str,
) -> (PathBuf, PathBuf) {
    panic!("'build-from-source' feature is not enabled. Enable it with --features build-from-source");
}

/// Search common library paths for MNN
fn search_common_paths(use_static: bool, target_os: &str) {
    let lib_name = if use_static { "libmnn.a" } else { "libmnn.so" };

    let common_paths = if target_os == "windows" {
        vec![
            "C:\\Program Files\\MNN\\lib",
            "C:\\MNN\\lib",
            "C:\\msys64\\mingw64\\lib",
            "C:\\msys64\\mingw32\\lib",
        ]
    } else if target_os == "macos" {
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
fn configure_platform(target_os: &str, target_env: &str) {
    if target_os == "windows" {
        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=cudart");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }

        if target_env == "gnu" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }

    if target_os == "linux" {
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

    if target_os == "macos" {
        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        }
    }

    if cfg!(feature = "android") || target_os == "android" {
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "ios") || target_os == "ios" {
        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }
}

/// Configure backend-specific linking
fn configure_backends() {
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