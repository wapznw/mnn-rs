//! Build script for mnn-sys
//!
//! This script handles linking to the MNN library, supporting:
//! - Static and dynamic linking
//! - Multiple backends (CPU, CUDA, OpenCL, Vulkan, Metal)
//! - Prebuilt binary download (default)
//! - Build from source or use pre-installed library
//! - Cross-platform builds
//!
//! # Build Modes (in priority order)
//!
//! 1. **use-prebuilt** (default): Download prebuilt binaries from GitHub Releases
//! 2. **build-from-source**: Build MNN locally using CMake
//! 3. **system-mnn**: Use system-installed MNN library
//!
//! ## Prebuilt Binaries
//!
//! Prebuilt binaries are automatically downloaded from GitHub Releases.
//! Set `MNN_PREBUILT_URL` to use a custom download URL.
//!
//! ## Building from Source
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
//! # Default: download prebuilt binaries
//! cargo build
//!
//! # Build from source instead
//! cargo build --features build-from-source --no-default-features
//!
//! # Use existing MNN source
//! MNN_SOURCE_PATH=/path/to/mnn cargo build --features build-from-source
//!
//! # Use pre-built MNN
//! MNN_LIB_DIR=/path/to/mnn/lib MNN_INCLUDE_DIR=/path/to/mnn/include cargo build
//! ```

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;

/// MNN version for prebuilt binaries - sync with crate version
const MNN_VERSION: &str = "0.1.1";
/// GitHub repository for prebuilt downloads
const GITHUB_REPO: &str = "wapznw/mnn-rs";

/// Get the NDK host tag based on the current host platform
fn get_ndk_host_tag() -> &'static str {
    // Check HOST environment variable first (set by Cargo)
    if let Ok(host) = env::var("HOST") {
        if host.contains("windows") {
            return "windows-x86_64";
        } else if host.contains("apple-darwin") {
            return "darwin-x86_64";
        } else if host.contains("linux") {
            return "linux-x86_64";
        }
    }

    // Fallback to checking the OS
    if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else {
        "linux-x86_64"
    }
}

/// Get the prebuilt variant name based on enabled features
fn get_prebuilt_variant() -> &'static str {
    // Priority: cuda > opencl > vulkan > cpu (default)
    // Metal is always included in macOS/iOS builds, no separate variant
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "opencl") {
        "opencl"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else {
        ""
    }
}

/// Download and extract prebuilt MNN binaries
#[cfg(feature = "use-prebuilt")]
fn download_prebuilt(target: &str, out_dir: &std::path::Path, debug: bool) -> Option<(PathBuf, PathBuf)> {
    use flate2::read::GzDecoder;

    let variant = get_prebuilt_variant();
    let prebuilt_dir = if variant.is_empty() {
        out_dir.join("mnn-prebuilt")
    } else {
        out_dir.join(format!("mnn-prebuilt-{}", variant))
    };
    let lib_dir = prebuilt_dir.join("lib");
    let include_dir = prebuilt_dir.join("include");

    // Check if already downloaded and valid
    if lib_dir.exists() && include_dir.exists() {
        // Verify library file exists
        let lib_exists = if cfg!(target_os = "windows") {
            lib_dir.join("mnn.lib").exists() || lib_dir.join("MNN.lib").exists()
        } else {
            lib_dir.join("libMNN.a").exists()
        };
        if lib_exists {
            if debug {
                println!("cargo:warning=mnn-sys: Using cached prebuilt at {:?}", prebuilt_dir);
            }
            return Some((lib_dir, include_dir));
        }
    }

    // Construct download URL with variant suffix
    let archive_name = if variant.is_empty() {
        format!("mnn-{}.tar.gz", target)
    } else {
        format!("mnn-{}-{}.tar.gz", target, variant)
    };

    let url = env::var("MNN_PREBUILT_URL").unwrap_or_else(|_| {
        format!(
            "https://github.com/{}/releases/download/v{}/{}",
            GITHUB_REPO, MNN_VERSION, archive_name
        )
    });

    println!("cargo:warning=mnn-sys: Downloading prebuilt MNN from: {}", url);

    // Download the archive
    let response = match ureq::get(&url).call() {
        Ok(r) => r,
        Err(e) => {
            println!("cargo:warning=mnn-sys: Failed to download prebuilt: {}", e);
            println!("cargo:warning=mnn-sys: Falling back to build-from-source or system-mnn");
            return None;
        }
    };

    if response.status() != 200 {
        println!("cargo:warning=mnn-sys: Download failed with HTTP {}", response.status());
        return None;
    }

    // Create prebuilt directory
    fs::create_dir_all(&prebuilt_dir).ok()?;

    // Extract the archive
    let reader = response.into_reader();
    let mut decoder = GzDecoder::new(reader);
    let mut archive = tar::Archive::new(&mut decoder);

    if let Err(e) = archive.unpack(&prebuilt_dir) {
        println!("cargo:warning=mnn-sys: Failed to extract archive: {}", e);
        // Clean up partial extraction
        let _ = fs::remove_dir_all(&prebuilt_dir);
        return None;
    }

    // Verify extraction was successful
    if !lib_dir.exists() {
        // Try to find extracted files (archive might not have top-level directory)
        if let Ok(entries) = fs::read_dir(&prebuilt_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Check if this directory contains lib/include
                    if path.join("lib").exists() {
                        return Some((path.join("lib"), path.join("include")));
                    }
                }
            }
        }
    }

    if debug {
        println!("cargo:warning=mnn-sys: Prebuilt extracted to {:?}", prebuilt_dir);
    }

    Some((lib_dir, include_dir))
}

/// Stub for when use-prebuilt feature is not enabled
#[cfg(not(feature = "use-prebuilt"))]
fn download_prebuilt(_target: &str, _out_dir: &std::path::Path, _debug: bool) -> Option<(PathBuf, PathBuf)> {
    None
}

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
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

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
    let use_prebuilt = cfg!(feature = "use-prebuilt");

    // Determine linking mode
    let static_link = cfg!(feature = "static");
    let dynamic_link = cfg!(feature = "dynamic");

    if static_link && dynamic_link {
        panic!("Cannot enable both 'static' and 'dynamic' features simultaneously");
    }

    // If no link mode specified, default to static
    let use_static = !dynamic_link;

    // Priority: env vars -> prebuilt -> build-from-source -> system-mnn
    // This allows testing with local artifacts without downloading
    let (lib_dir, include_dir) = if mnn_lib_dir.is_some() && mnn_include_dir.is_some() {
        // Use environment variables if both are set (highest priority for testing)
        if debug_build {
            println!("cargo:warning=mnn-sys: Using MNN_LIB_DIR and MNN_INCLUDE_DIR from environment");
        }
        (
            mnn_lib_dir.as_ref().map(|p| PathBuf::from(p)),
            mnn_include_dir.as_ref().map(|p| PathBuf::from(p)),
        )
    } else if use_prebuilt && !build_from_source {
        // Try to download prebuilt binaries
        if let Some((lib, inc)) = download_prebuilt(&target, &out_dir, debug_build) {
            if debug_build {
                println!("cargo:warning=mnn-sys: Using prebuilt binaries");
            }
            (Some(lib), Some(inc))
        } else {
            // Fall back to build-from-source or env vars
            if debug_build {
                println!("cargo:warning=mnn-sys: Prebuilt download failed, trying fallback");
            }
            get_lib_and_include_dirs(
                build_from_source,
                &mnn_source_path,
                &mnn_lib_dir,
                &mnn_include_dir,
                use_static,
                debug_build,
                &target,
                &target_os,
                &target_env,
                &target_arch,
            )
        }
    } else if build_from_source {
        get_lib_and_include_dirs(
            true,
            &mnn_source_path,
            &mnn_lib_dir,
            &mnn_include_dir,
            use_static,
            debug_build,
            &target,
            &target_os,
            &target_env,
            &target_arch,
        )
    } else {
        // Use env vars or system library
        (
            mnn_lib_dir.map(PathBuf::from),
            mnn_include_dir.map(PathBuf::from),
        )
    };

    // Compile the C++ wrapper
    let wrapper_dir = PathBuf::from("wrapper");
    let wrapper_cpp = wrapper_dir.join("mnn_wrapper.cpp");

    // Check if we have MNN headers available
    if include_dir.is_none() {
        panic!(
            "MNN headers not found!\n\
            \n\
            Please do one of the following:\n\
            1. Use default features to download prebuilt binaries:\n\
               cargo build\n\
            \n\
            2. Enable 'build-from-source' feature to build locally:\n\
               cargo build --features build-from-source --no-default-features\n\
            \n\
            3. Set MNN_INCLUDE_DIR and MNN_LIB_DIR environment variables:\n\
               set MNN_INCLUDE_DIR=/path/to/mnn/include\n\
               set MNN_LIB_DIR=/path/to/mnn/lib\n\
            \n\
            4. Set MNN_SOURCE_PATH to use existing MNN source:\n\
               set MNN_SOURCE_PATH=/path/to/mnn/source\n\
               cargo build --features build-from-source --no-default-features"
        );
    }

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

    // Configure Android NDK compiler for cross-compilation
    if target_os == "android" {
        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .expect("ANDROID_NDK_HOME or NDK_HOME must be set for Android builds");

        let ndk_path = std::path::Path::new(&ndk_home);

        // Android API level - must match what MNN was built with
        let api_level = 24;

        // Determine toolchain path and prefix based on target
        let (tool_prefix, target_triple) = match &target_arch[..] {
            "aarch64" => ("aarch64-linux-android", "aarch64-linux-android"),
            "armv7" | "arm" => ("arm-linux-androideabi", "armv7-linux-androideabi"),
            "x86_64" => ("x86_64-linux-android", "x86_64-linux-android"),
            "x86" => ("i686-linux-android", "i686-linux-android"),
            _ => panic!("Unsupported Android architecture: {}", target_arch),
        };

        // Find NDK toolchain directory
        let host_tag = get_ndk_host_tag();
        let toolchain_bin = ndk_path
            .join("toolchains")
            .join("llvm")
            .join("prebuilt")
            .join(host_tag)
            .join("bin");

        if !toolchain_bin.exists() {
            panic!(
                "Android NDK toolchain not found.\n\
                Expected at: {:?}\n\
                Make sure ANDROID_NDK_HOME is set correctly and NDK is installed.",
                toolchain_bin
            );
        }

        let exe_suffix = if cfg!(target_os = "windows") { ".exe" } else { "" };

        // Set AR tool path - use llvm-ar from NDK
        let ar_path = toolchain_bin.join(format!("llvm-ar{}", exe_suffix));
        if ar_path.exists() {
            // Set for cc-rs to find the archiver
            build.archiver(&ar_path);
            if debug_build {
                println!("cargo:warning=mnn-sys: Using archiver: {:?}", ar_path);
            }
        }

        // Use clang with proper target and API level
        let clang_path = toolchain_bin.join(format!("clang++{}", exe_suffix));
        if clang_path.exists() {
            build.compiler(&clang_path);
            // Set target with API level
            build.flag(format!("--target={}{}", target_triple, api_level));
            build.flag(format!("--sysroot={}", ndk_path.join("toolchains").join("llvm").join("prebuilt").join(host_tag).join("sysroot").display()));
        } else {
            panic!(
                "Could not find clang++ in NDK.\n\
                Expected at: {:?}",
                clang_path
            );
        }

        if debug_build {
            println!("cargo:warning=mnn-sys: Android NDK: {:?}", ndk_path);
            println!("cargo:warning=mnn-sys: Android target: {}{}", target_triple, api_level);
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
    } else {
        // Try common library paths (for system-mnn mode)
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

    // Print help message if library not found
    if lib_dir.is_none() {
        println!("cargo:warning=mnn-sys: MNN library not found in MNN_LIB_DIR or common paths");
        println!("cargo:warning=mnn-sys: Please do one of the following:");
        println!("cargo:warning=  1. Use default features to download prebuilt binaries: cargo build");
        println!("cargo:warning=  2. Set MNN_LIB_DIR to the directory containing libmnn.a or mnn.lib");
        println!("cargo:warning=  3. Use --features build-from-source to build MNN locally");
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
    println!("cargo:rerun-if-env-changed=MNN_PREBUILT_URL");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=NDK_HOME");

    // Re-run if wrapper changes
    println!("cargo:rerun-if-changed=wrapper/mnn_wrapper.h");
    println!("cargo:rerun-if-changed=wrapper/mnn_wrapper.cpp");

    if debug_build {
        println!("cargo:warning=mnn-sys: Build script completed successfully");
    }
}

/// Parse target OS from target triple
fn get_target_os(target: &str) -> String {
    // Check for Android first (e.g., aarch64-linux-android)
    // Android targets have "android" in the triple, but also contain "linux"
    if target.contains("android") {
        return "android".to_string();
    }
    if target.contains("ios") {
        return "ios".to_string();
    }

    let parts: Vec<&str> = target.split('-').collect();
    for part in &parts {
        match *part {
            "windows" => return "windows".to_string(),
            "linux" => return "linux".to_string(),
            "macos" | "darwin" => return "macos".to_string(),
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

/// Get library and include directories, with fallback to build-from-source
fn get_lib_and_include_dirs(
    build_from_source: bool,
    mnn_source_path: &Option<String>,
    mnn_lib_dir: &Option<String>,
    mnn_include_dir: &Option<String>,
    use_static: bool,
    debug: bool,
    target: &str,
    target_os: &str,
    target_env: &str,
    target_arch: &str,
) -> (Option<PathBuf>, Option<PathBuf>) {
    if build_from_source {
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
                    clone_mnn_from_github(&shared_source, debug);
                } else if debug {
                    println!("cargo:warning=mnn-sys: Using cached MNN source at {:?}", shared_source);
                }
                shared_source
            }
        };

        // Build directory: per-target to support cross-compilation
        // Format: target/mnn-build/<target-triple>/
        let build_dir = target_dir
            .join("mnn-build")
            .join(target);

        let (lib_dir, include_dir) = build_mnn_from_source(
            &source_path,
            &build_dir,
            use_static,
            debug,
            target,
            target_os,
            target_env,
            target_arch,
        );
        (Some(lib_dir), Some(include_dir))
    } else {
        // Use environment variables
        (
            mnn_lib_dir.as_ref().map(PathBuf::from),
            mnn_include_dir.as_ref().map(PathBuf::from),
        )
    }
}

/// Compute configuration hash for cache invalidation
#[cfg(feature = "build-from-source")]
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
    let (generator, extra_args) = if target_os == "android" {
        // Android requires Ninja build system
        let ninja_available = Command::new("ninja").arg("--version").output().is_ok();
        if !ninja_available {
            panic!(
                "Ninja build system is required for Android builds.\n\
                Install it with:\n\
                - Windows: choco install ninja\n\
                - macOS: brew install ninja\n\
                - Linux: apt install ninja-build"
            );
        }
        (Some("Ninja"), vec![])
    } else if target_os == "ios" {
        // iOS requires Ninja or Xcode
        let ninja_available = Command::new("ninja").arg("--version").output().is_ok();
        if ninja_available {
            (Some("Ninja"), vec![])
        } else {
            (Some("Xcode"), vec![])
        }
    } else if target_os == "windows" {
        if target_env == "msvc" {
            let arch = match &target_arch[..] {
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
        // Linux defaults to Unix Makefiles or Ninja
        (None, vec![])
    } else if target_os == "macos" {
        // macOS: prefer Ninja for consistency, fallback to default
        let ninja_available = Command::new("ninja").arg("--version").output().is_ok();
        if ninja_available {
            (Some("Ninja"), vec![])
        } else {
            // Let CMake choose (usually Unix Makefiles on macOS)
            (None, vec![])
        }
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

    // Android NDK cross-compilation support
    if target_os == "android" {
        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .expect("ANDROID_NDK_HOME or NDK_HOME must be set for Android builds");

        let ndk_path = std::path::Path::new(&ndk_home);

        // Determine Android ABI and platform
        let (abi, platform, _toolchain_name) = match &target_arch[..] {
            "aarch64" => ("arm64-v8a", "android-24", "aarch64-linux-android"),
            "armv7" | "arm" => ("armeabi-v7a", "android-24", "arm-linux-androideabi"),
            "x86_64" => ("x86_64", "android-24", "x86_64-linux-android"),
            "x86" => ("x86", "android-24", "i686-linux-android"),
            _ => panic!("Unsupported Android architecture: {}", target_arch),
        };

        // Use NDK toolchain file
        let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");
        if !toolchain_file.exists() {
            panic!("Android NDK toolchain file not found at {:?}", toolchain_file);
        }

        cmake_args.push(format!("-DCMAKE_TOOLCHAIN_FILE={}", toolchain_file.display()));
        cmake_args.push(format!("-DANDROID_ABI={}", abi));
        cmake_args.push(format!("-DANDROID_PLATFORM={}", platform));
        cmake_args.push("-DANDROID_STL=c++_static".to_string());
        cmake_args.push("-DCMAKE_SYSTEM_NAME=Android".to_string());

        if debug {
            println!("cargo:warning=mnn-sys: Android NDK: {:?}", ndk_path);
            println!("cargo:warning=mnn-sys: Android ABI: {}", abi);
            println!("cargo:warning=mnn-sys: Android Platform: {}", platform);
        }

        // Disable KleidiAI on Android as it has linking issues with stdout/stderr
        cmake_args.push("-DMNN_KLEIDIAI=OFF".to_string());
    }

    // iOS cross-compilation support
    if target_os == "ios" {
        // Determine iOS platform and architecture
        let (platform, arch, sdk_name) = match target {
            t if t.starts_with("aarch64-apple-ios") => ("OS64", "arm64", "iphoneos"),
            t if t.starts_with("aarch64-apple-ios-sim") => ("SIMULATORARM64", "arm64", "iphonesimulator"),
            t if t.starts_with("x86_64-apple-ios") => ("SIMULATOR64", "x86_64", "iphonesimulator"),
            t if t.starts_with("i386-apple-ios") || t.starts_with("i686-apple-ios") => ("SIMULATOR", "i386", "iphonesimulator"),
            t if t.starts_with("armv7-apple-ios") => ("OS", "armv7", "iphoneos"),
            t if t.starts_with("armv7s-apple-ios") => ("OS", "armv7s", "iphoneos"),
            _ => panic!("Unsupported iOS target: {}", target),
        };

        // Set CMake system name
        cmake_args.push("-DCMAKE_SYSTEM_NAME=iOS".to_string());
        cmake_args.push(format!("-DCMAKE_OSX_ARCHITECTURES={}", arch));
        cmake_args.push(format!("-DPLATFORM={}", platform));

        // Enable Metal backend by default for iOS
        if !cfg!(feature = "metal") {
            cmake_args.push("-DMNN_METAL=ON".to_string());
        }

        if debug {
            println!("cargo:warning=mnn-sys: iOS Platform: {}", platform);
            println!("cargo:warning=mnn-sys: iOS Arch: {}", arch);
            println!("cargo:warning=mnn-sys: iOS SDK: {}", sdk_name);
        }
    }

    if debug {
        println!("cargo:warning=mnn-sys: CMake args: {:?}", cmake_args);
        println!("cargo:warning=mnn-sys: Generator: {:?}", generator);
    }

    // Run CMake configure
    let mut cmd = Command::new("cmake");

    // Pass generator BEFORE other args (important for Android/Ninja)
    if let Some(gen) = generator {
        cmd.args(["-G", gen]);
    }

    cmd.args(&cmake_args);
    cmd.args(&extra_args);

    cmd.args(["-B", build_dir.to_str().unwrap()]);
    cmd.args(["-S", source_path.to_str().unwrap()]);

    if debug {
        println!("cargo:warning=mnn-sys: CMake configure command: {:?}", cmd);
    }

    let status = cmd.status().expect("Failed to run CMake configure. Make sure cmake is installed.");

    if !status.success() {
        panic!("CMake configure failed");
    }

    // Run CMake build with parallel jobs
    let num_jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus::get);

    // First, build the library
    let mut build_cmd = Command::new("cmake");
    build_cmd
        .args(["--build", build_dir.to_str().unwrap()])
        .args(["--config", "Release"]);

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

    // Then, install the library
    let mut install_cmd = Command::new("cmake");
    install_cmd
        .args(["--install", build_dir.to_str().unwrap()])
        .args(["--config", "Release"]);

    if debug {
        println!("cargo:warning=mnn-sys: Install command: {:?}", install_cmd);
    }

    let install_status = install_cmd
        .status()
        .expect("Failed to run CMake install");

    if !install_status.success() {
        // cmake --install might fail on some platforms, try manual installation
        if debug {
            println!("cargo:warning=mnn-sys: cmake --install failed, trying manual installation");
        }
    }

    // Check if install directory was created
    let installed_lib = install_dir.join("lib").join(if target_os == "windows" {
        if target_env == "gnu" { "libMNN.a" } else { "mnn.lib" }
    } else {
        "libmnn.a"
    });

    let (lib_dir, include_dir) = if installed_lib.exists() {
        // cmake --install worked
        (install_dir.join("lib"), install_dir.join("include"))
    } else {
        // Manual fallback: use build directory and source include
        if debug {
            println!("cargo:warning=mnn-sys: Using fallback paths for lib and include");
        }

        // Create install directories
        std::fs::create_dir_all(install_dir.join("lib")).ok();
        std::fs::create_dir_all(install_dir.join("include")).ok();

        // Copy library if it exists in build dir
        let lib_in_build = build_dir.join(if target_os == "windows" {
            if target_env == "gnu" { "libMNN.a" } else { "Release/mnn.lib" }
        } else {
            "libMNN.a"
        });

        if lib_in_build.exists() {
            let dest = install_dir.join("lib").join(installed_lib.file_name().unwrap());
            if !dest.exists() {
                if let Err(e) = std::fs::copy(&lib_in_build, &dest) {
                    if debug {
                        println!("cargo:warning=mnn-sys: Failed to copy library: {}", e);
                    }
                }
            }
        }

        // Use source include directory directly (most reliable)
        let src_include = source_path.join("include");
        if src_include.exists() {
            (install_dir.join("lib"), src_include)
        } else {
            (install_dir.join("lib"), install_dir.join("include"))
        }
    };

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
        // Android requires explicit linking to C++ standard library
        println!("cargo:rustc-link-lib=c++_static");
        println!("cargo:rustc-link-lib=log"); // Android logging

        // Link libc for stdout/stderr symbols (needed by KleidiAI)
        println!("cargo:rustc-link-lib=c");

        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "ios") || target_os == "ios" {
        // iOS always uses Metal by default
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
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