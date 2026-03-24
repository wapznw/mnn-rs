//! Backend configuration and management for MNN.
//!
//! This module provides types for configuring compute backends (CPU, GPU, etc.)
//! and querying backend capabilities.

/// Compute backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendType {
    /// CPU backend (always available)
    #[default]
    CPU,

    /// CUDA backend (NVIDIA GPUs)
    #[cfg(feature = "cuda")]
    Cuda,

    /// OpenCL backend (cross-platform GPU)
    #[cfg(feature = "opencl")]
    OpenCL,

    /// Vulkan backend (cross-platform GPU)
    #[cfg(feature = "vulkan")]
    Vulkan,

    /// Metal backend (Apple platforms)
    #[cfg(feature = "metal")]
    Metal,

    /// Auto-detect best available backend
    Auto,
}

impl BackendType {
    /// Convert to MNN forward type constant.
    pub(crate) fn to_mnn_type(&self) -> i32 {
        match self {
            BackendType::CPU => mnn_rs_sys::MNN_FORWARD_CPU,
            #[cfg(feature = "cuda")]
            BackendType::Cuda => mnn_rs_sys::MNN_FORWARD_CUDA,
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => mnn_rs_sys::MNN_FORWARD_OPENCL,
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => mnn_rs_sys::MNN_FORWARD_VULKAN,
            #[cfg(feature = "metal")]
            BackendType::Metal => mnn_rs_sys::MNN_FORWARD_METAL,
            BackendType::Auto => mnn_rs_sys::MNN_FORWARD_AUTO,
        }
    }

    /// Convert from MNN forward type constant.
    pub(crate) fn from_mnn_type(code: i32) -> Self {
        match code {
            #[cfg(feature = "cuda")]
            x if x == mnn_rs_sys::MNN_FORWARD_CUDA => BackendType::Cuda,
            #[cfg(feature = "opencl")]
            x if x == mnn_rs_sys::MNN_FORWARD_OPENCL => BackendType::OpenCL,
            #[cfg(feature = "vulkan")]
            x if x == mnn_rs_sys::MNN_FORWARD_VULKAN => BackendType::Vulkan,
            #[cfg(feature = "metal")]
            x if x == mnn_rs_sys::MNN_FORWARD_METAL => BackendType::Metal,
            _ => BackendType::CPU,
        }
    }

    /// Get the name of this backend.
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::CPU => "CPU",
            #[cfg(feature = "cuda")]
            BackendType::Cuda => "CUDA",
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => "OpenCL",
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => "Vulkan",
            #[cfg(feature = "metal")]
            BackendType::Metal => "Metal",
            BackendType::Auto => "Auto",
        }
    }

    /// Check if this backend is a GPU backend.
    pub fn is_gpu(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            BackendType::Cuda => true,
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => true,
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => true,
            #[cfg(feature = "metal")]
            BackendType::Metal => true,
            _ => false,
        }
    }

    /// Check if this backend is available on the current system.
    pub fn is_available(&self) -> bool {
        unsafe { mnn_rs_sys::mnn_is_backend_available(self.to_mnn_type()) != 0 }
    }

    /// Get all available backends on this system.
    pub fn available_backends() -> Vec<BackendType> {
        let mut backends = Vec::new();

        // CPU is always available
        backends.push(BackendType::CPU);

        #[cfg(feature = "cuda")]
        {
            if BackendType::Cuda.is_available() {
                backends.push(BackendType::Cuda);
            }
        }

        #[cfg(feature = "opencl")]
        {
            if BackendType::OpenCL.is_available() {
                backends.push(BackendType::OpenCL);
            }
        }

        #[cfg(feature = "vulkan")]
        {
            if BackendType::Vulkan.is_available() {
                backends.push(BackendType::Vulkan);
            }
        }

        #[cfg(feature = "metal")]
        {
            if BackendType::Metal.is_available() {
                backends.push(BackendType::Metal);
            }
        }

        backends
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Get list of available backends on this system.
pub fn available_backends() -> Vec<BackendType> {
    BackendType::available_backends()
}

/// Check if a specific backend is available.
pub fn is_backend_available(backend: BackendType) -> bool {
    backend.is_available()
}

/// Backend capabilities.
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    /// Maximum supported tensor dimensions
    pub max_tensor_dimensions: i32,

    /// Supports FP16 operations
    pub supports_fp16: bool,

    /// Supports INT8 operations
    pub supports_int8: bool,

    /// Supports BF16 operations
    pub supports_bf16: bool,
}

impl BackendCapabilities {
    /// Query capabilities for a specific backend.
    pub fn query(_backend: BackendType) -> Self {
        Self {
            max_tensor_dimensions: 8,
            supports_fp16: cfg!(feature = "fp16"),
            supports_int8: cfg!(feature = "int8"),
            supports_bf16: cfg!(feature = "bf16"),
        }
    }
}

/// Configuration for a compute backend.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// The backend type to use
    pub backend_type: BackendType,

    /// Device ID for GPU backends (default: 0)
    pub device_id: Option<i32>,

    /// Memory usage mode
    pub memory_mode: crate::config::MemoryMode,

    /// Power usage mode
    pub power_mode: crate::config::PowerMode,

    /// Precision mode
    pub precision_mode: crate::config::PrecisionMode,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::CPU,
            device_id: None,
            memory_mode: crate::config::MemoryMode::Normal,
            power_mode: crate::config::PowerMode::Normal,
            precision_mode: crate::config::PrecisionMode::Normal,
        }
    }
}

impl BackendConfig {
    /// Create a new backend config with default settings.
    pub fn new(backend_type: BackendType) -> Self {
        Self {
            backend_type,
            ..Default::default()
        }
    }

    /// Create a CPU backend config.
    pub fn cpu() -> Self {
        Self::new(BackendType::CPU)
    }

    /// Create a GPU backend config with auto-detection.
    pub fn gpu() -> Self {
        Self::new(BackendType::Auto)
    }

    /// Set the device ID.
    pub fn with_device_id(mut self, id: i32) -> Self {
        self.device_id = Some(id);
        self
    }

    /// Set the memory mode.
    pub fn with_memory_mode(mut self, mode: crate::config::MemoryMode) -> Self {
        self.memory_mode = mode;
        self
    }

    /// Set the power mode.
    pub fn with_power_mode(mut self, mode: crate::config::PowerMode) -> Self {
        self.power_mode = mode;
        self
    }

    /// Set the precision mode.
    pub fn with_precision_mode(mut self, mode: crate::config::PrecisionMode) -> Self {
        self.precision_mode = mode;
        self
    }
}

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DataType {
    /// 32-bit floating point
    #[default]
    Float32,

    /// 16-bit floating point (half precision)
    #[cfg(feature = "fp16")]
    Float16,

    /// Brain float 16
    #[cfg(feature = "bf16")]
    BFloat16,

    /// 32-bit signed integer
    Int32,

    /// 8-bit signed integer
    #[cfg(feature = "int8")]
    Int8,

    /// 8-bit unsigned integer
    UInt8,

    /// 16-bit signed integer
    Int16,

    /// 64-bit floating point
    Float64,
}

impl DataType {
    /// Get the size in bytes of this data type.
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            #[cfg(feature = "fp16")]
            DataType::Float16 => 2,
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => 2,
            DataType::Int32 => 4,
            #[cfg(feature = "int8")]
            DataType::Int8 => 1,
            DataType::UInt8 => 1,
            DataType::Int16 => 2,
            DataType::Float64 => 8,
        }
    }

    /// Get the name of this data type.
    pub fn name(&self) -> &'static str {
        match self {
            DataType::Float32 => "float32",
            #[cfg(feature = "fp16")]
            DataType::Float16 => "float16",
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => "bfloat16",
            DataType::Int32 => "int32",
            #[cfg(feature = "int8")]
            DataType::Int8 => "int8",
            DataType::UInt8 => "uint8",
            DataType::Int16 => "int16",
            DataType::Float64 => "float64",
        }
    }

    /// Check if this is a floating point type.
    pub fn is_float(&self) -> bool {
        match self {
            DataType::Float32 | DataType::Float64 => true,
            #[cfg(feature = "fp16")]
            DataType::Float16 => true,
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => true,
            _ => false,
        }
    }

    /// Check if this is an integer type.
    pub fn is_integer(&self) -> bool {
        match self {
            DataType::Int32 | DataType::Int16 | DataType::UInt8 => true,
            #[cfg(feature = "int8")]
            DataType::Int8 => true,
            _ => false,
        }
    }

    /// Check if this is a signed type.
    pub fn is_signed(&self) -> bool {
        !matches!(self, DataType::UInt8)
    }

    /// Convert to MNN type code.
    pub(crate) fn to_type_code(&self) -> i32 {
        // MNN uses halide_type_t codes: (code << 8) | bits
        match self {
            DataType::Float32 => (0 << 8) | 32,  // halide_type_float = 0
            DataType::Float64 => (0 << 8) | 64,
            DataType::Int32 => (1 << 8) | 32,    // halide_type_int = 1
            DataType::Int16 => (1 << 8) | 16,
            #[cfg(feature = "int8")]
            DataType::Int8 => (1 << 8) | 8,
            DataType::UInt8 => (2 << 8) | 8,     // halide_type_uint = 2
            #[cfg(feature = "fp16")]
            DataType::Float16 => (0 << 8) | 16,
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => (0 << 8) | 16,
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Get the MNN version string.
pub fn version() -> String {
    unsafe {
        let ptr = mnn_rs_sys::mnn_get_version();
        if ptr.is_null() {
            return String::from("unknown");
        }
        std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_name() {
        assert_eq!(BackendType::CPU.name(), "CPU");
        assert_eq!(BackendType::Auto.name(), "Auto");
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::Float64.size(), 8);
    }

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.backend_type, BackendType::CPU);
        assert!(config.device_id.is_none());
    }
}