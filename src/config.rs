//! Configuration types for MNN sessions and backends.
//!
//! This module provides configuration structures for creating interpreter
//! sessions with specific backend settings, thread counts, and data formats.

use crate::backend::{BackendConfig, BackendType};

/// Data format for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DataFormat {
    /// NHWC format: (batch, height, width, channels)
    /// Common in TensorFlow models
    #[default]
    Nhwc,

    /// NCHW format: (batch, channels, height, width)
    /// Common in PyTorch/ONNX models
    Nchw,

    /// NC4HW4 format: Optimized format for GPU backends
    Nc4hw4,
}

impl DataFormat {
    /// Get the name of this format
    pub fn name(&self) -> &'static str {
        match self {
            DataFormat::Nhwc => "NHWC",
            DataFormat::Nchw => "NCHW",
            DataFormat::Nc4hw4 => "NC4HW4",
        }
    }
}

/// Memory usage mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryMode {
    /// Normal memory usage (balanced)
    #[default]
    Normal,

    /// Low memory usage (may impact performance)
    Low,

    /// High memory usage for better performance
    High,
}

/// Power usage mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PowerMode {
    /// Normal power usage (balanced)
    #[default]
    Normal,

    /// Low power mode (may impact performance)
    Low,

    /// High power mode for maximum performance
    High,
}

/// Precision mode for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrecisionMode {
    /// Normal precision (default)
    #[default]
    Normal,

    /// Low precision (faster, may reduce accuracy)
    Low,

    /// High precision
    High,

    /// Low precision with BF16
    LowBf16,
}

/// Schedule configuration for creating sessions.
///
/// This configuration determines how MNN will execute the model,
/// including backend selection, thread count, and optimization settings.
#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    /// Backend configuration
    pub backend_config: BackendConfig,

    /// Number of threads for CPU backend (default: 4)
    pub num_threads: u32,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            backend_config: BackendConfig::default(),
            num_threads: 4,
        }
    }
}

impl ScheduleConfig {
    /// Create a new schedule config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a schedule config for CPU backend.
    pub fn cpu() -> Self {
        Self::default()
    }

    /// Create a schedule config for a specific backend type.
    pub fn with_backend(backend: BackendType) -> Self {
        Self {
            backend_config: BackendConfig::new(backend),
            ..Default::default()
        }
    }

    /// Set the backend type.
    pub fn backend(mut self, backend: BackendType) -> Self {
        self.backend_config.backend_type = backend;
        self
    }

    /// Set the number of threads for CPU backend.
    pub fn num_threads(mut self, threads: u32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the memory mode.
    pub fn memory_mode(mut self, mode: MemoryMode) -> Self {
        self.backend_config.memory_mode = mode;
        self
    }

    /// Set the power mode.
    pub fn power_mode(mut self, mode: PowerMode) -> Self {
        self.backend_config.power_mode = mode;
        self
    }

    /// Set the precision mode.
    pub fn precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.backend_config.precision_mode = mode;
        self
    }

    /// Set the device ID for GPU backends.
    pub fn device_id(mut self, id: i32) -> Self {
        self.backend_config.device_id = Some(id);
        self
    }
}

/// Builder for creating schedule configurations.
///
/// Provides a fluent interface for constructing [`ScheduleConfig`].
#[derive(Debug, Default)]
pub struct ScheduleConfigBuilder {
    config: ScheduleConfig,
}

impl ScheduleConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the backend type.
    pub fn backend(mut self, backend: BackendType) -> Self {
        self.config.backend_config.backend_type = backend;
        self
    }

    /// Set the number of threads.
    pub fn num_threads(mut self, threads: u32) -> Self {
        self.config.num_threads = threads;
        self
    }

    /// Set memory mode.
    pub fn memory_mode(mut self, mode: MemoryMode) -> Self {
        self.config.backend_config.memory_mode = mode;
        self
    }

    /// Set power mode.
    pub fn power_mode(mut self, mode: PowerMode) -> Self {
        self.config.backend_config.power_mode = mode;
        self
    }

    /// Set precision mode.
    pub fn precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.config.backend_config.precision_mode = mode;
        self
    }

    /// Set device ID.
    pub fn device_id(mut self, id: i32) -> Self {
        self.config.backend_config.device_id = Some(id);
        self
    }

    /// Build the schedule config.
    pub fn build(self) -> ScheduleConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ScheduleConfig::default();
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_builder() {
        let config = ScheduleConfigBuilder::new()
            .backend(BackendType::CPU)
            .num_threads(8)
            .memory_mode(MemoryMode::Low)
            .precision_mode(PrecisionMode::High)
            .build();

        assert_eq!(config.num_threads, 8);
        assert_eq!(config.backend_config.memory_mode, MemoryMode::Low);
        assert_eq!(config.backend_config.precision_mode, PrecisionMode::High);
    }
}