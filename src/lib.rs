//! # MNN Rust Bindings
//!
//! Safe Rust bindings for Alibaba's MNN (Mobile Neural Network) inference engine.
//!
//! MNN is a highly efficient and lightweight deep learning inference framework.
//! This crate provides idiomatic Rust bindings for running inference with MNN.
//!
//! ## Features
//!
//! - **Safe API**: All MNN operations are wrapped in safe Rust types
//! - **Multiple Backends**: CPU, CUDA, OpenCL, Vulkan, Metal support
//! - **Async Support**: Optional async API using tokio
//! - **Cross-Platform**: Windows, Linux, macOS, Android, iOS support
//!
//! ## Quick Start
//!
//! ```no_run
//! use mnn_rs::{Interpreter, ScheduleConfig, BackendType};
//!
//! // Load a model
//! let interpreter = Interpreter::from_file("model.mnn")?;
//!
//! // Create a session
//! let config = ScheduleConfig::new()
//!     .backend(BackendType::CPU)
//!     .num_threads(4);
//!
//! let mut session = interpreter.create_session(config)?;
//!
//! // Get input tensor
//! let input = session.get_input(None)?;
//!
//! // Fill input with data (example)
//! // input.write(&my_data)?;
//!
//! // Run inference
//! session.run()?;
//!
//! // Get output
//! let output = session.get_output(None)?;
//!
//! # Ok::<(), mnn_rs::MnnError>(())
//! ```
//!
//! ## Backend Configuration
//!
//! ```no_run
//! use mnn_rs::{ScheduleConfig, BackendType, MemoryMode, PrecisionMode};
//!
//! // CPU with custom settings
//! let cpu_config = ScheduleConfig::new()
//!     .backend(BackendType::CPU)
//!     .num_threads(8)
//!     .memory_mode(MemoryMode::Low);
//!
//! // GPU (auto-detect best backend)
//! let gpu_config = ScheduleConfig::new()
//!     .backend(BackendType::Auto)
//!     .precision_mode(PrecisionMode::Low);
//!
//! # Ok::<(), mnn_rs::MnnError>(())
//! ```
//!
//! ## Async API (requires "async" feature)
//!
//! ```ignore
//! use mnn_rs::{AsyncInterpreter, ScheduleConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mnn_rs::MnnError> {
//!     let interpreter = AsyncInterpreter::from_file("model.mnn").await?;
//!     let mut session = interpreter.create_session(ScheduleConfig::default()).await?;
//!
//!     session.run_async().await?;
//!
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub use mnn_rs_sys;

// Core modules
mod error;
pub use error::{MnnError, MnnResult};

mod backend;
pub use backend::{
    available_backends, is_backend_available, version, BackendCapabilities, BackendConfig,
    BackendType, DataType,
};

mod config;
pub use config::{DataFormat, MemoryMode, PowerMode, PrecisionMode, ScheduleConfig, ScheduleConfigBuilder, SessionMode};

mod tensor;
pub use tensor::{Tensor, TensorData, TensorInfo, TensorView};

#[cfg(feature = "image-process")]
mod image_process;
#[cfg(feature = "image-process")]
pub use image_process::{
    imread, imwrite, resize,
    Filter, ImageConfig, ImageFormat, ImageProcess, ImreadFlags, Matrix, ResizeFilter, Wrap,
};

mod session;
pub use session::{Session, SessionGuard};

mod interpreter;
pub use interpreter::Interpreter;
#[cfg(feature = "async")]
pub use interpreter::AsyncInterpreter;

mod utils;
pub use utils::{calculate_element_count, calculate_tensor_size, convert_format};

#[cfg(feature = "runtime")]
mod runtime;
#[cfg(feature = "runtime")]
pub use runtime::{InterpreterRuntimeExt, RuntimeInfo};

// Async module
#[cfg(feature = "async")]
mod async_mod;
#[cfg(feature = "async")]
pub use async_mod::{run_session_async, AsyncBatchInference, SessionPool};

// Re-export commonly used types at crate root
pub use crate::error::MnnError as Error;

/// Prelude for common MNN types.
pub mod prelude {
    //! Common types for MNN operations.
    //!
    //! This module re-exports the most commonly used types for convenience.
    //!
    //! ```
    //! use mnn_rs::prelude::*;
    //! ```

    pub use crate::backend::{BackendType, DataType};
    pub use crate::config::{DataFormat, MemoryMode, PowerMode, PrecisionMode, ScheduleConfig, ScheduleConfigBuilder};
    pub use crate::error::{MnnError, MnnResult};
    pub use crate::interpreter::Interpreter;
    pub use crate::session::Session;
    pub use crate::tensor::{Tensor, TensorData, TensorInfo};
    #[cfg(feature = "image-process")]
    pub use crate::image_process::{ImageConfig, ImageFormat, Filter};
    #[cfg(feature = "runtime")]
    pub use crate::runtime::RuntimeInfo;
}

/// Testing utilities (only available in tests).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        // Version should be a non-empty string
        assert!(!v.is_empty() || v == "unknown");
    }

    #[test]
    fn test_available_backends() {
        let backends = available_backends();
        // CPU should always be available
        assert!(backends.contains(&BackendType::CPU));
    }
}