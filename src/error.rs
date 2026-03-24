//! Error types for MNN operations.
//!
//! This module provides a comprehensive error type for all MNN operations.

use std::ffi::NulError;
use std::path::PathBuf;
use thiserror::Error;

/// Main error type for MNN operations.
#[derive(Debug, Error)]
pub enum MnnError {
    /// I/O error (file not found, permission denied, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid model file or corrupted model
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// Failed to create or use session
    #[error("Session error: {0}")]
    SessionError(String),

    /// Tensor operation failed
    #[error("Tensor error: {0}")]
    TensorError(String),

    /// Backend not available or misconfigured
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Operation not supported
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// Invalid input provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid path (non-UTF8 or invalid format)
    #[error("Invalid path")]
    InvalidPath,

    /// Internal MNN error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Null byte in string
    #[error("Null byte in string: {0}")]
    NullByte(#[from] NulError),

    /// Path-related error
    #[error("Path error: {path}")]
    PathError {
        /// The problematic path
        path: PathBuf,
        /// The source error
        #[source]
        source: std::io::Error,
    },

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<i32>,
        /// Actual shape
        actual: Vec<i32>,
    },

    /// Type mismatch
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Actual type
        actual: String,
    },

    /// Invalid dimension
    #[error("Invalid dimension: {message}")]
    InvalidDimension {
        /// Error message
        message: String,
    },

    /// Empty data
    #[error("Empty data provided")]
    EmptyData,

    /// Index out of bounds
    #[error("Index {index} out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds {
        /// The dimension
        dim: usize,
        /// The index
        index: i32,
        /// The size
        size: i32,
    },

    /// Backend not available
    #[error("Backend {backend:?} is not available on this platform")]
    BackendNotAvailable {
        /// The requested backend
        backend: String,
    },

    /// Model file not found
    #[error("Model file not found: {0}")]
    ModelNotFound(PathBuf),

    /// Session not initialized
    #[error("Session not initialized")]
    SessionNotInitialized,

    /// Interpreter not initialized
    #[error("Interpreter not initialized")]
    InterpreterNotInitialized,

    /// Feature not enabled
    #[error("Feature '{feature}' is not enabled. Add it to Cargo.toml features.")]
    FeatureNotEnabled {
        /// The feature name
        feature: &'static str,
    },

    /// Async runtime error
    #[cfg(feature = "async")]
    #[error("Async error: {0}")]
    AsyncError(String),
}

/// Result type alias for MNN operations.
pub type MnnResult<T> = Result<T, MnnError>;

impl MnnError {
    /// Create an invalid model error with context
    pub fn invalid_model<S: Into<String>>(msg: S) -> Self {
        MnnError::InvalidModel(msg.into())
    }

    /// Create a session error with context
    pub fn session_error<S: Into<String>>(msg: S) -> Self {
        MnnError::SessionError(msg.into())
    }

    /// Create a tensor error with context
    pub fn tensor_error<S: Into<String>>(msg: S) -> Self {
        MnnError::TensorError(msg.into())
    }

    /// Create a backend error with context
    pub fn backend_error<S: Into<String>>(msg: S) -> Self {
        MnnError::BackendError(msg.into())
    }

    /// Create an unsupported error
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        MnnError::Unsupported(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        MnnError::InvalidInput(msg.into())
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        MnnError::Internal(msg.into())
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[i32], actual: &[i32]) -> Self {
        MnnError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Create an out of memory error
    pub fn out_of_memory<S: Into<String>>(msg: S) -> Self {
        MnnError::OutOfMemory(msg.into())
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(dim: usize, index: i32, size: i32) -> Self {
        MnnError::IndexOutOfBounds { dim, index, size }
    }

    /// Create a backend not available error
    pub fn backend_not_available<S: Into<String>>(backend: S) -> Self {
        MnnError::BackendNotAvailable {
            backend: backend.into(),
        }
    }

    /// Check if this is an out of memory error
    pub fn is_oom(&self) -> bool {
        matches!(self, MnnError::OutOfMemory(_))
    }

    /// Check if this is a model error
    pub fn is_model_error(&self) -> bool {
        matches!(
            self,
            MnnError::InvalidModel(_) | MnnError::ModelNotFound(_)
        )
    }

    /// Check if this is a backend error
    pub fn is_backend_error(&self) -> bool {
        matches!(
            self,
            MnnError::BackendError(_) | MnnError::BackendNotAvailable { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = MnnError::invalid_model("test error");
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_shape_mismatch() {
        let err = MnnError::shape_mismatch(&[1, 2, 3], &[1, 2, 4]);
        assert!(err.to_string().contains("expected"));
        assert!(err.to_string().contains("got"));
    }

    #[test]
    fn test_error_checks() {
        let err = MnnError::out_of_memory("test");
        assert!(err.is_oom());

        let err = MnnError::invalid_model("test");
        assert!(err.is_model_error());

        let err = MnnError::backend_error("test");
        assert!(err.is_backend_error());
    }
}