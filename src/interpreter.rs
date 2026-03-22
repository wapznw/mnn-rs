//! Model interpreter for MNN inference.
//!
//! The interpreter holds the model and manages inference sessions.
//! It's the primary entry point for running neural network inference with MNN.

use crate::config::ScheduleConfig;
use crate::error::{MnnError, MnnResult};
use crate::session::Session;
use mnn_rs_sys::MNNInterpreter;
use std::ffi::CString;
use std::path::Path;

/// A model interpreter that holds a loaded neural network model.
///
/// The interpreter is the main entry point for MNN inference. It manages
/// the model and can create multiple sessions for concurrent inference.
///
/// # Thread Safety
/// The interpreter is thread-safe and can be shared across threads.
/// Multiple sessions can be created from a single interpreter.
///
/// # Example
/// ```no_run
/// use mnn_rs::{Interpreter, ScheduleConfig, BackendType};
///
/// // Load a model
/// let interpreter = Interpreter::from_file("model.mnn")?;
///
/// // Create a session with configuration
/// let config = ScheduleConfig::new()
///     .backend(BackendType::CPU)
///     .num_threads(4);
///
/// let mut session = interpreter.create_session(config)?;
///
/// // Get input tensor and fill with data
/// let input = session.get_input(None)?;
/// // ... fill input with data ...
///
/// // Run inference
/// session.run()?;
///
/// // Get output
/// let output = session.get_output(None)?;
/// # Ok::<(), mnn_rs::MnnError>(())
/// ```
pub struct Interpreter {
    inner: *mut MNNInterpreter,
    /// Model path (if loaded from file)
    model_path: Option<String>,
}

// Safety: Interpreter operations are thread-safe through MNN's internal synchronization
unsafe impl Send for Interpreter {}
unsafe impl Sync for Interpreter {}

impl Interpreter {
    /// Create a new interpreter from a model file.
    ///
    /// # Arguments
    /// * `path` - Path to the MNN model file
    ///
    /// # Returns
    /// A new interpreter on success, or an error if the model cannot be loaded.
    ///
    /// # Example
    /// ```no_run
    /// use mnn_rs::Interpreter;
    ///
    /// let interpreter = Interpreter::from_file("model.mnn")?;
    /// # Ok::<(), mnn_rs::MnnError>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> MnnResult<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();

        // Check if file exists
        if !path.as_ref().exists() {
            return Err(MnnError::ModelNotFound(path.as_ref().to_path_buf()));
        }

        let c_path = CString::new(path_str.as_str())?;

        let inner = unsafe { mnn_rs_sys::mnn_interpreter_create_from_file(c_path.as_ptr()) };

        if inner.is_null() {
            return Err(MnnError::invalid_model(format!(
                "Failed to load model from: {}",
                path_str
            )));
        }

        Ok(Self {
            inner,
            model_path: Some(path_str),
        })
    }

    /// Create a new interpreter from in-memory model data.
    ///
    /// This is useful for embedding models in the binary or loading
    /// models from non-filesystem sources.
    ///
    /// # Arguments
    /// * `data` - The model data as bytes
    ///
    /// # Returns
    /// A new interpreter on success, or an error if the model cannot be loaded.
    ///
    /// # Example
    /// ```ignore
    /// use mnn_rs::Interpreter;
    ///
    /// let model_data = include_bytes!("model.mnn");
    /// let interpreter = Interpreter::from_bytes(model_data)?;
    /// # Ok::<(), mnn_rs::MnnError>(())
    /// ```
    pub fn from_bytes(data: &[u8]) -> MnnResult<Self> {
        if data.is_empty() {
            return Err(MnnError::invalid_model("Model data is empty"));
        }

        let inner = unsafe {
            mnn_rs_sys::mnn_interpreter_create_from_buffer(
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
            )
        };

        if inner.is_null() {
            return Err(MnnError::invalid_model("Failed to load model from memory"));
        }

        Ok(Self {
            inner,
            model_path: None,
        })
    }

    /// Create a new inference session.
    ///
    /// Sessions hold the runtime state for inference and can be created
    /// with different backend configurations.
    ///
    /// # Arguments
    /// * `config` - Schedule configuration specifying backend and settings
    ///
    /// # Returns
    /// A new session on success, or an error if session creation fails.
    pub fn create_session(&self, config: ScheduleConfig) -> MnnResult<Session> {
        unsafe { Session::new(self.inner, config) }
    }

    /// Get the model path (if loaded from file).
    pub fn model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
    }

    /// Get the business code (model identifier).
    ///
    /// # Returns
    /// The business code string.
    pub fn business_code(&self) -> String {
        unsafe {
            let ptr = mnn_rs_sys::mnn_interpreter_get_biz_code(self.inner);
            if ptr.is_null() {
                return String::new();
            }
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Get the model UUID.
    ///
    /// # Returns
    /// The UUID string.
    pub fn uuid(&self) -> String {
        unsafe {
            let ptr = mnn_rs_sys::mnn_interpreter_get_uuid(self.inner);
            if ptr.is_null() {
                return String::new();
            }
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                mnn_rs_sys::mnn_interpreter_destroy(self.inner);
            }
        }
    }
}

impl std::fmt::Debug for Interpreter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interpreter")
            .field("model_path", &self.model_path)
            .field("business_code", &self.business_code())
            .finish()
    }
}

#[cfg(feature = "async")]
mod async_impl {
    use super::*;
    use std::sync::Arc;

    /// An asynchronous interpreter wrapper.
    ///
    /// This wraps an interpreter for use in async contexts.
    #[derive(Clone)]
    pub struct AsyncInterpreter {
        inner: Arc<Interpreter>,
    }

    impl AsyncInterpreter {
        /// Create an async interpreter from a file.
        pub async fn from_file<P: AsRef<Path> + Send + 'static>(path: P) -> MnnResult<Self> {
            let path = path.as_ref().to_path_buf();
            tokio::task::spawn_blocking(move || Interpreter::from_file(path))
                .await
                .map_err(|e| MnnError::AsyncError(e.to_string()))?
                .map(Self::new)
        }

        /// Create an async interpreter from bytes.
        pub async fn from_bytes(data: Vec<u8>) -> MnnResult<Self> {
            tokio::task::spawn_blocking(move || Interpreter::from_bytes(&data))
                .await
                .map_err(|e| MnnError::AsyncError(e.to_string()))?
                .map(Self::new)
        }

        /// Create a new async interpreter from an existing interpreter.
        pub fn new(interpreter: Interpreter) -> Self {
            Self {
                inner: Arc::new(interpreter),
            }
        }

        /// Create a session.
        pub async fn create_session(&self, config: ScheduleConfig) -> MnnResult<Session> {
            let interpreter = Arc::clone(&self.inner);
            tokio::task::spawn_blocking(move || interpreter.create_session(config))
                .await
                .map_err(|e| MnnError::AsyncError(e.to_string()))?
        }

        /// Get the inner interpreter.
        pub fn inner(&self) -> &Interpreter {
            &self.inner
        }
    }
}

#[cfg(feature = "async")]
pub use async_impl::AsyncInterpreter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpreter_missing_file() {
        let result = Interpreter::from_file("nonexistent_model.mnn");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MnnError::ModelNotFound(_)));
    }

    #[test]
    fn test_interpreter_empty_bytes() {
        let result = Interpreter::from_bytes(&[]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MnnError::InvalidModel(_)));
    }
}