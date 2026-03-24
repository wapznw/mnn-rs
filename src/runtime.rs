//! Runtime management for multi-session sharing.
//!
//! This module provides runtime manager for sharing resources across multiple sessions,
//! which can improve memory efficiency and performance when running multiple models.

use crate::backend::BackendType;
use crate::error::{MnnError, MnnResult};
use crate::interpreter::Interpreter;
use crate::session::Session;
use mnn_rs_sys::*;

/// Runtime manager for sharing resources across sessions.
///
/// The runtime manager allows multiple sessions to share backend resources,
/// which can reduce memory usage and improve performance when running
/// multiple models on the same device.
///
/// # Example
/// ```no_run
/// use mnn_rs::{RuntimeInfo, Interpreter, ScheduleConfig};
///
/// // Create runtime manager for GPU
/// let runtime = RuntimeInfo::new(BackendType::Auto, 4)?;
///
/// // Load multiple models
/// let interpreter1 = Interpreter::from_file("model1.mnn")?;
/// let interpreter2 = Interpreter::from_file("model2.mnn")?;
///
/// // Create sessions sharing the same runtime
/// let session1 = interpreter1.create_session_with_runtime(&runtime)?;
/// let session2 = interpreter2.create_session_with_runtime(&runtime)?;
/// # Ok::<(), mnn_rs::MnnError>(())
/// ```
pub struct RuntimeInfo {
    inner: *mut MNNRuntimeManager,
}

// Safety: RuntimeManager is thread-safe through MNN's internal synchronization
unsafe impl Send for RuntimeInfo {}
unsafe impl Sync for RuntimeInfo {}

impl RuntimeInfo {
    /// Create a new runtime manager.
    ///
    /// # Arguments
    /// * `backend_type` - The backend type to use (CPU, GPU, etc.)
    /// * `num_threads` - Number of threads for CPU backend
    ///
    /// # Returns
    /// A new runtime manager on success, or an error if creation fails.
    ///
    /// # Note
    /// Runtime manager requires MNN to be built with runtime support.
    /// If not available, this will return an error.
    pub fn new(backend_type: BackendType, num_threads: i32) -> MnnResult<Self> {
        let inner = unsafe {
            mnn_runtime_manager_create(backend_type.to_mnn_type(), num_threads)
        };

        if inner.is_null() {
            return Err(MnnError::internal(
                "Failed to create runtime manager. MNN may not be built with runtime support."
            ));
        }

        Ok(Self { inner })
    }

    /// Create a session with this runtime manager.
    ///
    /// # Arguments
    /// * `interpreter` - The interpreter to create session from
    /// * `config` - Schedule configuration
    ///
    /// # Returns
    /// A new session on success, or an error if creation fails.
    pub fn create_session(
        &self,
        interpreter: &Interpreter,
        config: crate::config::ScheduleConfig,
    ) -> MnnResult<Session> {
        unsafe {
            let inner = mnn_interpreter_create_session_with_runtime(
                interpreter.inner(),
                self.inner,
                config.backend_config.backend_type.to_mnn_type(),
                config.num_threads as i32,
            );

            if inner.is_null() {
                return Err(MnnError::session_error("Failed to create session with runtime"));
            }

            Ok(Session::from_ptr(inner, interpreter.inner()))
        }
    }

    /// Get the raw pointer to the underlying MNN RuntimeManager.
    ///
    /// # Safety
    /// The returned pointer is owned by this RuntimeInfo and must not be freed.
    pub unsafe fn as_ptr(&self) -> *const MNNRuntimeManager {
        self.inner
    }
}

impl Drop for RuntimeInfo {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                mnn_runtime_manager_destroy(self.inner);
            }
        }
    }
}

impl std::fmt::Debug for RuntimeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeInfo").finish_non_exhaustive()
    }
}

/// Extension trait for creating sessions with runtime manager.
pub trait InterpreterRuntimeExt {
    /// Create a session with a shared runtime manager.
    ///
    /// # Arguments
    /// * `runtime` - The runtime manager to share
    ///
    /// # Returns
    /// A new session on success, or an error if creation fails.
    fn create_session_with_runtime(&self, runtime: &RuntimeInfo) -> MnnResult<Session>;
}

impl InterpreterRuntimeExt for Interpreter {
    fn create_session_with_runtime(&self, runtime: &RuntimeInfo) -> MnnResult<Session> {
        // Use default CPU config
        let config = crate::config::ScheduleConfig::new()
            .backend(BackendType::CPU);
        runtime.create_session(self, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_info_creation() {
        // Runtime manager may not be available in all builds
        let result = RuntimeInfo::new(BackendType::CPU, 4);
        // If runtime is not enabled, this will fail - which is expected
        if result.is_ok() {
            // If successful, verify we can drop it without issues
            drop(result.unwrap());
        }
    }
}
