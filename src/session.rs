//! Session management for MNN inference.
//!
//! A session represents an inference context with allocated resources.
//! Sessions are created from interpreters and are used to run inference.

use crate::config::ScheduleConfig;
use crate::error::{MnnError, MnnResult};
use crate::tensor::Tensor;
use mnn_rs_sys::MNNSession;
use std::ffi::CString;

/// An inference session.
///
/// Sessions hold the runtime state for model inference, including
/// allocated memory and intermediate tensors.
///
/// # Thread Safety
/// Sessions can be used from multiple threads, but `run()` must not be
/// called concurrently on the same session.
pub struct Session {
    inner: *mut MNNSession,
    /// Interpreter pointer (not owned)
    interpreter: *mut mnn_rs_sys::MNNInterpreter,
    /// Whether the session has been run at least once
    has_run: bool,
}

// Safety: Session operations are thread-safe through MNN's internal synchronization
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    /// Create a new session.
    ///
    /// # Safety
    /// The interpreter pointer must be valid.
    pub(crate) unsafe fn new(
        interpreter: *mut mnn_rs_sys::MNNInterpreter,
        config: ScheduleConfig,
    ) -> MnnResult<Self> {
        // SAFETY: Caller ensures interpreter pointer is valid
        let inner = unsafe {
            mnn_rs_sys::mnn_interpreter_create_session(
                interpreter,
                config.backend_config.backend_type.to_mnn_type(),
                config.num_threads as i32,
            )
        };

        if inner.is_null() {
            return Err(MnnError::session_error("Failed to create session"));
        }

        Ok(Self {
            inner,
            interpreter,
            has_run: false,
        })
    }

    /// Get an input tensor by name.
    ///
    /// # Arguments
    /// * `name` - The name of the input tensor (None for the first input)
    ///
    /// # Returns
    /// A mutable reference to the input tensor.
    pub fn get_input(&self, name: Option<&str>) -> MnnResult<Tensor> {
        unsafe {
            let name_ptr = match name {
                Some(n) => {
                    let c_name = CString::new(n)?;
                    c_name.as_ptr()
                }
                None => std::ptr::null(),
            };

            let tensor_ptr =
                mnn_rs_sys::mnn_interpreter_get_session_input(self.interpreter, self.inner, name_ptr);

            if tensor_ptr.is_null() {
                return Err(MnnError::tensor_error(match name {
                    Some(n) => format!("Input tensor '{}' not found", n),
                    None => "No input tensor found".to_string(),
                }));
            }

            Ok(Tensor::from_ptr_with_name(
                tensor_ptr,
                name.map(|s| s.to_string()),
            ))
        }
    }

    /// Get an output tensor by name.
    ///
    /// # Arguments
    /// * `name` - The name of the output tensor (None for the first output)
    ///
    /// # Returns
    /// A reference to the output tensor.
    pub fn get_output(&self, name: Option<&str>) -> MnnResult<Tensor> {
        unsafe {
            let name_ptr = match name {
                Some(n) => {
                    let c_name = CString::new(n)?;
                    c_name.as_ptr()
                }
                None => std::ptr::null(),
            };

            let tensor_ptr = mnn_rs_sys::mnn_interpreter_get_session_output(
                self.interpreter,
                self.inner,
                name_ptr,
            );

            if tensor_ptr.is_null() {
                return Err(MnnError::tensor_error(match name {
                    Some(n) => format!("Output tensor '{}' not found", n),
                    None => "No output tensor found".to_string(),
                }));
            }

            Ok(Tensor::from_ptr_with_name(
                tensor_ptr,
                name.map(|s| s.to_string()),
            ))
        }
    }

    /// Run inference.
    ///
    /// This executes the model on the current input tensors and
    /// populates the output tensors.
    ///
    /// # Returns
    /// Ok(()) on success, or an error on failure.
    pub fn run(&mut self) -> MnnResult<()> {
        let result =
            unsafe { mnn_rs_sys::mnn_interpreter_run_session(self.interpreter, self.inner) };

        match result {
            x if x == mnn_rs_sys::MNN_ERROR_NONE => {
                self.has_run = true;
                Ok(())
            }
            x if x == mnn_rs_sys::MNN_ERROR_OUT_OF_MEMORY => {
                Err(MnnError::out_of_memory("Out of memory during inference"))
            }
            x if x == mnn_rs_sys::MNN_ERROR_NOT_SUPPORT => {
                Err(MnnError::unsupported("Operation not supported"))
            }
            x if x == mnn_rs_sys::MNN_ERROR_EXECUTION => {
                Err(MnnError::internal("Execution error during inference"))
            }
            code => Err(MnnError::internal(format!(
                "Inference failed with error code: {}",
                code
            ))),
        }
    }

    /// Check if the session has been run at least once.
    pub fn has_run(&self) -> bool {
        self.has_run
    }

    /// Get the memory usage of this session in bytes.
    pub fn memory_usage(&self) -> usize {
        let memory_mb = unsafe {
            mnn_rs_sys::mnn_interpreter_get_session_memory(self.interpreter, self.inner)
        };
        (memory_mb * 1024.0 * 1024.0) as usize
    }

    /// Get the FLOPS count of this session.
    pub fn flops(&self) -> f32 {
        unsafe { mnn_rs_sys::mnn_interpreter_get_session_flops(self.interpreter, self.inner) }
    }

    /// Get the raw pointer to the underlying MNN session.
    ///
    /// # Safety
    /// The returned pointer is owned by this Session and must not be freed.
    pub fn inner(&self) -> *mut MNNSession {
        self.inner
    }

    /// Get the mutable raw pointer to the underlying MNN session.
    ///
    /// # Safety
    /// The returned pointer is owned by this Session and must not be freed.
    pub fn inner_mut(&mut self) -> *mut MNNSession {
        self.inner
    }

    /// Get the interpreter pointer (not owned).
    ///
    /// # Safety
    /// The returned pointer is owned by the Interpreter.
    pub fn interpreter(&self) -> *mut mnn_rs_sys::MNNInterpreter {
        self.interpreter
    }

    /// Create a new session from raw pointers.
    ///
    /// # Safety
    /// The pointers must be valid and the interpreter must outlive the session.
    pub unsafe fn from_ptr(inner: *mut MNNSession, interpreter: *mut mnn_rs_sys::MNNInterpreter) -> Self {
        Self {
            inner,
            interpreter,
            has_run: false,
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.inner.is_null() && !self.interpreter.is_null() {
            unsafe {
                mnn_rs_sys::mnn_interpreter_release_session(self.interpreter, self.inner);
            }
        }
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("has_run", &self.has_run)
            .finish()
    }
}

/// A guard for ensuring session resources are properly managed.
///
/// When dropped, this will release the session resources.
pub struct SessionGuard<'a> {
    session: &'a mut Session,
}

impl std::fmt::Debug for SessionGuard<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionGuard").finish_non_exhaustive()
    }
}

impl<'a> SessionGuard<'a> {
    /// Create a new guard for a session.
    pub fn new(session: &'a mut Session) -> Self {
        Self { session }
    }

    /// Run inference.
    pub fn run(&mut self) -> MnnResult<()> {
        self.session.run()
    }
}

impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        // Session will be cleaned up when dropped
    }
}

#[cfg(feature = "async")]
mod async_impl {
    use super::*;

    impl Session {
        /// Run inference asynchronously.
        pub async fn run_async(&mut self) -> MnnResult<()> {
            let inner = self.inner;
            let interpreter = self.interpreter;

            let result = tokio::task::spawn_blocking(move || unsafe {
                mnn_rs_sys::mnn_interpreter_run_session(interpreter, inner)
            })
            .await
            .map_err(|e| MnnError::AsyncError(e.to_string()))?;

            match result {
                x if x == mnn_rs_sys::MNN_ERROR_NONE => {
                    self.has_run = true;
                    Ok(())
                }
                x if x == mnn_rs_sys::MNN_ERROR_OUT_OF_MEMORY => {
                    Err(MnnError::out_of_memory("Out of memory during inference"))
                }
                x if x == mnn_rs_sys::MNN_ERROR_NOT_SUPPORT => {
                    Err(MnnError::unsupported("Operation not supported"))
                }
                x if x == mnn_rs_sys::MNN_ERROR_EXECUTION => {
                    Err(MnnError::internal("Execution error during inference"))
                }
                code => Err(MnnError::internal(format!(
                    "Inference failed with error code: {}",
                    code
                ))),
            }
        }
    }
}

#[cfg(test)]
mod tests {}