//! Raw FFI bindings for MNN (Mobile Neural Network) inference engine.
//!
//! This crate provides unsafe raw bindings to the MNN C wrapper API.
//! Users should prefer the safe `mnn-rs` crate instead.
//!
//! # Safety
//!
//! All functions and types in this module are unsafe and directly map to
//! the MNN C API. Proper usage requires understanding of the MNN documentation
//! and careful memory management.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// ============================================================================
// Forward declarations (opaque handles)
// ============================================================================

/// Opaque handle to an MNN Interpreter
/// The Interpreter holds the model and can create multiple sessions
#[repr(C)]
pub struct MNNInterpreter {
    _private: [u8; 0],
}

/// Opaque handle to an MNN Session
/// A Session represents an inference context with allocated resources
#[repr(C)]
pub struct MNNSession {
    _private: [u8; 0],
}

/// Opaque handle to an MNN Tensor
/// Tensors hold multi-dimensional array data for inference
#[repr(C)]
pub struct MNNTensor {
    _private: [u8; 0],
}

// ============================================================================
// Enum Definitions
// ============================================================================

/// Backend type for computation (matches MNNForwardType)
pub const MNN_FORWARD_CPU: c_int = 0;
pub const MNN_FORWARD_AUTO: c_int = 0;
pub const MNN_FORWARD_OPENCL: c_int = 1;
pub const MNN_FORWARD_OPENGL: c_int = 2;
pub const MNN_FORWARD_VULKAN: c_int = 3;
pub const MNN_FORWARD_METAL: c_int = 4;
pub const MNN_FORWARD_CUDA: c_int = 5;
pub const MNN_FORWARD_NPU: c_int = 6;

/// Error codes
pub const MNN_ERROR_NONE: c_int = 0;
pub const MNN_ERROR_OUT_OF_MEMORY: c_int = 1;
pub const MNN_ERROR_NOT_SUPPORT: c_int = 2;
pub const MNN_ERROR_EXECUTION: c_int = 9;

/// Data format
pub const MNN_DATA_FORMAT_NHWC: c_int = 0;
pub const MNN_DATA_FORMAT_NC4HW4: c_int = 1;
pub const MNN_DATA_FORMAT_NCHW: c_int = 2;

// ============================================================================
// FFI Function Declarations (C Wrapper Functions)
// ============================================================================

extern "C" {
    // ========================================================================
    // Version and Info
    // ========================================================================

    /// Get MNN version string
    pub fn mnn_get_version() -> *const c_char;

    /// Check if a backend is available
    pub fn mnn_is_backend_available(type_: c_int) -> c_int;

    // ========================================================================
    // Interpreter Functions
    // ========================================================================

    /// Create interpreter from file
    pub fn mnn_interpreter_create_from_file(file: *const c_char) -> *mut MNNInterpreter;

    /// Create interpreter from buffer
    pub fn mnn_interpreter_create_from_buffer(buffer: *const c_void, size: usize) -> *mut MNNInterpreter;

    /// Destroy interpreter
    pub fn mnn_interpreter_destroy(interpreter: *mut MNNInterpreter);

    /// Create session
    pub fn mnn_interpreter_create_session(
        interpreter: *mut MNNInterpreter,
        type_: c_int,
        num_thread: c_int,
    ) -> *mut MNNSession;

    /// Release session
    pub fn mnn_interpreter_release_session(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    );

    /// Run session
    pub fn mnn_interpreter_run_session(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> c_int;

    /// Get session input tensor
    pub fn mnn_interpreter_get_session_input(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
        name: *const c_char,
    ) -> *mut MNNTensor;

    /// Get session output tensor
    pub fn mnn_interpreter_get_session_output(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
        name: *const c_char,
    ) -> *mut MNNTensor;

    /// Resize session
    pub fn mnn_interpreter_resize_session(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    );

    /// Get session memory in MB
    pub fn mnn_interpreter_get_session_memory(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> f32;

    /// Get session FLOPS in M
    pub fn mnn_interpreter_get_session_flops(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> f32;

    /// Get business code
    pub fn mnn_interpreter_get_biz_code(interpreter: *mut MNNInterpreter) -> *const c_char;

    /// Get UUID
    pub fn mnn_interpreter_get_uuid(interpreter: *mut MNNInterpreter) -> *const c_char;

    // ========================================================================
    // Tensor Functions
    // ========================================================================

    /// Get tensor dimensions count
    pub fn mnn_tensor_get_dimensions(tensor: *const MNNTensor) -> c_int;

    /// Get tensor shape element at index
    pub fn mnn_tensor_get_dim(tensor: *const MNNTensor, index: c_int) -> c_int;

    /// Get tensor element count
    pub fn mnn_tensor_get_element_count(tensor: *const MNNTensor) -> c_int;

    /// Get tensor size in bytes
    pub fn mnn_tensor_get_size(tensor: *const MNNTensor) -> c_int;

    /// Get tensor host data pointer
    pub fn mnn_tensor_get_host_data(tensor: *mut MNNTensor) -> *mut c_void;

    /// Get tensor type code
    pub fn mnn_tensor_get_type_code(tensor: *const MNNTensor) -> c_int;

    /// Get tensor dimension type
    pub fn mnn_tensor_get_dimension_type(tensor: *const MNNTensor) -> c_int;
}

#[cfg(test)]
mod tests {}