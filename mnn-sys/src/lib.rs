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

/// Opaque handle to an MNN ImageProcess
/// ImageProcess handles image preprocessing and conversion
#[repr(C)]
pub struct MNNImageProcess {
    _private: [u8; 0],
}

/// Opaque handle to an MNN Matrix
/// Matrix represents a 3x3 affine transformation matrix
#[repr(C)]
pub struct MNNMatrix {
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
// Image Format Constants (matches MNN::CV::ImageFormat)
// ============================================================================

pub const MNN_IMAGE_FORMAT_RGBA: c_int = 0;
pub const MNN_IMAGE_FORMAT_RGB: c_int = 1;
pub const MNN_IMAGE_FORMAT_BGR: c_int = 2;
pub const MNN_IMAGE_FORMAT_GRAY: c_int = 3;
pub const MNN_IMAGE_FORMAT_BGRA: c_int = 4;
pub const MNN_IMAGE_FORMAT_YCRCB: c_int = 5;
pub const MNN_IMAGE_FORMAT_YUV: c_int = 6;
pub const MNN_IMAGE_FORMAT_HSV: c_int = 7;
pub const MNN_IMAGE_FORMAT_XYZ: c_int = 8;
pub const MNN_IMAGE_FORMAT_BGR555: c_int = 9;
pub const MNN_IMAGE_FORMAT_BGR565: c_int = 10;
pub const MNN_IMAGE_FORMAT_YUV_NV21: c_int = 11;
pub const MNN_IMAGE_FORMAT_YUV_NV12: c_int = 12;
pub const MNN_IMAGE_FORMAT_YUV_I420: c_int = 13;
pub const MNN_IMAGE_FORMAT_HSV_FULL: c_int = 14;

// ============================================================================
// Filter Type Constants (matches MNN::CV::Filter)
// ============================================================================

pub const MNN_FILTER_NEAREST: c_int = 0;
pub const MNN_FILTER_BILINEAR: c_int = 1;
pub const MNN_FILTER_BICUBIC: c_int = 2;

// ============================================================================
// Wrap Type Constants (matches MNN::CV::Wrap)
// ============================================================================

pub const MNN_WRAP_CLAMP_TO_EDGE: c_int = 0;
pub const MNN_WRAP_ZERO: c_int = 1;
pub const MNN_WRAP_REPEAT: c_int = 2;

// ============================================================================
// ImageProcess Config Structure
// ============================================================================

/// ImageProcess config structure for image preprocessing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNImageProcessConfig {
    pub filterType: c_int,
    pub sourceFormat: c_int,
    pub destFormat: c_int,
    pub mean: [f32; 4],
    pub normal: [f32; 4],
    pub wrap: c_int,
}

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

    // ========================================================================
    // ImageProcess Functions
    // ========================================================================

    /// Create image process with config
    pub fn mnn_image_process_create(config: *const MNNImageProcessConfig) -> *mut MNNImageProcess;

    /// Destroy image process
    pub fn mnn_image_process_destroy(process: *mut MNNImageProcess);

    /// Set transform matrix
    pub fn mnn_image_process_set_matrix(process: *mut MNNImageProcess, matrix: *const MNNMatrix);

    /// Convert image to tensor
    pub fn mnn_image_process_convert(
        process: *mut MNNImageProcess,
        source: *const u8,
        iw: c_int,
        ih: c_int,
        stride: c_int,
        tensor: *mut MNNTensor,
    ) -> c_int;

    /// Create image tensor
    pub fn mnn_image_tensor_create(w: c_int, h: c_int, bpp: c_int, data: *mut c_void) -> *mut MNNTensor;

    /// Destroy image tensor
    pub fn mnn_image_tensor_destroy(tensor: *mut MNNTensor);

    // ========================================================================
    // Matrix Functions
    // ========================================================================

    /// Create identity matrix
    pub fn mnn_matrix_create_identity() -> *mut MNNMatrix;

    /// Create scale matrix
    pub fn mnn_matrix_create_scale(sx: f32, sy: f32) -> *mut MNNMatrix;

    /// Create translate matrix
    pub fn mnn_matrix_create_translate(dx: f32, dy: f32) -> *mut MNNMatrix;

    /// Create rotate matrix (degrees)
    pub fn mnn_matrix_create_rotate(degrees: f32) -> *mut MNNMatrix;

    /// Create matrix from raw data (9 floats)
    pub fn mnn_matrix_create(data: *const f32) -> *mut MNNMatrix;

    /// Clone matrix
    pub fn mnn_matrix_clone(matrix: *const MNNMatrix) -> *mut MNNMatrix;

    /// Destroy matrix
    pub fn mnn_matrix_destroy(matrix: *mut MNNMatrix);

    /// Get matrix element at (row, col)
    pub fn mnn_matrix_get(matrix: *const MNNMatrix, row: c_int, col: c_int) -> f32;

    /// Set matrix element at (row, col)
    pub fn mnn_matrix_set(matrix: *mut MNNMatrix, row: c_int, col: c_int, value: f32);

    /// Multiply two matrices
    pub fn mnn_matrix_multiply(a: *const MNNMatrix, b: *const MNNMatrix) -> *mut MNNMatrix;

    /// Invert matrix
    pub fn mnn_matrix_invert(matrix: *const MNNMatrix) -> *mut MNNMatrix;

    // ========================================================================
    // Tensor Advanced Functions (GPU Memory Operations)
    // ========================================================================

    /// Copy data from host tensor to device tensor
    pub fn mnn_tensor_copy_from_host(dest: *mut MNNTensor, host_tensor: *const MNNTensor) -> c_int;

    /// Copy data from device tensor to host tensor
    pub fn mnn_tensor_copy_to_host(host_tensor: *mut MNNTensor, dest: *const MNNTensor) -> c_int;

    /// Create a device tensor with given shape
    pub fn mnn_tensor_create_device(
        shape: *const c_int,
        dimensions: c_int,
        type_code: c_int,
        format: c_int,
    ) -> *mut MNNTensor;

    /// Clone a tensor
    pub fn mnn_tensor_clone(tensor: *const MNNTensor, deep_copy: c_int) -> *mut MNNTensor;

    /// Destroy a user-created tensor
    pub fn mnn_tensor_destroy(tensor: *mut MNNTensor);

    /// Get tensor device ID (for GPU tensors)
    pub fn mnn_tensor_device_id(tensor: *const MNNTensor) -> u64;

    /// Get tensor backend type
    pub fn mnn_tensor_get_backend(tensor: *const MNNTensor) -> c_int;
}

// ============================================================================
// Session Advanced Types
// ============================================================================

/// String array for returning names
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNStringArray {
    pub names: *mut *mut c_char,
    pub count: c_int,
}

// Session mode constants
pub const MNN_SESSION_MODE_DEBUG: c_int = 0;
pub const MNN_SESSION_MODE_RELEASE: c_int = 1;
pub const MNN_SESSION_MODE_INPUT_INSIDE: c_int = 2;
pub const MNN_SESSION_MODE_INPUT_USER: c_int = 3;
pub const MNN_SESSION_MODE_OUTPUT_INSIDE: c_int = 4;
pub const MNN_SESSION_MODE_OUTPUT_USER: c_int = 5;
pub const MNN_SESSION_MODE_RESIZE_DIRECT: c_int = 6;
pub const MNN_SESSION_MODE_RESIZE_DEFER: c_int = 7;
pub const MNN_SESSION_MODE_BACKEND_FIX: c_int = 8;
pub const MNN_SESSION_MODE_BACKEND_AUTO: c_int = 9;

extern "C" {
    // ========================================================================
    // Session Advanced Functions
    // ========================================================================

    /// Set session mode
    pub fn mnn_interpreter_set_session_mode(interpreter: *mut MNNInterpreter, mode: c_int);

    /// Set cache file for optimization
    pub fn mnn_interpreter_set_cache_file(interpreter: *mut MNNInterpreter, path: *const c_char, key_size: usize);

    /// Update cache from session
    pub fn mnn_interpreter_update_cache(interpreter: *mut MNNInterpreter, session: *mut MNNSession) -> c_int;

    /// Set external file for model
    pub fn mnn_interpreter_set_external_file(interpreter: *mut MNNInterpreter, path: *const c_char, flag: usize);

    /// Get input tensor names
    pub fn mnn_interpreter_get_input_names(interpreter: *mut MNNInterpreter, session: *mut MNNSession) -> MNNStringArray;

    /// Get output tensor names
    pub fn mnn_interpreter_get_output_names(interpreter: *mut MNNInterpreter, session: *mut MNNSession) -> MNNStringArray;

    /// Free string array
    pub fn mnn_string_array_free(array: *mut MNNStringArray);

    /// Resize tensor with new shape
    pub fn mnn_interpreter_resize_tensor(
        interpreter: *mut MNNInterpreter,
        tensor: *mut MNNTensor,
        shape: *const c_int,
        dims: c_int,
    );

    /// Get session operator count
    pub fn mnn_interpreter_get_session_op_count(interpreter: *mut MNNInterpreter, session: *mut MNNSession) -> c_int;
}

// ============================================================================
// Runtime Management Types
// ============================================================================

/// Opaque handle to MNN RuntimeManager
#[repr(C)]
pub struct MNNRuntimeManager {
    _private: [u8; 0],
}

extern "C" {
    // ========================================================================
    // Runtime Management
    // ========================================================================

    /// Create runtime manager from config
    pub fn mnn_runtime_manager_create(type_: c_int, num_threads: c_int) -> *mut MNNRuntimeManager;

    /// Destroy runtime manager
    pub fn mnn_runtime_manager_destroy(manager: *mut MNNRuntimeManager);

    /// Create session with shared runtime
    pub fn mnn_interpreter_create_session_with_runtime(
        interpreter: *mut MNNInterpreter,
        runtime: *mut MNNRuntimeManager,
        type_: c_int,
        num_threads: c_int,
    ) -> *mut MNNSession;
}

#[cfg(test)]
mod tests {}