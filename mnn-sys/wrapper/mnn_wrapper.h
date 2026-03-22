/**
 * @file mnn_wrapper.h
 * @brief C wrapper for MNN C++ API
 *
 * This header provides C-compatible functions that wrap the MNN C++ API,
 * allowing FFI bindings from Rust and other languages.
 */

#ifndef MNN_WRAPPER_H
#define MNN_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ============================================================================
 * Type Definitions (using MNN's types)
 * ============================================================================ */

/** Opaque handle to MNN Interpreter */
typedef struct MNNInterpreterHandle MNNInterpreter;

/** Opaque handle to MNN Session */
typedef struct MNNSessionHandle MNNSession;

/** Opaque handle to MNN Tensor */
typedef struct MNNTensorHandle MNNTensor;

/** Forward/Backend type (matches MNN's definition) */
typedef int MNNForwardTypeWrapper;

/** Data type (matches MNN's halide_type_t codes) */
typedef int MNNDataTypeWrapper;

/** Error code */
typedef int MNNErrorCodeWrapper;

/* Error codes */
#define MNN_ERROR_NONE 0
#define MNN_ERROR_OUT_OF_MEMORY 1
#define MNN_ERROR_NOT_SUPPORT 2
#define MNN_ERROR_EXECUTION 9

/* ============================================================================
 * Version and Info
 * ============================================================================ */

/** Get MNN version string */
const char* mnn_get_version(void);

/** Check if a backend is available (0: not available, 1: available) */
int mnn_is_backend_available(int type);

/* ============================================================================
 * Interpreter Functions
 * ============================================================================ */

/** Create interpreter from file */
MNNInterpreter* mnn_interpreter_create_from_file(const char* file);

/** Create interpreter from buffer */
MNNInterpreter* mnn_interpreter_create_from_buffer(const void* buffer, size_t size);

/** Destroy interpreter */
void mnn_interpreter_destroy(MNNInterpreter* interpreter);

/** Create session with config */
MNNSession* mnn_interpreter_create_session(MNNInterpreter* interpreter, int type, int num_thread);

/** Destroy session */
void mnn_interpreter_release_session(MNNInterpreter* interpreter, MNNSession* session);

/** Run session */
int mnn_interpreter_run_session(MNNInterpreter* interpreter, MNNSession* session);

/** Get session input tensor */
MNNTensor* mnn_interpreter_get_session_input(MNNInterpreter* interpreter, MNNSession* session, const char* name);

/** Get session output tensor */
MNNTensor* mnn_interpreter_get_session_output(MNNInterpreter* interpreter, MNNSession* session, const char* name);

/** Resize session */
void mnn_interpreter_resize_session(MNNInterpreter* interpreter, MNNSession* session);

/** Get session memory in MB */
float mnn_interpreter_get_session_memory(MNNInterpreter* interpreter, MNNSession* session);

/** Get session FLOPS in M */
float mnn_interpreter_get_session_flops(MNNInterpreter* interpreter, MNNSession* session);

/** Get business code */
const char* mnn_interpreter_get_biz_code(MNNInterpreter* interpreter);

/** Get UUID */
const char* mnn_interpreter_get_uuid(MNNInterpreter* interpreter);

/* ============================================================================
 * Tensor Functions
 * ============================================================================ */

/** Get tensor dimensions count */
int mnn_tensor_get_dimensions(const MNNTensor* tensor);

/** Get tensor shape element at index */
int mnn_tensor_get_dim(const MNNTensor* tensor, int index);

/** Get tensor element count */
int mnn_tensor_get_element_count(const MNNTensor* tensor);

/** Get tensor size in bytes */
int mnn_tensor_get_size(const MNNTensor* tensor);

/** Get tensor host data pointer */
void* mnn_tensor_get_host_data(MNNTensor* tensor);

/** Get tensor data type code (halide_type) */
int mnn_tensor_get_type_code(const MNNTensor* tensor);

/** Get tensor dimension type (NHWC/NCHW/NC4HW4) */
int mnn_tensor_get_dimension_type(const MNNTensor* tensor);

#ifdef __cplusplus
}
#endif

#endif /* MNN_WRAPPER_H */