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
#include <stdint.h>

/* ============================================================================
 * Type Definitions (using MNN's types)
 * ============================================================================ */

/** Opaque handle to MNN Interpreter */
typedef struct MNNInterpreterHandle MNNInterpreter;

/** Opaque handle to MNN Session */
typedef struct MNNSessionHandle MNNSession;

/** Opaque handle to MNN Tensor */
typedef struct MNNTensorHandle MNNTensor;

/** Opaque handle to MNN ImageProcess */
typedef struct MNNImageProcessHandle MNNImageProcess;

/** Opaque handle to MNN Matrix */
typedef struct MNNMatrixHandle MNNMatrix;

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

/* Image format (matches MNN::CV::ImageFormat) */
#define MNN_IMAGE_FORMAT_RGBA      0
#define MNN_IMAGE_FORMAT_RGB       1
#define MNN_IMAGE_FORMAT_BGR       2
#define MNN_IMAGE_FORMAT_GRAY      3
#define MNN_IMAGE_FORMAT_BGRA      4
#define MNN_IMAGE_FORMAT_YCRCB     5
#define MNN_IMAGE_FORMAT_YUV       6
#define MNN_IMAGE_FORMAT_HSV       7
#define MNN_IMAGE_FORMAT_XYZ       8
#define MNN_IMAGE_FORMAT_BGR555    9
#define MNN_IMAGE_FORMAT_BGR565    10
#define MNN_IMAGE_FORMAT_YUV_NV21  11
#define MNN_IMAGE_FORMAT_YUV_NV12  12
#define MNN_IMAGE_FORMAT_YUV_I420  13
#define MNN_IMAGE_FORMAT_HSV_FULL  14

/* Filter type (matches MNN::CV::Filter) */
#define MNN_FILTER_NEAREST   0
#define MNN_FILTER_BILINEAR  1
#define MNN_FILTER_BICUBIC   2

/* Wrap type (matches MNN::CV::Wrap) */
#define MNN_WRAP_CLAMP_TO_EDGE  0
#define MNN_WRAP_ZERO           1
#define MNN_WRAP_REPEAT         2

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

/* ============================================================================
 * ImageProcess Functions
 * ============================================================================ */

/** ImageProcess config structure */
typedef struct {
    int filterType;           /**< Filter type: NEAREST, BILINEAR, BICUBIC */
    int sourceFormat;         /**< Source image format */
    int destFormat;           /**< Destination image format */
    float mean[4];            /**< Mean values for normalization */
    float normal[4];          /**< Normalization scales */
    int wrap;                 /**< Edge wrap type */
} MNNImageProcessConfig;

/** Create image process with config */
MNNImageProcess* mnn_image_process_create(const MNNImageProcessConfig* config);

/** Destroy image process */
void mnn_image_process_destroy(MNNImageProcess* process);

/** Set transform matrix */
void mnn_image_process_set_matrix(MNNImageProcess* process, const MNNMatrix* matrix);

/** Convert image to tensor
 * @param process image process
 * @param source source image data
 * @param iw source image width
 * @param ih source image height
 * @param stride bytes per row
 * @param tensor destination tensor
 * @return error code
 */
int mnn_image_process_convert(MNNImageProcess* process, const uint8_t* source,
                               int iw, int ih, int stride, MNNTensor* tensor);

/** Create image tensor
 * @param w image width
 * @param h image height
 * @param bpp bytes per pixel
 * @param data optional pixel data
 * @return created tensor
 */
MNNTensor* mnn_image_tensor_create(int w, int h, int bpp, void* data);

/** Destroy image tensor */
void mnn_image_tensor_destroy(MNNTensor* tensor);

/**
 * @brief Read image from file using MNN CV
 * @param path Image file path
 * @param flags Read flags (same as OpenCV imread flags)
 * @return created tensor (uint8 type), NULL on failure
 *
 * Flags:
 *   -1 = Unchanged
 *    0 = Grayscale
 *    1 = Color (BGR)
 */
MNNTensor* mnn_imread(const char* path, int flags);

/**
 * @brief Write image to file using MNN CV
 * @param path Output file path
 * @param tensor Input tensor (uint8 type)
 * @param params Optional parameters (not used currently)
 * @return 0 on success, error code on failure
 */
int mnn_imwrite(const char* path, const MNNTensor* tensor, const void* params);

/**
 * @brief Resize image tensor
 * @param src Source tensor
 * @param dstWidth Destination width
 * @param dstHeight Destination height
 * @param filter Filter type (0=nearest, 1=bilinear, 2=bicubic)
 * @return New resized tensor, NULL on failure
 */
MNNTensor* mnn_resize(const MNNTensor* src, int dstWidth, int dstHeight, int filter);

/* ============================================================================
 * Matrix Functions
 * ============================================================================ */

/** Create identity matrix */
MNNMatrix* mnn_matrix_create_identity(void);

/** Create scale matrix */
MNNMatrix* mnn_matrix_create_scale(float sx, float sy);

/** Create translate matrix */
MNNMatrix* mnn_matrix_create_translate(float dx, float dy);

/** Create rotate matrix (degrees) */
MNNMatrix* mnn_matrix_create_rotate(float degrees);

/** Create matrix from raw data (9 floats) */
MNNMatrix* mnn_matrix_create(const float* data);

/** Clone matrix */
MNNMatrix* mnn_matrix_clone(const MNNMatrix* matrix);

/** Destroy matrix */
void mnn_matrix_destroy(MNNMatrix* matrix);

/** Get matrix element at (row, col) */
float mnn_matrix_get(const MNNMatrix* matrix, int row, int col);

/** Set matrix element at (row, col) */
void mnn_matrix_set(MNNMatrix* matrix, int row, int col, float value);

/** Multiply two matrices */
MNNMatrix* mnn_matrix_multiply(const MNNMatrix* a, const MNNMatrix* b);

/** Invert matrix */
MNNMatrix* mnn_matrix_invert(const MNNMatrix* matrix);

/* ============================================================================
 * Tensor Advanced Functions (GPU Memory Operations)
 * ============================================================================ */

/** Map type for GPU memory access */
#define MNN_MAP_TYPE_READ   0
#define MNN_MAP_TYPE_WRITE  1

/** Copy data from host tensor to device tensor */
int mnn_tensor_copy_from_host(MNNTensor* dest, const MNNTensor* host_tensor);

/** Copy data from device tensor to host tensor */
int mnn_tensor_copy_to_host(MNNTensor* host_tensor, const MNNTensor* dest);

/** Create a device tensor with given shape */
MNNTensor* mnn_tensor_create_device(
    const int* shape,
    int dimensions,
    int type_code,
    int format
);

/** Clone a tensor
 * @param tensor source tensor
 * @param deep_copy if 1, copy data; if 0, only copy metadata
 * @return cloned tensor
 */
MNNTensor* mnn_tensor_clone(const MNNTensor* tensor, int deep_copy);

/** Destroy a user-created tensor */
void mnn_tensor_destroy(MNNTensor* tensor);

/** Get tensor device ID (for GPU tensors) */
uint64_t mnn_tensor_device_id(const MNNTensor* tensor);

/** Get tensor backend type */
int mnn_tensor_get_backend(const MNNTensor* tensor);

/* ============================================================================
 * Session Advanced Functions
 * ============================================================================ */

/** Session mode */
#define MNN_SESSION_MODE_DEBUG          0
#define MNN_SESSION_MODE_RELEASE        1
#define MNN_SESSION_MODE_INPUT_INSIDE   2
#define MNN_SESSION_MODE_INPUT_USER     3
#define MNN_SESSION_MODE_OUTPUT_INSIDE  4
#define MNN_SESSION_MODE_OUTPUT_USER    5
#define MNN_SESSION_MODE_RESIZE_DIRECT  6
#define MNN_SESSION_MODE_RESIZE_DEFER   7
#define MNN_SESSION_MODE_BACKEND_FIX    8
#define MNN_SESSION_MODE_BACKEND_AUTO   9

/** Set session mode */
void mnn_interpreter_set_session_mode(MNNInterpreter* interpreter, int mode);

/** Set cache file for optimization */
void mnn_interpreter_set_cache_file(MNNInterpreter* interpreter, const char* path, size_t key_size);

/** Update cache from session */
int mnn_interpreter_update_cache(MNNInterpreter* interpreter, MNNSession* session);

/** Set external file for model */
void mnn_interpreter_set_external_file(MNNInterpreter* interpreter, const char* path, size_t flag);

/** Get input tensor names */
typedef struct {
    char** names;
    int count;
} MNNStringArray;

MNNStringArray mnn_interpreter_get_input_names(MNNInterpreter* interpreter, MNNSession* session);
MNNStringArray mnn_interpreter_get_output_names(MNNInterpreter* interpreter, MNNSession* session);

/** Free string array */
void mnn_string_array_free(MNNStringArray* array);

/** Resize tensor with new shape */
void mnn_interpreter_resize_tensor(MNNInterpreter* interpreter, MNNTensor* tensor, const int* shape, int dims);

/** Get operator info count */
int mnn_interpreter_get_session_op_count(MNNInterpreter* interpreter, MNNSession* session);

/* ============================================================================
 * Runtime Management (Multi-Session Sharing)
 * ============================================================================ */

/** Opaque handle to MNN RuntimeManager */
typedef struct MNNRuntimeManagerHandle MNNRuntimeManager;

/** Create runtime manager from config */
MNNRuntimeManager* mnn_runtime_manager_create(int type, int num_threads);

/** Destroy runtime manager */
void mnn_runtime_manager_destroy(MNNRuntimeManager* manager);

/** Create session with shared runtime
 * @param interpreter Interpreter
 * @param runtime Runtime manager
 * @param type Backend type
 * @param num_threads Number of threads
 * @return Session or NULL on failure
 */
MNNSession* mnn_interpreter_create_session_with_runtime(
    MNNInterpreter* interpreter,
    MNNRuntimeManager* runtime,
    int type,
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif /* MNN_WRAPPER_H */