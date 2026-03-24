/**
 * @file mnn_module_wrapper.h
 * @brief C wrapper for MNN Module/Expr API (Dynamic Computation Graph)
 */

#ifndef MNN_MODULE_WRAPPER_H
#define MNN_MODULE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

/** Opaque handle to MNN Module */
typedef struct MNNModuleHandle MNNModule;

/** Opaque handle to MNN Variable (VARP) */
typedef struct MNNVarHandle MNNVar;

/** Opaque handle to MNN Executor */
typedef struct MNNExecutorHandle MNNExecutor;

/** Data format for Module */
typedef enum {
    MNN_DIMENSION_FORMAT_NHWC = 0,
    MNN_DIMENSION_FORMAT_NC4HW4 = 1,
    MNN_DIMENSION_FORMAT_NCHW = 2
} MNNDimensionFormat;

/** Module config */
typedef struct {
    bool dynamic;       /**< Load module as dynamic */
    bool shape_mutable; /**< Shape is mutable */
    bool rearrange;     /**< Pre-rearrange weights */
    int backend_type;   /**< Backend type (MNN_FORWARD_*) */
} MNNModuleConfig;

/** Variable info */
typedef struct {
    int* shape;
    int dim_count;
    int type_code;
    int type_bits;
    MNNDimensionFormat format;
} MNNVarInfo;

/* ============================================================================
 * Module Functions
 * ============================================================================ */

/** Load module from file
 * @param inputs Input tensor names
 * @param input_count Number of inputs
 * @param outputs Output tensor names
 * @param output_count Number of outputs
 * @param file_path Path to model file
 * @param config Optional config (can be NULL)
 * @return Loaded module or NULL on failure
 */
MNNModule* mnn_module_load_from_file(
    const char** inputs, int input_count,
    const char** outputs, int output_count,
    const char* file_path,
    const MNNModuleConfig* config
);

/** Load module from buffer
 * @param inputs Input tensor names
 * @param input_count Number of inputs
 * @param outputs Output tensor names
 * @param output_count Number of outputs
 * @param buffer Model data buffer
 * @param buffer_size Buffer size
 * @param config Optional config (can be NULL)
 * @return Loaded module or NULL on failure
 */
MNNModule* mnn_module_load_from_buffer(
    const char** inputs, int input_count,
    const char** outputs, int output_count,
    const uint8_t* buffer, size_t buffer_size,
    const MNNModuleConfig* config
);

/** Destroy module */
void mnn_module_destroy(MNNModule* module);

/** Clone module
 * @param module Source module
 * @param share_params Share parameters or copy
 * @return Cloned module or NULL on failure
 */
MNNModule* mnn_module_clone(const MNNModule* module, bool share_params);

/** Forward single input
 * @param module Module
 * @param input Input variable
 * @return Output variable or NULL on failure
 */
MNNVar* mnn_module_forward(MNNModule* module, const MNNVar* input);

/** Forward multiple inputs
 * @param module Module
 * @param inputs Input variables
 * @param input_count Number of inputs
 * @param output_count Number of outputs
 * @return Output variables array (caller must free)
 */
MNNVar** mnn_module_forward_multi(MNNModule* module,
                                   const MNNVar** inputs, int input_count,
                                   int* output_count);

/** Get module parameter count */
int mnn_module_get_parameter_count(const MNNModule* module);

/** Get module parameters */
MNNVar** mnn_module_get_parameters(const MNNModule* module, int* count);

/** Set training mode */
void mnn_module_set_training(MNNModule* module, bool is_training);

/** Get training mode */
bool mnn_module_is_training(const MNNModule* module);

/** Get module name */
const char* mnn_module_get_name(const MNNModule* module);

/** Set module name */
void mnn_module_set_name(MNNModule* module, const char* name);

/* ============================================================================
 * Variable (Expr) Functions
 * ============================================================================ */

/** Create input variable
 * @param shape Shape array
 * @param dim_count Number of dimensions
 * @param format Data format
 * @param type_code Data type code
 * @param type_bits Data type bits
 * @return New variable or NULL
 */
MNNVar* mnn_var_create_input(const int* shape, int dim_count,
                              MNNDimensionFormat format,
                              int type_code, int type_bits);

/** Create constant variable from float data
 * @param data Float data array
 * @param shape Shape array
 * @param dim_count Number of dimensions
 * @param format Data format
 * @return New variable or NULL
 */
MNNVar* mnn_var_create_constant_float(const float* data,
                                       const int* shape, int dim_count,
                                       MNNDimensionFormat format);

/** Destroy variable */
void mnn_var_destroy(MNNVar* var);

/** Get variable info */
MNNVarInfo* mnn_var_get_info(const MNNVar* var);

/** Free variable info */
void mnn_var_info_free(MNNVarInfo* info);

/** Read variable data as float
 * @param var Variable
 * @param count Number of elements
 * @return Float array (caller must free)
 */
float* mnn_var_read_float(const MNNVar* var, int* count);

/** Write float data to variable
 * @param var Variable
 * @param data Float data array
 * @param count Number of elements
 * @return true on success
 */
bool mnn_var_write_float(MNNVar* var, const float* data, int count);

/** Get variable shape
 * @param var Variable
 * @param count Output shape element count
 * @return Shape array (caller must free)
 */
int* mnn_var_get_shape(const MNNVar* var, int* count);

/** Variable arithmetic operations */
MNNVar* mnn_var_add(const MNNVar* a, const MNNVar* b);
MNNVar* mnn_var_sub(const MNNVar* a, const MNNVar* b);
MNNVar* mnn_var_mul(const MNNVar* a, const MNNVar* b);
MNNVar* mnn_var_div(const MNNVar* a, const MNNVar* b);

/** Variable reduce operations
 * @param var Variable
 * @param axes Axes to reduce (NULL for all)
 * @param axis_count Number of axes
 * @param keep_dims Keep reduced dimensions
 * @return Reduced variable
 */
MNNVar* mnn_var_sum(const MNNVar* var, const int* axes, int axis_count, bool keep_dims);
MNNVar* mnn_var_mean(const MNNVar* var, const int* axes, int axis_count, bool keep_dims);
MNNVar* mnn_var_max(const MNNVar* var, const int* axes, int axis_count, bool keep_dims);
MNNVar* mnn_var_min(const MNNVar* var, const int* axes, int axis_count, bool keep_dims);

/** Variable shape operations */
MNNVar* mnn_var_reshape(const MNNVar* var, const int* shape, int dim_count);
MNNVar* mnn_var_transpose(const MNNVar* var, const int* perm, int dim_count);

/** Variable concatenate */
MNNVar* mnn_var_concat(const MNNVar** vars, int var_count, int axis);

/** Variable split */
MNNVar** mnn_var_split(const MNNVar* var, int splits, int axis, int* output_count);

#ifdef __cplusplus
}
#endif

#endif /* MNN_MODULE_WRAPPER_H */
