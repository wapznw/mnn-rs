/**
 * @file mnn_module_wrapper.cpp
 * @brief C wrapper for MNN Module/Expr API implementation
 */

#include "mnn_module_wrapper.h"

#ifdef MNN_MODULE_ENABLED
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#else
// Stub implementation when MNN_MODULE_ENABLED is not defined
#endif

#include <vector>
#include <cstring>
#include <algorithm>

using namespace MNN::Express;

/* ============================================================================
 * Helper functions
 * ============================================================================ */

static std::vector<std::string> to_string_vector(const char** arr, int count) {
    std::vector<std::string> result;
    if (!arr || count <= 0) return result;

    result.reserve(count);
    for (int i = 0; i < count; i++) {
        if (arr[i]) {
            result.push_back(std::string(arr[i]));
        }
    }
    return result;
}

static std::vector<int> to_int_vector(const int* arr, int count) {
    std::vector<int> result;
    if (!arr || count <= 0) return result;

    result.assign(arr, arr + count);
    return result;
}

/* ============================================================================
 * Module Functions
 * ============================================================================ */

MNNModule* mnn_module_load_from_file(
    const char** inputs, int input_count,
    const char** outputs, int output_count,
    const char* file_path,
    const MNNModuleConfig* config
) {
#ifdef MNN_MODULE_ENABLED
    if (!file_path || !inputs || !outputs || input_count <= 0 || output_count <= 0) {
        return nullptr;
    }

    auto input_names = to_string_vector(inputs, input_count);
    auto output_names = to_string_vector(outputs, output_count);

    Module::Config module_config;
    if (config) {
        module_config.dynamic = config->dynamic;
        module_config.shapeMutable = config->shape_mutable;
        module_config.rearrange = config->rearrange;
    }

    Module* module = Module::load(input_names, output_names, file_path, &module_config);
    return reinterpret_cast<MNNModule*>(module);
#else
    return nullptr;
#endif
}

MNNModule* mnn_module_load_from_buffer(
    const char** inputs, int input_count,
    const char** outputs, int output_count,
    const uint8_t* buffer, size_t buffer_size,
    const MNNModuleConfig* config
) {
#ifdef MNN_MODULE_ENABLED
    if (!buffer || buffer_size == 0 || !inputs || !outputs || input_count <= 0 || output_count <= 0) {
        return nullptr;
    }

    auto input_names = to_string_vector(inputs, input_count);
    auto output_names = to_string_vector(outputs, output_count);

    Module::Config module_config;
    if (config) {
        module_config.dynamic = config->dynamic;
        module_config.shapeMutable = config->shape_mutable;
        module_config.rearrange = config->rearrange;
    }

    Module* module = Module::load(input_names, output_names, buffer, buffer_size, &module_config);
    return reinterpret_cast<MNNModule*>(module);
#else
    return nullptr;
#endif
}

void mnn_module_destroy(MNNModule* module) {
#ifdef MNN_MODULE_ENABLED
    if (module) {
        delete reinterpret_cast<Module*>(module);
    }
#else
    (void)module;
#endif
}

MNNModule* mnn_module_clone(const MNNModule* module, bool share_params) {
#ifdef MNN_MODULE_ENABLED
    if (!module) return nullptr;

    Module* cloned = Module::clone(reinterpret_cast<const Module*>(module), share_params);
    return reinterpret_cast<MNNModule*>(cloned);
#else
    return nullptr;
#endif
}

MNNVar* mnn_module_forward(MNNModule* module, const MNNVar* input) {
#ifdef MNN_MODULE_ENABLED
    if (!module || !input) return nullptr;

    auto* m = reinterpret_cast<Module*>(module);
    auto* v = reinterpret_cast<const VARP*>(input);

    VARP result = m->forward(*v);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar** mnn_module_forward_multi(MNNModule* module,
                                   const MNNVar** inputs, int input_count,
                                   int* output_count) {
#ifdef MNN_MODULE_ENABLED
    if (!module || !inputs || input_count <= 0 || !output_count) {
        if (output_count) *output_count = 0;
        return nullptr;
    }

    auto* m = reinterpret_cast<Module*>(module);

    std::vector<VARP> input_vars;
    input_vars.reserve(input_count);
    for (int i = 0; i < input_count; i++) {
        if (inputs[i]) {
            input_vars.push_back(*reinterpret_cast<const VARP*>(inputs[i]));
        }
    }

    auto output_vars = m->onForward(input_vars);
    *output_count = static_cast<int>(output_vars.size());

    MNNVar** result = new MNNVar*[output_vars.size()];
    for (size_t i = 0; i < output_vars.size(); i++) {
        result[i] = reinterpret_cast<MNNVar*>(new VARP(output_vars[i]));
    }

    return result;
#else
    if (output_count) *output_count = 0;
    return nullptr;
#endif
}

int mnn_module_get_parameter_count(const MNNModule* module) {
#ifdef MNN_MODULE_ENABLED
    if (!module) return 0;

    auto* m = reinterpret_cast<const Module*>(module);
    auto params = m->parameters();
    return static_cast<int>(params.size());
#else
    return 0;
#endif
}

MNNVar** mnn_module_get_parameters(const MNNModule* module, int* count) {
#ifdef MNN_MODULE_ENABLED
    if (!module || !count) {
        if (count) *count = 0;
        return nullptr;
    }

    auto* m = reinterpret_cast<const Module*>(module);
    auto params = m->parameters();

    *count = static_cast<int>(params.size());
    if (params.empty()) return nullptr;

    MNNVar** result = new MNNVar*[params.size()];
    for (size_t i = 0; i < params.size(); i++) {
        result[i] = reinterpret_cast<MNNVar*>(new VARP(params[i]));
    }

    return result;
#else
    if (count) *count = 0;
    return nullptr;
#endif
}

void mnn_module_set_training(MNNModule* module, bool is_training) {
#ifdef MNN_MODULE_ENABLED
    if (!module) return;

    auto* m = reinterpret_cast<Module*>(module);
    m->setIsTraining(is_training);
#else
    (void)module;
    (void)is_training;
#endif
}

bool mnn_module_is_training(const MNNModule* module) {
#ifdef MNN_MODULE_ENABLED
    if (!module) return false;

    auto* m = reinterpret_cast<const Module*>(module);
    return m->getIsTraining();
#else
    return false;
#endif
}

const char* mnn_module_get_name(const MNNModule* module) {
#ifdef MNN_MODULE_ENABLED
    if (!module) return "";

    auto* m = reinterpret_cast<const Module*>(module);
    static thread_local std::string name_storage;
    name_storage = m->name();
    return name_storage.c_str();
#else
    return "";
#endif
}

void mnn_module_set_name(MNNModule* module, const char* name) {
#ifdef MNN_MODULE_ENABLED
    if (!module || !name) return;

    auto* m = reinterpret_cast<Module*>(module);
    m->setName(std::string(name));
#else
    (void)module;
    (void)name;
#endif
}

/* ============================================================================
 * Variable (Expr) Functions
 * ============================================================================ */

MNNVar* mnn_var_create_input(const int* shape, int dim_count,
                              MNNDimensionFormat format,
                              int type_code, int type_bits) {
#ifdef MNN_MODULE_ENABLED
    if (!shape || dim_count <= 0) return nullptr;

    auto shape_vec = to_int_vector(shape, dim_count);

    // Create input variable
    VARP var = Variable::createInput(shape_vec, static_cast<Dimensionformat>(format));
    return reinterpret_cast<MNNVar*>(new VARP(var));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_create_constant_float(const float* data,
                                       const int* shape, int dim_count,
                                       MNNDimensionFormat format) {
#ifdef MNN_MODULE_ENABLED
    if (!data || !shape || dim_count <= 0) return nullptr;

    auto shape_vec = to_int_vector(shape, dim_count);

    // Create constant variable from data
    VARP var = _Const(data, shape_vec, static_cast<Dimensionformat>(format));
    return reinterpret_cast<MNNVar*>(new VARP(var));
#else
    return nullptr;
#endif
}

void mnn_var_destroy(MNNVar* var) {
#ifdef MNN_MODULE_ENABLED
    if (var) {
        delete reinterpret_cast<VARP*>(var);
    }
#else
    (void)var;
#endif
}

MNNVarInfo* mnn_var_get_info(const MNNVar* var) {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    if (!v->get()) return nullptr;

    MNNVarInfo* info = new MNNVarInfo;
    memset(info, 0, sizeof(MNNVarInfo));

    auto shape = v->getInfo()->dim;
    info->dim_count = static_cast<int>(shape.size());
    info->shape = new int[info->dim_count];
    std::copy(shape.begin(), shape.end(), info->shape);

    auto type = v->getInfo()->type;
    info->type_code = static_cast<int>(type.code);
    info->type_bits = type.bits;

    info->format = static_cast<MNNDimensionFormat>(v->getInfo()->order);

    return info;
#else
    return nullptr;
#endif
}

void mnn_var_info_free(MNNVarInfo* info) {
    if (!info) return;

    if (info->shape) {
        delete[] info->shape;
    }
    delete info;
}

float* mnn_var_read_float(const MNNVar* var, int* count) {
#ifdef MNN_MODULE_ENABLED
    if (!var || !count) {
        if (count) *count = 0;
        return nullptr;
    }

    auto* v = reinterpret_cast<const VARP*>(var);

    // Get info
    auto info = v->getInfo();
    if (!info) {
        *count = 0;
        return nullptr;
    }

    int element_count = 1;
    for (int dim : info->dim) {
        element_count *= dim;
    }
    *count = element_count;

    // Note: This is a simplified implementation
    // Actual reading requires proper tensor access
    float* data = new float[element_count];
    // In real implementation, we need to get the underlying tensor data
    // This requires MNN's internal API access
    memset(data, 0, element_count * sizeof(float));

    return data;
#else
    if (count) *count = 0;
    return nullptr;
#endif
}

bool mnn_var_write_float(MNNVar* var, const float* data, int count) {
#ifdef MNN_MODULE_ENABLED
    if (!var || !data || count <= 0) return false;

    auto* v = reinterpret_cast<VARP*>(var);
    // Note: Writing requires proper tensor access which is complex
    // This is a placeholder
    (void)v;
    return false;
#else
    return false;
#endif
}

int* mnn_var_get_shape(const MNNVar* var, int* count) {
#ifdef MNN_MODULE_ENABLED
    if (!var || !count) {
        if (count) *count = 0;
        return nullptr;
    }

    auto* v = reinterpret_cast<const VARP*>(var);
    if (!v->get()) {
        *count = 0;
        return nullptr;
    }

    auto shape = v->getInfo()->dim;
    *count = static_cast<int>(shape.size());

    int* result = new int[*count];
    std::copy(shape.begin(), shape.end(), result);

    return result;
#else
    if (count) *count = 0;
    return nullptr;
#endif
}

/* ============================================================================
 * Arithmetic Operations
 * ============================================================================ */

MNNVar* mnn_var_add(const MNNVar* a, const MNNVar* b) {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va + *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_sub(const MNNVar* a, const MNNVar* b) {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va - *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_mul(const MNNVar* a, const MNNVar* b) {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va * *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_div(const MNNVar* a, const MNNVar* b) {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va / *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

/* ============================================================================
 * Reduce Operations
 * ============================================================================ */

MNNVar* mnn_var_sum(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceSum(*v, dims);
    if (keep_dims) {
        // Keep dims handling would require additional MNN API
    }

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_mean(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceMean(*v, dims);

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_max(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceMax(*v, dims);

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_min(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceMin(*v, dims);

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

/* ============================================================================
 * Shape Operations
 * ============================================================================ */

MNNVar* mnn_var_reshape(const MNNVar* var, const int* shape, int dim_count) {
#ifdef MNN_MODULE_ENABLED
    if (!var || !shape || dim_count <= 0) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    auto shape_vec = to_int_vector(shape, dim_count);

    VARP result = _Reshape(*v, shape_vec);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_transpose(const MNNVar* var, const int* perm, int dim_count) {
#ifdef MNN_MODULE_ENABLED
    if (!var || !perm || dim_count <= 0) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    auto perm_vec = to_int_vector(perm, dim_count);

    VARP result = _Transpose(*v, perm_vec);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar* mnn_var_concat(const MNNVar** vars, int var_count, int axis) {
#ifdef MNN_MODULE_ENABLED
    if (!vars || var_count <= 0) return nullptr;

    std::vector<VARP> var_list;
    var_list.reserve(var_count);
    for (int i = 0; i < var_count; i++) {
        if (vars[i]) {
            var_list.push_back(*reinterpret_cast<const VARP*>(vars[i]));
        }
    }

    VARP result = _Concat(var_list, axis);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif
}

MNNVar** mnn_var_split(const MNNVar* var, int splits, int axis, int* output_count) {
#ifdef MNN_MODULE_ENABLED
    if (!var || !output_count || splits <= 0) {
        if (output_count) *output_count = 0;
        return nullptr;
    }

    auto* v = reinterpret_cast<const VARP*>(var);

    auto results = _Split(*v, splits, axis);
    *output_count = static_cast<int>(results.size());

    MNNVar** result = new MNNVar*[results.size()];
    for (size_t i = 0; i < results.size(); i++) {
        result[i] = reinterpret_cast<MNNVar*>(new VARP(results[i]));
    }

    return result;
#else
    if (output_count) *output_count = 0;
    return nullptr;
#endif
}
