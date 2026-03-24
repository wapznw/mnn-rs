/**
 * @file mnn_wrapper.cpp
 * @brief C wrapper implementation for MNN C++ API
 */

#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Matrix.h>
#ifdef MNN_IMGCODECS
#include <MNN/cv/imgcodecs.hpp>
#endif
#ifdef MNN_BUILD_OPENCV
#include <MNN/cv/imgproc/geometric.hpp>
#endif
#ifdef MNN_MODULE_ENABLED
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#endif
#include <cstring>
#include <cstdio>
#include <memory>

using namespace MNN;

/* ============================================================================
 * Version and Info
 * ============================================================================ */

const char* mnn_get_version(void) {
    return getVersion();
}

int mnn_is_backend_available(int type) {
    // CPU is always available
    if (type == MNN_FORWARD_CPU || type == 0) {
        return 1;
    }
    return 0;
}

/* ============================================================================
 * Interpreter Functions
 * ============================================================================ */

MNNInterpreter* mnn_interpreter_create_from_file(const char* file) {
    if (!file) {
        return nullptr;
    }
    Interpreter* interpreter = Interpreter::createFromFile(file);
    return reinterpret_cast<MNNInterpreter*>(interpreter);
}

MNNInterpreter* mnn_interpreter_create_from_buffer(const void* buffer, size_t size) {
    if (!buffer || size == 0) {
        return nullptr;
    }
    Interpreter* interpreter = Interpreter::createFromBuffer(buffer, size);
    return reinterpret_cast<MNNInterpreter*>(interpreter);
}

void mnn_interpreter_destroy(MNNInterpreter* interpreter) {
    if (interpreter) {
        Interpreter::destroy(reinterpret_cast<Interpreter*>(interpreter));
    }
}

MNNSession* mnn_interpreter_create_session(MNNInterpreter* interpreter, int type, int num_thread) {
    if (!interpreter) {
        return nullptr;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);

    ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(type);
    config.numThread = num_thread > 0 ? num_thread : 4;
    // Leave backendConfig as nullptr (default)

    Session* session = interp->createSession(config);
    return reinterpret_cast<MNNSession*>(session);
}

void mnn_interpreter_release_session(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    interp->releaseSession(sess);
}

int mnn_interpreter_run_session(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return MNN_ERROR_EXECUTION;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    ErrorCode result = interp->runSession(sess);
    return static_cast<int>(result);
}

MNNTensor* mnn_interpreter_get_session_input(MNNInterpreter* interpreter, MNNSession* session, const char* name) {
    if (!interpreter || !session) {
        return nullptr;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    Tensor* tensor = interp->getSessionInput(sess, name);
    return reinterpret_cast<MNNTensor*>(tensor);
}

MNNTensor* mnn_interpreter_get_session_output(MNNInterpreter* interpreter, MNNSession* session, const char* name) {
    if (!interpreter || !session) {
        return nullptr;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    Tensor* tensor = interp->getSessionOutput(sess, name);
    return reinterpret_cast<MNNTensor*>(tensor);
}

void mnn_interpreter_resize_session(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    interp->resizeSession(sess);
}

float mnn_interpreter_get_session_memory(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return 0.0f;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    float memory = 0.0f;
    interp->getSessionInfo(sess, Interpreter::MEMORY, &memory);
    return memory;
}

float mnn_interpreter_get_session_flops(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return 0.0f;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    float flops = 0.0f;
    interp->getSessionInfo(sess, Interpreter::FLOPS, &flops);
    return flops;
}

const char* mnn_interpreter_get_biz_code(MNNInterpreter* interpreter) {
    if (!interpreter) {
        return "";
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    return interp->bizCode();
}

const char* mnn_interpreter_get_uuid(MNNInterpreter* interpreter) {
    if (!interpreter) {
        return "";
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    return interp->uuid();
}

/* ============================================================================
 * Tensor Functions
 * ============================================================================ */

int mnn_tensor_get_dimensions(const MNNTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    auto t = reinterpret_cast<const Tensor*>(tensor);
    return t->dimensions();
}

int mnn_tensor_get_dim(const MNNTensor* tensor, int index) {
    if (!tensor || index < 0) {
        return 0;
    }

    auto t = reinterpret_cast<const Tensor*>(tensor);
    if (index >= t->dimensions()) {
        return 0;
    }
    return t->length(index);
}

int mnn_tensor_get_element_count(const MNNTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    auto t = reinterpret_cast<const Tensor*>(tensor);
    return t->elementSize();
}

int mnn_tensor_get_size(const MNNTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    auto t = reinterpret_cast<const Tensor*>(tensor);
    return t->size();
}

void* mnn_tensor_get_host_data(MNNTensor* tensor) {
    if (!tensor) {
        return nullptr;
    }

    auto t = reinterpret_cast<Tensor*>(tensor);
    return t->host<void>();
}

int mnn_tensor_get_type_code(const MNNTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    auto t = reinterpret_cast<const Tensor*>(tensor);
    auto type = t->getType();
    return (type.code << 8) | type.bits;
}

int mnn_tensor_get_dimension_type(const MNNTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    auto t = reinterpret_cast<const Tensor*>(tensor);
    return static_cast<int>(t->getDimensionType());
}

/* ============================================================================
 * ImageProcess Functions
 * ============================================================================ */

#ifdef MNN_IMAGE_PROCESS

MNNImageProcess* mnn_image_process_create(const MNNImageProcessConfig* config) {
    if (!config) {
        return nullptr;
    }

    MNN::CV::ImageProcess::Config cfg;
    cfg.filterType = static_cast<MNN::CV::Filter>(config->filterType);
    cfg.sourceFormat = static_cast<MNN::CV::ImageFormat>(config->sourceFormat);
    cfg.destFormat = static_cast<MNN::CV::ImageFormat>(config->destFormat);
    cfg.wrap = static_cast<MNN::CV::Wrap>(config->wrap);

    for (int i = 0; i < 4; ++i) {
        cfg.mean[i] = config->mean[i];
        cfg.normal[i] = config->normal[i];
    }

    auto process = MNN::CV::ImageProcess::create(cfg);
    return reinterpret_cast<MNNImageProcess*>(process);
}

void mnn_image_process_destroy(MNNImageProcess* process) {
    if (process) {
        MNN::CV::ImageProcess::destroy(reinterpret_cast<MNN::CV::ImageProcess*>(process));
    }
}

void mnn_image_process_set_matrix(MNNImageProcess* process, const MNNMatrix* matrix) {
    if (!process || !matrix) {
        return;
    }

    auto proc = reinterpret_cast<MNN::CV::ImageProcess*>(process);
    auto mat = reinterpret_cast<const MNN::CV::Matrix*>(matrix);
    proc->setMatrix(*mat);
}

int mnn_image_process_convert(MNNImageProcess* process, const uint8_t* source,
                               int iw, int ih, int stride, MNNTensor* tensor) {
    if (!process || !source || !tensor) {
        return MNN_ERROR_EXECUTION;
    }

    auto proc = reinterpret_cast<MNN::CV::ImageProcess*>(process);
    auto t = reinterpret_cast<MNN::Tensor*>(tensor);

    MNN::ErrorCode result = proc->convert(source, iw, ih, stride, t);
    return static_cast<int>(result);
}

MNNTensor* mnn_image_tensor_create(int w, int h, int bpp, void* data) {
    if (w <= 0 || h <= 0 || bpp <= 0) {
        return nullptr;
    }

    auto tensor = MNN::CV::ImageProcess::createImageTensor<uint8_t>(w, h, bpp, data);
    return reinterpret_cast<MNNTensor*>(tensor);
}

void mnn_image_tensor_destroy(MNNTensor* tensor) {
    if (tensor) {
        // For user-created tensors, we need to delete them
        delete reinterpret_cast<MNN::Tensor*>(tensor);
    }
}

/* ============================================================================
 * MNN CV Image Codecs (imread/imwrite/resize)
 * Requires: -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON
 * ============================================================================ */

MNNTensor* mnn_imread(const char* path, int flags) {
#ifdef MNN_IMGCODECS
    if (!path) {
        return nullptr;
    }

    auto varp = MNN::CV::imread(std::string(path), flags);
    if (varp == nullptr) {
        return nullptr;
    }

    // Compute the VARP to get actual data
    MNN::Express::Variable::compute({varp});

    // Get tensor from VARP after computation
    auto tensor = varp->getTensor();
    if (!tensor) {
        return nullptr;
    }

    // Get tensor info
    auto type = tensor->getType();
    auto dims = tensor->shape();

    // Create a new tensor with the same shape and type
    auto owned_tensor = MNN::Tensor::create(dims, type, nullptr, MNN::Tensor::DimensionType::TENSORFLOW);
    if (!owned_tensor) {
        return nullptr;
    }

    // Copy data
    auto srcHost = tensor->host<uint8_t>();
    auto dstHost = owned_tensor->host<uint8_t>();
    if (srcHost && dstHost) {
        auto bytes = owned_tensor->size();
        memcpy(dstHost, srcHost, bytes);
    }

    // Return the owned tensor (caller must destroy it using mnn_tensor_destroy)
    return reinterpret_cast<MNNTensor*>(owned_tensor);
#else
    (void)path;
    (void)flags;
    return nullptr;
#endif
}

int mnn_imwrite(const char* path, const MNNTensor* tensor, const void* params) {
#ifdef MNN_IMGCODECS
    if (!path || !tensor) {
        return -1;
    }

    auto t = reinterpret_cast<const MNN::Tensor*>(tensor);

    // Create VARP from tensor: use Expr::create() and Variable::create()
    // const_cast is safe here as we don't modify the tensor and don't take ownership
    auto expr = MNN::Express::Expr::create(const_cast<MNN::Tensor*>(t), false);
    auto varp = MNN::Express::Variable::create(expr, 0);

    // Convert params to std::vector<int> (not currently used from Rust)
    std::vector<int> empty_params;
    bool result = MNN::CV::imwrite(std::string(path), varp, empty_params);
    return result ? 0 : -1;
#else
    (void)path;
    (void)tensor;
    (void)params;
    return -1;
#endif
}

MNNTensor* mnn_resize(const MNNTensor* src, int dstWidth, int dstHeight, int filter) {
#ifdef MNN_BUILD_OPENCV
    if (!src || dstWidth <= 0 || dstHeight <= 0) {
        return nullptr;
    }

    auto srcTensor = reinterpret_cast<const MNN::Tensor*>(src);

    // Filter: 0=nearest, 1=bilinear, 2=bicubic
    int method = MNN::CV::INTER_LINEAR;
    if (filter == 0) {
        method = MNN::CV::INTER_NEAREST;
    } else if (filter == 2) {
        method = MNN::CV::INTER_CUBIC;
    }

    // Create VARP from tensor: use Expr::create() and Variable::create()
    // const_cast is safe here as we don't modify the tensor and don't take ownership
    auto expr = MNN::Express::Expr::create(const_cast<MNN::Tensor*>(srcTensor), false);
    auto varp = MNN::Express::Variable::create(expr, 0);

    // MNN::CV::resize signature: resize(VARP src, Size dsize, double fx, double fy, int interpolation, ...)
    MNN::CV::Size dsize = {dstWidth, dstHeight};
    auto resized = MNN::CV::resize(varp, dsize, 0, 0, method);

    if (resized == nullptr) {
        return nullptr;
    }

    // Compute the VARP to get actual data
    MNN::Express::Variable::compute({resized});

    // Get tensor from VARP after computation
    auto tensor = resized->getTensor();
    if (!tensor) {
        return nullptr;
    }

    // Get tensor info
    auto type = tensor->getType();
    auto dims = tensor->shape();

    // Create a new tensor with the same shape and type
    auto owned_tensor = MNN::Tensor::create(dims, type, nullptr, MNN::Tensor::DimensionType::TENSORFLOW);
    if (!owned_tensor) {
        return nullptr;
    }

    // Copy data
    auto srcHost = tensor->host<uint8_t>();
    auto dstHost = owned_tensor->host<uint8_t>();
    if (srcHost && dstHost) {
        auto bytes = owned_tensor->size();
        memcpy(dstHost, srcHost, bytes);
    }

    // Return the owned tensor (caller must destroy it using mnn_tensor_destroy)
    return reinterpret_cast<MNNTensor*>(owned_tensor);
#else
    (void)src;
    (void)dstWidth;
    (void)dstHeight;
    (void)filter;
    return nullptr;
#endif
}

#endif /* MNN_IMAGE_PROCESS */

/* ============================================================================
 * Matrix Functions
 * ============================================================================ */

MNNMatrix* mnn_matrix_create_identity(void) {
    auto matrix = new MNN::CV::Matrix();
    matrix->setIdentity();
    return reinterpret_cast<MNNMatrix*>(matrix);
}

MNNMatrix* mnn_matrix_create_scale(float sx, float sy) {
    auto matrix = new MNN::CV::Matrix();
    matrix->setScale(sx, sy);
    return reinterpret_cast<MNNMatrix*>(matrix);
}

MNNMatrix* mnn_matrix_create_translate(float dx, float dy) {
    auto matrix = new MNN::CV::Matrix();
    matrix->setTranslate(dx, dy);
    return reinterpret_cast<MNNMatrix*>(matrix);
}

MNNMatrix* mnn_matrix_create_rotate(float degrees) {
    auto matrix = new MNN::CV::Matrix();
    matrix->setRotate(degrees);
    return reinterpret_cast<MNNMatrix*>(matrix);
}

MNNMatrix* mnn_matrix_create(const float* data) {
    if (!data) {
        return nullptr;
    }

    auto matrix = new MNN::CV::Matrix();
    matrix->set9(data);
    return reinterpret_cast<MNNMatrix*>(matrix);
}

MNNMatrix* mnn_matrix_clone(const MNNMatrix* matrix) {
    if (!matrix) {
        return nullptr;
    }

    auto src = reinterpret_cast<const MNN::CV::Matrix*>(matrix);
    auto dst = new MNN::CV::Matrix(*src);
    return reinterpret_cast<MNNMatrix*>(dst);
}

void mnn_matrix_destroy(MNNMatrix* matrix) {
    if (matrix) {
        delete reinterpret_cast<MNN::CV::Matrix*>(matrix);
    }
}

float mnn_matrix_get(const MNNMatrix* matrix, int row, int col) {
    if (!matrix || row < 0 || row > 2 || col < 0 || col > 2) {
        return 0.0f;
    }

    auto m = reinterpret_cast<const MNN::CV::Matrix*>(matrix);
    int index = row * 3 + col;
    return m->get(index);
}

void mnn_matrix_set(MNNMatrix* matrix, int row, int col, float value) {
    if (!matrix || row < 0 || row > 2 || col < 0 || col > 2) {
        return;
    }

    auto m = reinterpret_cast<MNN::CV::Matrix*>(matrix);
    int index = row * 3 + col;
    m->set(index, value);
}

MNNMatrix* mnn_matrix_multiply(const MNNMatrix* a, const MNNMatrix* b) {
    if (!a || !b) {
        return nullptr;
    }

    auto ma = reinterpret_cast<const MNN::CV::Matrix*>(a);
    auto mb = reinterpret_cast<const MNN::CV::Matrix*>(b);
    auto result = new MNN::CV::Matrix();
    result->setConcat(*ma, *mb);
    return reinterpret_cast<MNNMatrix*>(result);
}

MNNMatrix* mnn_matrix_invert(const MNNMatrix* matrix) {
    if (!matrix) {
        return nullptr;
    }

    auto m = reinterpret_cast<const MNN::CV::Matrix*>(matrix);
    auto result = new MNN::CV::Matrix();
    if (!m->invert(result)) {
        delete result;
        return nullptr;
    }
    return reinterpret_cast<MNNMatrix*>(result);
}

/* ============================================================================
 * Tensor Advanced Functions (GPU Memory Operations)
 * ============================================================================ */

int mnn_tensor_copy_from_host(MNNTensor* dest, const MNNTensor* host_tensor) {
    if (!dest || !host_tensor) {
        return MNN_ERROR_EXECUTION;
    }

    auto d = reinterpret_cast<MNN::Tensor*>(dest);
    auto h = reinterpret_cast<const MNN::Tensor*>(host_tensor);

    bool result = d->copyFromHostTensor(h);
    return result ? MNN_ERROR_NONE : MNN_ERROR_EXECUTION;
}

int mnn_tensor_copy_to_host(MNNTensor* host_tensor, const MNNTensor* dest) {
    if (!host_tensor || !dest) {
        return MNN_ERROR_EXECUTION;
    }

    auto h = reinterpret_cast<MNN::Tensor*>(host_tensor);
    auto d = reinterpret_cast<const MNN::Tensor*>(dest);

    bool result = d->copyToHostTensor(h);
    return result ? MNN_ERROR_NONE : MNN_ERROR_EXECUTION;
}

MNNTensor* mnn_tensor_create_device(
    const int* shape,
    int dimensions,
    int type_code,
    int format
) {
    if (!shape || dimensions <= 0) {
        return nullptr;
    }

    // Convert to std::vector<int>
    std::vector<int> shape_vec;
    for (int i = 0; i < dimensions; i++) {
        shape_vec.push_back(shape[i]);
    }

    // Create device tensor using MNN's static method with default float type
    auto tensor = MNN::Tensor::createDevice<float>(shape_vec,
                         static_cast<MNN::Tensor::DimensionType>(format));
    return reinterpret_cast<MNNTensor*>(tensor);
}

MNNTensor* mnn_tensor_clone(const MNNTensor* tensor, int deep_copy) {
    if (!tensor) {
        return nullptr;
    }

    auto t = reinterpret_cast<const MNN::Tensor*>(tensor);

    // Use MNN's static clone method
    auto cloned = MNN::Tensor::clone(t, deep_copy != 0);
    return reinterpret_cast<MNNTensor*>(cloned);
}

void mnn_tensor_destroy(MNNTensor* tensor) {
    if (tensor) {
        MNN::Tensor::destroy(reinterpret_cast<MNN::Tensor*>(tensor));
    }
}

uint64_t mnn_tensor_device_id(const MNNTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    // MNN doesn't directly expose device ID per tensor
    // This is a placeholder - actual implementation would need MNN backend info
    return 0;
}

int mnn_tensor_get_backend(const MNNTensor* tensor) {
    if (!tensor) {
        return MNN_FORWARD_CPU;
    }

    // Return the backend type - this requires MNN internal access
    // For now, return CPU as default
    return MNN_FORWARD_CPU;
}

/* ============================================================================
 * Session Advanced Functions
 * ============================================================================ */

void mnn_interpreter_set_session_mode(MNNInterpreter* interpreter, int mode) {
    if (!interpreter) {
        return;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    // Mode mapping depends on MNN's SessionMode enum
    interp->setSessionMode(static_cast<Interpreter::SessionMode>(mode));
}

void mnn_interpreter_set_cache_file(MNNInterpreter* interpreter, const char* path, size_t key_size) {
    if (!interpreter || !path) {
        return;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    interp->setCacheFile(path, key_size);
}

int mnn_interpreter_update_cache(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return MNN_ERROR_EXECUTION;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);
    ErrorCode result = interp->updateCacheFile(sess);
    return static_cast<int>(result);
}

void mnn_interpreter_set_external_file(MNNInterpreter* interpreter, const char* path, size_t flag) {
    if (!interpreter || !path) {
        return;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    interp->setExternalFile(path, flag);
}

MNNStringArray mnn_interpreter_get_input_names(MNNInterpreter* interpreter, MNNSession* session) {
    MNNStringArray result = { nullptr, 0 };

    if (!interpreter || !session) {
        return result;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);

    auto& inputs = interp->getSessionInputAll(sess);
    if (inputs.empty()) {
        return result;
    }

    result.count = static_cast<int>(inputs.size());
    result.names = new char*[inputs.size()];
    size_t idx = 0;
    for (const auto& pair : inputs) {
        result.names[idx] = new char[pair.first.length() + 1];
        strcpy(result.names[idx], pair.first.c_str());
        idx++;
    }

    return result;
}

MNNStringArray mnn_interpreter_get_output_names(MNNInterpreter* interpreter, MNNSession* session) {
    MNNStringArray result = { nullptr, 0 };

    if (!interpreter || !session) {
        return result;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);

    auto& outputs = interp->getSessionOutputAll(sess);
    if (outputs.empty()) {
        return result;
    }

    result.count = static_cast<int>(outputs.size());
    result.names = new char*[outputs.size()];
    size_t idx = 0;
    for (const auto& pair : outputs) {
        result.names[idx] = new char[pair.first.length() + 1];
        strcpy(result.names[idx], pair.first.c_str());
        idx++;
    }

    return result;
}

void mnn_string_array_free(MNNStringArray* array) {
    if (!array || !array->names) {
        return;
    }

    for (int i = 0; i < array->count; i++) {
        delete[] array->names[i];
    }
    delete[] array->names;
    array->names = nullptr;
    array->count = 0;
}

void mnn_interpreter_resize_tensor(MNNInterpreter* interpreter, MNNTensor* tensor, const int* shape, int dims) {
    if (!interpreter || !tensor || !shape || dims <= 0) {
        return;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto t = reinterpret_cast<Tensor*>(tensor);

    std::vector<int> shape_vec(shape, shape + dims);
    interp->resizeTensor(t, shape_vec);
}

int mnn_interpreter_get_session_op_count(MNNInterpreter* interpreter, MNNSession* session) {
    if (!interpreter || !session) {
        return 0;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto sess = reinterpret_cast<Session*>(session);

    // Get FLOPS as a measure of operator complexity
    float flops = 0.0f;
    interp->getSessionInfo(sess, Interpreter::FLOPS, &flops);
    return static_cast<int>(flops);
}

/* ============================================================================
 * Runtime Management (Multi-Session Sharing)
 * Note: Requires MNN with full Express/Module support
 * ============================================================================ */

MNNRuntimeManager* mnn_runtime_manager_create(int type, int num_threads) {
#ifdef MNN_RUNTIME_ENABLED
    // Create schedule config
    ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(type);
    config.numThread = num_threads > 0 ? num_threads : 4;

    // Create runtime manager using MNN's Executor
    // Note: Requires MNN with Express::Executor support
    auto* rtmgr = MNN::Express::Executor::RuntimeManager::createRuntimeManager(config);
    return reinterpret_cast<MNNRuntimeManager*>(rtmgr);
#else
    // Runtime manager not enabled, return nullptr
    (void)type;
    (void)num_threads;
    return nullptr;
#endif
}

void mnn_runtime_manager_destroy(MNNRuntimeManager* manager) {
#ifdef MNN_RUNTIME_ENABLED
    if (manager) {
        MNN::Express::Executor::RuntimeManager::destroy(
            reinterpret_cast<MNN::Express::Executor::RuntimeManager*>(manager)
        );
    }
#else
    (void)manager;
#endif
}

MNNSession* mnn_interpreter_create_session_with_runtime(
    MNNInterpreter* interpreter,
    MNNRuntimeManager* runtime,
    int type,
    int num_threads
) {
#ifdef MNN_RUNTIME_ENABLED
    if (!interpreter || !runtime) {
        return nullptr;
    }

    auto interp = reinterpret_cast<Interpreter*>(interpreter);
    auto rtmgr = reinterpret_cast<MNN::Express::Executor::RuntimeManager*>(runtime);

    // Create session config
    ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(type);
    config.numThread = num_threads > 0 ? num_threads : 4;

    // Create session with runtime manager
    // Note: This requires MNN with RuntimeManager support
    Session* session = interp->createSession(config, rtmgr);
    return reinterpret_cast<MNNSession*>(session);
#else
    (void)interpreter;
    (void)runtime;
    (void)type;
    (void)num_threads;
    return nullptr;
#endif
}