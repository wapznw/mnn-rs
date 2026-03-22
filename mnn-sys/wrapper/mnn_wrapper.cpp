/**
 * @file mnn_wrapper.cpp
 * @brief C wrapper implementation for MNN C++ API
 */

#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>
#include <cstring>
#include <stdio.h>

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