/**
 * @file mnn_llm_wrapper.cpp
 * @brief C wrapper implementation for MNN LLM C++ API
 *
 * Wraps MNN::Transformer::Llm and MNN::Transformer::Embedding behind opaque
 * C handles. Requires MNN built with MNN_BUILD_LLM=ON. Compiles as C++14.
 */

#include "mnn_llm_wrapper.h"

#include <llm/llm.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

namespace {

/** Duplicate a std::string into a malloc'd NUL-terminated buffer.
 *  The result is released with mnn_string_free() (a plain free()). */
char* dup_string(const std::string& s) {
    size_t len = s.size();
    char* p = static_cast<char*>(malloc(len + 1));
    if (p == nullptr) {
        return nullptr;
    }
    memcpy(p, s.c_str(), len);
    p[len] = '\0';
    return p;
}

/** Copy a float VARP into a caller-provided buffer.
 *  Returns the number of elements written (>= 0) or -1 on error. */
int copy_embedding(MNN::Express::VARP var, float* out, int out_cap) {
    if (var == nullptr) {
        return -1;
    }
    const MNN::Express::Variable::Info* info = var->getInfo();
    if (info == nullptr) {
        return -1;
    }
    const float* data = var->readMap<float>();
    if (data == nullptr) {
        return -1;
    }
    const int size = static_cast<int>(info->size);
    if (size <= 0) {
        return -1;
    }
    const int written = size < out_cap ? size : out_cap;
    memcpy(out, data, sizeof(float) * static_cast<size_t>(written));
    return written;
}

/** Build a ChatMessages vector from parallel role/content C arrays. */
MNN::Transformer::ChatMessages build_messages(const char* const* roles,
                                              const char* const* contents, int n) {
    MNN::Transformer::ChatMessages messages;
    if (roles == nullptr || contents == nullptr || n <= 0) {
        return messages;
    }
    messages.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (roles[i] != nullptr && contents[i] != nullptr) {
            messages.push_back(std::make_pair(std::string(roles[i]),
                                              std::string(contents[i])));
        }
    }
    return messages;
}

}  // namespace

/* ============================================================================
 * Streaming Stream Buffer
 * ============================================================================ */

/** std::streambuf that forwards every chunk of generated text to a C callback. */
class LlmStreamBuf : public std::streambuf {
public:
    LlmStreamBuf(mnn_llm_text_cb cb, void* userdata) : mCb(cb), mUserdata(userdata) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (mCb != nullptr && s != nullptr && n > 0) {
            // Callback expects a NUL-terminated string.
            std::string chunk(s, static_cast<size_t>(n));
            mCb(chunk.c_str(), mUserdata);
        }
        return n;
    }

    virtual int overflow(int c) override {
        if (mCb != nullptr && c != traits_type::eof()) {
            char ch = static_cast<char>(c);
            mCb(&ch, mUserdata);
        }
        return c;
    }

private:
    mnn_llm_text_cb mCb;
    void* mUserdata;
};

/* ============================================================================
 * Opaque Handle Definitions
 * ============================================================================ */

struct MNNLlmHandle {
    MNN::Transformer::Llm* llm = nullptr;
    /** Streaming sink; only alive while a streaming session is active. */
    std::shared_ptr<LlmStreamBuf> stream;
    /** std::ostream view over `stream`; Llm::generate_init requires an ostream*. */
    std::shared_ptr<std::ostream> os;
};

struct MNNEmbeddingHandle {
    MNN::Transformer::Embedding* emb = nullptr;
};

/* ============================================================================
 * LLM Lifecycle
 * ============================================================================ */

MNNLlm* mnn_llm_create(const char* config_path) {
    if (config_path == nullptr) {
        return nullptr;
    }
    MNN::Transformer::Llm* llm = MNN::Transformer::Llm::createLLM(config_path);
    if (llm == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = new MNNLlmHandle();
    handle->llm = llm;
    return reinterpret_cast<MNNLlm*>(handle);
}

void mnn_llm_destroy(MNNLlm* llm) {
    if (llm == nullptr) {
        return;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    // Drop streaming resources before releasing the LLM itself.
    handle->os.reset();
    handle->stream.reset();
    if (handle->llm != nullptr) {
        // MUST use the static destroy; it owns internal lifetime bookkeeping.
        MNN::Transformer::Llm::destroy(handle->llm);
        handle->llm = nullptr;
    }
    delete handle;
}

bool mnn_llm_load(MNNLlm* llm) {
    if (llm == nullptr) {
        return false;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return false;
    }
    return handle->llm->load();
}

/* ============================================================================
 * Blocking Generation
 * ============================================================================ */

char* mnn_llm_response_text(MNNLlm* llm, const char* text, int max_new_tokens) {
    if (llm == nullptr || text == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    std::ostringstream oss;
    handle->llm->response(std::string(text), &oss, nullptr, max_new_tokens);
    return dup_string(oss.str());
}

char* mnn_llm_response_messages(MNNLlm* llm, const char* const* roles,
                                const char* const* contents, int n,
                                int max_new_tokens) {
    if (llm == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    MNN::Transformer::ChatMessages messages = build_messages(roles, contents, n);
    if (messages.empty()) {
        return nullptr;
    }
    std::ostringstream oss;
    handle->llm->response(messages, &oss, nullptr, max_new_tokens);
    return dup_string(oss.str());
}

/* ============================================================================
 * Token Generation (arrays)
 * ============================================================================ */

int mnn_llm_generate_tokens(MNNLlm* llm, const int* input_ids, int n,
                            int max_new_tokens, int* out, int* out_n) {
    if (llm == nullptr || input_ids == nullptr || n <= 0 || out == nullptr ||
        out_n == nullptr) {
        return -1;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return -1;
    }
    const int capacity = *out_n;
    if (capacity <= 0) {
        return -1;
    }
    std::vector<int> ids(input_ids, input_ids + n);
    std::vector<int> result = handle->llm->generate(ids, max_new_tokens);
    const int actual = static_cast<int>(result.size());
    const int copied = actual < capacity ? actual : capacity;
    if (copied > 0) {
        memcpy(out, result.data(), sizeof(int) * static_cast<size_t>(copied));
    }
    *out_n = actual;
    return actual;
}

/* ============================================================================
 * Streaming Generation
 * ============================================================================ */

void mnn_llm_generate_init(MNNLlm* llm, const char* text, mnn_llm_text_cb cb,
                           void* userdata, const char* end_with) {
    if (llm == nullptr || text == nullptr) {
        return;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return;
    }
    handle->stream = std::make_shared<LlmStreamBuf>(cb, userdata);
    handle->os = std::make_shared<std::ostream>(handle->stream.get());

    // Feed the prompt (prefill) without generating any token, mirroring
    // MNN's `Llm::response(input_ids, os, end_with, 0)` used by llm_demo:
    // generate_init + prefill, then the caller loops `generate(1)`.
    // Apply the chat template exactly like `Llm::response(std::string, ...)`
    // would, but avoid its unconditional `std::cout << "prompt: ..."`.
    std::string templated = handle->llm->apply_chat_template(std::string(text));
    std::vector<int> input_ids = handle->llm->tokenizer_encode(
        templated.empty() ? std::string(text) : templated);
    handle->llm->response(input_ids, handle->os.get(), end_with, 0);
}

bool mnn_llm_generate_step(MNNLlm* llm, int max_token) {
    if (llm == nullptr) {
        return false;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return false;
    }
    handle->llm->generate(max_token);
    return handle->llm->stoped();
}

bool mnn_llm_stoped(MNNLlm* llm) {
    if (llm == nullptr) {
        return false;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return false;
    }
    return handle->llm->stoped();
}

/* ============================================================================
 * Tokenizer
 * ============================================================================ */

int* mnn_llm_tokenizer_encode(MNNLlm* llm, const char* text, int* out_n) {
    if (out_n != nullptr) {
        *out_n = 0;
    }
    if (llm == nullptr || text == nullptr || out_n == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    std::vector<int> ids = handle->llm->tokenizer_encode(std::string(text));
    if (ids.empty()) {
        return nullptr;
    }
    const size_t count = ids.size();
    int* arr = static_cast<int*>(malloc(sizeof(int) * count));
    if (arr == nullptr) {
        return nullptr;
    }
    memcpy(arr, ids.data(), sizeof(int) * count);
    *out_n = static_cast<int>(count);
    return arr;
}

char* mnn_llm_tokenizer_decode(MNNLlm* llm, int token) {
    if (llm == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    return dup_string(handle->llm->tokenizer_decode(token));
}

bool mnn_llm_is_stop(MNNLlm* llm, int token) {
    if (llm == nullptr) {
        return false;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return false;
    }
    return handle->llm->is_stop(token);
}

/* ============================================================================
 * Chat Template & Config
 * ============================================================================ */

char* mnn_llm_apply_chat_template(MNNLlm* llm, const char* text) {
    if (llm == nullptr || text == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    return dup_string(handle->llm->apply_chat_template(std::string(text)));
}

char* mnn_llm_apply_chat_template_messages(MNNLlm* llm,
                                           const char* const* roles,
                                           const char* const* contents, int n) {
    if (llm == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    MNN::Transformer::ChatMessages messages = build_messages(roles, contents, n);
    if (messages.empty()) {
        return nullptr;
    }
    return dup_string(handle->llm->apply_chat_template(messages));
}

bool mnn_llm_set_config(MNNLlm* llm, const char* json) {
    if (llm == nullptr || json == nullptr) {
        return false;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return false;
    }
    return handle->llm->set_config(std::string(json));
}

char* mnn_llm_dump_config(MNNLlm* llm) {
    if (llm == nullptr) {
        return nullptr;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return nullptr;
    }
    return dup_string(handle->llm->dump_config());
}

/* ============================================================================
 * State & Metrics
 * ============================================================================ */

void mnn_llm_reset(MNNLlm* llm) {
    if (llm == nullptr) {
        return;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return;
    }
    handle->llm->reset();
}

bool mnn_llm_reuse_kv(MNNLlm* llm) {
    if (llm == nullptr) {
        return false;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return false;
    }
    return handle->llm->reuse_kv();
}

int mnn_llm_get_status(MNNLlm* llm) {
    if (llm == nullptr) {
        return -2;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return -2;
    }
    const MNN::Transformer::LlmContext* ctx = handle->llm->getContext();
    if (ctx == nullptr) {
        return -2;
    }
    return static_cast<int>(ctx->status);
}

int mnn_llm_get_context(MNNLlm* llm, MNNLlmContext* out) {
    if (llm == nullptr || out == nullptr) {
        return -1;
    }
    MNNLlmHandle* handle = reinterpret_cast<MNNLlmHandle*>(llm);
    if (handle->llm == nullptr) {
        return -1;
    }
    const MNN::Transformer::LlmContext* ctx = handle->llm->getContext();
    if (ctx == nullptr) {
        return -1;
    }
    out->prompt_len = ctx->prompt_len;
    out->gen_seq_len = ctx->gen_seq_len;
    out->all_seq_len = ctx->all_seq_len;
    out->load_us = ctx->load_us;
    out->prefill_us = ctx->prefill_us;
    out->decode_us = ctx->decode_us;
    out->sample_us = ctx->sample_us;
    return 0;
}

/* ============================================================================
 * Memory Helpers
 * ============================================================================ */

void mnn_string_free(char* s) {
    free(s);
}

void mnn_int_array_free(int* p) {
    free(p);
}

/* ============================================================================
 * Embedding Model
 * ============================================================================ */

MNNEmbedding* mnn_embedding_create(const char* config_path, bool load) {
    if (config_path == nullptr) {
        return nullptr;
    }
    MNN::Transformer::Embedding* emb =
        MNN::Transformer::Embedding::createEmbedding(config_path, load);
    if (emb == nullptr) {
        return nullptr;
    }
    MNNEmbeddingHandle* handle = new MNNEmbeddingHandle();
    handle->emb = emb;
    return reinterpret_cast<MNNEmbedding*>(handle);
}

void mnn_embedding_destroy(MNNEmbedding* embedding) {
    if (embedding == nullptr) {
        return;
    }
    MNNEmbeddingHandle* handle = reinterpret_cast<MNNEmbeddingHandle*>(embedding);
    if (handle->emb != nullptr) {
        // Embedding derives from Llm, so the static Llm::destroy must be used.
        MNN::Transformer::Llm::destroy(handle->emb);
        handle->emb = nullptr;
    }
    delete handle;
}

int mnn_embedding_dim(MNNEmbedding* embedding) {
    if (embedding == nullptr) {
        return -1;
    }
    MNNEmbeddingHandle* handle = reinterpret_cast<MNNEmbeddingHandle*>(embedding);
    if (handle->emb == nullptr) {
        return -1;
    }
    return handle->emb->dim();
}

int mnn_embedding_txt(MNNEmbedding* embedding, const char* text, float* out,
                      int out_cap) {
    if (embedding == nullptr || text == nullptr || out == nullptr ||
        out_cap <= 0) {
        return -1;
    }
    MNNEmbeddingHandle* handle = reinterpret_cast<MNNEmbeddingHandle*>(embedding);
    if (handle->emb == nullptr) {
        return -1;
    }
    MNN::Express::VARP var = handle->emb->txt_embedding(std::string(text));
    return copy_embedding(var, out, out_cap);
}

int mnn_embedding_ids(MNNEmbedding* embedding, const int* ids, int n, float* out,
                      int out_cap) {
    if (embedding == nullptr || ids == nullptr || n <= 0 || out == nullptr ||
        out_cap <= 0) {
        return -1;
    }
    MNNEmbeddingHandle* handle = reinterpret_cast<MNNEmbeddingHandle*>(embedding);
    if (handle->emb == nullptr) {
        return -1;
    }
    std::vector<int> id_vec(ids, ids + n);
    MNN::Express::VARP var = handle->emb->ids_embedding(id_vec);
    return copy_embedding(var, out, out_cap);
}
