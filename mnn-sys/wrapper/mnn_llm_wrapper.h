/**
 * @file mnn_llm_wrapper.h
 * @brief C wrapper for MNN LLM C++ API (MNN::Transformer::Llm / Embedding)
 *
 * This header provides C-compatible functions that wrap the MNN LLM C++ API,
 * allowing FFI bindings from Rust and other languages. It requires a build of
 * MNN compiled with MNN_BUILD_LLM=ON.
 */

#ifndef MNN_LLM_WRAPPER_H
#define MNN_LLM_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

/** Opaque handle to an MNN LLM instance (wraps MNN::Transformer::Llm) */
typedef struct MNNLlmHandle MNNLlm;

/** Opaque handle to an MNN embedding model (wraps MNN::Transformer::Embedding) */
typedef struct MNNEmbeddingHandle MNNEmbedding;

/** Callback receiving incremental text chunks during streaming generation */
typedef void (*mnn_llm_text_cb)(const char* text, void* userdata);

/** Snapshot of the LLM context metrics (readable subset of MNN's LlmContext) */
typedef struct {
    int prompt_len;       /**< Length of the prompt (prefill) in tokens */
    int gen_seq_len;      /**< Number of tokens generated this turn */
    int all_seq_len;      /**< Total sequence length including history */
    int64_t load_us;      /**< Model load time in microseconds */
    int64_t prefill_us;   /**< Prefill time in microseconds */
    int64_t decode_us;    /**< Decode time in microseconds */
    int64_t sample_us;    /**< Sampling time in microseconds */
} MNNLlmContext;

/* ============================================================================
 * Status Constants (matches MNN::Transformer::LlmStatus)
 * ============================================================================ */

#define MNN_LLM_STATUS_NOT_LOADED          (-1)
#define MNN_LLM_STATUS_RUNNING             0
#define MNN_LLM_STATUS_NORMAL_FINISHED     1
#define MNN_LLM_STATUS_MAX_TOKENS_FINISHED 2
#define MNN_LLM_STATUS_USER_CANCEL         3
#define MNN_LLM_STATUS_INTERNAL_ERROR      4
#define MNN_LLM_STATUS_TIMEOUT             5

/* ============================================================================
 * LLM Lifecycle
 * ============================================================================ */

/** Create LLM instance from config file
 * @param config_path Path to the model config.json
 * @return LLM handle or NULL on failure
 */
MNNLlm* mnn_llm_create(const char* config_path);

/** Destroy LLM instance (uses the static Llm::destroy, never delete)
 * @param llm LLM handle (may be NULL)
 */
void mnn_llm_destroy(MNNLlm* llm);

/** Load model weights
 * @param llm LLM handle
 * @return true on success
 */
bool mnn_llm_load(MNNLlm* llm);

/* ============================================================================
 * Blocking Generation
 * ============================================================================ */

/** Generate the full text response for a single prompt (blocking)
 * @param llm LLM handle
 * @param text User prompt
 * @param max_new_tokens Maximum tokens to generate (<= 0 uses config default)
 * @return NUL-terminated response string (free with mnn_string_free) or NULL on failure
 */
char* mnn_llm_response_text(MNNLlm* llm, const char* text, int max_new_tokens);

/** Generate the full text response for a chat message list (blocking)
 * @param llm LLM handle
 * @param roles Array of n role strings ("system"/"user"/"assistant"/...)
 * @param contents Array of n content strings
 * @param n Number of role/content pairs
 * @param max_new_tokens Maximum tokens to generate (<= 0 uses config default)
 * @return NUL-terminated response string (free with mnn_string_free) or NULL on failure
 */
char* mnn_llm_response_messages(MNNLlm* llm, const char* const* roles,
                                const char* const* contents, int n,
                                int max_new_tokens);

/* ============================================================================
 * Token Generation (arrays)
 * ============================================================================ */

/** Generate raw token ids for given input token ids (blocking)
 * @param llm LLM handle
 * @param input_ids Input token id array
 * @param n Number of input tokens
 * @param max_new_tokens Maximum tokens to generate (<= 0 uses config default)
 * @param out Output buffer for generated token ids
 * @param out_n On input: capacity of `out`; on output: actual number of generated tokens
 * @return Number of generated tokens (>= 0) or a negative value on error.
 *         If the result exceeds the output capacity it is truncated but the
 *         full count is still reported through *out_n.
 */
int mnn_llm_generate_tokens(MNNLlm* llm, const int* input_ids, int n,
                            int max_new_tokens, int* out, int* out_n);

/* ============================================================================
 * Streaming Generation
 * ============================================================================ */

/** Initialize streaming generation with a text callback
 * @param llm LLM handle
 * @param text User prompt (chat template is applied automatically)
 * @param cb Callback invoked with each chunk of generated text
 * @param userdata Opaque pointer forwarded to the callback
 * @param end_with Marker string appended at the end of generation (may be NULL)
 *
 * This feeds the prompt (prefill) and leaves the model ready for repeated
 * `mnn_llm_generate_step` calls, which decode one token per step and stream
 * the decoded text through `cb`.
 */
void mnn_llm_generate_init(MNNLlm* llm, const char* text, mnn_llm_text_cb cb,
                           void* userdata, const char* end_with);

/** Run one streaming generation step
 * @param llm LLM handle
 * @param max_token Maximum number of tokens to generate in this step
 * @return true when generation has stopped, false otherwise
 */
bool mnn_llm_generate_step(MNNLlm* llm, int max_token);

/** Check whether generation has stopped
 * @param llm LLM handle
 * @return true if the last generated token was a stop token
 */
bool mnn_llm_stoped(MNNLlm* llm);

/* ============================================================================
 * Tokenizer
 * ============================================================================ */

/** Encode text into token ids
 * @param llm LLM handle
 * @param text Input text
 * @param out_n On output: number of tokens (0 on failure)
 * @return Malloc'd token id array (free with mnn_int_array_free) or NULL on failure
 */
int* mnn_llm_tokenizer_encode(MNNLlm* llm, const char* text, int* out_n);

/** Decode a single token id into text
 * @param llm LLM handle
 * @param token Token id
 * @return NUL-terminated string (free with mnn_string_free) or NULL on failure
 */
char* mnn_llm_tokenizer_decode(MNNLlm* llm, int token);

/** Check whether a token id is a stop token
 * @param llm LLM handle
 * @param token Token id
 * @return true if the token terminates generation
 */
bool mnn_llm_is_stop(MNNLlm* llm, int token);

/* ============================================================================
 * Chat Template & Config
 * ============================================================================ */

/** Apply the chat template to a single user message
 * @param llm LLM handle
 * @param text User message
 * @return NUL-terminated templated prompt (free with mnn_string_free) or NULL on failure
 */
char* mnn_llm_apply_chat_template(MNNLlm* llm, const char* text);

/** Apply the chat template to a chat message list
 * @param llm LLM handle
 * @param roles Array of n role strings
 * @param contents Array of n content strings
 * @param n Number of role/content pairs
 * @return NUL-terminated templated prompt (free with mnn_string_free) or NULL on failure
 */
char* mnn_llm_apply_chat_template_messages(MNNLlm* llm,
                                           const char* const* roles,
                                           const char* const* contents, int n);

/** Set runtime configuration from a JSON string
 * @param llm LLM handle
 * @param json JSON configuration string (e.g. {"backend_type":"cpu"})
 * @return true on success
 */
bool mnn_llm_set_config(MNNLlm* llm, const char* json);

/** Dump the current runtime configuration as a JSON string
 * @param llm LLM handle
 * @return NUL-terminated JSON string (free with mnn_string_free) or NULL on failure
 */
char* mnn_llm_dump_config(MNNLlm* llm);

/* ============================================================================
 * State & Metrics
 * ============================================================================ */

/** Reset generation state (history, counters)
 * @param llm LLM handle
 */
void mnn_llm_reset(MNNLlm* llm);

/** Check whether KV cache reuse is enabled
 * @param llm LLM handle
 * @return true if KV cache is reused across turns
 */
bool mnn_llm_reuse_kv(MNNLlm* llm);

/** Get the current LLM status
 * @param llm LLM handle
 * @return One of the MNN_LLM_STATUS_* values, or -2 if the handle is NULL
 */
int mnn_llm_get_status(MNNLlm* llm);

/** Snapshot LLM context metrics
 * @param llm LLM handle
 * @param out Output structure to fill
 * @return 0 on success, non-zero on failure (NULL handle/out)
 */
int mnn_llm_get_context(MNNLlm* llm, MNNLlmContext* out);

/* ============================================================================
 * Memory Helpers
 * ============================================================================ */

/** Free a string previously returned by this API
 * @param s String pointer (may be NULL)
 */
void mnn_string_free(char* s);

/** Free an int array previously returned by this API
 * @param p Array pointer (may be NULL)
 */
void mnn_int_array_free(int* p);

/* ============================================================================
 * Embedding Model
 * ============================================================================ */

/** Create an embedding model from config file
 * @param config_path Path to the model config.json
 * @param load Load weights immediately (true) or defer to a later call
 * @return Embedding handle or NULL on failure
 */
MNNEmbedding* mnn_embedding_create(const char* config_path, bool load);

/** Destroy an embedding model (Embedding derives from Llm, uses Llm::destroy)
 * @param embedding Embedding handle (may be NULL)
 */
void mnn_embedding_destroy(MNNEmbedding* embedding);

/** Get the embedding vector dimension
 * @param embedding Embedding handle
 * @return Dimension (>= 0) or a negative value on failure
 */
int mnn_embedding_dim(MNNEmbedding* embedding);

/** Embed a text string into a float vector
 * @param embedding Embedding handle
 * @param text Input text
 * @param out Output buffer for the embedding vector
 * @param out_cap Capacity of `out` in floats
 * @return Number of elements written (>= 0) or a negative value on error.
 *         If the embedding is larger than `out_cap` it is truncated.
 */
int mnn_embedding_txt(MNNEmbedding* embedding, const char* text, float* out,
                      int out_cap);

/** Embed token ids into a float vector
 * @param embedding Embedding handle
 * @param ids Input token id array
 * @param n Number of input tokens
 * @param out Output buffer for the embedding vector
 * @param out_cap Capacity of `out` in floats
 * @return Number of elements written (>= 0) or a negative value on error.
 *         If the embedding is larger than `out_cap` it is truncated.
 */
int mnn_embedding_ids(MNNEmbedding* embedding, const int* ids, int n, float* out,
                      int out_cap);

#ifdef __cplusplus
}
#endif

#endif /* MNN_LLM_WRAPPER_H */
