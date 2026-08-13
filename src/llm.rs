//! LLM (Large Language Model) inference support.
//!
//! This module provides safe Rust bindings for MNN's LLM engine
//! (`MNN::Transformer::Llm` and `MNN::Transformer::Embedding`), including
//! single-turn and multi-turn text generation, closure-based token streaming,
//! tokenizer encode/decode, chat templates, runtime JSON configuration, and
//! text/ID embedding.
//!
//! # Feature requirement
//!
//! This module is compiled with the `llm` feature. The MNN library must have
//! been built with the LLM engine enabled (`MNN_BUILD_LLM=ON`), which the
//! prebuilt binaries ship with. When building MNN yourself, combine `llm` with
//! `build-from-source`:
//!
//! ```text
//! cargo build --features llm,build-from-source --no-default-features
//! ```
//!
//! Using a prebuilt MNN that ships the LLM engine:
//!
//! ```text
//! cargo build --features llm --no-default-features
//! ```
//!
//! # Thread safety
//!
//! MNN's `Llm`/`Embedding` do not guarantee concurrent generation safety, so
//! both wrappers are designed for single-threaded exclusive use (they are
//! `!Send` and `!Sync`). For concurrent sessions, create multiple instances.

use crate::error::{MnnError, MnnResult};
use mnn_rs_sys::{MNNEmbedding, MNNLlm, MNNLlmContext};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;

/// Capacity of the token output buffer used by [`Llm::generate_tokens`].
const TOKEN_BUFFER_CAPACITY: usize = 4096;

/// Maximum size the token output buffer may grow to.
const TOKEN_BUFFER_MAX: usize = 1 << 20;

/// MNN LLM status, mirrored from the C `LlmStatus` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LlmStatus {
    /// Model not loaded yet
    NotLoaded = -1,
    /// Generation is currently running
    Running = 0,
    /// Generation finished normally
    NormalFinished = 1,
    /// Generation stopped because the maximum token count was reached
    MaxTokensFinished = 2,
    /// Generation was cancelled by the user
    UserCancel = 3,
    /// An internal error occurred
    InternalError = 4,
    /// Generation timed out
    Timeout = 5,
}

impl TryFrom<i32> for LlmStatus {
    type Error = MnnError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(LlmStatus::NotLoaded),
            0 => Ok(LlmStatus::Running),
            1 => Ok(LlmStatus::NormalFinished),
            2 => Ok(LlmStatus::MaxTokensFinished),
            3 => Ok(LlmStatus::UserCancel),
            4 => Ok(LlmStatus::InternalError),
            5 => Ok(LlmStatus::Timeout),
            other => Err(MnnError::invalid_input(format!(
                "Unknown LLM status value: {other}"
            ))),
        }
    }
}

/// A single chat message with a role and content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role of the message sender (e.g. "system", "user" or "assistant")
    pub role: String,
    /// Text content of the message
    pub content: String,
}

/// LLM inference performance statistics.
#[derive(Debug, Clone, Copy)]
pub struct LlmPerformance {
    /// Length of the prompt (prefill) in tokens
    pub prompt_len: u32,
    /// Number of tokens generated this turn
    pub gen_seq_len: u32,
    /// Total sequence length including history
    pub all_seq_len: u32,
    /// Model load time in microseconds
    pub load_us: u64,
    /// Prefill time in microseconds
    pub prefill_us: u64,
    /// Decode time in microseconds
    pub decode_us: u64,
    /// Sampling time in microseconds
    pub sample_us: u64,
}

/// Safe wrapper around an MNN LLM instance.
///
/// Provides blocking and streaming text generation, tokenizer and chat
/// template operations, runtime configuration, and performance metrics.
///
/// This type is `!Send` and `!Sync` because MNN's LLM engine is not safe to
/// use concurrently; use one instance per thread.
pub struct Llm {
    inner: *mut MNNLlm,
    /// Marker that opts the handle out of `Send`/`Sync` (single-threaded use only)
    _not_send: PhantomData<*mut ()>,
}

impl std::fmt::Debug for Llm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Llm")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Llm {
    /// Create an LLM instance from a model directory's `config.json` path.
    ///
    /// # Arguments
    /// * `config_path` - Path to the model's `config.json`
    ///
    /// # Returns
    /// A new `Llm` on success, or an error if creation fails.
    pub fn create<P: AsRef<Path>>(config_path: P) -> MnnResult<Self> {
        let path_str = config_path.as_ref().to_string_lossy().into_owned();
        let c_path = CString::new(path_str.as_str())?;
        // SAFETY: `c_path` is a valid NUL-terminated string for the call.
        let inner = unsafe { mnn_rs_sys::mnn_llm_create(c_path.as_ptr()) };
        if inner.is_null() {
            return Err(MnnError::invalid_model(format!(
                "Failed to create LLM from config: {path_str}"
            )));
        }
        Ok(Self {
            inner,
            _not_send: PhantomData,
        })
    }

    /// Load the model weights into memory.
    pub fn load(&mut self) -> MnnResult<()> {
        check_bool(
            // SAFETY: `self.inner` is a valid LLM handle created by `create`.
            unsafe { mnn_rs_sys::mnn_llm_load(self.inner) },
            "Failed to load LLM model",
        )
    }

    /// Generate a full response to a single text prompt (blocking).
    ///
    /// # Arguments
    /// * `text` - The user prompt
    /// * `max_new_tokens` - Maximum tokens to generate; `None` (or `<= 0`)
    ///   uses the model's configured default
    ///
    /// # Returns
    /// The full generated response text.
    pub fn response(&mut self, text: &str, max_new_tokens: Option<i32>) -> MnnResult<String> {
        let c_text = CString::new(text)?;
        // SAFETY: `c_text` is a valid NUL-terminated string for the call.
        let ptr = unsafe {
            mnn_rs_sys::mnn_llm_response_text(
                self.inner,
                c_text.as_ptr(),
                max_new_tokens.unwrap_or(0),
            )
        };
        // SAFETY: the wrapper returns a malloc'd string (or NULL); ownership
        // is transferred to `read_c_string`, which frees it exactly once.
        unsafe { read_c_string(ptr) }
    }

    /// Generate a full response from a multi-turn chat history (blocking).
    ///
    /// # Arguments
    /// * `messages` - Ordered chat messages (roles and contents)
    /// * `max_new_tokens` - Maximum tokens to generate; `None` (or `<= 0`)
    ///   uses the model's configured default
    ///
    /// # Returns
    /// The full generated response text.
    pub fn response_messages(
        &mut self,
        messages: &[ChatMessage],
        max_new_tokens: Option<i32>,
    ) -> MnnResult<String> {
        let roles: Vec<CString> = messages
            .iter()
            .map(|m| CString::new(m.role.as_str()))
            .collect::<Result<_, _>>()?;
        let contents: Vec<CString> = messages
            .iter()
            .map(|m| CString::new(m.content.as_str()))
            .collect::<Result<_, _>>()?;
        let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
        let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();

        // SAFETY: the two pointer arrays are valid for `n` elements and point
        // at NUL-terminated C strings that outlive the call.
        let ptr = unsafe {
            mnn_rs_sys::mnn_llm_response_messages(
                self.inner,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                max_new_tokens.unwrap_or(0),
            )
        };
        // SAFETY: the wrapper returns a malloc'd string (or NULL); ownership
        // is transferred to `read_c_string`, which frees it exactly once.
        unsafe { read_c_string(ptr) }
    }

    /// Generate raw output token IDs from input token IDs (blocking).
    ///
    /// The output buffer starts at 4096 tokens and grows (doubling) if the
    /// result is truncated.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    /// * `max_new_tokens` - Maximum tokens to generate; `None` (or `<= 0`)
    ///   uses the model's configured default
    ///
    /// # Returns
    /// The generated token IDs.
    pub fn generate_tokens(
        &mut self,
        input_ids: &[i32],
        max_new_tokens: Option<i32>,
    ) -> MnnResult<Vec<i32>> {
        let mut capacity = TOKEN_BUFFER_CAPACITY;
        loop {
            let mut out = vec![0i32; capacity];
            let mut out_n: c_int = capacity as c_int;
            // SAFETY: `input_ids` and `out` are valid for the given lengths;
            // `out_n` is capacity-in / count-out.
            let ret = unsafe {
                mnn_rs_sys::mnn_llm_generate_tokens(
                    self.inner,
                    input_ids.as_ptr(),
                    input_ids.len() as c_int,
                    max_new_tokens.unwrap_or(0),
                    out.as_mut_ptr(),
                    &mut out_n,
                )
            };
            if ret < 0 {
                return Err(MnnError::internal(format!(
                    "LLM token generation failed with code {ret}"
                )));
            }
            let count = out_n as usize;
            if count <= capacity {
                out.truncate(count);
                return Ok(out);
            }
            // The result was truncated; grow the buffer and retry.
            capacity = capacity.saturating_mul(2);
            if capacity > TOKEN_BUFFER_MAX {
                return Err(MnnError::internal(
                    "LLM token output exceeds the maximum buffer size",
                ));
            }
        }
    }

    /// Stream generated text chunks through a closure.
    ///
    /// The chat template is applied to `prompt`, then the model prefill runs
    /// and tokens are decoded one step at a time; each decoded text chunk is
    /// passed to `on_chunk`.
    ///
    /// # Arguments
    /// * `prompt` - The user prompt (chat template applied automatically)
    /// * `max_tokens` - Maximum number of tokens to generate per step
    /// * `on_chunk` - Closure invoked with each incremental text chunk
    ///
    /// The closure is called synchronously from the current thread for the
    /// whole duration of the generation loop and is never moved across
    /// threads.
    pub fn generate_stream<F>(&mut self, prompt: &str, max_tokens: i32, on_chunk: F) -> MnnResult<()>
    where
        F: FnMut(&str),
    {
        // SAFETY: the concrete monomorphization has the exact C callback ABI.
        unsafe extern "C" fn stream_cb<F>(text: *const c_char, userdata: *mut c_void)
        where
            F: FnMut(&str),
        {
            if text.is_null() || userdata.is_null() {
                return;
            }
            // SAFETY: `userdata` points at the `Box<F>` owned by the calling
            // `generate_stream`, which stays alive for the whole loop.
            let cb: &mut F = unsafe { &mut *(userdata as *mut F) };
            // SAFETY: `text` is a NUL-terminated string owned by MNN for the
            // duration of the callback.
            let chunk = unsafe { CStr::from_ptr(text) }.to_string_lossy();
            cb(&chunk);
        }

        let mut boxed = Box::new(on_chunk);
        let userdata: *mut c_void = (&mut *boxed as *mut F).cast();
        let c_prompt = CString::new(prompt)?;

        // SAFETY: `boxed` owns the closure and stays alive for the whole
        // generation loop; MNN only invokes `stream_cb` synchronously from
        // within these calls on the current thread. `end_with` is NULL, so
        // generation stops only via `mnn_llm_stoped`. `mnn_llm_generate_init`
        // feeds the prompt (prefill) and may already emit text through the
        // callback.
        unsafe {
            mnn_rs_sys::mnn_llm_generate_init(
                self.inner,
                c_prompt.as_ptr(),
                Some(stream_cb::<F>),
                userdata,
                std::ptr::null(),
            );
            while !mnn_rs_sys::mnn_llm_stoped(self.inner) {
                mnn_rs_sys::mnn_llm_generate_step(self.inner, max_tokens);
            }
        }

        drop(boxed);
        Ok(())
    }

    /// Encode a text string into token IDs using the model tokenizer.
    pub fn tokenize(&self, text: &str) -> MnnResult<Vec<i32>> {
        let c_text = CString::new(text)?;
        let mut out_n: c_int = 0;
        // SAFETY: `c_text` is a valid NUL-terminated string for the call.
        let ptr = unsafe {
            mnn_rs_sys::mnn_llm_tokenizer_encode(self.inner, c_text.as_ptr(), &mut out_n)
        };
        if ptr.is_null() {
            return Err(MnnError::internal("LLM tokenizer encode failed"));
        }
        // SAFETY: the wrapper malloc'd `out_n` ints; ownership is transferred
        // to `read_int_array`, which frees it exactly once.
        Ok(unsafe { read_int_array(ptr, out_n as usize) })
    }

    /// Decode a single token ID back into text.
    pub fn decode(&self, token: i32) -> MnnResult<String> {
        // SAFETY: the wrapper returns a malloc'd string (or NULL); ownership
        // is transferred to `read_c_string`, which frees it exactly once.
        let ptr = unsafe { mnn_rs_sys::mnn_llm_tokenizer_decode(self.inner, token) };
        unsafe { read_c_string(ptr) }
    }

    /// Check whether a token ID is a stop token for this model.
    pub fn is_stop(&self, token: i32) -> bool {
        // SAFETY: `self.inner` is a valid LLM handle created by `create`.
        unsafe { mnn_rs_sys::mnn_llm_is_stop(self.inner, token) }
    }

    /// Apply the model's chat template to a single user message.
    pub fn apply_chat_template(&self, text: &str) -> MnnResult<String> {
        let c_text = CString::new(text)?;
        // SAFETY: `c_text` is a valid NUL-terminated string for the call.
        let ptr = unsafe { mnn_rs_sys::mnn_llm_apply_chat_template(self.inner, c_text.as_ptr()) };
        // SAFETY: the wrapper returns a malloc'd string (or NULL); ownership
        // is transferred to `read_c_string`, which frees it exactly once.
        unsafe { read_c_string(ptr) }
    }

    /// Update the runtime configuration from a JSON string.
    ///
    /// Supported keys include `backend_type`, `thread_num`, sampler settings,
    /// and `max_new_tokens`.
    pub fn set_config(&mut self, json: &str) -> MnnResult<()> {
        let c_json = CString::new(json)?;
        check_bool(
            // SAFETY: `c_json` is a valid NUL-terminated string for the call.
            unsafe { mnn_rs_sys::mnn_llm_set_config(self.inner, c_json.as_ptr()) },
            "Failed to set LLM config",
        )
    }

    /// Dump the current runtime configuration as a JSON string.
    pub fn dump_config(&self) -> MnnResult<String> {
        // SAFETY: the wrapper returns a malloc'd string (or NULL); ownership
        // is transferred to `read_c_string`, which frees it exactly once.
        let ptr = unsafe { mnn_rs_sys::mnn_llm_dump_config(self.inner) };
        unsafe { read_c_string(ptr) }
    }

    /// Reset the generation state (history and counters).
    pub fn reset(&mut self) {
        // SAFETY: `self.inner` is a valid LLM handle created by `create`.
        unsafe { mnn_rs_sys::mnn_llm_reset(self.inner) };
    }

    /// Check whether KV-cache reuse across turns is enabled.
    pub fn reuse_kv(&self) -> bool {
        // SAFETY: `self.inner` is a valid LLM handle created by `create`.
        unsafe { mnn_rs_sys::mnn_llm_reuse_kv(self.inner) }
    }

    /// Get the current generation status.
    ///
    /// Raw status codes that are not in [`LlmStatus`] (e.g. a NULL handle)
    /// are mapped to [`LlmStatus::InternalError`].
    pub fn status(&self) -> LlmStatus {
        // SAFETY: `self.inner` is a valid LLM handle created by `create`.
        let raw = unsafe { mnn_rs_sys::mnn_llm_get_status(self.inner) };
        LlmStatus::try_from(raw).unwrap_or(LlmStatus::InternalError)
    }

    /// Get inference performance statistics.
    ///
    /// # Errors
    /// Returns an error if the model is not loaded yet.
    pub fn performance(&self) -> MnnResult<LlmPerformance> {
        let mut ctx: MNNLlmContext = unsafe { std::mem::zeroed() };
        // SAFETY: `ctx` is a valid, initialized `MNNLlmContext` buffer.
        let ret = unsafe { mnn_rs_sys::mnn_llm_get_context(self.inner, &mut ctx) };
        if ret != 0 {
            return Err(MnnError::internal(
                "Failed to read LLM context (model not loaded?)",
            ));
        }
        Ok(LlmPerformance {
            prompt_len: ctx.prompt_len as u32,
            gen_seq_len: ctx.gen_seq_len as u32,
            all_seq_len: ctx.all_seq_len as u32,
            load_us: ctx.load_us as u64,
            prefill_us: ctx.prefill_us as u64,
            decode_us: ctx.decode_us as u64,
            sample_us: ctx.sample_us as u64,
        })
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            // SAFETY: `inner` was created by `mnn_llm_create` and is destroyed
            // exactly once here.
            unsafe { mnn_rs_sys::mnn_llm_destroy(self.inner) };
        }
    }
}

/// Safe wrapper around an MNN embedding model.
///
/// Embedding models map text or token ID sequences to dense float vectors.
/// Like [`Llm`], this type is `!Send` and `!Sync` and should be used from a
/// single thread.
pub struct Embedding {
    inner: *mut MNNEmbedding,
    /// Marker that opts the handle out of `Send`/`Sync` (single-threaded use only)
    _not_send: PhantomData<*mut ()>,
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Embedding {
    /// Create an embedding model from a model directory's `config.json`.
    ///
    /// # Arguments
    /// * `config_path` - Path to the model's `config.json`
    /// * `load` - Load model weights immediately (`true`) or defer loading
    ///
    /// # Returns
    /// A new `Embedding` on success, or an error if creation fails.
    pub fn create<P: AsRef<Path>>(config_path: P, load: bool) -> MnnResult<Self> {
        let path_str = config_path.as_ref().to_string_lossy().into_owned();
        let c_path = CString::new(path_str.as_str())?;
        // SAFETY: `c_path` is a valid NUL-terminated string for the call.
        let inner = unsafe { mnn_rs_sys::mnn_embedding_create(c_path.as_ptr(), load) };
        if inner.is_null() {
            return Err(MnnError::invalid_model(format!(
                "Failed to create embedding model from config: {path_str}"
            )));
        }
        Ok(Self {
            inner,
            _not_send: PhantomData,
        })
    }

    /// Get the embedding vector dimension.
    ///
    /// Returns 0 if the dimension is unavailable (negative raw value).
    pub fn dim(&self) -> usize {
        // SAFETY: `self.inner` is a valid embedding handle created by `create`.
        let d = unsafe { mnn_rs_sys::mnn_embedding_dim(self.inner) };
        if d < 0 {
            0
        } else {
            d as usize
        }
    }

    /// Embed a text string into a dense float vector.
    pub fn embed_text(&self, text: &str) -> MnnResult<Vec<f32>> {
        let dim = self.dim();
        if dim == 0 {
            return Err(MnnError::internal(
                "Embedding model has zero or unavailable dimension",
            ));
        }
        let c_text = CString::new(text)?;
        let mut out = vec![0.0f32; dim];
        // SAFETY: `c_text` is a valid NUL-terminated string and `out` has
        // `dim` capacity, matching `out_cap`.
        let count = unsafe {
            mnn_rs_sys::mnn_embedding_txt(
                self.inner,
                c_text.as_ptr(),
                out.as_mut_ptr(),
                dim as c_int,
            )
        };
        if count < 0 {
            return Err(MnnError::internal(format!(
                "Embedding text failed with code {count}"
            )));
        }
        out.truncate((count as usize).min(dim));
        Ok(out)
    }

    /// Embed a sequence of token IDs into a dense float vector.
    pub fn embed_ids(&self, ids: &[i32]) -> MnnResult<Vec<f32>> {
        let dim = self.dim();
        if dim == 0 {
            return Err(MnnError::internal(
                "Embedding model has zero or unavailable dimension",
            ));
        }
        let mut out = vec![0.0f32; dim];
        // SAFETY: `ids` is valid for `ids.len()` elements and `out` has `dim`
        // capacity, matching `out_cap`.
        let count = unsafe {
            mnn_rs_sys::mnn_embedding_ids(
                self.inner,
                ids.as_ptr(),
                ids.len() as c_int,
                out.as_mut_ptr(),
                dim as c_int,
            )
        };
        if count < 0 {
            return Err(MnnError::internal(format!(
                "Embedding ids failed with code {count}"
            )));
        }
        out.truncate((count as usize).min(dim));
        Ok(out)
    }

    /// Compute the cosine similarity between two embedding vectors.
    ///
    /// # Errors
    /// Returns an error if either slice is empty, if the lengths differ, or
    /// if either vector has zero norm.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> MnnResult<f32> {
        if a.is_empty() || b.is_empty() {
            return Err(MnnError::EmptyData);
        }
        if a.len() != b.len() {
            return Err(MnnError::invalid_input(format!(
                "Cosine similarity requires equal lengths, got {} and {}",
                a.len(),
                b.len()
            )));
        }
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += f64::from(x) * f64::from(y);
            norm_a += f64::from(x) * f64::from(x);
            norm_b += f64::from(y) * f64::from(y);
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 {
            return Err(MnnError::invalid_input(
                "Cannot compute cosine similarity of zero vectors",
            ));
        }
        Ok((dot / denom) as f32)
    }
}

impl Drop for Embedding {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            // SAFETY: `inner` was created by `mnn_embedding_create` and is
            // destroyed exactly once here.
            unsafe { mnn_rs_sys::mnn_embedding_destroy(self.inner) };
        }
    }
}

/// Convert a C `bool` result into an `MnnResult`, mapping `false` to an
/// internal error carrying `context`.
fn check_bool(ok: bool, context: &str) -> MnnResult<()> {
    if ok {
        Ok(())
    } else {
        Err(MnnError::internal(context))
    }
}

/// Copy a NUL-terminated C string and free it with `mnn_string_free`.
///
/// # Safety
/// `ptr` must be a non-null pointer returned by the LLM C wrapper (allocated
/// with `strdup`), and must be freed exactly once.
unsafe fn read_c_string(ptr: *mut c_char) -> MnnResult<String> {
    if ptr.is_null() {
        return Err(MnnError::internal("LLM returned a null string"));
    }
    // SAFETY: the caller guarantees `ptr` is a valid NUL-terminated string.
    let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    // SAFETY: the caller guarantees `ptr` was allocated for `mnn_string_free`.
    unsafe { mnn_rs_sys::mnn_string_free(ptr) };
    Ok(text)
}

/// Copy `len` ints out of a C array and free it with `mnn_int_array_free`.
///
/// # Safety
/// `ptr` must point to at least `len` allocated ints and be freed exactly
/// once.
unsafe fn read_int_array(ptr: *mut c_int, len: usize) -> Vec<i32> {
    // SAFETY: the caller guarantees `ptr` is valid for `len` elements.
    let data = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: the caller guarantees `ptr` was allocated for `mnn_int_array_free`.
    unsafe { mnn_rs_sys::mnn_int_array_free(ptr) };
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_status_roundtrip() {
        assert!(matches!(LlmStatus::try_from(-1), Ok(LlmStatus::NotLoaded)));
        assert!(matches!(LlmStatus::try_from(0), Ok(LlmStatus::Running)));
        assert!(matches!(LlmStatus::try_from(1), Ok(LlmStatus::NormalFinished)));
        assert!(matches!(LlmStatus::try_from(2), Ok(LlmStatus::MaxTokensFinished)));
        assert!(matches!(LlmStatus::try_from(3), Ok(LlmStatus::UserCancel)));
        assert!(matches!(LlmStatus::try_from(4), Ok(LlmStatus::InternalError)));
        assert!(matches!(LlmStatus::try_from(5), Ok(LlmStatus::Timeout)));
        assert!(LlmStatus::try_from(42).is_err());
    }

    #[test]
    fn cosine_similarity_same_vectors() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        let s = Embedding::cosine_similarity(&a, &b).unwrap();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let s = Embedding::cosine_similarity(&a, &b).unwrap();
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_errors() {
        assert!(Embedding::cosine_similarity(&[], &[]).is_err());
        assert!(Embedding::cosine_similarity(&[1.0, 2.0], &[3.0]).is_err());
        assert!(Embedding::cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]).is_err());
    }
}
