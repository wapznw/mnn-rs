//! LLM embedding example.
//!
//! Demonstrates embedding text into dense float vectors and computing the
//! cosine similarity between them. Embedding models map text to vectors such
//! that semantically related sentences have higher similarity.
//!
//! Requires the `llm` feature (which implies `build-from-source`).
//!
//! Usage:
//! ```bash
//! cargo run --example llm_embedding --features llm,build-from-source --no-default-features -- /path/to/model/config.json
//! ```

use mnn_rs::{Embedding, MnnResult};

fn main() -> MnnResult<()> {
    let config_path = std::env::args()
        .nth(1)
        .expect("Usage: llm_embedding <config.json>");

    println!("=== MNN LLM Embedding Example ===\n");

    println!("[Step 1] Creating embedding model from: {}", config_path);
    let embedding = Embedding::create(&config_path, true)?;
    println!("  Embedding model created");
    println!("  Embedding dimension: {}", embedding.dim());

    let a = "What is the capital of France?";
    let b = "Paris is the capital of France.";
    let c = "The weather is sunny today.";

    println!("\n[Step 2] Embedding texts");
    let vec_a = embedding.embed_text(a)?;
    let vec_b = embedding.embed_text(b)?;
    let vec_c = embedding.embed_text(c)?;
    println!("  Text A embedded: {} floats", vec_a.len());
    println!("  Text B embedded: {} floats", vec_b.len());
    println!("  Text C embedded: {} floats", vec_c.len());

    println!("\n[Step 3] Cosine similarities");
    let sim_ab = Embedding::cosine_similarity(&vec_a, &vec_b)?;
    let sim_ac = Embedding::cosine_similarity(&vec_a, &vec_c)?;
    println!("  sim(A, B) = {:.4}", sim_ab);
    println!("  sim(A, C) = {:.4}", sim_ac);
    println!("  (related sentences should have higher similarity)");

    println!("\n=== Embedding example finished ===");
    Ok(())
}
