//! LLM streaming generation example.
//!
//! Demonstrates incremental text generation using a closure callback, i.e.
//! tokens are delivered to the caller as they are decoded rather than
//! waiting for the full response.
//!
//! Requires the `llm` feature (which implies `build-from-source`).
//!
//! Usage:
//! ```bash
//! cargo run --example llm_stream --features llm,build-from-source --no-default-features -- /path/to/model/config.json [prompt]
//! ```

use mnn_rs::{Llm, MnnResult};

fn main() -> MnnResult<()> {
    let config_path = std::env::args()
        .nth(1)
        .expect("Usage: llm_stream <config.json> [prompt]");
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Hello, please introduce yourself.".to_string());

    println!("=== MNN LLM Streaming Example ===\n");

    let mut llm = Llm::create(&config_path)?;
    llm.load()?;

    println!("Prompt: {prompt}\n");
    print!("Response: ");
    std::io::Write::flush(&mut std::io::stdout()).expect("Failed to flush stdout");

    // Stream generated text chunk by chunk through a closure. The closure is
    // called synchronously from the current thread for each decoded chunk.
    llm.generate_stream(&prompt, 1, |chunk| {
        print!("{chunk}");
        std::io::Write::flush(&mut std::io::stdout()).expect("Failed to flush stdout");
    })?;

    println!("\n\n=== Streaming finished ===");
    Ok(())
}
