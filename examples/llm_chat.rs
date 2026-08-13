//! LLM multi-turn chat example.
//!
//! Demonstrates loading an MNN LLM model, enabling KV-cache reuse, and
//! running an interactive multi-turn conversation.
//!
//! Requires the `llm` feature (which implies `build-from-source`, since
//! prebuilt MNN binaries do not ship the LLM engine).
//!
//! Usage:
//! ```bash
//! cargo run --example llm_chat --features llm,build-from-source --no-default-features -- /path/to/model/config.json
//! ```

use mnn_rs::{ChatMessage, Llm, MnnResult};

fn main() -> MnnResult<()> {
    let config_path = std::env::args()
        .nth(1)
        .expect("Usage: llm_chat <config.json>");

    println!("=== MNN LLM Chat Example ===\n");

    println!("[Step 1] Creating LLM from config: {}", config_path);
    let mut llm = Llm::create(&config_path)?;
    println!("  LLM created");

    println!("[Step 2] Loading model weights");
    llm.load()?;
    println!("  Model loaded");

    // Enable KV-cache reuse so the multi-turn history stays in memory
    // across turns instead of re-processing the whole prompt each time.
    println!("[Step 3] Enabling KV-cache reuse");
    llm.set_config(r#"{"reuse_kv": true}"#)?;
    println!("  KV-cache reuse enabled (reuse_kv = {})", llm.reuse_kv());

    let mut history = vec![ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful assistant.".to_string(),
    }];

    println!("\n[Step 4] Interactive chat (type 'exit' or 'quit' to leave)\n");

    loop {
        print!("User: ");
        std::io::Write::flush(&mut std::io::stdout()).expect("Failed to flush stdout");

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let input = input.trim().to_string();
        if input.is_empty() || input == "exit" || input == "quit" {
            break;
        }

        history.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });

        let reply = llm.response_messages(&history, Some(128))?;
        println!("Assistant: {reply}");

        history.push(ChatMessage {
            role: "assistant".to_string(),
            content: reply,
        });

        if let Ok(perf) = llm.performance() {
            println!(
                "  [perf] prompt={} gen={} prefill={}us decode={}us",
                perf.prompt_len, perf.gen_seq_len, perf.prefill_us, perf.decode_us
            );
        }
    }

    println!("\n=== Chat finished ===");
    Ok(())
}
