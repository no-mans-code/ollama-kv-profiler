//! Synthetic filler text, sized to a target token count via an empirically
//! measured chars-per-token ratio - a fixed guess (e.g. "4 chars/token")
//! is off enough across model families and this specific word pool to be
//! worth actually measuring per model rather than assuming.

use anyhow::{bail, Result};
use rand::Rng;

use crate::ollama::{Client, GenerateOptions, GenerateRequest};

/// Real-ish words (not a lorem-ipsum block a model may have memorized
/// wholesale) to build varied filler text from - varied enough that a
/// model can't shortcut on repetition, cheap enough to generate a lot of
/// it fast.
const WORDS: &[&str] = &[
    "kingdom", "river", "mountain", "algorithm", "harvest", "engine", "orbit", "signal",
    "forest", "market", "voltage", "compass", "granite", "horizon", "lantern", "current",
    "meadow", "circuit", "glacier", "treaty", "harbor", "reactor", "prairie", "satellite",
    "canyon", "furnace", "tunnel", "archive", "reservoir", "beacon", "quarry", "spectrum",
];

/// Builds text of roughly `target_chars` characters from the word pool.
pub fn filler_text(target_chars: usize) -> String {
    let mut rng = rand::thread_rng();
    let mut out = String::with_capacity(target_chars + 32);
    while out.len() < target_chars {
        let w = WORDS[rng.gen_range(0..WORDS.len())];
        out.push_str(w);
        out.push(' ');
    }
    out
}

/// Empirically finds this model's chars-per-token ratio for this word
/// pool by sending a calibration prompt and reading back how many tokens
/// Ollama reports it as.
pub fn calibrate_chars_per_token(client: &Client, model: &str, num_gpu: i32, num_ctx: u32) -> Result<f64> {
    let sample = filler_text(4000);
    let resp = client.generate(&GenerateRequest {
        model,
        prompt: &sample,
        stream: false,
        context: None,
        options: GenerateOptions { num_gpu, num_ctx },
    })?;
    if resp.prompt_eval_count == 0 {
        bail!("calibration call to `{model}` reported zero prompt tokens");
    }
    Ok(sample.chars().count() as f64 / resp.prompt_eval_count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filler_text_reaches_target_length() {
        let text = filler_text(500);
        assert!(text.len() >= 500);
        assert!(text.len() < 500 + 20); // one word's worth of overshoot at most
    }

    #[test]
    fn filler_text_is_not_a_single_repeated_word() {
        let text = filler_text(2000);
        let unique_words: std::collections::HashSet<&str> = text.split_whitespace().collect();
        assert!(unique_words.len() > 5, "expected real variety, got {unique_words:?}");
    }
}
