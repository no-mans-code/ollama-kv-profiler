//! Deterministic, hardware-only measurements, combined into a roofline-style
//! crossover prediction: would a *real* disk-persisted KV-cache swap (the
//! kind [`crate::bench`] cannot actually exercise, since Ollama has no such
//! path - see the README) beat cold reingestion, for a given model on this
//! machine? This doesn't require running the full profiling sweep - just
//! one disk benchmark (model-independent) and the "cold" tokens/sec this
//! tool already measures (or a single quick calibration call).
//!
//! The formula: swap-in time per token is `kv_bytes_per_token /
//! disk_bandwidth`; reingestion time per token is `1 / prefill_tokens_per_sec`
//! (in the compute-bound, linear-layers regime). Swap wins when
//! `kv_bytes_per_token * prefill_tokens_per_sec < disk_bandwidth` - a
//! condition that, notably, does not depend on context length at all, in
//! that regime. It stops holding at very long context: attention cost
//! grows quadratically with context length while KV-cache transfer cost
//! stays linear, so real disk swap becomes *relatively* more attractive
//! the longer the context gets, even in a case this formula calls a
//! reingest win at short/medium length. This matches published analysis
//! (see README references) - it is not this project's own conjecture.

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;

/// Measures real sequential read bandwidth to a fresh temp file - the
/// practical proxy for "how fast could a real KV-cache-to-disk swap read
/// its saved state back," since a KV blob is read sequentially in one
/// chunk, not in small random pages. Reports read bandwidth specifically
/// (not write): saving a cache happens once, off the hot path; reading it
/// back is paid on every reuse, which is the side that matters here.
pub fn measure_disk_read_bandwidth_bytes_per_sec(size_bytes: usize) -> Result<f64> {
    let path = std::env::temp_dir().join(format!("ollama-kv-profiler-diskbench-{}.bin", std::process::id()));
    let buf = vec![0xABu8; size_bytes];

    {
        let mut f = std::fs::File::create(&path).context("failed to create disk bench file")?;
        f.write_all(&buf).context("failed to write disk bench file")?;
        f.sync_all().context("failed to flush disk bench file to disk")?;
    }
    // Drop and reopen so the read below at least has to go through the
    // filesystem layer fresh, though on a system with a large page cache
    // this may still be served from RAM rather than real media - noted as
    // a real limitation in the README rather than hidden.
    let t0 = Instant::now();
    let read_back = std::fs::read(&path).context("failed to read disk bench file back")?;
    let read_secs = t0.elapsed().as_secs_f64().max(1e-9);
    anyhow::ensure!(read_back.len() == size_bytes, "read back a different size than written");
    let _ = std::fs::remove_file(&path);

    Ok(size_bytes as f64 / read_secs)
}

/// KV-cache bytes per token: `2 (K and V) * num_layers * num_kv_heads *
/// head_dim * kv_dtype_bytes`. `kv_dtype_bytes` is 2 for Ollama's default
/// f16 KV cache; pass a smaller value if KV-cache quantization is
/// configured (e.g. 1 for q8_0). `num_kv_heads`/`head_dim` take `f64`
/// since some architectures (Gemma's interleaved local/global attention)
/// vary head count per layer - `ollama::Client::architecture_info` passes
/// the per-layer average, and `num_layers * average` reproduces the true
/// cross-layer total exactly.
pub fn kv_bytes_per_token(num_layers: u32, num_kv_heads: f64, head_dim: f64, kv_dtype_bytes: u32) -> u64 {
    (2.0 * num_layers as f64 * num_kv_heads * head_dim * kv_dtype_bytes as f64).round() as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CrossoverPrediction {
    pub kv_bytes_per_token: u64,
    pub disk_bandwidth_bytes_per_sec: f64,
    pub prefill_tokens_per_sec: f64,
    /// The disk bandwidth a real swap would need to match reingestion speed.
    pub required_bandwidth_bytes_per_sec: f64,
    /// True in the linear (non-attention-dominated) regime this formula
    /// covers - see module docs for why real disk swap can still win at
    /// very long context even when this is false.
    pub disk_swap_would_win_at_short_to_medium_context: bool,
}

pub fn predict_crossover(kv_bytes_per_token: u64, disk_bandwidth_bytes_per_sec: f64, prefill_tokens_per_sec: f64) -> CrossoverPrediction {
    let required = kv_bytes_per_token as f64 * prefill_tokens_per_sec;
    CrossoverPrediction {
        kv_bytes_per_token,
        disk_bandwidth_bytes_per_sec,
        prefill_tokens_per_sec,
        required_bandwidth_bytes_per_sec: required,
        disk_swap_would_win_at_short_to_medium_context: disk_bandwidth_bytes_per_sec > required,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_bytes_matches_hand_calculation_for_qwen2_5_3b() {
        // qwen2.5:3b, from real /api/show output: block_count=36,
        // head_count_kv=2, embedding_length=2048, head_count=16 ->
        // head_dim=128. f16 KV cache (2 bytes).
        let bytes = kv_bytes_per_token(36, 2.0, 128.0, 2);
        assert_eq!(bytes, 36_864);
    }

    #[test]
    fn kv_bytes_matches_hand_calculation_for_gemma4_26b_per_layer_heads() {
        // gemma4:26b, from real /api/show output: block_count=30,
        // head_count_kv=[8,8,8,8,8,2] repeated 5x (interleaved local/global
        // attention - NOT one scalar for the whole model), embedding_length
        // =2816, head_count=16 -> head_dim=176. Average kv_heads across the
        // 30-entry array = (25*8 + 5*2) / 30 = 7.0 exactly.
        let avg_kv_heads = (25.0 * 8.0 + 5.0 * 2.0) / 30.0;
        assert_eq!(avg_kv_heads, 7.0);
        let bytes = kv_bytes_per_token(30, avg_kv_heads, 176.0, 2);
        // 2 * 30 * 7.0 * 176 * 2 = 147,840 - more than double what a naive
        // "use the query head count (16) for every layer" fallback would
        // have given (2*30*16*176*2 = 337,920), which is exactly the bug
        // this test exists to catch a regression of.
        assert_eq!(bytes, 147_840);
    }

    #[test]
    fn crossover_favors_swap_when_disk_is_fast_enough() {
        // 36KB/token, 10,000 tok/s -> needs 360 MB/s; a 3.5 GB/s NVMe clears that easily.
        let p = predict_crossover(36_864, 3_500_000_000.0, 10_000.0);
        assert!(p.disk_swap_would_win_at_short_to_medium_context);
    }

    #[test]
    fn crossover_favors_reingest_when_disk_is_too_slow() {
        // Same model/speed, but a slow eMMC-class device (~100 MB/s).
        let p = predict_crossover(36_864, 100_000_000.0, 10_000.0);
        assert!(!p.disk_swap_would_win_at_short_to_medium_context);
    }
}
