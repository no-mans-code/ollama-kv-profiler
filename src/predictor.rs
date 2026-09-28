//! Real-time prediction: given a LIVE system snapshot (VRAM/RAM free right
//! now, not a one-time measurement) and a document's real token count,
//! estimate how long ingesting it fresh would take versus swapping its
//! context in from an on-disk LRU cache (if one exists) - then let the
//! caller compare that estimate against what actually happens. The
//! crossover formula in `hardware.rs` made concrete and live, backed by a
//! real cache (`kvcache` - the same crate and key shape docuzent's own
//! `Session` uses) rather than a one-time hardware characterization.
//!
//! Honesty note: the "swap" measurement here loads a real context from a
//! real on-disk cache, then hands it to Ollama exactly as the rest of this
//! tool already does elsewhere - but Ollama's own reuse only sometimes
//! actually saves compute (see the README's reuse-window findings). So the
//! *actual* swap time measured here reflects "disk read + whatever Ollama
//! does with the reused context today," not a pure disk-bandwidth figure.
//!
//! An earlier version of [`predict`] estimated swap time as pure
//! disk-bandwidth (`document_tokens * kv_bytes_per_token / bandwidth`) and
//! was found to underestimate real swap latency by 73-86% - worse on
//! *faster* models, not slower ones. The reason: every swap call still
//! pays Ollama's real per-call cost to reprocess the short follow-up
//! question and decode the answer, and that cost is fixed in *token*
//! terms (a handful of prefill tokens plus however many answer tokens it
//! actually generates), not milliseconds - so on a fast model it dwarfs
//! the (also tiny) disk-only estimate, while on a slow model it's
//! comparable to the disk-only estimate's own scale. `predict` now models
//! that overhead explicitly via [`ModelSpeed`] instead of treating it as
//! unexplained error.
//!
//! A first attempt at this fix assumed the answer always uses the full
//! `num_predict` token budget - that overshot badly (measured 157-209%
//! error) because "summarize in five words" makes most models stop at a
//! real EOS well before the cap (observed: 9 tokens actually generated
//! against a budget of 16, confirmed via a direct `eval_count` check).
//! `ModelSpeed::expected_answer_tokens` is therefore *measured* - one real
//! calibration call with a generous budget, reading back how many tokens
//! the model actually chose to generate for this exact prompt - not
//! assumed from the cap.
//!
//! A second, deeper bug in the same direction: the disk term originally
//! used `kv_bytes_per_token` from `hardware.rs`'s crossover formula - the
//! size of a *real* per-layer KV tensor cache (tens to hundreds of KB per
//! token). But that's not what this module actually swaps. Ollama's only
//! reuse mechanism is the `context` field, a plain array of token IDs
//! (~8 bytes/token before JSON overhead) - what `DocumentCache` genuinely
//! writes to and reads from disk here. Using the KV-tensor size overshot
//! the real disk cost by 4-5 orders of magnitude for larger models (a
//! 14B model's 192 KB/token KV tensor vs. its actual ~8 byte/token
//! context array), which is why swap predictions kept overshooting even
//! after the answer-length fix. `ModelSpeed::disk_bytes_per_token` is
//! measured directly from a real serialized context, not derived from
//! model architecture. The practical finding this surfaces: under
//! Ollama's real `context`-array reuse, disk time is nearly always
//! negligible regardless of model size - swap latency is dominated almost
//! entirely by Ollama's own per-call overhead (follow-up prefill +
//! answer decode), not by disk bandwidth at all. The crossover formula in
//! `hardware.rs` remains a valid *hardware* characterization (relevant if
//! a real KV-tensor cache, e.g. llama.cpp's own prompt-cache files, were
//! ever used instead) - it just isn't what this live path measures.
//!
//! Two more real refinements, both found by comparing residual error
//! (actual minus predicted) across model sizes rather than assuming the
//! formula above was the end of it:
//!
//! - `fixed_overhead_ms` was originally calibrated from a trivial
//!   context-free `"hi"` call (~15-35ms measured) - but the real swap call
//!   attaches a multi-thousand-token context array, and the residual
//!   overhead measured against *realistically-shaped* calls (real context
//!   attached, real follow-up prompt) was consistently larger (~45-50ms)
//!   and, crucially, nearly identical between a 3B and a 14B model despite
//!   very different architectures - the signature of a real fixed cost the
//!   trivial calibration was underestimating, not something the earlier
//!   version's formula terms already captured. Calibration now reuses the
//!   same real-context draws already needed for `expected_answer_tokens`,
//!   no extra round trips.
//! - `prefill_tokens_per_sec`/`eval_tokens_per_sec` were originally
//!   measured once from a single seed call and held fixed for an entire
//!   `validate()` run. Under real contention that assumption breaks: a
//!   30b-model run under 97% VRAM contention showed a swap residual 7-8x
//!   larger in absolute terms than smaller models under lighter
//!   contention, consistent with the model's *own* real decode speed
//!   drifting between calibration time and each rep rather than a missing
//!   formula term. Every rep's ingest call already pays for a fresh real
//!   prefill+eval measurement as a side effect (it processes the document,
//!   then generates `DEFAULT_NUM_PREDICT` tokens) - `validate()` now
//!   refreshes `speed` from each rep's own ingest timing before predicting
//!   that rep's swap, at no extra request cost.

use std::path::Path;

use anyhow::Result;
use kvcache::Cache;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::bench::timed_generate;
use crate::hardware;
use crate::ollama::{Client, GenerateOptions, GenerateRequest};
use crate::textgen::{calibrate_chars_per_token, filler_text};

/// What the system looks like right now - queried fresh, not cached,
/// since VRAM/RAM usage genuinely changes moment to moment.
#[derive(Debug, Clone, Serialize)]
pub struct SystemSnapshot {
    pub vram_free_fraction: Option<f64>,
    pub ram_free_fraction: Option<f64>,
    pub disk_bandwidth_bytes_per_sec: f64,
}

pub fn snapshot_now(disk_bandwidth_bytes_per_sec: f64) -> SystemSnapshot {
    SystemSnapshot {
        vram_free_fraction: hardware::vram_free_fraction(),
        ram_free_fraction: hardware::ram_free_fraction(),
        disk_bandwidth_bytes_per_sec,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TimePrediction {
    pub predicted_ingest_ms: f64,
    /// `None` when there's no cache entry to swap in - nothing to predict.
    pub predicted_swap_ms: Option<f64>,
}

/// The exact follow-up prompt `validate()` sends after a cache hit - its
/// own (small) prefill cost is part of what "swap" really pays for, so the
/// prediction needs to know its shape, not just the cached document's size.
pub const FOLLOW_UP_PROMPT: &str = "\n\nQuestion: summarize the above in five words.\nAnswer:";

/// A model+device's measured speed. `prefill_tokens_per_sec` and
/// `eval_tokens_per_sec` are refreshed every rep from that rep's own real
/// ingest timing (see module doc comment - a one-time measurement was
/// found to drift badly under real contention); the other fields are
/// calibrated once up front, since they're stable properties of the
/// prompt shape and serialization format rather than moment-to-moment
/// hardware conditions.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpeed {
    pub prefill_tokens_per_sec: f64,
    /// Decode/eval tok/s - governs how long generating the answer itself
    /// takes, which is the dominant real cost of a "swap" call for slow
    /// models (see [`predict`]'s doc comment).
    pub eval_tokens_per_sec: f64,
    /// How many tokens the model actually generates for [`FOLLOW_UP_PROMPT`]
    /// before hitting a real stop (EOS/newline) - measured, not assumed
    /// from `num_predict` (see module doc comment for why that overshot).
    pub expected_answer_tokens: f64,
    /// Real bytes-per-token of the *actual* on-disk cache artifact (the
    /// serialized `context` token-ID array `DocumentCache` writes) - not
    /// `hardware::kv_bytes_per_token`'s theoretical KV-tensor size (see
    /// module doc comment for why conflating the two overshot badly).
    pub disk_bytes_per_token: f64,
    /// Fixed per-call latency that neither Ollama's own `prompt_eval_duration`
    /// nor `eval_duration` accounts for - measured as the residual
    /// (`wall_ms` minus both reported durations) on calls shaped like the
    /// real swap call (real context attached, real follow-up prompt), not
    /// a trivial context-free call (see module doc comment for why that
    /// distinction matters). Only added to the swap prediction, since
    /// swap's actual measurement is real wall-clock time (it must pay this
    /// cost) while ingest's actual measurement uses Ollama's own reported
    /// duration (which doesn't).
    pub fixed_overhead_ms: f64,
}

/// `speed` is the caller's most recent real measurement for this
/// model+device (this module doesn't own a rolling estimate) - what's
/// genuinely live here is the VRAM/RAM snapshot and the cache-hit check,
/// not tokens/sec, which is stable enough per model+device that
/// re-measuring it every single prediction would mostly just add noise.
///
/// `follow_up_tokens` is the caller's - not baked in as [`FOLLOW_UP_PROMPT`]
/// used to be, since a real caller's "follow-up" isn't necessarily that
/// fixed calibration prompt (e.g. docuzent's is an arbitrary real user
/// question, wildly variable in length) - callers that *are* replaying
/// `FOLLOW_UP_PROMPT` (this crate's own `validate()`) compute it the same
/// way `predict` used to internally: `FOLLOW_UP_PROMPT`'s char count over
/// a measured chars-per-token ratio.
///
/// The swap estimate is disk read + Ollama's own real per-call overhead:
/// reprocessing the short follow-up question (prefill-bound) and
/// generating the answer (eval/decode-bound, `num_predict` tokens). A
/// pure disk-bandwidth estimate ignores both and was measured to
/// underestimate real swap time by 73-86% for fast/small models (where
/// the KV payload is tiny but the model still has to decode 16 tokens) -
/// both terms scale with how fast *this model itself* is, which is why
/// the earlier version's relative error got worse, not better, on faster
/// models: the overhead is roughly fixed in token-count terms, not in
/// milliseconds, so it dominates the (also small) disk-only estimate.
pub fn predict(
    document_tokens: u64,
    speed: &ModelSpeed,
    follow_up_tokens: f64,
    snapshot: &SystemSnapshot,
    cache_hit: bool,
) -> TimePrediction {
    let predicted_ingest_ms = document_tokens as f64 / speed.prefill_tokens_per_sec.max(1e-9) * 1000.0;
    let predicted_swap_ms = if cache_hit {
        let disk_ms = document_tokens as f64 * speed.disk_bytes_per_token / snapshot.disk_bandwidth_bytes_per_sec.max(1e-9) * 1000.0;
        let follow_up_tokens = follow_up_tokens.max(1.0);
        let follow_up_prefill_ms = follow_up_tokens / speed.prefill_tokens_per_sec.max(1e-9) * 1000.0;
        let answer_eval_ms = speed.expected_answer_tokens / speed.eval_tokens_per_sec.max(1e-9) * 1000.0;
        Some(disk_ms + follow_up_prefill_ms + answer_eval_ms + speed.fixed_overhead_ms)
    } else {
        None
    };
    TimePrediction { predicted_ingest_ms, predicted_swap_ms }
}

pub fn hash_document(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A thin, document-shaped wrapper over `kvcache::Cache`: stores/loads the
/// resumable generation context per `(model, context size, document
/// hash)` - the same key shape docuzent's own `Session` uses.
pub struct DocumentCache {
    cache: Cache,
}

impl DocumentCache {
    pub fn open(path: &Path, capacity_bytes: u64) -> Result<Self> {
        Ok(Self { cache: Cache::open(path, capacity_bytes)? })
    }

    fn key(model: &str, num_ctx: u32, doc_hash: &str) -> String {
        format!("{model}|{num_ctx}|{doc_hash}")
    }

    pub fn get(&self, model: &str, num_ctx: u32, doc_hash: &str) -> Result<Option<Vec<i64>>> {
        match self.cache.get(&Self::key(model, num_ctx, doc_hash))? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn put(&self, model: &str, num_ctx: u32, doc_hash: &str, context: &[i64]) -> Result<()> {
        self.cache.put(&Self::key(model, num_ctx, doc_hash), &serde_json::to_vec(context)?)
    }
}

/// One real prediction-vs-actual comparison.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationOutcome {
    pub rep: usize,
    pub scenario: &'static str, // "ingest" | "swap"
    pub document_tokens: u64,
    pub predicted_ms: f64,
    pub actual_ms: f64,
    pub error_pct: f64,
    pub vram_free_fraction: Option<f64>,
    pub ram_free_fraction: Option<f64>,
}

/// Runs `reps` rounds, each producing one real "ingest" outcome (a fresh,
/// never-seen document - tests the ingest-time prediction) immediately
/// followed by one real "swap" outcome (the same document, now cached -
/// tests the swap-time prediction). `prefill_tokens_per_sec` and
/// `eval_tokens_per_sec` start from a real seed calibration call, then get
/// refreshed every rep from that rep's own real ingest timing (see module
/// doc comment for why a one-time measurement wasn't enough).
pub fn validate(
    client: &Client,
    cache: &DocumentCache,
    model: &str,
    num_gpu: Option<i32>,
    num_ctx: u32,
    target_tokens: u64,
    disk_bandwidth: f64,
    reps: usize,
    on_progress: &dyn Fn(&str),
) -> Result<Vec<ValidationOutcome>> {
    on_progress("warmup + calibration...");
    let _ = client.generate(&GenerateRequest {
        model,
        prompt: "warmup",
        stream: false,
        context: None,
        options: GenerateOptions { num_gpu, num_ctx, num_predict: crate::ollama::DEFAULT_NUM_PREDICT },
    })?;
    let chars_per_token = calibrate_chars_per_token(client, model, num_gpu, num_ctx)?;
    let target_chars = (target_tokens as f64 * chars_per_token).round() as usize;

    // One real cold measurement seeds both prefill and eval tok/s for
    // every prediction this run makes - the same call already pays for
    // both (a cold prefill of the document, then DEFAULT_NUM_PREDICT
    // tokens of eval/decode), so no extra round trip is needed.
    let (seed_resp, seed_timing) = timed_generate(client, model, &filler_text(target_chars), None, num_gpu, num_ctx)?;
    let prefill_tokens_per_sec = seed_timing.prompt_eval_count as f64 / (seed_timing.prompt_eval_duration_ms / 1000.0).max(1e-9);
    let eval_tokens_per_sec = seed_timing.eval_count as f64 / (seed_timing.eval_duration_ms / 1000.0).max(1e-9);
    // The real on-disk artifact size for this model's context array, not
    // the theoretical KV-tensor size (see module doc comment) - measured
    // from the same serialization DocumentCache::put actually writes.
    let disk_bytes_per_token = serde_json::to_vec(&seed_resp.context)?.len() as f64 / seed_resp.context.len().max(1) as f64;

    // A dedicated calibration for FOLLOW_UP_PROMPT specifically, with a
    // generous budget so the model can hit its own real stop (EOS) -
    // reading back the true answer length rather than assuming the
    // num_predict cap is always reached (it usually isn't for a
    // "summarize in five words"-style prompt; see module doc comment).
    // Two things matter for this to be representative: (1) it must attach
    // a real document context, not `None` - with nothing to summarize the
    // model behaves differently (observed: 25 tokens with no context vs.
    // 6-11 with a real one attached); (2) Ollama's default sampling isn't
    // greedy, so a single draw is noisy (the same real prompt was
    // observed to produce 6, 11, 7, 7, and 7 tokens across five repeated
    // calls) - averaging several draws is needed for a stable estimate.
    //
    // The same real-context-attached draws also give a much more honest
    // `fixed_overhead_ms` than a trivial `"hi"`/no-context call would: the
    // residual (`wall_ms` minus both reported durations) on a call shaped
    // exactly like the real swap call captures whatever Ollama pays to
    // attach a large context that a context-free call never would (see
    // module doc comment - this alone closed most of the remaining gap on
    // 3B/14B testing).
    const ANSWER_CALIBRATION_DRAWS: usize = 5;
    let mut answer_token_draws = Vec::with_capacity(ANSWER_CALIBRATION_DRAWS);
    let mut overhead_draws = Vec::with_capacity(ANSWER_CALIBRATION_DRAWS);
    for _ in 0..ANSWER_CALIBRATION_DRAWS {
        let t0 = std::time::Instant::now();
        let draw = client.generate(&GenerateRequest {
            model,
            prompt: FOLLOW_UP_PROMPT,
            stream: false,
            context: Some(&seed_resp.context),
            options: GenerateOptions { num_gpu, num_ctx, num_predict: 64 },
        })?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let reported_ms = draw.prompt_eval_duration as f64 / 1e6 + draw.eval_duration as f64 / 1e6;
        answer_token_draws.push(draw.eval_count as f64);
        overhead_draws.push((wall_ms - reported_ms).max(0.0));
    }
    let expected_answer_tokens = (answer_token_draws.iter().sum::<f64>() / answer_token_draws.len() as f64).max(1.0);
    let fixed_overhead_ms = overhead_draws.iter().sum::<f64>() / overhead_draws.len() as f64;

    let mut speed = ModelSpeed { prefill_tokens_per_sec, eval_tokens_per_sec, expected_answer_tokens, disk_bytes_per_token, fixed_overhead_ms };
    // `predict` takes follow-up token count from the caller now (see its
    // doc comment) - this is the one real shape `validate` itself uses.
    let follow_up_tokens = (FOLLOW_UP_PROMPT.chars().count() as f64 / chars_per_token.max(1e-9)).max(1.0);
    on_progress(&format!(
        "seeded prefill speed: {prefill_tokens_per_sec:.0} tok/s, eval speed: {eval_tokens_per_sec:.0} tok/s, expected answer length: {expected_answer_tokens:.0} tokens, real disk payload: {disk_bytes_per_token:.1} bytes/token, fixed overhead: {fixed_overhead_ms:.1}ms"
    ));

    let mut outcomes = Vec::with_capacity(reps * 2);

    for rep in 0..reps {
        on_progress(&format!("rep {}/{reps}: ingest...", rep + 1));
        let text = filler_text(target_chars);
        let doc_hash = hash_document(&text);

        let snapshot = snapshot_now(disk_bandwidth);
        if let Some(w) = hardware::fairness_warning("VRAM", snapshot.vram_free_fraction) {
            on_progress(&w);
        }
        if let Some(w) = hardware::fairness_warning("system RAM", snapshot.ram_free_fraction) {
            on_progress(&w);
        }
        let prediction = predict(target_tokens, &speed, follow_up_tokens, &snapshot, false);

        let (resp, timing) = timed_generate(client, model, &text, None, num_gpu, num_ctx)?;
        let actual_ms = timing.prompt_eval_duration_ms;

        // Refresh from this rep's own real timing rather than trusting the
        // one-time seed measurement for the rest of the run - real decode
        // speed can drift under contention (see module doc comment for why
        // a static estimate badly underpredicted a 30B run's swap time
        // under heavy VRAM pressure). Free: this call already pays for
        // both measurements as a side effect of ingesting the document.
        if timing.prompt_eval_duration_ms > 0.0 {
            speed.prefill_tokens_per_sec = timing.prompt_eval_count as f64 / (timing.prompt_eval_duration_ms / 1000.0);
        }
        if timing.eval_duration_ms > 0.0 {
            speed.eval_tokens_per_sec = timing.eval_count as f64 / (timing.eval_duration_ms / 1000.0);
        }

        outcomes.push(ValidationOutcome {
            rep,
            scenario: "ingest",
            document_tokens: timing.prompt_eval_count,
            predicted_ms: prediction.predicted_ingest_ms,
            actual_ms,
            error_pct: (actual_ms - prediction.predicted_ingest_ms).abs() / actual_ms.max(1e-9) * 100.0,
            vram_free_fraction: snapshot.vram_free_fraction,
            ram_free_fraction: snapshot.ram_free_fraction,
        });
        on_progress(&format!(
            "  ingest: predicted {:.0}ms, actual {:.0}ms ({:.1}% off)",
            outcomes.last().unwrap().predicted_ms,
            outcomes.last().unwrap().actual_ms,
            outcomes.last().unwrap().error_pct
        ));

        cache.put(model, num_ctx, &doc_hash, &resp.context)?;

        on_progress(&format!("rep {}/{reps}: swap...", rep + 1));
        let snapshot = snapshot_now(disk_bandwidth);
        let t0 = std::time::Instant::now();
        let cached = cache.get(model, num_ctx, &doc_hash)?;
        let cache_hit = cached.is_some();
        let prediction = predict(target_tokens, &speed, follow_up_tokens, &snapshot, cache_hit);

        if let Some(context) = cached {
            let _ = timed_generate(client, model, FOLLOW_UP_PROMPT, Some(&context), num_gpu, num_ctx)?;
            let actual_ms = t0.elapsed().as_secs_f64() * 1000.0; // disk read + the generate call, the full real swap path
            let predicted_ms = prediction.predicted_swap_ms.unwrap_or(0.0);
            outcomes.push(ValidationOutcome {
                rep,
                scenario: "swap",
                document_tokens: target_tokens,
                predicted_ms,
                actual_ms,
                error_pct: (actual_ms - predicted_ms).abs() / actual_ms.max(1e-9) * 100.0,
                vram_free_fraction: snapshot.vram_free_fraction,
                ram_free_fraction: snapshot.ram_free_fraction,
            });
            on_progress(&format!(
                "  swap: predicted {:.0}ms, actual {:.0}ms ({:.1}% off)",
                outcomes.last().unwrap().predicted_ms,
                outcomes.last().unwrap().actual_ms,
                outcomes.last().unwrap().error_pct
            ));
        }
    }

    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> SystemSnapshot {
        SystemSnapshot { vram_free_fraction: Some(1.0), ram_free_fraction: Some(1.0), disk_bandwidth_bytes_per_sec: 3_000_000_000.0 }
    }

    fn speed() -> ModelSpeed {
        ModelSpeed {
            prefill_tokens_per_sec: 5000.0,
            eval_tokens_per_sec: 100.0,
            expected_answer_tokens: 9.0,
            disk_bytes_per_token: 8.0,
            fixed_overhead_ms: 35.0,
        }
    }

    const FOLLOW_UP_TOKENS: f64 = 14.0; // FOLLOW_UP_PROMPT.chars().count() / 4.0 chars-per-token, rounded up

    #[test]
    fn predict_gives_no_swap_estimate_without_a_cache_hit() {
        let p = predict(1000, &speed(), FOLLOW_UP_TOKENS, &snapshot(), false);
        assert_eq!(p.predicted_ingest_ms, 200.0); // 1000 tokens / 5000 tok/s = 0.2s = 200ms
        assert!(p.predicted_swap_ms.is_none());
    }

    #[test]
    fn predict_gives_a_swap_estimate_on_a_cache_hit() {
        let p = predict(1000, &speed(), FOLLOW_UP_TOKENS, &snapshot(), true);
        // disk: 1000 tokens * 8 B/token / 3e9 B/s * 1000 = 0.00267 ms
        // follow-up prefill: 14 tokens / 5000 tok/s * 1000
        // answer eval: 9 tokens (measured expected_answer_tokens) / 100 tok/s * 1000 = 90ms
        // fixed overhead: 35ms (measured)
        let disk_ms = 1000.0 * 8.0 / 3_000_000_000.0 * 1000.0;
        let follow_up_ms = FOLLOW_UP_TOKENS / 5000.0 * 1000.0;
        let answer_ms = 9.0 / 100.0 * 1000.0;
        let expected = disk_ms + follow_up_ms + answer_ms + 35.0;
        assert!((p.predicted_swap_ms.unwrap() - expected).abs() < 0.001);
    }

    #[test]
    fn predict_swap_time_is_dominated_by_call_overhead_not_disk() {
        // Under Ollama's real context-array reuse, disk time is
        // negligible regardless of model size (see module doc comment) -
        // the follow-up prefill + answer decode terms should dwarf the
        // disk term even for a large document.
        let p = predict(100_000, &speed(), FOLLOW_UP_TOKENS, &snapshot(), true);
        let disk_only_ms = 100_000.0 * 8.0 / 3_000_000_000.0 * 1000.0;
        assert!(p.predicted_swap_ms.unwrap() > disk_only_ms * 10.0);
    }

    #[test]
    fn hash_document_is_deterministic_and_distinct() {
        assert_eq!(hash_document("hello"), hash_document("hello"));
        assert_ne!(hash_document("hello"), hash_document("world"));
    }

    #[test]
    fn document_cache_round_trips_a_real_context() {
        let path = std::env::temp_dir().join(format!("predictor-test-cache-{}.redb", std::process::id()));
        let cache = DocumentCache::open(&path, 10_000_000).unwrap();
        let ctx = vec![1i64, 2, 3, 4, 5];
        assert!(cache.get("model", 4096, "hash").unwrap().is_none());
        cache.put("model", 4096, "hash", &ctx).unwrap();
        assert_eq!(cache.get("model", 4096, "hash").unwrap(), Some(ctx));
        let _ = std::fs::remove_file(&path);
    }
}
