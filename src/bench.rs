//! The core measurement: for one (model, GPU/CPU split, context size), how
//! much does reusing Ollama's `context` token array - genuinely
//! round-tripped through a disk file each time, not kept in a Rust
//! variable, to faithfully match how a real caller (e.g. docuzent's
//! `kvcache`-backed `Session`) actually persists and reloads it - save
//! versus a cold call that reingests the document from scratch. Measured
//! immediately after and again after a delay with a distraction call in
//! between (an attempt to actually evict whatever internal state made the
//! immediate case fast, rather than just letting time pass and hoping).
//! Runs identically whether the model is fully VRAM-resident or offloaded
//! to RAM (or split between the two) - that's the `gpu_label`/
//! `actual_vram_fraction` axis on [`ProfileResult`], not a separate code path.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::ollama::{Client, GenerateOptions, GenerateRequest, GenerateResponse};
use crate::textgen::{calibrate_chars_per_token, filler_text};

pub struct RunConfig {
    pub model: String,
    /// What was requested: 999 for "force full GPU", 0 for "force full CPU".
    pub num_gpu: Option<i32>,
    /// Human label for `num_gpu`, e.g. "gpu" or "cpu".
    pub gpu_label: String,
    pub num_ctx: u32,
    pub target_tokens: u64,
    pub reps: usize,
    pub delay_secs: u64,
}

#[derive(Serialize, Clone, Debug)]
pub struct Timing {
    pub prompt_eval_count: u64,
    pub prompt_eval_duration_ms: f64,
    pub eval_count: u64,
    pub eval_duration_ms: f64,
    pub load_duration_ms: f64,
    pub wall_ms: f64,
}

#[derive(Serialize, Clone, Debug)]
pub struct Aggregate {
    pub reps: usize,
    pub mean_prompt_eval_ms: f64,
    pub stdev_prompt_eval_ms: f64,
    pub mean_tokens_per_sec: f64,
}

fn aggregate(timings: &[Timing]) -> Aggregate {
    let n = timings.len().max(1) as f64;
    let mean_ms = timings.iter().map(|t| t.prompt_eval_duration_ms).sum::<f64>() / n;
    let variance = timings.iter().map(|t| (t.prompt_eval_duration_ms - mean_ms).powi(2)).sum::<f64>() / n;
    let mean_tps = timings
        .iter()
        .map(|t| {
            if t.prompt_eval_duration_ms > 0.0 {
                t.prompt_eval_count as f64 / (t.prompt_eval_duration_ms / 1000.0)
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n;
    Aggregate {
        reps: timings.len(),
        mean_prompt_eval_ms: mean_ms,
        stdev_prompt_eval_ms: variance.sqrt(),
        mean_tokens_per_sec: mean_tps,
    }
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Even reuse after a delay + distraction call is much faster than cold - a real,
    /// persistent benefit (only possible if Ollama's underlying runner never actually
    /// evicted the shared prefix between calls).
    ReuseHelpsPersistently,
    /// Reuse is fast immediately, but degrades back to cold-like cost after a delay -
    /// the short-lived window this project has already observed with small models.
    ReuseHelpsImmediatelyOnly,
    /// Reuse is not meaningfully faster even immediately - reingest and reuse cost the same.
    NoReuseBenefit,
}

const SPEEDUP_THRESHOLD: f64 = 2.0; // "meaningfully faster" = at least 2x

fn helps(cold: &Aggregate, candidate: &Aggregate) -> bool {
    candidate.mean_prompt_eval_ms > 0.0 && cold.mean_prompt_eval_ms / candidate.mean_prompt_eval_ms >= SPEEDUP_THRESHOLD
}

fn decide(cold: &Aggregate, immediate: &Aggregate, delayed: &Aggregate) -> Verdict {
    match (helps(cold, immediate), helps(cold, delayed)) {
        (true, true) => Verdict::ReuseHelpsPersistently,
        (true, false) => Verdict::ReuseHelpsImmediatelyOnly,
        _ => Verdict::NoReuseBenefit,
    }
}

#[derive(Serialize, Debug)]
pub struct ProfileResult {
    pub model: String,
    pub gpu_label: String,
    pub requested_num_gpu: Option<i32>,
    /// Ground truth from `/api/ps`: fraction of the model's weights
    /// actually resident in VRAM during this run, if Ollama reported it.
    pub actual_vram_fraction: Option<f64>,
    pub num_ctx: u32,
    pub target_tokens: u64,
    pub delay_secs: u64,
    pub cold: Aggregate,
    pub immediate_reuse: Aggregate,
    pub delayed_reuse: Aggregate,
    pub verdict: Verdict,
}

/// Writes a context array to a real file and reads it back, rather than
/// keeping it in a Rust variable across calls - a faithful stand-in for
/// the actual disk-persisted swap this tool exists to evaluate (matching
/// how a real caller, e.g. docuzent's `kvcache`-backed `Session`, would
/// genuinely round-trip it through disk between requests), not an
/// in-memory shortcut that happens to look equivalent.
fn round_trip_context_through_disk(context: &[i64], label: &str) -> Result<Vec<i64>> {
    let path = std::env::temp_dir().join(format!("ollama-kv-profiler-context-{label}-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(context)?).with_context(|| format!("failed to write {}", path.display()))?;
    let bytes = std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let _ = std::fs::remove_file(&path);
    serde_json::from_slice(&bytes).context("failed to deserialize context read back from disk")
}

pub(crate) fn timed_generate(client: &Client, model: &str, prompt: &str, context: Option<&[i64]>, num_gpu: Option<i32>, num_ctx: u32) -> Result<(GenerateResponse, Timing)> {
    let t0 = Instant::now();
    let resp = client.generate(&GenerateRequest {
        model,
        prompt,
        stream: false,
        context,
        options: GenerateOptions { num_gpu, num_ctx, num_predict: crate::ollama::DEFAULT_NUM_PREDICT },
    })?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let timing = Timing {
        prompt_eval_count: resp.prompt_eval_count,
        prompt_eval_duration_ms: resp.prompt_eval_duration as f64 / 1e6,
        eval_count: resp.eval_count,
        eval_duration_ms: resp.eval_duration as f64 / 1e6,
        load_duration_ms: resp.load_duration as f64 / 1e6,
        wall_ms,
    };
    Ok((resp, timing))
}

/// A fast path for [`crate::hardware`]'s crossover prediction: just enough
/// to get a real cold prefill tokens/sec for this model+device+context -
/// one warmup call plus one timed cold call, skipping the immediate/
/// delayed/distraction measurements a full [`profile`] run does.
pub fn quick_cold_tokens_per_sec(client: &Client, model: &str, num_gpu: Option<i32>, num_ctx: u32, target_tokens: u64) -> Result<f64> {
    let _ = client.generate(&GenerateRequest {
        model,
        prompt: "warmup",
        stream: false,
        context: None,
        options: GenerateOptions { num_gpu, num_ctx, num_predict: crate::ollama::DEFAULT_NUM_PREDICT },
    })?;
    let chars_per_token = calibrate_chars_per_token(client, model, num_gpu, num_ctx)?;
    let target_chars = (target_tokens as f64 * chars_per_token).round() as usize;
    let text = filler_text(target_chars);
    let (_, timing) = timed_generate(client, model, &text, None, num_gpu, num_ctx)?;
    Ok(timing.prompt_eval_count as f64 / (timing.prompt_eval_duration_ms / 1000.0).max(1e-9))
}

pub fn profile(client: &Client, cfg: &RunConfig, on_progress: &dyn Fn(&str)) -> Result<ProfileResult> {
    on_progress(&format!(
        "loading {} ({}, num_ctx={})...",
        cfg.model, cfg.gpu_label, cfg.num_ctx
    ));
    // One throwaway call first so the GPU split + context size take effect
    // (Ollama reloads the runner when these change) before any timed rep.
    let _ = client.generate(&GenerateRequest {
        model: &cfg.model,
        prompt: "warmup",
        stream: false,
        context: None,
        options: GenerateOptions { num_gpu: cfg.num_gpu, num_ctx: cfg.num_ctx, num_predict: crate::ollama::DEFAULT_NUM_PREDICT },
    })?;
    let actual_vram_fraction = client.vram_fraction(&cfg.model)?;

    // Fairness checks (#1/#2): whichever resource this run's real split
    // actually depends on, warn (never abort) if something else is using
    // enough of it that timings may not reflect an idle system. Checked
    // against the REAL post-warmup split, not the requested one, since
    // `auto` mode's real split is only known after the warmup call.
    let vram_used = actual_vram_fraction.unwrap_or(if cfg.num_gpu == Some(0) { 0.0 } else { 1.0 });
    if vram_used > 0.0 {
        if let Some(w) = crate::hardware::fairness_warning("VRAM", crate::hardware::vram_free_fraction()) {
            on_progress(&w);
        }
    }
    if vram_used < 1.0 {
        if let Some(w) = crate::hardware::fairness_warning("system RAM", crate::hardware::ram_free_fraction()) {
            on_progress(&w);
        }
    }

    let chars_per_token = calibrate_chars_per_token(client, &cfg.model, cfg.num_gpu, cfg.num_ctx)?;
    let target_chars = (cfg.target_tokens as f64 * chars_per_token).round() as usize;

    let mut cold_timings = Vec::with_capacity(cfg.reps);
    let mut immediate_timings = Vec::with_capacity(cfg.reps);
    let mut delayed_timings = Vec::with_capacity(cfg.reps);

    for rep in 0..cfg.reps {
        on_progress(&format!("  rep {}/{}: cold prefill...", rep + 1, cfg.reps));
        let text = filler_text(target_chars);
        let (cold_resp, cold_t) = timed_generate(client, &cfg.model, &text, None, cfg.num_gpu, cfg.num_ctx)?;
        if cold_t.load_duration_ms > 1000.0 {
            on_progress(&format!(
                "  WARNING: this rep paid {:.0}ms of model load time - something evicted the runner between reps, cold timings may be inflated",
                cold_t.load_duration_ms
            ));
        }
        cold_timings.push(cold_t);

        on_progress("  immediate reuse (context round-tripped through disk)...");
        let disk_context = round_trip_context_through_disk(&cold_resp.context, "immediate")?;
        let (_, imm_t) = timed_generate(
            client,
            &cfg.model,
            "\n\nQuestion: summarize the above in five words.\nAnswer:",
            Some(&disk_context),
            cfg.num_gpu,
            cfg.num_ctx,
        )?;
        immediate_timings.push(imm_t);

        if cfg.delay_secs > 0 {
            on_progress(&format!("  waiting {}s...", cfg.delay_secs));
            std::thread::sleep(Duration::from_secs(cfg.delay_secs));
        }
        on_progress("  distraction call (attempting eviction)...");
        let distraction = filler_text(target_chars.min(4000));
        let _ = timed_generate(client, &cfg.model, &distraction, None, cfg.num_gpu, cfg.num_ctx)?;

        on_progress("  delayed reuse (context round-tripped through disk)...");
        let disk_context = round_trip_context_through_disk(&cold_resp.context, "delayed")?;
        let (_, delayed_t) = timed_generate(
            client,
            &cfg.model,
            "\n\nQuestion: name one topic mentioned.\nAnswer:",
            Some(&disk_context),
            cfg.num_gpu,
            cfg.num_ctx,
        )?;
        delayed_timings.push(delayed_t);
    }

    let cold = aggregate(&cold_timings);
    let immediate_reuse = aggregate(&immediate_timings);
    let delayed_reuse = aggregate(&delayed_timings);
    let verdict = decide(&cold, &immediate_reuse, &delayed_reuse);

    Ok(ProfileResult {
        model: cfg.model.clone(),
        gpu_label: cfg.gpu_label.clone(),
        requested_num_gpu: cfg.num_gpu,
        actual_vram_fraction,
        num_ctx: cfg.num_ctx,
        target_tokens: cfg.target_tokens,
        delay_secs: cfg.delay_secs,
        cold,
        immediate_reuse,
        delayed_reuse,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agg(ms: f64) -> Aggregate {
        Aggregate { reps: 1, mean_prompt_eval_ms: ms, stdev_prompt_eval_ms: 0.0, mean_tokens_per_sec: 0.0 }
    }

    #[test]
    fn persistent_benefit_when_both_immediate_and_delayed_are_fast() {
        let v = decide(&agg(1000.0), &agg(50.0), &agg(60.0));
        assert_eq!(v, Verdict::ReuseHelpsPersistently);
    }

    #[test]
    fn immediate_only_when_delayed_regresses_to_cold() {
        let v = decide(&agg(1000.0), &agg(50.0), &agg(950.0));
        assert_eq!(v, Verdict::ReuseHelpsImmediatelyOnly);
    }

    #[test]
    fn no_benefit_when_nothing_is_meaningfully_faster() {
        let v = decide(&agg(1000.0), &agg(800.0), &agg(900.0));
        assert_eq!(v, Verdict::NoReuseBenefit);
    }
}
