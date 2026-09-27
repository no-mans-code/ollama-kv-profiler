mod bench;
mod ollama;
mod textgen;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use bench::{profile, RunConfig, Verdict};
use ollama::Client;

/// Profiles a local Ollama server to decide, per (model, context size,
/// GPU/CPU split), whether reusing its short-lived generation context
/// beats cold reprocessing - and by how much, and for how long.
#[derive(Parser)]
struct Cli {
    /// Comma-separated Ollama model tags to profile
    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,
    /// Ollama host
    #[arg(long, default_value = "http://localhost:11434")]
    host: String,
    /// Comma-separated GPU/CPU modes to test: "gpu" (num_gpu=999, force
    /// full GPU offload) and/or "cpu" (num_gpu=0, force full CPU)
    #[arg(long, value_delimiter = ',', default_value = "gpu,cpu")]
    gpu_modes: Vec<String>,
    /// Comma-separated fractions of each model's own max context length to
    /// test (e.g. 0.25,0.75 tests a quarter-full and mostly-full window)
    #[arg(long, value_delimiter = ',', default_value = "0.25,0.75")]
    context_fractions: Vec<f64>,
    /// Repetitions per (model, gpu mode, context fraction) - more reps
    /// gives tighter mean/stdev at the cost of a longer run
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Seconds to wait, plus one distraction call, before the delayed-reuse
    /// measurement - an attempt to actually evict whatever internal state
    /// made immediate reuse fast, not just letting time pass. Calibrated
    /// empirically against qwen2.5:3b: reuse still fully worked at a 5s
    /// delay and had fully degraded by 45s, so 30s is a safe default to
    /// reliably land past the threshold rather than guess at it.
    #[arg(long, default_value_t = 30)]
    delay_secs: u64,
    /// Where to write the full JSON results (in addition to the printed summary)
    #[arg(long)]
    output: Option<PathBuf>,
}

fn gpu_mode_setting(label: &str) -> Result<i32> {
    match label {
        "gpu" => Ok(999),
        "cpu" => Ok(0),
        other => anyhow::bail!("unknown gpu mode `{other}` - expected \"gpu\" or \"cpu\""),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.models.is_empty() {
        anyhow::bail!("pass at least one model with --models");
    }

    let client = Client::new(&cli.host);
    let mut results = Vec::new();

    for model in &cli.models {
        let max_ctx = client
            .max_context_length(model)
            .with_context(|| format!("failed to read `{model}`'s context length - has it been pulled?"))?;
        println!("\n=== {model} (max context {max_ctx} tokens) ===");

        for gpu_label in &cli.gpu_modes {
            let num_gpu = gpu_mode_setting(gpu_label)?;

            for &fraction in &cli.context_fractions {
                let num_ctx = ((max_ctx as f64 * fraction).round() as u32).max(512);
                let target_tokens = (num_ctx as u64).saturating_sub(200);

                let cfg = RunConfig {
                    model: model.clone(),
                    num_gpu,
                    gpu_label: gpu_label.clone(),
                    num_ctx,
                    target_tokens,
                    reps: cli.reps,
                    delay_secs: cli.delay_secs,
                };

                let result = profile(&client, &cfg, &|msg| println!("{msg}"))?;
                print_result(&result);
                results.push(result);
            }
        }
    }

    print_summary_table(&results);

    if let Some(path) = &cli.output {
        std::fs::write(path, serde_json::to_string_pretty(&results)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("\nFull results written to {}", path.display());
    }

    Ok(())
}

fn print_result(r: &bench::ProfileResult) {
    let vram = r
        .actual_vram_fraction
        .map(|f| format!("{:.0}% in VRAM", f * 100.0))
        .unwrap_or_else(|| "VRAM split unknown".to_string());
    println!(
        "  ctx={} tokens={} [{}] -> cold {:.0}ms ({:.0} tok/s), immediate {:.0}ms, delayed {:.0}ms -> {:?}",
        r.num_ctx,
        r.target_tokens,
        vram,
        r.cold.mean_prompt_eval_ms,
        r.cold.mean_tokens_per_sec,
        r.immediate_reuse.mean_prompt_eval_ms,
        r.delayed_reuse.mean_prompt_eval_ms,
        r.verdict
    );
}

fn print_summary_table(results: &[bench::ProfileResult]) {
    println!("\n=== Decision summary ===");
    println!("{:<22} {:<5} {:>8} {:>10} {:>14} {:>14} {:<28}", "model", "gpu", "ctx", "vram%", "cold ms", "delayed ms", "verdict");
    for r in results {
        let vram_pct = r.actual_vram_fraction.map(|f| format!("{:.0}", f * 100.0)).unwrap_or_else(|| "?".to_string());
        println!(
            "{:<22} {:<5} {:>8} {:>10} {:>14.0} {:>14.0} {:<28}",
            r.model, r.gpu_label, r.num_ctx, vram_pct, r.cold.mean_prompt_eval_ms, r.delayed_reuse.mean_prompt_eval_ms, verdict_label(r.verdict)
        );
    }
}

fn verdict_label(v: Verdict) -> &'static str {
    match v {
        Verdict::ReuseHelpsPersistently => "reuse persistently (swap worth it)",
        Verdict::ReuseHelpsImmediatelyOnly => "reuse only immediately (reingest after)",
        Verdict::NoReuseBenefit => "always reingest",
    }
}
