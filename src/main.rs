use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use ollama_kv_profiler::bench::{self, profile, quick_cold_tokens_per_sec, RunConfig, Verdict};
use ollama_kv_profiler::hardware::{kv_bytes_per_token, measure_disk_read_bandwidth_bytes_per_sec, predict_crossover, CrossoverPrediction};
use ollama_kv_profiler::ollama::Client;

/// Profiles a local Ollama server to decide, per (model, context size,
/// GPU/CPU split), whether reusing its short-lived generation context
/// beats cold reprocessing - and by how much, and for how long. Also
/// predicts, from real hardware measurements alone (no full sweep needed),
/// whether a genuine disk-persisted KV-cache swap (the kind Ollama itself
/// cannot do - see README) would be worth building at all.
#[derive(Parser)]
struct Cli {
    /// Comma-separated Ollama model tags to profile
    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,
    /// Ollama host
    #[arg(long, default_value = "http://localhost:11434")]
    host: String,
    /// Comma-separated device modes to test: "gpu" (num_gpu=999, force full
    /// GPU offload), "auto" (omit num_gpu - Ollama spills whatever doesn't
    /// fit into RAM on its own, the realistic case for an oversized model),
    /// and/or "cpu" (num_gpu=0, force full CPU - rarely what you actually
    /// want; "auto" is the representative RAM-offloading test)
    #[arg(long, value_delimiter = ',', default_value = "gpu,auto")]
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
    /// Skip the full empirical sweep entirely - just measure disk
    /// bandwidth and one cold prefill per (model, gpu mode), then print
    /// the deterministic crossover prediction. Much faster; see README
    /// for what this formula does and does not cover.
    #[arg(long)]
    predict_only: bool,
    /// KV cache dtype size in bytes for the crossover formula - 2 for f16
    /// (Ollama's default), smaller if KV-cache quantization is configured
    #[arg(long, default_value_t = 2)]
    kv_dtype_bytes: u32,
    /// Size of the temp file used to measure real disk read bandwidth
    #[arg(long, default_value_t = 256)]
    disk_bench_mb: usize,
}

fn gpu_mode_setting(label: &str) -> Result<Option<i32>> {
    match label {
        "gpu" => Ok(Some(999)),
        "cpu" => Ok(Some(0)),
        "auto" => Ok(None),
        other => anyhow::bail!("unknown gpu mode `{other}` - expected \"gpu\", \"auto\", or \"cpu\""),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.models.is_empty() {
        anyhow::bail!("pass at least one model with --models");
    }

    println!("Measuring real disk read bandwidth ({} MB, sequential)...", cli.disk_bench_mb);
    let disk_bandwidth = measure_disk_read_bandwidth_bytes_per_sec(cli.disk_bench_mb * 1024 * 1024)?;
    println!("  {:.0} MB/s\n", disk_bandwidth / 1e6);

    let client = Client::new(&cli.host);
    let mut results = Vec::new();
    let mut predictions = Vec::new();

    for model in &cli.models {
        let max_ctx = client
            .max_context_length(model)
            .with_context(|| format!("failed to read `{model}`'s context length - has it been pulled?"))?;
        let arch = client
            .architecture_info(model)
            .with_context(|| format!("failed to read `{model}`'s architecture info"))?;
        let kv_bytes = kv_bytes_per_token(arch.num_layers, arch.num_kv_heads, arch.head_dim, cli.kv_dtype_bytes);
        println!(
            "\n=== {model} (max context {max_ctx} tokens, {} layers, {:.1} avg KV heads x {:.0} dim -> {:.1} KB/token KV cache) ===",
            arch.num_layers,
            arch.num_kv_heads,
            arch.head_dim,
            kv_bytes as f64 / 1024.0
        );

        for gpu_label in &cli.gpu_modes {
            let num_gpu = gpu_mode_setting(gpu_label)?;
            let fraction = cli.context_fractions.first().copied().unwrap_or(0.25);
            let num_ctx = ((max_ctx as f64 * fraction).round() as u32).max(512);
            let target_tokens = (num_ctx as u64).saturating_sub(200);

            if cli.predict_only {
                println!("  [{gpu_label}] measuring cold prefill speed...");
                let tokens_per_sec = quick_cold_tokens_per_sec(&client, model, num_gpu, num_ctx, target_tokens)?;
                let prediction = predict_crossover(kv_bytes, disk_bandwidth, tokens_per_sec);
                print_prediction(model, gpu_label, &prediction);
                predictions.push((model.clone(), gpu_label.clone(), prediction));
                continue;
            }

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

                if fraction == cli.context_fractions[0] {
                    let prediction = predict_crossover(kv_bytes, disk_bandwidth, result.cold.mean_tokens_per_sec);
                    print_prediction(model, gpu_label, &prediction);
                    predictions.push((model.clone(), gpu_label.clone(), prediction));
                }
                results.push(result);
            }
        }
    }

    if !results.is_empty() {
        print_summary_table(&results);
    }
    print_prediction_summary(&predictions);

    if let Some(path) = &cli.output {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&serde_json::json!({
                "disk_bandwidth_bytes_per_sec": disk_bandwidth,
                "sweep_results": results,
                "crossover_predictions": predictions,
            }))?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        println!("\nFull results written to {}", path.display());
    }

    println!(
        "\nIf you found this useful: please consider opening an issue at \
         https://github.com/no-mans-code/ollama-kv-profiler/issues with your \
         --output JSON attached, noting your GPU/CPU/disk. This tool's crossover \
         formula and reuse-window defaults are calibrated on one machine so far - \
         more hardware data makes both better for everyone. (Your data is used only \
         to improve this project; nothing is collected automatically.)"
    );

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
    println!("\n=== Empirical decision summary (Ollama's own short-lived reuse) ===");
    println!("{:<22} {:<5} {:>8} {:>10} {:>14} {:>14} {:<28}", "model", "gpu", "ctx", "vram%", "cold ms", "delayed ms", "verdict");
    for r in results {
        let vram_pct = r.actual_vram_fraction.map(|f| format!("{:.0}", f * 100.0)).unwrap_or_else(|| "?".to_string());
        println!(
            "{:<22} {:<5} {:>8} {:>10} {:>14.0} {:>14.0} {:<28}",
            r.model, r.gpu_label, r.num_ctx, vram_pct, r.cold.mean_prompt_eval_ms, r.delayed_reuse.mean_prompt_eval_ms, verdict_label(r.verdict)
        );
    }
}

fn print_prediction(model: &str, gpu_label: &str, p: &CrossoverPrediction) {
    println!(
        "  [{model}/{gpu_label}] {:.1} KB/token x {:.0} tok/s prefill = needs {:.0} MB/s disk to beat reingest; measured disk gives {:.0} MB/s -> {}",
        p.kv_bytes_per_token as f64 / 1024.0,
        p.prefill_tokens_per_sec,
        p.required_bandwidth_bytes_per_sec / 1e6,
        p.disk_bandwidth_bytes_per_sec / 1e6,
        if p.disk_swap_would_win_at_short_to_medium_context {
            "a real disk swap WOULD beat reingestion here"
        } else {
            "reingestion wins here (at short/medium context - see README on long-context attention cost)"
        }
    );
}

fn print_prediction_summary(predictions: &[(String, String, CrossoverPrediction)]) {
    if predictions.is_empty() {
        return;
    }
    println!("\n=== Deterministic crossover prediction (hardware-only, no full sweep needed) ===");
    println!("{:<22} {:<5} {:>14} {:>16} {:>16} {:<10}", "model", "gpu", "KB/token", "needs MB/s", "disk MB/s", "verdict");
    for (model, gpu_label, p) in predictions {
        println!(
            "{:<22} {:<5} {:>14.1} {:>16.0} {:>16.0} {:<10}",
            model,
            gpu_label,
            p.kv_bytes_per_token as f64 / 1024.0,
            p.required_bandwidth_bytes_per_sec / 1e6,
            p.disk_bandwidth_bytes_per_sec / 1e6,
            if p.disk_swap_would_win_at_short_to_medium_context { "swap" } else { "reingest" }
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
