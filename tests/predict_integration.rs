//! Real, non-mocked check of the predict-only path: live Ollama for
//! architecture info + one cold prefill, a real disk read benchmark, and
//! the crossover formula applied to genuine numbers - not just the
//! formula's own unit tests against hand-picked inputs.

use ollama_kv_profiler::hardware::{kv_bytes_per_token, measure_disk_read_bandwidth_bytes_per_sec, predict_crossover};
use ollama_kv_profiler::ollama::Client;

const MODEL: &str = "qwen2.5:3b";
const HOST: &str = "http://localhost:11434";

#[test]
fn predicts_a_real_crossover_from_live_measurements() {
    let client = Client::new(HOST);
    let max_ctx = client.max_context_length(MODEL).expect("ollama must be running with the model pulled");
    let arch = client.architecture_info(MODEL).unwrap();
    assert!(arch.num_layers > 0);
    assert!(arch.num_kv_heads > 0);
    assert!(arch.head_dim > 0);

    let kv_bytes = kv_bytes_per_token(arch.num_layers, arch.num_kv_heads, arch.head_dim, 2);
    // qwen2.5:3b is a known architecture: 36 layers, 2 KV heads (GQA), 128 head_dim, f16 -> 36864 B/token exactly.
    assert_eq!(kv_bytes, 36_864, "unexpected architecture read back for {MODEL}: {arch:?}");

    let disk_bandwidth = measure_disk_read_bandwidth_bytes_per_sec(64 * 1024 * 1024).unwrap();
    assert!(disk_bandwidth > 1_000_000.0, "disk bandwidth measurement looks implausibly slow: {disk_bandwidth} B/s");

    let num_ctx = (max_ctx / 8).max(512);
    let tokens_per_sec =
        ollama_kv_profiler::bench::quick_cold_tokens_per_sec(&client, MODEL, Some(999), num_ctx, (num_ctx as u64).saturating_sub(200)).unwrap();
    assert!(tokens_per_sec > 0.0);

    let prediction = predict_crossover(kv_bytes, disk_bandwidth, tokens_per_sec);
    println!(
        "real prediction: {:.1} KB/token, {:.0} tok/s, needs {:.0} MB/s, disk gives {:.0} MB/s -> swap wins: {}",
        prediction.kv_bytes_per_token as f64 / 1024.0,
        tokens_per_sec,
        prediction.required_bandwidth_bytes_per_sec / 1e6,
        disk_bandwidth / 1e6,
        prediction.disk_swap_would_win_at_short_to_medium_context
    );
    // No fixed assertion on the verdict itself - that's real hardware-dependent
    // data, not something to hardcode. The test's job is to prove every input
    // to the formula came from a real, live measurement without panicking or
    // producing a nonsensical (zero/negative/NaN) number.
    assert!(prediction.required_bandwidth_bytes_per_sec.is_finite() && prediction.required_bandwidth_bytes_per_sec > 0.0);
}

/// Real predictions for the specific models this project is being profiled
/// against: fully GPU-resident, and "auto" (Ollama's own natural GPU/RAM
/// split when nothing fits fully in VRAM) - answers "is disk swap worth it
/// when everything fits in VRAM?" and "does it help once there's real RAM
/// offloading?" with actual numbers, not a guess. Deliberately does NOT
/// force full-CPU (num_gpu=0) - that is not a scenario anyone actually
/// runs for a model this size; "auto" (real partial offload) is the
/// representative RAM-offloading case.
/// Ignored by default: needs devstral-small-2:24b and qwen3-coder:30b pulled.
#[test]
#[ignore]
fn real_predictions_for_devstral_and_qwen_coder() {
    let client = Client::new(HOST);
    for model in ["devstral-small-2:24b", "qwen3-coder:30b"] {
        let max_ctx = client.max_context_length(model).unwrap();
        let arch = client.architecture_info(model).unwrap();
        let kv_bytes = kv_bytes_per_token(arch.num_layers, arch.num_kv_heads, arch.head_dim, 2);
        let disk_bandwidth = measure_disk_read_bandwidth_bytes_per_sec(64 * 1024 * 1024).unwrap();
        let num_ctx = ((max_ctx as f64 * 0.15) as u32).max(512);
        let target_tokens = (num_ctx as u64).saturating_sub(200);

        for (label, num_gpu) in [("gpu", Some(999)), ("auto", None)] {
            let tokens_per_sec = ollama_kv_profiler::bench::quick_cold_tokens_per_sec(&client, model, num_gpu, num_ctx, target_tokens).unwrap();
            let actual_vram = client.vram_fraction(model).unwrap();
            let prediction = predict_crossover(kv_bytes, disk_bandwidth, tokens_per_sec);
            println!(
                "{model} [{label}, vram={actual_vram:?}] {:.1} KB/token, {tokens_per_sec:.0} tok/s, needs {:.0} MB/s, disk {:.0} MB/s -> swap wins: {}",
                kv_bytes as f64 / 1024.0,
                prediction.required_bandwidth_bytes_per_sec / 1e6,
                disk_bandwidth / 1e6,
                prediction.disk_swap_would_win_at_short_to_medium_context
            );
        }
    }
}
