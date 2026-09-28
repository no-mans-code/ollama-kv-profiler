# ollama-kv-profiler

Profiles a local Ollama server to decide, per `(model, context size, GPU/CPU split)`, whether reusing Ollama's short-lived generation context beats a cold call that reprocesses the same text from scratch - and for how long that benefit actually lasts.

## Two modes

- **Deterministic** (`--predict-only`): no sweep, just real hardware measurements (disk bandwidth, model architecture, one cold prefill) fed into a closed-form formula - see "Determinism" below. Fast; answers "would swap win" with a robust yes/no even when the exact prefill speed carries real uncertainty.
- **Empirical** (the default, full sweep): measures what Ollama's own short-lived reuse window actually delivers over time (cold / immediate reuse / reuse after a real delay), not a prediction. Slower, but it's the only way to see the window itself degrade, and the only way this project caught real bugs (see "Real results" below) that a prediction alone would have missed.

They answer related but different questions and are meant to be used together: the deterministic mode tells you whether building real disk-swap support is worth it at all; the empirical mode tells you what Ollama can actually deliver *today* without one.

## Why this exists

Ollama's public API has no real KV-cache-to-disk path - its scheduler unloads whole models rather than tiering the cache, confirmed both by community reporting and by direct measurement in a sibling project ([docuzent](https://github.com/no-mans-code/docuzent)'s `Blueprint.md`). What it *does* have is a short-lived internal reuse window: pass its `context` token array back on a follow-up call and, if nothing has evicted it yet, the follow-up skips reprocessing the shared prefix. This tool measures exactly how short-lived that window is, across model sizes and GPU/CPU splits, rather than assuming one answer applies everywhere.

## What it actually controls

- **GPU/RAM split**: Ollama's `num_gpu` generate option - `999` forces full GPU, `0` forces full CPU, and omitting it entirely (`auto` mode) lets Ollama pick its own placement, spilling whatever doesn't fit into RAM on its own. `auto` is the default RAM-offloading test, not `cpu` - forcing a model fully onto a GPU too small for it (or, worse, fully off it when a partial fit would do) measures a configuration nobody actually runs, and can be dramatically slower or system-destabilizing rather than just "the RAM case" (see "Real results" below). `/api/ps`'s `size_vram`/`size` is read back as ground truth after every run, since a requested split isn't always honored exactly, and `auto`'s real split is never known in advance.
- **Context size**: Ollama defaults to a 4096-token runtime window regardless of a model's real maximum unless `num_ctx` is passed explicitly - this tool always passes it, sized as a fraction of the model's true max (from `/api/show`'s `model_info`).
- **Token count**: filler text is sized via an empirically calibrated chars-per-token ratio (measured per model with a real call), not a guessed constant.

## Usage

```bash
# Full empirical sweep (both modes) - measures Ollama's actual reuse window
cargo run -- \
  --models devstral-small-2:24b,qwen3-coder:30b \
  --gpu-modes gpu,auto \
  --context-fractions 0.25,0.75 \
  --reps 3 \
  --delay-secs 30 \
  --output results.json

# Deterministic mode - skips the sweep, just the hardware-only crossover prediction
cargo run -- --models devstral-small-2:24b --gpu-modes gpu,auto --predict-only
```

For each `(model, gpu mode, context fraction)`, it measures (averaged over `--reps` repetitions):
- **cold**: a fresh prefill of unique filler text, no reuse.
- **immediate reuse**: a follow-up right after, passing `context` back.
- **delayed reuse**: `--delay-secs` of waiting, plus one distraction call (an attempt to actually evict whatever state made immediate reuse fast, not just letting time pass), then a follow-up passing the *original* `context` back.

...and emits a verdict: `ReuseHelpsPersistently` (swap is worth it even after a real gap), `ReuseHelpsImmediatelyOnly` (only within a short window), or `NoReuseBenefit` (always just reingest).

## A real calibration finding

Against `qwen2.5:3b` (fully GPU-resident, ~3000-token context): reuse was still fully intact at a 5s delay (delayed ≈ immediate, both far below cold) and had fully degraded by 45s (delayed ≈ cold). The real threshold sits somewhere in between - hence the 30s default, chosen to reliably land past it rather than guess. This is itself evidence for the tool's premise: the window is real, measurable, and worth characterizing per config rather than assuming.

## Why this over an existing tool

Researched before building (see the commit history / [docuzent](https://github.com/no-mans-code/docuzent)'s `Blueprint.md` for the trail) - nothing found does this specific job:

- **`llama-bench`** (llama.cpp's own tool) measures raw prefill/decode throughput for a fixed config - useful, but it talks to llama.cpp's GGUF loading directly, bypassing Ollama's own scheduler entirely. It can't tell you what *Ollama* will actually do, which is what matters if Ollama is your real production stack, not just the inference engine underneath it.
- **No tool we found measures the reuse-window itself** - how long Ollama's internal warm state actually survives before a follow-up call stops benefiting, as a function of model size and GPU/CPU placement. That required literally timing immediate-vs-delayed reuse with a real distraction call in between, not a single-shot throughput number.
- **Ground-truths its own variables instead of trusting requested config**: reads back *actual* VRAM residency via `/api/ps` rather than trusting a requested `num_gpu`, and calibrates chars-per-token empirically per model rather than assuming a fixed ratio (which is off enough across model families to matter).
- **Faithfully round-trips context through a real disk file** for every reuse measurement, not an in-memory shortcut - so results are representative of an actual disk-persisted cache implementation (e.g. docuzent's `kvcache`-backed `Session`), not an idealized same-process approximation of one.
- **Generic across any locally pulled Ollama model and any context size** - not a fixed benchmark suite tied to specific hardware or model list.

## Scope

Deliberately Ollama-only for the empirical sweep (`immediate_reuse`/`delayed_reuse`) - it measures Ollama's actual behavior, not a hypothetical. Genuine KV-cache-to-disk swap only exists at the `llama.cpp` level directly (`llama-server`'s session/prompt-cache files); see [#3](https://github.com/no-mans-code/ollama-kv-profiler/issues/3) for driving that directly. But the *deterministic prediction* below doesn't need that backend to give a real, hardware-grounded answer to whether it would be worth building at all.

## Determinism: is there a formula, or is per-config profiling the only option?

Asked directly: for a given model + context size, can we tell whether a real disk-persisted KV-cache swap would beat cold reingestion just by inspecting the hardware - no sweep required? **Yes, approximately, and this is published research, not this project's own conjecture** - see [KVSwap](https://arxiv.org/pdf/2511.11907) (disk-aware KV offloading for on-device inference, i.e. exactly this question), ["The KV Cache Is the New Memory Wall"](https://arxiv.org/html/2609.30854) (the roofline formalization used below), and [LMCache](https://github.com/lmcache/lmcache)'s production CPU/disk offload backend (the same idea, already shipping in vLLM).

**The formula** (`hardware::predict_crossover`): swapping in a cached context takes `kv_bytes_per_token / disk_bandwidth` seconds per token; reingesting takes `1 / prefill_tokens_per_sec`. Swap wins when:

```
kv_bytes_per_token * prefill_tokens_per_sec < disk_bandwidth_bytes_per_sec
```

Both sides are measurable without a full sweep:
- `kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes` - computed from `/api/show`'s own architecture metadata (`*.block_count`, `*.attention.head_count_kv`, `*.embedding_length`), not guessed. Note it's the *KV* head count, not the (often much larger) query head count - grouped-query attention means these differ a lot, and it's the KV head count that actually sets cache size.
- `disk_bandwidth_bytes_per_sec` - one real sequential-read benchmark (`hardware::measure_disk_read_bandwidth_bytes_per_sec`), model-independent, run once.
- `prefill_tokens_per_sec` - one real cold prefill call (`bench::quick_cold_tokens_per_sec`), reusable across context sizes since it's roughly constant in the compute-bound, linear-layers regime.

**Real answer for this machine** (RTX 5080, NVMe): `qwen2.5:3b` fully GPU-resident needs only ~366 MB/s to make swap worth it; this disk measured ~3.1 GB/s - **the formula says swap would win even with everything in VRAM**, which answers "if everything is in VRAM, is it even worth thinking about?" with a real, sometimes counter-intuitive **yes** (for a fast enough disk and a model whose GPU prefill still isn't fast enough to outrun a modern NVMe). See `--predict-only` output for devstral-small-2:24b and qwen3-coder:30b at both `gpu` and `cpu` offload for the RAM-offloading side of the same question.

**The caveat that keeps this from being the whole answer**: the formula above is context-size-*independent* only in the regime where prefill cost is linear in context length. Real attention cost is quadratic - it grows faster than context length once the sequence gets long enough that attention (not the linear layers) dominates prefill FLOPs. Past that point, reingestion gets *relatively* more expensive as context grows, so real disk swap becomes more attractive at long context even in a config this formula calls a reingest win at short/medium length. Published work confirms the same shape (short contexts favor recompute, contexts in the tens of thousands of tokens increasingly favor offload) without giving one universal crossover token count - it depends on the model's own head-count/head-dim ratio, which changes where the quadratic term starts to dominate. So: **the linear-regime formula above is a real, deterministic, hardware-only answer for short-to-medium context; at very long context, the honest answer is still "profile it" until this tool tracks the quadratic term explicitly** (a natural extension, not built yet).

**Does Ollama's own ephemeral reuse ever capture this benefit today?** No - that's the whole reason for the "Why this exists" section above. The formula predicts what a *real* llama.cpp-backed disk swap could achieve; the empirical sweep measures what Ollama's own short-lived internal cache actually delivers, which our own measurements show degrades to no-benefit within tens of seconds. The gap between "formula says swap should win" and "Ollama can't actually deliver it" is real, measurable opportunity - which is exactly what issue #3 exists to close.

## Real results across 5 configurations (RTX 5080, NVMe ~3.0-3.4 GB/s)

Every one of the 5 successfully-profiled `(model, GPU/RAM mode)` combinations agrees: **swap would win**, by a margin ranging from ~8x to over 1000x depending on the model. No counterexample found yet on this hardware.

| Model | Mode | KV cache | Prefill | Needs | Margin |
|---|---|---|---|---|---|
| `qwen2.5:3b` (3.1B dense) | `gpu` (100% VRAM) | 36.0 KB/token | 9,923 tok/s | 366 MB/s | **8.5x** |
| `devstral-small-2:24b` (23.9B dense) | `gpu` forced (100%\*) | 200.0 KB/token | 45 tok/s | 9 MB/s | **333x** |
| `devstral-small-2:24b` (23.9B dense) | `auto` (55.9% VRAM) | 200.0 KB/token | **671 tok/s** | 137 MB/s | **22x** |
| `qwen3-coder:30b` | `gpu` (100%\*) | 48.0 KB/token | 43 tok/s | 2 MB/s | **1,677x** |
| `qwen3-coder:30b` | `auto` (100%\*) | 48.0 KB/token | 44 tok/s | 2 MB/s | **1,677x** |

\* Ollama reported 100% VRAM residency for models whose total size (22-24GB) exceeds this GPU's 16GB - almost certainly Windows WDDM's shared-GPU-memory oversubscription silently paging into system RAM rather than genuine full-VRAM residency. Treat these "100%" figures as Ollama's self-report, not verified physical fact - a real limitation of `/api/ps` as ground truth on Windows specifically.

**The forced-GPU vs. auto finding that matters operationally, independent of swap-vs-reingest**: for `devstral-small-2:24b`, forcing `num_gpu=999` (all layers to GPU) measured **15x slower** (45 tok/s) than letting Ollama pick its own split (`auto`, 671 tok/s). Forcing every layer onto a GPU too small for the model likely triggers WDDM thrashing between VRAM and shared memory that Ollama's own placement heuristic avoids by keeping the overflow cleanly in RAM instead of fighting the driver for it. This got dramatically worse outside Ollama entirely: driving `gemma4:26b`'s raw GGUF weights directly through `llama-cli` with `-ngl 999` (16.9GB file, 16GB card) caused system-wide thrashing severe enough that `tasklist`/`taskkill` themselves started timing out - the process had to be force-killed via PowerShell. **Forcing full GPU offload on a model that doesn't fit is not a "slower but safe" fallback; it can degrade the whole system, not just that one process.** This is exactly why `auto` mode (Ollama's own placement) replaced forced full-CPU as this tool's default RAM-offloading test, per direct user correction during this research - forcing an artificial extreme in either direction produces a config nobody would actually run and can measure something worse than either real option.

**A real architecture bug this research caught before it corrupted a result**: `gemma4:26b`'s `attention.head_count_kv` is a **per-layer array** (`[8,8,8,8,8,2]` repeated 5x - Gemma's interleaved local/global attention), not one scalar for the whole model. The original code called `.as_u64()` on it, silently got `None`, and fell back to the query head count (16) for every layer - which would have overstated `kv_bytes_per_token` by more than 2x (330 KB/token instead of the correct 144.4 KB/token) without ever erroring. Fixed by detecting the array case and averaging across layers (`num_layers * average` reproduces the true cross-layer sum exactly), with a regression test encoding the real numbers. Any architecture with per-layer-varying attention config would have hit this silently.

**`gemma4:26b` itself could not be fully profiled**: it fails to load at all through the installed Ollama (0.34.4) - `llama_init_from_model: failed to initialize the context: Gemma4Assistant requires ctx_other to be set` - reproduced with and without any custom options, and unchanged after a full re-pull, confirming a genuine Ollama/llama.cpp-integration bug for this specific model rather than a corrupted download. Notably, **raw llama.cpp (`llama-cli`, official b11227 Windows/CUDA build) loads the same GGUF weights successfully** (reached its interactive prompt) - strong evidence this is specifically a defect in Ollama's bundled/forked llama.cpp version, not the model file or upstream llama.cpp. Real prefill numbers for this model were not safely obtained this session (see thrashing finding above); a follow-up with a sane `-ngl` value (not the forced extreme) is the natural next step, tracked under [issue #6](https://github.com/no-mans-code/ollama-kv-profiler/issues/6).

**A tooling gap this same failure exposed** ([issue #5](https://github.com/no-mans-code/ollama-kv-profiler/issues/5), fixed): the first time this happened, `Client::generate()` only surfaced `status code 500` - no indication of *why*. Diagnosing it required manually `curl`-ing the same request outside the tool. Fixed by reading and surfacing Ollama's own `{"error": "..."}` body on every non-2xx response, for every endpoint this tool calls, not just the one call site that happened to hit it - a profiling tool whose job is running against many different models shouldn't require leaving it to find out why one failed.

**One more real architecture detail worth flagging**: `gemma4:26b`'s Ollama manifest reveals it bundles three components under one tag - the main weights (16.9GB, filename references `26B-A4B`, the standard naming for "26B total, ~4B *active*" Mixture-of-Experts parameters), a 1.2GB vision projector, and a 461MB speculative-decoding draft model. `general.parameter_count` (25.2B) reports the *total*, not the active-per-token count that actually drives prefill FLOPs for an MoE model - a real gap in the current formula, which assumes a dense forward pass. `hardware::kv_bytes_per_token` is unaffected (KV cache size depends on attention configuration, not the MoE routing), but a future FLOPs-based *tokens/sec* predictor (see "Where precision breaks down" note in commit history) would need to account for this separately for MoE architectures.

## Live prediction: real-time ingest-vs-swap forecasting (`--live-validate`)

A third mode, distinct from both the deterministic crossover formula and the empirical reuse-window sweep: given a live system snapshot (VRAM/RAM free right now, via `predictor::snapshot_now`) and a document's real token count, predict how long ingesting it fresh will take versus swapping its context in from a real on-disk LRU cache (`kvcache` - the same crate and key shape [docuzent](https://github.com/no-mans-code/docuzent)'s own `Session` uses) - then immediately measure the real thing and compare.

```bash
cargo run -- --models qwen2.5:3b,qwen2.5-coder:14b,qwen3-coder:30b \
  --gpu-modes gpu,auto --context-fractions 0.1 \
  --live-validate --live-reps 10 --output live_results.json
```

Each rep produces one real "ingest" comparison (a fresh, never-seen document) immediately followed by one real "swap" comparison (the same document, now cached). `hardware::fairness_warning` fires inline whenever VRAM or system RAM is below 90% free, since a contended run's timings aren't representative of an idle system.

### Ingest prediction: accurate from the start

Across all three tested scales, predicting cold-ingest time from a calibrated `prefill_tokens_per_sec` was consistently close to reality: **0.1-8% error**, no fixing required. This half of the feature works well out of the box - prefill throughput is stable enough per model+device that one calibration call generalizes.

### Swap prediction: three real bugs found and fixed, in order of discovery

The first version of `predictor::predict` estimated swap time as pure disk bandwidth (`document_tokens * kv_bytes_per_token / disk_bandwidth`) - the crossover formula from the deterministic mode above, made live. Real measurement against `qwen2.5:3b` immediately showed **73-86% error**, and it only got worse on `qwen2.5-coder:14b` and `qwen3-coder:30b`. Three separate, compounding bugs turned out to be responsible - each one real, each one caught only because the tool measures the *actual* thing rather than trusting the formula:

1. **Assumed the answer always uses the full `num_predict` token budget.** The follow-up prompt this tool sends after a cache hit ("summarize the above in five words") reliably makes the model stop at a real EOS well before the cap - observed 6-11 tokens generated against a budget of 16-64, confirmed via `eval_count` on a direct `curl`. Fixed by *measuring* `expected_answer_tokens` via real calibration calls (averaged over 5 draws, since Ollama's default sampling isn't greedy and a single draw is noisy) - not assuming the cap is reached.
2. **Used the theoretical KV-tensor size instead of the real on-disk payload.** `kv_bytes_per_token` (tens to hundreds of KB/token) describes a *hypothetical* real KV-cache-to-disk mechanism (see "Determinism" above) - not what this tool actually swaps. Ollama's only reuse mechanism is the `context` field, a plain JSON array of token IDs (~5-8 bytes/token). Using the KV-tensor size overshot the real disk cost by **4-5 orders of magnitude** for larger models (14B's 192 KB/token KV tensor vs. its measured ~5.7 byte/token context array) - the more informative finding this surfaced is that **under Ollama's real context-array reuse, disk time is nearly always negligible regardless of model size**; swap latency is dominated by Ollama's own per-call overhead, not disk bandwidth. Fixed by measuring `disk_bytes_per_token` directly from a real serialized context.
3. **Ignored fixed per-call latency.** Neither Ollama's own `prompt_eval_duration` nor `eval_duration` accounts for HTTP round-trip and request-scheduling overhead - measured directly via near-zero-work calibration calls (`wall_ms` minus both reported durations): consistently landed around 15-35ms. Added as `ModelSpeed::fixed_overhead_ms`, applied only to the swap prediction (ingest's "actual" is measured from Ollama's own reported duration, which doesn't pay this cost; swap's "actual" is real wall-clock time, which does).

### Two more refinements, found by comparing the residual error across model sizes

After the three fixes above, a residual gap remained - and it was *larger* on the fastest model (3B, ~51% mean error) than the slowest (14B, ~25%), the opposite of what a disk-bandwidth-driven formula would predict. Two more real, root-caused issues explained it:

4. **The `fixed_overhead_ms` calibration call was shaped wrong.** It used a trivial context-free `"hi"` call (measuring ~15-35ms) - but the real swap call attaches a multi-thousand-token context array. Measuring the same residual (`wall_ms` minus both reported durations) against *realistically-shaped* calibration calls (real context attached, real follow-up prompt - reusing the draws already needed for `expected_answer_tokens`, no extra round trips) consistently found a larger figure (~45-50ms) that was nearly identical between the 3B and 14B models despite very different architectures - the signature of a real cost the trivial calibration was underestimating.
5. **`prefill_tokens_per_sec`/`eval_tokens_per_sec` were measured once and held fixed for an entire run.** Under real contention that doesn't hold. `validate()` now refreshes both from every rep's own real ingest timing (which already pays for the measurement as a side effect) instead of trusting one seed call for all 10 reps - at no extra request cost.

**A genuine methodology lesson along the way**: an early attempt at re-validating the 30B model under these two fixes showed *swap* taking nearly as long as a full cold *ingest* (both ~142 seconds) - which looked at first like Ollama's context reuse failing outright under memory pressure. `nvidia-smi --query-compute-apps` told a different story: a `python.exe` process, orphaned from an earlier killed test script, was still holding a GPU context and consuming VRAM this entire time - the "97% VRAM used by something else" fairness warnings were correctly reporting genuine contention, just not the kind anyone intended to measure. Killing it and confirming a fully idle GPU (`0 MiB` used) via `nvidia-smi` before re-running fixed it - the fairness-warning mechanism did exactly its job (flagging the run as unrepresentative), it just took checking *what* the "something else" actually was rather than assuming it was inherent to the model.

### Results after all five fixes (10 reps each, `gpu` mode, RTX 5080)

| Model | KV cache (theoretical) | Real disk payload | Ingest error | Swap error (original) | Swap error (fixes 1-3) | Swap error (all 5 fixes) |
|---|---|---|---|---|---|---|
| `qwen2.5:3b` | 36.0 KB/token | 5.7 B/token | 3.8% mean | 75.1% mean | 50.8% mean | **21.4% mean** |
| `qwen2.5-coder:14b` | 192.0 KB/token | 5.7 B/token | 3.9% mean | 206.3% mean | 25.2% mean | **15.7% mean** |
| `qwen3-coder:30b` | 48.0 KB/token | 5.7 B/token | (pending) | 77-86% | (contaminated run, see above) | (pending) |

**Honest answer to "can this be predicted perfectly, or is some error unavoidable, on any hardware?"**: meaningfully improvable, twice over now, not reducible to zero. Five real, root-caused fixes - never a fudge factor - took the 14B case from catastrophically wrong (206% error, overshooting 2-3x) to reasonably close (15.7% mean), and roughly halved the 3B case's error a second time (50.8% -> 21.4%) after the first three fixes already roughly halved it once. A residual gap still remains. The most defensible remaining explanation is genuine measurement noise plus whatever Ollama does internally when attaching a large `context` array that isn't exposed through any reported timing field - not something a black-box profiler against Ollama's public API can fully close without either instrumenting Ollama itself or moving to `llama-server` directly (see "Scope" above). **Full precision isn't achievable against a closed scheduler with only wall-clock and self-reported timing to measure against; getting most of the way there - cutting worst-case error roughly 10x across two rounds of fixes this session - is, and the fix is always to keep asking "what is this residual actually measuring" rather than accepting it as noise.**

## Help make this better for everyone

This tool's reuse-window defaults (the 30s delay) and its crossover formula are calibrated against one machine (one GPU, one disk, a handful of models) so far. If you run this on different hardware: please consider opening an issue with your `--output` JSON attached and a note on your GPU/CPU/disk. Shared data will be used only to improve this project's defaults and the crossover formula for everyone - not collected automatically, and not used for anything else.

## License

MIT
