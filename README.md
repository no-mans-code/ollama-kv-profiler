# ollama-kv-profiler

Profiles a local Ollama server to decide, per `(model, context size, GPU/CPU split)`, whether reusing Ollama's short-lived generation context beats a cold call that reprocesses the same text from scratch - and for how long that benefit actually lasts.

## Why this exists

Ollama's public API has no real KV-cache-to-disk path - its scheduler unloads whole models rather than tiering the cache, confirmed both by community reporting and by direct measurement in a sibling project ([docuzent](https://github.com/no-mans-code/docuzent)'s `Blueprint.md`). What it *does* have is a short-lived internal reuse window: pass its `context` token array back on a follow-up call and, if nothing has evicted it yet, the follow-up skips reprocessing the shared prefix. This tool measures exactly how short-lived that window is, across model sizes and GPU/CPU splits, rather than assuming one answer applies everywhere.

## What it actually controls

- **GPU/CPU split**: Ollama's `num_gpu` generate option (`999` for "force full GPU", `0` for "force full CPU") - changing it forces a runner reload at that split, so the same model can be tested fully-VRAM-resident and fully-RAM-resident without needing different-sized models to force the difference. `/api/ps`'s `size_vram`/`size` is read back as ground truth, since a requested split isn't always honored exactly.
- **Context size**: Ollama defaults to a 4096-token runtime window regardless of a model's real maximum unless `num_ctx` is passed explicitly - this tool always passes it, sized as a fraction of the model's true max (from `/api/show`'s `model_info`).
- **Token count**: filler text is sized via an empirically calibrated chars-per-token ratio (measured per model with a real call), not a guessed constant.

## Usage

```bash
cargo run -- \
  --models devstral-small-2:24b,qwen3-coder:30b \
  --gpu-modes gpu,cpu \
  --context-fractions 0.25,0.75 \
  --reps 3 \
  --delay-secs 30 \
  --output results.json
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

## Help make this better for everyone

This tool's reuse-window defaults (the 30s delay) and its crossover formula are calibrated against one machine (one GPU, one disk, a handful of models) so far. If you run this on different hardware: please consider opening an issue with your `--output` JSON attached and a note on your GPU/CPU/disk. Shared data will be used only to improve this project's defaults and the crossover formula for everyone - not collected automatically, and not used for anything else.

## License

MIT
