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

## Scope

Deliberately Ollama-only for now - it measures Ollama's actual behavior, not a hypothetical. Genuine KV-cache-to-disk swap only exists at the `llama.cpp` level directly (`llama-server`'s session/prompt-cache files), which this tool does not drive - a real "should we swap" verdict against that backend would need a second, llama.cpp-speaking backend, deliberately left as a documented extension point rather than built speculatively here.

## License

MIT
