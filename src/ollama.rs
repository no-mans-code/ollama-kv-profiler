//! A minimal, standalone Ollama client - deliberately not shared with any
//! other project's own client, so this profiler stays a single, portable
//! binary usable against any local Ollama server.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Every call in this tool only cares about prefill cost, never the
/// generated text - a small, uniform cap keeps the eval (decode) phase
/// fast and bounded everywhere, rather than left open-ended.
pub const DEFAULT_NUM_PREDICT: i32 = 16;

pub struct Client {
    host: String,
}

/// The architecture parameters that determine a model's KV-cache size per
/// token: `2 * num_layers * num_kv_heads * head_dim * dtype_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArchitectureInfo {
    pub num_layers: u32,
    /// The *key/value* head count, not the (often larger) query head count
    /// grouped-query attention models report separately.
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl Client {
    pub fn new(host: impl Into<String>) -> Self {
        Self { host: host.into() }
    }

    /// The model's real maximum context window, from `/api/show`'s
    /// `model_info["<family>.context_length"]` (found by suffix - the
    /// family prefix varies per architecture).
    pub fn max_context_length(&self, model: &str) -> Result<u32> {
        let url = format!("{}/api/show", self.host.trim_end_matches('/'));
        let resp: Value = ureq::post(&url)
            .send_json(serde_json::json!({ "model": model }))
            .with_context(|| format!("ollama show request to {url} failed - is `ollama serve` running?"))?
            .into_json()
            .context("failed to parse ollama show response")?;
        let model_info = resp.get("model_info").context("no model_info in show response")?;
        let context_length = model_info
            .as_object()
            .context("model_info was not an object")?
            .iter()
            .find(|(k, _)| k.ends_with(".context_length"))
            .and_then(|(_, v)| v.as_u64())
            .with_context(|| format!("no *.context_length field for `{model}`"))?;
        Ok(context_length as u32)
    }

    /// The architecture parameters that determine KV-cache size per token -
    /// read from `/api/show`'s `model_info`, not assumed. Field names carry
    /// an architecture-family prefix (`qwen2.*`, `llama.*`, ...), found by
    /// suffix rather than hardcoded per family, same as `max_context_length`.
    pub fn architecture_info(&self, model: &str) -> Result<ArchitectureInfo> {
        let url = format!("{}/api/show", self.host.trim_end_matches('/'));
        let resp: Value = ureq::post(&url)
            .send_json(serde_json::json!({ "model": model }))
            .with_context(|| format!("ollama show request to {url} failed"))?
            .into_json()
            .context("failed to parse ollama show response")?;
        let model_info = resp
            .get("model_info")
            .and_then(|v| v.as_object())
            .context("no model_info object in show response")?;

        let find_u64 = |suffix: &str| -> Option<u64> {
            model_info.iter().find(|(k, _)| k.ends_with(suffix)).and_then(|(_, v)| v.as_u64())
        };

        let num_layers = find_u64(".block_count").with_context(|| format!("no *.block_count for `{model}`"))?;
        let embedding_length = find_u64(".embedding_length").with_context(|| format!("no *.embedding_length for `{model}`"))?;
        let head_count = find_u64(".attention.head_count").with_context(|| format!("no *.attention.head_count for `{model}`"))?;
        // Grouped-query attention models report a smaller KV head count
        // separately - that's what actually sets KV-cache size, not the
        // (larger) query head count. Falls back to head_count for
        // architectures that don't use GQA and so don't report it.
        let num_kv_heads = find_u64(".attention.head_count_kv").unwrap_or(head_count);
        let head_dim = embedding_length / head_count.max(1);

        Ok(ArchitectureInfo {
            num_layers: num_layers as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
        })
    }

    /// Ground truth for how much of a currently-loaded model actually sits
    /// in VRAM, from `/api/ps` - more reliable than trusting a requested
    /// `num_gpu` blindly, since not every split is honored exactly.
    pub fn vram_fraction(&self, model: &str) -> Result<Option<f64>> {
        let url = format!("{}/api/ps", self.host.trim_end_matches('/'));
        let resp: PsResponse = ureq::get(&url)
            .call()
            .with_context(|| format!("ollama ps request to {url} failed"))?
            .into_json()
            .context("failed to parse ollama ps response")?;
        Ok(resp
            .models
            .into_iter()
            .find(|m| m.model == model || m.name == model)
            .map(|m| {
                if m.size == 0 {
                    0.0
                } else {
                    m.size_vram as f64 / m.size as f64
                }
            }))
    }

    pub fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let url = format!("{}/api/generate", self.host.trim_end_matches('/'));
        let resp: GenerateResponse = ureq::post(&url)
            .send_json(req)
            .with_context(|| format!("ollama generate request to {url} failed"))?
            .into_json()
            .context("failed to parse ollama generate response")?;
        if resp.response.is_empty() && resp.prompt_eval_count == 0 {
            bail!("empty response from ollama - model may have failed to load");
        }
        Ok(resp)
    }
}

#[derive(Serialize)]
pub struct GenerateOptions {
    /// `None` omits the field entirely, letting Ollama pick its own
    /// GPU/RAM split - the realistic "spill whatever doesn't fit into RAM"
    /// behavior, as opposed to a forced extreme.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    pub num_ctx: u32,
    /// Caps how many tokens the model generates in its response. This
    /// tool only cares about prefill (`prompt_eval_*`) cost, never the
    /// answer text itself - left unbounded, a model can ramble on for an
    /// open-ended filler-text prompt, inflating wall time and eval_count
    /// unpredictably for a measurement that isn't even about generation.
    pub num_predict: i32,
}

#[derive(Serialize)]
pub struct GenerateRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<&'a [i64]>,
    pub options: GenerateOptions,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct GenerateResponse {
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub context: Vec<i64>,
    #[serde(default)]
    pub prompt_eval_count: u64,
    /// Nanoseconds, as Ollama reports it.
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_duration: u64,
    /// Nanoseconds spent (re)loading the model into the runner - should be
    /// ~0 on every timed rep, since [`crate::bench::profile`] pays this
    /// cost once in an untimed warmup call before any rep starts.
    #[serde(default)]
    pub load_duration: u64,
}

#[derive(Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<PsModel>,
}

#[derive(Deserialize)]
struct PsModel {
    name: String,
    model: String,
    size: u64,
    size_vram: u64,
}
