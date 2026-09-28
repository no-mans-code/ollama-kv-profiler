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
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ArchitectureInfo {
    pub num_layers: u32,
    /// The *key/value* head count, not the (often larger) query head count
    /// grouped-query attention models report separately.
    /// Not always a whole number: some architectures (Gemma's interleaved
    /// local/global attention) vary this per layer, and this is the
    /// average across layers - `num_layers * this` still reproduces the
    /// true total exactly.
    pub num_kv_heads: f64,
    pub head_dim: f64,
}

impl Client {
    pub fn new(host: impl Into<String>) -> Self {
        Self { host: host.into() }
    }

    /// POSTs a JSON body and returns the parsed JSON response. On a
    /// non-2xx status, surfaces Ollama's own `{"error": "..."}` body
    /// rather than just the HTTP status code - the status code alone
    /// (e.g. "status code 500") gives no clue what actually went wrong,
    /// which cost real time during this project's own research (had to
    /// manually `curl` the same request outside the tool to discover why
    /// a model failed to load). See issue #5.
    fn post_json<T: for<'de> Deserialize<'de>>(&self, path: &str, body: impl Serialize) -> Result<T> {
        let url = format!("{}{path}", self.host.trim_end_matches('/'));
        match ureq::post(&url).send_json(body) {
            Ok(resp) => resp.into_json().with_context(|| format!("failed to parse response from {url}")),
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_else(|_| "<no body>".to_string());
                bail!("{url} returned HTTP {code}: {body}")
            }
            Err(e) => Err(e).with_context(|| format!("request to {url} failed - is `ollama serve` running?")),
        }
    }

    /// Same as [`Self::post_json`] but for a plain GET (no request body).
    fn get_json<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T> {
        let url = format!("{}{path}", self.host.trim_end_matches('/'));
        match ureq::get(&url).call() {
            Ok(resp) => resp.into_json().with_context(|| format!("failed to parse response from {url}")),
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_else(|_| "<no body>".to_string());
                bail!("{url} returned HTTP {code}: {body}")
            }
            Err(e) => Err(e).with_context(|| format!("request to {url} failed - is `ollama serve` running?")),
        }
    }

    /// The model's real maximum context window, from `/api/show`'s
    /// `model_info["<family>.context_length"]` (found by suffix - the
    /// family prefix varies per architecture).
    pub fn max_context_length(&self, model: &str) -> Result<u32> {
        let resp: Value = self.post_json("/api/show", serde_json::json!({ "model": model }))?;
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
        let resp: Value = self.post_json("/api/show", serde_json::json!({ "model": model }))?;
        let model_info = resp
            .get("model_info")
            .and_then(|v| v.as_object())
            .context("no model_info object in show response")?;

        let find_u64 = |suffix: &str| -> Option<u64> {
            model_info.iter().find(|(k, _)| k.ends_with(suffix)).and_then(|(_, v)| v.as_u64())
        };
        // Some architectures (Gemma's interleaved local/global attention is
        // the real example that forced this) report head counts as a
        // per-layer array, not one scalar for the whole model - e.g.
        // gemma4.attention.head_count_kv = [8,8,8,8,8,2,...]. Averaging the
        // array and multiplying by num_layers elsewhere reproduces the true
        // total exactly (num_layers * average = sum), so this is not an
        // approximation as long as the array's length matches block_count -
        // checked below rather than assumed.
        let find_avg_u64 = |suffix: &str| -> Option<f64> {
            let v = model_info.iter().find(|(k, _)| k.ends_with(suffix)).map(|(_, v)| v)?;
            if let Some(n) = v.as_u64() {
                return Some(n as f64);
            }
            let arr = v.as_array()?;
            if arr.is_empty() {
                return None;
            }
            let sum: u64 = arr.iter().filter_map(|x| x.as_u64()).sum();
            Some(sum as f64 / arr.len() as f64)
        };

        let num_layers = find_u64(".block_count").with_context(|| format!("no *.block_count for `{model}`"))?;
        let embedding_length = find_avg_u64(".embedding_length").with_context(|| format!("no *.embedding_length for `{model}`"))?;
        let head_count = find_avg_u64(".attention.head_count").with_context(|| format!("no *.attention.head_count for `{model}`"))?;
        // Grouped-query attention models report a smaller KV head count
        // separately - that's what actually sets KV-cache size, not the
        // (larger) query head count. Falls back to head_count for
        // architectures that don't use GQA and so don't report it.
        let num_kv_heads = find_avg_u64(".attention.head_count_kv").unwrap_or(head_count);
        let head_dim = embedding_length / head_count.max(1.0);

        Ok(ArchitectureInfo {
            num_layers: num_layers as u32,
            num_kv_heads,
            head_dim,
        })
    }

    /// Ground truth for how much of a currently-loaded model actually sits
    /// in VRAM, from `/api/ps` - more reliable than trusting a requested
    /// `num_gpu` blindly, since not every split is honored exactly.
    pub fn vram_fraction(&self, model: &str) -> Result<Option<f64>> {
        let resp: PsResponse = self.get_json("/api/ps")?;
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
        let resp: GenerateResponse = self.post_json("/api/generate", req)?;
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
