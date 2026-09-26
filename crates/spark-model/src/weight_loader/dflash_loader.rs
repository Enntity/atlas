// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter weight loader.
//!
//! Loads `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`-style drafter checkpoints into
//! the typed [`DflashWeights`] structure consumed by
//! [`crate::layers::BlockDiffusionDraftHead`]. The drafter is a small
//! Qwen3-architecture transformer (8 layers, hidden=2048, GQA 32:4) with
//! these distinctive parts vs. a vanilla Qwen3:
//!
//!  * `model.fc` — `[len(target_layer_ids) * target_hidden, draft_hidden]`
//!    BF16 projection that maps the stack of captured target hidden states
//!    into the drafter's input space.
//!  * `model.hidden_norm` — RMSNorm applied to the projected target context
//!    before mixing with token embeddings.
//!  * `lm_head` — drafter ships its own (NOT tied to target's), so
//!    `tie_word_embeddings=false`.
//!  * Optional `d2t` — draft-vocab → target-vocab id remap (absent when
//!    drafter shares vocab with target, as in Qwen3.6-35B-A3B-DFlash where
//!    both = 248320).
//!  * Special `mask_token_id` (`248070` for Qwen3.6-DFlash) used for the γ
//!    "to-be-predicted" positions in block diffusion.
//!
//! Under TP the drafter is **not sharded** — it's small (~1–2 GB BF16),
//! every rank loads the full set. Mirrors the existing MTP-under-TP pattern
//! (`MTP loads ALL experts on every rank — no EP all_reduce needed`).

use anyhow::{Context, Result};
use serde::Deserialize;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense_auto};

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    /// HF architecture identities; absent on older generic DFlash configs.
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Lightning DSpark's required anchor/sampling declarations.
    #[serde(default)]
    pub dspark_bonus_anchor: Option<bool>,
    #[serde(default)]
    pub sample_from_anchor: Option<bool>,
    #[serde(default)]
    pub attention_sink_bias: Option<bool>,
    #[serde(default)]
    pub dspark_markov_rank: Option<usize>,
    #[serde(default)]
    pub target_layer_ids: Option<Vec<usize>>,
    #[serde(
        default,
        alias = "dspark_confidence_head",
        alias = "enable_confidence_head"
    )]
    pub confidence_head: Option<bool>,
    #[serde(default, alias = "dspark_adaptive", alias = "adaptive_verification")]
    pub adaptive: Option<bool>,
    #[serde(default)]
    pub quantization_config: Option<DflashQuantizationConfig>,
    /// Block size γ. Qwen3.6-DFlash ships `block_size: 16`.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Drafter base RoPE θ. Defaults to 10M (matches Qwen3.6-DFlash).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// HF-style `rope_scaling` block. `None` ⇒ plain RoPE (the v2 2026-04-27
    /// Qwen3.6-DFlash drafter ships `rope_scaling: null`). When present and
    /// `rope_type == "yarn"`, the drafter's YaRN parameters are used to
    /// build the inv_freq table at construction time.
    #[serde(default)]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// DSpark Markov rank. `None` / 0 = DFlash-only (no sequential fixup).
    #[serde(default)]
    pub markov_rank: Option<usize>,
    /// Drafter RMSNorm epsilon (Qwen3 default 1e-6; DFlash2 ships 1e-5).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// transformers-5 `rope_parameters` block. Folded into `rope_theta` /
    /// `rope_scaling` by [`parse_dflash_config`] when those are absent.
    #[serde(default)]
    pub rope_parameters: Option<DflashRopeParameters>,
    /// HF `is_causal`. Takes precedence over `dflash_config.causal`
    /// (DFlash2 ships `false`: bidirectional γ-block attention).
    #[serde(default)]
    pub is_causal: Option<bool>,
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

/// HF architecture identity of the DFlash2 drafter (grouped dynamic conv +
/// candidate selector on top of the DFlash backbone).
pub const DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";

/// transformers-5 `rope_parameters`: `rope_theta` plus the scaling fields
/// that older configs carried in `rope_scaling`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(flatten)]
    pub scaling: DflashRopeScaling,
}

impl DflashConfig {
    /// DFlash2 checkpoint: declared architecture or selector metadata.
    pub fn is_dflash2(&self) -> bool {
        self.architectures
            .as_ref()
            .is_some_and(|a| a.iter().any(|a| a == DFLASH2_ARCHITECTURE))
            || self
                .dflash_config
                .as_ref()
                .is_some_and(|c| c.selector_rank.is_some())
    }

    /// Causal γ-block attention: HF `is_causal` wins over
    /// `dflash_config.causal`; both absent ⇒ bidirectional.
    pub fn query_causal(&self) -> bool {
        self.is_causal
            .or_else(|| self.dflash_config.as_ref().and_then(|c| c.causal))
            .unwrap_or(false)
    }

    /// Checkpoint sliding window: `dflash_config.swa_window_size`, else the
    /// HF `sliding_window` when `use_sliding_window` is set or every
    /// `layer_types` entry is `sliding_attention`.
    pub fn sliding_window(&self) -> Option<usize> {
        let sub = self.dflash_config.as_ref();
        if let Some(window) = sub.and_then(|c| c.swa_window_size) {
            return Some(window);
        }
        let all_sliding = self.layer_types.as_ref().is_some_and(|types| {
            !types.is_empty() && types.iter().all(|t| t == "sliding_attention")
        });
        let sliding = self.use_sliding_window == Some(true)
            || sub.and_then(|c| c.use_swa) == Some(true)
            || all_sliding;
        if sliding { self.sliding_window } else { None }
    }
}

/// Subset of HF `rope_scaling` block consumed by Atlas. Mirrors the field
/// names in `transformers`' Qwen3 config so `serde_json::from_str` works
/// directly on the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    /// Currently only `"yarn"` is recognised; anything else falls back to
    /// plain RoPE with a warning logged at construction time.
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<f32>,
}

/// Minimal nested quantization metadata needed for runtime admission.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashQuantizationConfig {
    #[serde(default)]
    pub kv_cache_quant_algo: Option<String>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash. Order matters:
    /// shallow-to-deep concatenation is what `fc` expects.
    pub target_layer_ids: Vec<usize>,
    /// When `Some(true)`, every drafter layer uses causal γ-block attention.
    /// Lightning DSpark sets this; Qwen-DFlash leaves it unset (bidirectional).
    #[serde(default)]
    pub causal: Option<bool>,
    /// Force sliding-window attention on every drafter layer.
    #[serde(default)]
    pub use_swa: Option<bool>,
    /// Sliding-window size when `use_swa` or `layer_types` request SWA.
    #[serde(default)]
    pub swa_window_size: Option<usize>,
    /// When `Some(true)`, each layer must ship `self_attn.attention_sink_bias`.
    #[serde(default)]
    pub attention_sink_bias: Option<bool>,
    #[serde(default)]
    pub sample_from_anchor: Option<bool>,
    #[serde(
        default,
        alias = "dspark_confidence_head",
        alias = "enable_confidence_head"
    )]
    pub confidence_head: Option<bool>,
    #[serde(default, alias = "dspark_adaptive", alias = "adaptive_verification")]
    pub adaptive: Option<bool>,
    /// Nested block size (DFlash2). Used when the root `block_size` is absent.
    #[serde(default)]
    pub block_size: Option<usize>,
    /// DFlash2 grouped dynamic conv: channels per group.
    #[serde(default)]
    pub conv_group_size: Option<usize>,
    /// DFlash2 grouped dynamic conv: taps (in-block causal kernel width).
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    /// DFlash2 candidate-selector codebook rank.
    #[serde(default)]
    pub selector_rank: Option<usize>,
    /// DFlash2 candidates per draft position.
    #[serde(default)]
    pub selector_top_k: Option<usize>,
    /// DFlash2 optional scalings; only the identity values are supported.
    #[serde(default)]
    pub input_embedding_scale: Option<f32>,
    #[serde(default)]
    pub output_multiplier: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
}

/// Raw weight bundle for the DFlash drafter, post-load.
///
/// Verified against `z-lab/Qwen3.6-35B-A3B-DFlash` (commit 42d3b34, May 2026):
/// the checkpoint ships 91 BF16 tensors — `fc.weight`, `hidden_norm.weight`,
/// `norm.weight`, plus 11 weights per drafter layer × 8 layers. **No
/// `embed_tokens` or `lm_head` are in the checkpoint** — the drafter shares
/// the target's embedding and LM head at construction time. This matches the
/// vLLM PR #40898 flow: when those keys are absent, vLLM's `AutoWeightsLoader`
/// adds them to `skip_substrs`, leaving the runtime to slot in the target's
/// pointers.
#[allow(dead_code)]
pub struct DflashWeights {
    pub config: DflashConfig,

    /// `[draft_hidden, len(target_layer_ids) * target_hidden]`.
    /// Qwen3.6-35B-A3B-DFlash: `[2048, 10240]`.
    pub fc: DenseWeight,
    /// `[draft_hidden]` — RMSNorm applied to the projected target context
    /// before mixing with token embeddings.
    pub hidden_norm: DenseWeight,
    /// `[draft_hidden]` — final RMSNorm before LM head.
    pub norm: DenseWeight,

    pub layers: Vec<DflashLayerWeights>,
    pub draft_id_to_target_id: Option<Vec<i64>>,
    /// DSpark Markov embedding `[vocab, rank]` BF16. None for DFlash-only.
    pub markov_w1: Option<DenseWeight>,
    /// DSpark Markov projection `[vocab, rank]` BF16 (NVFP4 dequanted).
    pub markov_w2: Option<DenseWeight>,
    pub markov_rank: usize,
    /// Optional drafter-owned embed. None → share the target's.
    pub embed_tokens: Option<DenseWeight>,
    /// DFlash2 grouped conv + candidate selector. None for DFlash v1 / DSpark.
    pub dflash2: Option<Dflash2Weights>,
}

/// One DFlash2 grouped dynamic conv (`attention_conv` / `mlp_conv`).
pub struct Dflash2ConvWeights {
    /// `[2 sides, taps, hidden]` BF16 static coefficients.
    pub base_kernel: DenseWeight,
    /// `[2 * taps * groups, hidden]` BF16 per-row coefficient deltas.
    pub kernel_projection: DenseWeight,
}

/// DFlash2 additions on top of the DFlash backbone.
pub struct Dflash2Weights {
    /// Per drafter layer: `(attention_conv, mlp_conv)`.
    pub layers: Vec<(Dflash2ConvWeights, Dflash2ConvWeights)>,
    /// `candidate_selector.hidden_projection` `[rank, hidden]` BF16.
    pub hidden_projection: DenseWeight,
    /// `candidate_selector.predecessor_codebook` `[vocab, rank]` BF16.
    pub predecessor_codebook: DenseWeight,
    /// `candidate_selector.successor_codebook` `[vocab, rank]` BF16.
    pub successor_codebook: DenseWeight,
    pub conv_group_size: usize,
    pub conv_taps: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
}

/// Per-drafter-layer raw weights (BF16). Same shape across all 8 layers.
#[allow(dead_code)]
pub struct DflashLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    /// Per-q-head FlashAttention sink logits `[num_q_heads]` BF16.
    /// None when the checkpoint has no sink tensor (Qwen-DFlash).
    pub attention_sink_bias: Option<DenseWeight>,
}

/// Probe a [`WeightStore`] for the presence of DFlash drafter weights.
/// Returns true if the store contains the unique `fc.weight` tensor that
/// DFlash drafters ship — a lightweight detection that doesn't load any
/// data. Both bare-key and `model.`-prefixed layouts are accepted; the
/// canonical `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash` checkpoints ship the
/// bare layout (verified against commit 42d3b34, May 2026).
pub fn store_has_dflash_weights(store: &WeightStore) -> bool {
    store.contains("fc.weight") || store.contains("model.fc.weight")
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Parsing DFlash drafter config.json")?;
    let mut config: DflashConfig =
        serde_json::from_value(raw.clone()).context("Parsing DFlash drafter config.json")?;
    // transformers-5 checkpoints may carry rope settings ONLY in
    // `rope_parameters`, and DFlash2 nests `block_size` in `dflash_config`.
    // Root keys, when present, keep their historical precedence.
    if let Some(params) = config.rope_parameters.clone() {
        if raw.get("rope_theta").is_none()
            && let Some(theta) = params.rope_theta
        {
            config.rope_theta = theta;
        }
        let scaled = params
            .scaling
            .rope_type
            .as_deref()
            .is_some_and(|t| t != "default");
        if config.rope_scaling.is_none() && scaled {
            config.rope_scaling = Some(params.scaling);
        }
    }
    if raw.get("block_size").is_none()
        && let Some(block_size) = config.dflash_config.as_ref().and_then(|c| c.block_size)
    {
        config.block_size = block_size;
    }
    Ok(config)
}

/// Load DFlash drafter weights from a separate [`WeightStore`] pointing at
/// the drafter checkpoint.
///
/// The drafter ships its weights at the **root** of the safetensors file
/// (no `model.` prefix), in the same naming convention as a vanilla Qwen3
/// transformer minus `embed_tokens` and `lm_head`. Atlas's runtime fills
/// those two from the *target* model's embedding / LM head at construction
/// time — exactly mirroring vLLM's "absent in checkpoint → skip_substrs →
/// share with parent" flow.
///
/// The probed key list (verified against `z-lab/Qwen3.6-35B-A3B-DFlash`):
///
/// ```text
///   fc.weight                                              [H, 5*H_target]
///   hidden_norm.weight                                     [H]
///   norm.weight                                            [H]
///   layers.{0..L-1}.input_layernorm.weight                 [H]
///   layers.{0..L-1}.post_attention_layernorm.weight        [H]
///   layers.{0..L-1}.self_attn.q_proj.weight                [Q*Hd, H]
///   layers.{0..L-1}.self_attn.k_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.v_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.o_proj.weight                [H, Q*Hd]
///   layers.{0..L-1}.self_attn.q_norm.weight                [Hd]
///   layers.{0..L-1}.self_attn.k_norm.weight                [Hd]
///   layers.{0..L-1}.mlp.gate_proj.weight                   [I, H]
///   layers.{0..L-1}.mlp.up_proj.weight                     [I, H]
///   layers.{0..L-1}.mlp.down_proj.weight                   [H, I]
/// ```
///
/// where `H=2048`, `H_target=2048`, `Q=32`, `Kv=4`, `Hd=128`, `I=6144`,
/// `L=8` for Qwen3.6-35B-A3B-DFlash.
///
/// Under TP the drafter is replicated, not sharded — `tp_size>1` produces
/// the same per-rank result as `tp_size=1`. Memory cost: ~948 MB BF16
/// per rank, trivially below the 119 GB GB10 budget.
pub fn load_dflash_weights(
    drafter_store: &WeightStore,
    drafter_config: &DflashConfig,
    gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DflashWeights>> {
    if !store_has_dflash_weights(drafter_store) {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    }

    let prefix = if drafter_store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };

    // dense_auto: BF16 as-is, packed NVFP4 (Lightning DSpark MLP/fc/markov_w2)
    // dequanted once at load.
    let fc = dense_auto(drafter_store, &format!("{prefix}fc.weight"), gpu)
        .context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense_auto(drafter_store, &format!("{prefix}hidden_norm.weight"), gpu)
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense_auto(drafter_store, &format!("{prefix}norm.weight"), gpu)
        .context("DFlash drafter: load norm.weight")?;

    let require_sinks = drafter_config
        .dflash_config
        .as_ref()
        .and_then(|c| c.attention_sink_bias)
        == Some(true);
    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    let mut sink_layers = 0usize;
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        let sink_name = format!("{lp}.self_attn.attention_sink_bias");
        let attention_sink_bias = if drafter_store.contains(&sink_name) {
            sink_layers += 1;
            Some(dense_auto(drafter_store, &sink_name, gpu)?)
        } else if require_sinks {
            anyhow::bail!(
                "dflash_config.attention_sink_bias=true but {sink_name} is missing from the checkpoint"
            );
        } else {
            None
        };
        let layer = DflashLayerWeights {
            input_layernorm: dense_auto(
                drafter_store,
                &format!("{lp}.input_layernorm.weight"),
                gpu,
            )?,
            post_attention_layernorm: dense_auto(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
                gpu,
            )?,
            q_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.q_proj.weight"), gpu)?,
            k_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.k_proj.weight"), gpu)?,
            v_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.v_proj.weight"), gpu)?,
            o_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.o_proj.weight"), gpu)?,
            q_norm: dense_auto(drafter_store, &format!("{lp}.self_attn.q_norm.weight"), gpu)?,
            k_norm: dense_auto(drafter_store, &format!("{lp}.self_attn.k_norm.weight"), gpu)?,
            gate_proj: dense_auto(drafter_store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
            up_proj: dense_auto(drafter_store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
            down_proj: dense_auto(drafter_store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
            attention_sink_bias,
        };
        layers.push(layer);
    }

    let draft_id_to_target_id = if drafter_store.contains(&format!("{prefix}d2t"))
        || drafter_store.contains(&format!("{prefix}draft_id_to_target_id"))
    {
        tracing::warn!(
            "DFlash drafter has draft-id→target-id mapping; remapping path is not yet wired (Phase 2.5 follow-up)"
        );
        Some(Vec::new())
    } else {
        None
    };

    let markov_w1_name = format!("{prefix}markov_head.markov_w1.weight");
    let markov_w2_name = format!("{prefix}markov_head.markov_w2.weight");
    let (markov_w1, markov_w2, markov_rank) =
        if drafter_store.contains(&markov_w1_name) && drafter_store.contains(&markov_w2_name) {
            let rank = drafter_config.markov_rank.unwrap_or(512);
            tracing::info!("DSpark Markov head: rank={rank} (w1 BF16, w2 auto/NVFP4)");
            (
                Some(dense_auto(drafter_store, &markov_w1_name, gpu)?),
                Some(dense_auto(drafter_store, &markov_w2_name, gpu)?),
                rank,
            )
        } else {
            (None, None, 0)
        };

    let embed_name = format!("{prefix}embed_tokens.weight");
    let embed_tokens = if drafter_store.contains(&embed_name) {
        Some(dense_auto(drafter_store, &embed_name, gpu)?)
    } else {
        None
    };

    let dflash2 = if drafter_config.is_dflash2() {
        Some(load_dflash2_weights(
            drafter_store,
            drafter_config,
            prefix,
            gpu,
        )?)
    } else {
        None
    };

    let sub = drafter_config.dflash_config.as_ref();
    if let Ok(fc_meta) = drafter_store.get(&format!("{prefix}fc.weight"))
        && fc_meta.dtype == spark_runtime::weights::WeightDtype::UInt8
        && fc_meta.shape.len() == 2
    {
        tracing::info!(
            "DFlash fc NVFP4 unpack: on-disk {:?} U8 → logical [{}, {}] BF16",
            fc_meta.shape,
            fc_meta.shape[0],
            fc_meta.shape[1] * 2
        );
    }
    tracing::info!(
        "DFlash/DSpark drafter loaded: {} layers, hidden={}, vocab={}, γ={}, markov_rank={}, own_embed={}, dflash2={}, causal={}, swa={:?}, sinks={}/{}, target_layers={:?}",
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.block_size,
        markov_rank,
        embed_tokens.is_some(),
        dflash2.is_some(),
        drafter_config.query_causal(),
        drafter_config.sliding_window(),
        sink_layers,
        layers.len(),
        sub.map(|c| c.target_layer_ids.as_slice()).unwrap_or(&[]),
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        markov_w1,
        markov_w2,
        markov_rank,
        embed_tokens,
        dflash2,
    }))
}

/// Load the DFlash2 grouped-conv and candidate-selector tensors, checking
/// every shape against the config (a transposed or mis-sized tensor would
/// otherwise draft silent garbage).
fn load_dflash2_weights(
    store: &WeightStore,
    config: &DflashConfig,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<Dflash2Weights> {
    let sub = config
        .dflash_config
        .as_ref()
        .context("DFlash2 drafter config.json is missing `dflash_config`")?;
    let required = |value: Option<usize>, field: &str| {
        value.with_context(|| format!("DFlash2 drafter config is missing `dflash_config.{field}`"))
    };
    let group_size = required(sub.conv_group_size, "conv_group_size")?;
    let taps = required(sub.conv_kernel_size, "conv_kernel_size")?;
    let rank = required(sub.selector_rank, "selector_rank")?;
    let top_k = required(sub.selector_top_k, "selector_top_k")?;
    for (value, field) in [
        (sub.input_embedding_scale, "input_embedding_scale"),
        (sub.output_multiplier, "output_multiplier"),
    ] {
        anyhow::ensure!(
            value.is_none_or(|v| v == 1.0),
            "DFlash2 `dflash_config.{field}`={value:?} is not supported (only 1.0)"
        );
    }
    anyhow::ensure!(
        sub.final_logit_softcapping.is_none_or(|v| v <= 0.0),
        "DFlash2 `dflash_config.final_logit_softcapping` is not supported"
    );
    let hidden = config.hidden_size;
    anyhow::ensure!(
        group_size > 0 && hidden % group_size == 0,
        "DFlash2 conv_group_size={group_size} must divide hidden_size={hidden}"
    );
    let groups = hidden / group_size;
    let load = |name: String, shape: &[usize]| -> Result<DenseWeight> {
        let meta = store
            .get(&name)
            .with_context(|| format!("DFlash2 drafter: missing {name}"))?;
        anyhow::ensure!(
            meta.shape == shape,
            "DFlash2 drafter: {name} has shape {:?}, expected {shape:?}",
            meta.shape
        );
        dense_auto(store, &name, gpu).with_context(|| format!("DFlash2 drafter: load {name}"))
    };
    let conv = |layer: usize, site: &str| -> Result<Dflash2ConvWeights> {
        let p = format!("{prefix}layers.{layer}.{site}");
        Ok(Dflash2ConvWeights {
            base_kernel: load(format!("{p}.base_kernel"), &[2, taps, hidden])?,
            kernel_projection: load(
                format!("{p}.kernel_projection.weight"),
                &[2 * taps * groups, hidden],
            )?,
        })
    };
    let layers = (0..config.num_hidden_layers)
        .map(|i| Ok((conv(i, "attention_conv")?, conv(i, "mlp_conv")?)))
        .collect::<Result<Vec<_>>>()?;
    let selector = format!("{prefix}candidate_selector");
    let vocab = config.vocab_size;
    tracing::info!(
        "DFlash2 drafter: grouped conv (groups={groups}, taps={taps}) × {} layers, candidate selector (rank={rank}, top_k={top_k})",
        layers.len()
    );
    Ok(Dflash2Weights {
        layers,
        hidden_projection: load(
            format!("{selector}.hidden_projection.weight"),
            &[rank, hidden],
        )?,
        predecessor_codebook: load(format!("{selector}.predecessor_codebook"), &[vocab, rank])?,
        successor_codebook: load(format!("{selector}.successor_codebook"), &[vocab, rank])?,
        conv_group_size: group_size,
        conv_taps: taps,
        selector_rank: rank,
        selector_top_k: top_k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test the DFlash drafter `config.json` parser against the live
    /// `z-lab/Qwen3.6-35B-A3B-DFlash` checkpoint downloaded into the user's
    /// HF cache. Skipped when the cache directory isn't populated — keeps
    /// CI hermetic. Asserts the locked drafter dimensions: 8 layers,
    /// hidden=2048, vocab=248320, γ=16, mask=248070, layer_ids=[1,10,19,28,37].
    #[test]
    fn parse_qwen3_6_35b_dflash_config() {
        const SNAP: &str = "/workspace/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719/config.json";
        let json = match std::fs::read_to_string(SNAP) {
            Ok(s) => s,
            Err(_) => {
                tracing::warn!("Skipping: drafter snapshot not in cache");
                return;
            }
        };
        let config = parse_dflash_config(&json).expect("parse drafter config");
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 248320);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.block_size, 16);
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 248070);
        assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
    }

    #[test]
    fn parse_lightning_dspark_config() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_data/lightning_dspark_config.json"
        ));
        let config = parse_dflash_config(json).expect("parse Lightning DSpark config");
        assert_eq!(config.num_hidden_layers, 6);
        assert_eq!(config.hidden_size, 2688);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 131072);
        assert_eq!(config.block_size, 8);
        assert_eq!(config.markov_rank, Some(512));
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 990);
        assert_eq!(sub.target_layer_ids, vec![1, 5, 19, 29, 41, 51]);
        assert_eq!(sub.causal, Some(true));
        assert_eq!(sub.use_swa, Some(true));
        assert_eq!(sub.swa_window_size, Some(1024));
        assert_eq!(sub.attention_sink_bias, Some(true));
    }

    /// `incoai/GLM-5.3-Flash-DFlash2` config.json (transformers 5: rope_theta
    /// only inside `rope_parameters`, block_size only inside `dflash_config`).
    const DFLASH2_CONFIG_JSON: &str = r#"{
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 154856,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      5,
      14,
      24,
      33,
      42
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": [
    154820,
    154827,
    154829
  ],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "is_causal": false,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 1048576,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 45,
  "pad_token_id": 154820,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {
    "rope_theta": 10000.0,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.7.0",
  "use_cache": false,
  "use_sliding_window": true,
  "vocab_size": 154880
}"#;

    #[test]
    fn parse_dflash2_config() {
        let config = parse_dflash_config(DFLASH2_CONFIG_JSON).expect("parse DFlash2 config");
        assert!(config.is_dflash2());
        assert_eq!(config.num_hidden_layers, 5);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.vocab_size, 154880);
        assert_eq!(config.block_size, 8);
        assert_eq!(config.rope_theta, 10_000.0);
        assert!(
            config.rope_scaling.is_none(),
            "rope_type=default is plain RoPE"
        );
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert!(!config.query_causal());
        assert_eq!(config.sliding_window(), Some(2048));
        let sub = config.dflash_config.as_ref().expect("dflash_config");
        assert_eq!(sub.mask_token_id, 154856);
        assert_eq!(sub.target_layer_ids, vec![5, 14, 24, 33, 42]);
        assert_eq!(sub.conv_group_size, Some(16));
        assert_eq!(sub.conv_kernel_size, Some(2));
        assert_eq!(sub.selector_rank, Some(256));
        assert_eq!(sub.selector_top_k, Some(16));
    }

    /// Root keys keep precedence over the transformers-5 / nested fallbacks,
    /// and a v1-style config without them keeps the historical defaults.
    #[test]
    fn dflash_config_fallbacks_keep_v1_defaults() {
        let base = serde_json::json!({
            "hidden_size": 2048, "num_hidden_layers": 8, "intermediate_size": 6144,
            "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
            "vocab_size": 248320,
            "layer_types": ["full_attention"], "use_sliding_window": false, "sliding_window": 4096,
            "dflash_config": {"mask_token_id": 248070, "target_layer_ids": [1, 10]}
        });
        let v1 = parse_dflash_config(&base.to_string()).unwrap();
        assert!(!v1.is_dflash2());
        assert_eq!(v1.block_size, 16);
        assert_eq!(v1.rope_theta, 10_000_000.0);
        assert_eq!(v1.rms_norm_eps, 1e-6);
        assert!(!v1.query_causal());
        assert_eq!(v1.sliding_window(), None);

        let mut root = base.clone();
        root["block_size"] = 16.into();
        root["rope_theta"] = 5e6.into();
        root["is_causal"] = true.into();
        root["rope_parameters"] = serde_json::json!({
            "rope_theta": 1e4, "rope_type": "yarn", "factor": 4.0
        });
        root["dflash_config"]["block_size"] = 8.into();
        root["dflash_config"]["causal"] = false.into();
        let root = parse_dflash_config(&root.to_string()).unwrap();
        assert_eq!(root.block_size, 16);
        assert_eq!(root.rope_theta, 5e6);
        assert!(
            root.query_causal(),
            "is_causal wins over dflash_config.causal"
        );
        let scaling = root
            .rope_scaling
            .expect("yarn rope_parameters fold into rope_scaling");
        assert_eq!(scaling.rope_type.as_deref(), Some("yarn"));
        assert_eq!(scaling.factor, Some(4.0));
    }
}
