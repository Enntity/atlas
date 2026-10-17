// SPDX-License-Identifier: AGPL-3.0-only

//! F4 phase 1: DFlash2 draft-only observer for the native GLM-5.3 EXL3
//! target (`ATLAS_GLM_EXL3_DFLASH=1`, TP=2/EP=1).
//!
//! The serial decode is untouched: after every serial token the replicated
//! DFlash2 drafter proposes `k` tokens, which are only scored against the
//! tokens the serial decode subsequently produces (`DFLASH_ACCEPT` lines).
//! Both TP ranks hold the whole drafter, capture the same post-all-reduce
//! hiddens and see the same tokens, so they compute identical drafts with
//! no exchange.
//!
//! Capture contract (vLLM `get_eagle3_aux_layers_from_config` adds 1 to the
//! DFlash `target_layer_ids`; the GLM EAGLE3 overlay then captures after
//! layer `idx` when `idx + 1` is an aux layer): for each 0-indexed target
//! layer `i` in `target_layer_ids` the drafter context row is the mean over
//! the 4 mHC streams of the highway AFTER layer `i` (its FFN `hc_post`), in
//! BF16, concatenated in `target_layer_ids` order -> `[5 * 4096]` per
//! position. On the native EXL3 path every layer finishes with its post
//! combine into `hc_streams` (FP32), so the highway read right after
//! `layer.decode/prefill` returns is the true post-layer state.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::WeightStore;

use super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::layers::dflash_head::{CaptureDescriptor, DsparkStartupExecution, SequenceGeneration};
use crate::layers::{BlockDiffusionDraftHead, DflashProposerState};
use crate::speculative::{DraftProposer, ProposerState};
use crate::traits::SequenceState;

mod decode;
mod head;
mod hooks;
mod observer;
pub(crate) mod scorer;
mod spec;

pub(crate) use spec::EP_CMD_EXL3_DFLASH_SPEC;

pub use head::Exl3ReplicatedVocabHead;
pub use observer::Exl3DflashObserver;
use observer::{ChunkRows, SeqObs, request_label};
use scorer::{AcceptStats, DraftScorer, VerifyParity};

/// Master switch for the observer.
pub const ENV: &str = "ATLAS_GLM_EXL3_DFLASH";
/// Drafter snapshot directory (config.json + model.safetensors).
pub const ENV_DRAFTER: &str = "ATLAS_GLM_EXL3_DFLASH_DRAFTER";
/// Drafts per step (`k`; the drafter block is `k + 1` rows).
pub const ENV_K: &str = "ATLAS_GLM_EXL3_DFLASH_K";
/// Per-request `DFLASH_ACCEPT` cadence in decode steps (0 = end only).
pub const ENV_LOG_EVERY: &str = "ATLAS_GLM_EXL3_DFLASH_LOG_EVERY";
/// `1` logs every step's drafts (`DFLASH_DRAFT`).
pub const ENV_TRACE: &str = "ATLAS_GLM_EXL3_DFLASH_TRACE";
/// Capture contraction (bisection knob): `fp32mean` (default: mean the F32
/// streams, round once) or `bf16mean` (round each stream to BF16, then mean,
/// as vLLM's materialized BF16 post streams do).
pub const ENV_CAPTURE: &str = "ATLAS_GLM_EXL3_DFLASH_CAPTURE";
/// Signed offset added to every capture layer (bisection knob, default 0).
pub const ENV_LAYER_SHIFT: &str = "ATLAS_GLM_EXL3_DFLASH_LAYER_SHIFT";
/// `1`: every LOG_EVERY steps, time the drafter phases (`DFLASH_PROFILE`).
/// Adds synchronizations and one repeated proposal on those steps only.
pub const ENV_PROFILE: &str = "ATLAS_GLM_EXL3_DFLASH_PROFILE";
/// Phase 2 verify parity probe: every N-th draft step (N >= 1) also runs the
/// `k + 1`-row EXL3 verify forward over `[bonus, drafts..]`, restores the
/// target state, and checks each checkable verify row's argmax against the
/// serial token (`DFLASH_VERIFY_PARITY`). Allocates the KDA verify pools.
pub const ENV_VERIFY_PROBE: &str = "ATLAS_GLM_EXL3_DFLASH_VERIFY_PROBE";

fn parse_probe(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.trim().parse().ok()).unwrap_or(0)
}

/// Phase 2 speculative lane (`1`): greedy single-sequence decode steps
/// verify `[bonus, d1..dk]` in one `k + 1`-row target forward and emit the
/// accepted drafts plus the bonus. Numerics-changing (verify rows run at
/// M = k + 1); output is self-consistent with the verify argmax chain, not
/// byte-identical to serial decode.
pub const ENV_SPEC: &str = "ATLAS_GLM_EXL3_DFLASH_SPEC";

/// Rollback test knob: cap every speculative step's accepted drafts at `j`
/// (0..=k). With correct rollback the emitted chain equals the unforced one.
pub const ENV_FORCE_ACCEPT: &str = "ATLAS_GLM_EXL3_DFLASH_FORCE_ACCEPT";
/// `1`: at sequence free, rank 0 logs the generated token ids
/// (`DFLASH_TOKENS`) so runs can be compared token by token.
pub const ENV_TOKEN_LOG: &str = "ATLAS_GLM_EXL3_DFLASH_TOKEN_LOG";

/// Exact rollback check every N speculative steps (diagnostic, slow): after
/// a partial accept, restore the pre-verify state, re-verify with every
/// rejected row replaced by different tokens, commit again and require the
/// accepted rows' argmax and the full KDA/conv state to be bit-identical.
pub const ENV_ROLLBACK_CHECK: &str = "ATLAS_GLM_EXL3_DFLASH_ROLLBACK_CHECK";

/// Drafter weight precision: `bf16` (default, the reference drafter),
/// `mxfp8` (needs `ATLAS_DFLASH_MXFP8=1`) or `nvfp4` (the load-time NVFP4
/// twins; `ATLAS_DFLASH_NVFP4_TC=1` + `ATLAS_W4A16_TC=1` select the
/// tensor-core tier). The quantized modes change drafter numerics only.
pub const ENV_DRAFTER_WEIGHTS: &str = "ATLAS_GLM_EXL3_DFLASH_DRAFTER_WEIGHTS";

fn drafter_weights_mode() -> Result<&'static str> {
    let mxfp8_env = std::env::var("ATLAS_DFLASH_MXFP8").as_deref() == Ok("1");
    let nvfp4_off = std::env::var("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_ok();
    match std::env::var(ENV_DRAFTER_WEIGHTS).as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("bf16") => Ok("bf16"),
        Ok("mxfp8") => {
            ensure!(
                mxfp8_env,
                "{ENV_DRAFTER_WEIGHTS}=mxfp8 needs ATLAS_DFLASH_MXFP8=1"
            );
            Ok("mxfp8")
        }
        Ok("nvfp4") => {
            ensure!(
                !mxfp8_env && !nvfp4_off,
                "{ENV_DRAFTER_WEIGHTS}=nvfp4 needs ATLAS_DFLASH_MXFP8 unset and \
                 ATLAS_NO_DFLASH_DRAFTER_NVFP4 unset"
            );
            Ok("nvfp4")
        }
        Ok(other) => {
            anyhow::bail!("{ENV_DRAFTER_WEIGHTS}={other:?}: expected bf16, mxfp8 or nvfp4")
        }
    }
}

/// `ATLAS_GLM_EXL3_DFLASH=1` and `ATLAS_GLM_EXL3_DFLASH_SPEC=1`.
pub fn spec_enabled() -> bool {
    enabled() && std::env::var(ENV_SPEC).as_deref() == Ok("1")
}

/// Verify rows the SSM pools must snapshot for the EXL3 observer (`k + 1`
/// when the verify probe or the speculative lane is on, else 0). Read by
/// model construction and the server's memory preflight so both size the
/// same pools.
pub fn verify_rows() -> usize {
    let probe = parse_probe(std::env::var(ENV_VERIFY_PROBE).ok().as_deref()) > 0;
    if !enabled() || !(probe || spec_enabled()) {
        return 0;
    }
    parse_k(std::env::var(ENV_K).ok().as_deref()).map_or(0, |k| k + 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureMode {
    Fp32Mean,
    Bf16Mean,
}

fn parse_capture(raw: Option<&str>) -> Result<CaptureMode> {
    match raw.map(str::trim) {
        None | Some("") | Some("fp32mean") => Ok(CaptureMode::Fp32Mean),
        Some("bf16mean") => Ok(CaptureMode::Bf16Mean),
        Some(other) => anyhow::bail!("{ENV_CAPTURE}={other:?}: expected fp32mean or bf16mean"),
    }
}

fn shifted_layers(ids: &[usize], shift: i64, layers: usize) -> Result<Vec<usize>> {
    ids.iter()
        .map(|&l| {
            let s = l as i64 + shift;
            ensure!(
                (0..layers as i64).contains(&s),
                "{ENV_LAYER_SHIFT}={shift} moves capture layer {l} outside 0..{layers}"
            );
            Ok(s as usize)
        })
        .collect()
}

/// Container path of the pinned drafter (`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`).
pub const DEFAULT_DRAFTER_DIR: &str = "/root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2-pin/snapshots/dc77ff1c99eeb2df044ee3d4f0094eb033fee410";

const DEFAULT_K: usize = 7;
const DEFAULT_LOG_EVERY: u64 = 64;

/// `ATLAS_GLM_EXL3_DFLASH=1`.
pub fn enabled() -> bool {
    std::env::var(ENV).as_deref() == Ok("1")
}

/// Refusal used by the EXL3 speculation guards: phase 1 is draft-only, so a
/// CLI speculative mode would route the EXL3 target into an unimplemented
/// verify lane.
pub fn cli_spec_refusal() -> String {
    format!(
        "{ENV}=1 is the draft-only DFlash2 observer (F4 phase 1): drop --dflash/--speculative/\
         --self-speculative and configure it with {ENV_DRAFTER}, {ENV_K}, {ENV_LOG_EVERY}, {ENV_TRACE}; \
         EXL3 verify is phase 2"
    )
}

fn parse_k(raw: Option<&str>) -> Result<usize> {
    let k = match raw {
        None => DEFAULT_K,
        Some(v) => v
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{ENV_K}={v:?} is not an integer"))?,
    };
    ensure!(
        (1..=7).contains(&k),
        "{ENV_K}={k} outside 1..=7 (drafter block_size 8)"
    );
    Ok(k)
}

fn parse_log_every(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_LOG_EVERY)
}

/// Drafter, its replicated vocabulary head and weights, built before the KV
/// budget so their memory is accounted, then attached to the model.
pub struct Exl3DflashParts {
    head: BlockDiffusionDraftHead,
    drafter_store: WeightStore,
    k: usize,
}

/// Load the DFlash2 drafter on this rank (every rank holds all of it).
pub fn load(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    embed_tokens: DevicePtr,
    max_seq_len: usize,
    max_batch_size: usize,
) -> Result<Exl3DflashParts> {
    use spark_runtime::weights::WeightLoader;
    let checkpoint = store
        .exl3_checkpoint()
        .context("ATLAS_GLM_EXL3_DFLASH requires the native EXL3 checkpoint")?;
    ensure!(
        config.model_type == "glm5_next"
            && config.hc_mult == 4
            && config.tp_world_size == 2
            && config.ep_world_size == 1,
        "ATLAS_GLM_EXL3_DFLASH requires GLM-5 EXL3 with TP=2/EP=1 and 4 mHC streams"
    );
    ensure!(
        std::env::var("ATLAS_GLM_PREFILL_SP").as_deref() != Ok("1"),
        "ATLAS_GLM_EXL3_DFLASH needs every prefill row's highway on both ranks; unset ATLAS_GLM_PREFILL_SP"
    );
    ensure!(
        !embed_tokens.is_null(),
        "EXL3 DFlash needs the target embedding"
    );
    let k = parse_k(std::env::var(ENV_K).ok().as_deref())?;
    let dir = std::path::PathBuf::from(
        std::env::var(ENV_DRAFTER).unwrap_or_else(|_| DEFAULT_DRAFTER_DIR.to_string()),
    );
    let json = std::fs::read_to_string(dir.join("config.json")).with_context(|| {
        format!(
            "EXL3 DFlash: read {}/config.json ({ENV_DRAFTER})",
            dir.display()
        )
    })?;
    let drafter_config = crate::weight_loader::dflash_loader::parse_dflash_config(&json)?;
    let sub = drafter_config
        .dflash_config
        .as_ref()
        .context("EXL3 DFlash: drafter config has no dflash_config")?;
    ensure!(
        !sub.target_layer_ids.is_empty()
            && sub
                .target_layer_ids
                .iter()
                .all(|&l| l < config.num_hidden_layers),
        "EXL3 DFlash: target_layer_ids {:?} outside the {}-layer target",
        sub.target_layer_ids,
        config.num_hidden_layers
    );
    ensure!(
        k < drafter_config.block_size,
        "{ENV_K}={k} needs k + 1 <= drafter block_size {}",
        drafter_config.block_size
    );
    if k + 1 != drafter_config.block_size {
        tracing::warn!(
            "EXL3 DFlash: k={k} -> block of {} rows, drafter trained with block_size {}",
            k + 1,
            drafter_config.block_size
        );
    }
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    let drafter_store = loader
        .load(&dir, gpu, 0)
        .with_context(|| format!("EXL3 DFlash: load drafter weights from {}", dir.display()))?;
    let weights =
        crate::weight_loader::load_dflash_weights(&drafter_store, &drafter_config, gpu, 1)?
            .context("EXL3 DFlash: drafter store has no fc.weight")?;
    ensure!(
        weights.candidate_selector.is_some(),
        "EXL3 DFlash: drafter is not DFlash2 (no candidate-selector weights)"
    );
    let mut startup = DsparkStartupExecution::from_env_lenient();
    startup.native_batch_authoritative = false;
    startup.option_b_enabled = true;
    startup.proposal_lane_count = 1;
    startup.draft_cap_override = None;
    startup.option_b_no_ctx = false;
    let weights_mode = drafter_weights_mode()?;
    startup.bf16_weights_only = weights_mode == "bf16";
    let gamma = k + 1;
    let mut head = BlockDiffusionDraftHead::from_weights(
        weights,
        embed_tokens,
        DevicePtr::NULL,
        None,
        config.hidden_size,
        config.vocab_size,
        Some(gamma),
        None,
        gpu,
        max_seq_len,
        // One extra drafter KV region: a request's state can briefly
        // coexist with the next one's.
        max_batch_size + 1,
        startup,
    )?;
    let vocab_head = Exl3ReplicatedVocabHead::load(
        checkpoint,
        config,
        gpu,
        head.vocab_size,
        gamma,
        gpu.default_stream(),
    )?;
    tracing::info!(
        "EXL3 DFlash: drafter {} ({} layers, k={k}, block {gamma}, target layers {:?}, window {:?}, \
         weights {weights_mode} -> {:?}); head = both packed EXL3 lm_head TP2 halves replicated \
         ({:.2} GB read/step)",
        dir.display(),
        head.num_layers,
        head.target_layer_ids,
        head.window_size,
        head.quant,
        vocab_head.bytes_per_step() as f64 / 1e9,
    );
    head.external_head = Some(Box::new(vocab_head));
    Ok(Exl3DflashParts {
        head,
        drafter_store,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_parsing_defaults_and_bounds() {
        assert_eq!(parse_k(None).unwrap(), 7);
        assert_eq!(parse_k(Some("3")).unwrap(), 3);
        assert!(parse_k(Some("0")).is_err());
        assert!(parse_k(Some("8")).is_err());
        assert!(parse_k(Some("x")).is_err());
        assert_eq!(parse_log_every(None), DEFAULT_LOG_EVERY);
        assert_eq!(parse_log_every(Some("0")), 0);
        assert_eq!(parse_log_every(Some("bad")), DEFAULT_LOG_EVERY);
    }

    #[test]
    fn probe_parsing() {
        assert_eq!(parse_probe(None), 0);
        assert_eq!(parse_probe(Some("16")), 16);
        assert_eq!(parse_probe(Some("x")), 0);
    }

    #[test]
    fn capture_knobs_parse() {
        assert_eq!(parse_capture(None).unwrap(), CaptureMode::Fp32Mean);
        assert_eq!(
            parse_capture(Some("bf16mean")).unwrap(),
            CaptureMode::Bf16Mean
        );
        assert!(parse_capture(Some("mean")).is_err());
        assert_eq!(
            shifted_layers(&[5, 14, 42], 1, 45).unwrap(),
            vec![6, 15, 43]
        );
        assert_eq!(shifted_layers(&[5, 14], -1, 45).unwrap(), vec![4, 13]);
        assert!(shifted_layers(&[0], -1, 45).is_err());
        assert!(shifted_layers(&[44], 1, 45).is_err());
    }

    #[test]
    fn refusal_names_the_env_recipe() {
        let msg = cli_spec_refusal();
        assert!(msg.contains(ENV) && msg.contains(ENV_DRAFTER) && msg.contains("phase 2"));
    }
}
