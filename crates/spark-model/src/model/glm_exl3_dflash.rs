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

mod head;
pub(crate) mod scorer;
mod spec;

pub(crate) use spec::EP_CMD_EXL3_DFLASH_SPEC;

pub use head::Exl3ReplicatedVocabHead;
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

/// One prefill chunk's context rows, reserved at the first capture layer and
/// committed at the last.
#[derive(Clone, Copy)]
struct ChunkRows {
    /// Absolute position of the first kept row.
    first: usize,
    /// Highway row of `first`.
    row0: usize,
    rows: usize,
    /// Accumulator row of `first`.
    acc_row: usize,
}

struct SeqObs {
    owner: SequenceGeneration,
    state: Box<dyn ProposerState>,
    scorer: DraftScorer,
    /// Position whose hiddens the last decode forward left in `save`.
    pending_capture: Option<usize>,
    chunk: Option<ChunkRows>,
    broken: Option<String>,
    steps: u64,
    parity: VerifyParity,
    /// Last position whose emitted token the scorers have seen.
    observed_through: Option<usize>,
    /// Speculative-lane steps and the tokens they emitted.
    spec_steps: u64,
    spec_emitted: u64,
    /// Rollback checks: (run, state identical, accepted argmax identical).
    rollback_checks: (u64, u64, u64),
    /// Thinking-budget hard-gate engagement (`DFLASH_THINKBUDGET`):
    /// truncation events, committed emissions and dropped governed
    /// emissions. Logged unconditionally at retire so a plain run proves
    /// whether the gate engaged.
    think_budget_fired: u64,
    think_budget_kept: u64,
    think_budget_dropped: u64,
    /// Summed synchronized ms of (draft, verify, commit) under PROFILE.
    spec_ms: [f64; 3],
}

impl SeqObs {
    /// Feed the token emitted at `pos` to the scorers once.
    fn observe(&mut self, pos: usize, token: u32) {
        if self.observed_through.is_some_and(|p| pos <= p) {
            return;
        }
        self.scorer.observe(pos, token);
        self.parity.observe(pos, token);
        self.observed_through = Some(pos);
    }

    fn dstate(&mut self) -> &mut DflashProposerState {
        self.state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .expect("EXL3 DFlash state is a DflashProposerState")
    }
}

pub struct Exl3DflashObserver {
    head: BlockDiffusionDraftHead,
    _drafter_store: WeightStore,
    capture_layers: Vec<usize>,
    k: usize,
    /// `[n_capture * hidden]` BF16: the last decoded position's context row.
    save: DevicePtr,
    slot_bytes: usize,
    contract: KernelHandle,
    hidden: usize,
    hc_mult: usize,
    hc_elem: usize,
    rank: usize,
    log_every: u64,
    trace: bool,
    profile: bool,
    probe_every: u64,
    /// Speculative lane on (`ATLAS_GLM_EXL3_DFLASH_SPEC=1`).
    spec: bool,
    /// `ATLAS_GLM_EXL3_DFLASH_FORCE_ACCEPT` cap.
    force_accept: Option<usize>,
    token_log: bool,
    rollback_check_every: u64,
    /// `[k + 1, n_capture * hidden]` BF16 context rows of the last verify
    /// (NULL unless the probe or the speculative lane is on).
    verify_save: DevicePtr,
    generation: AtomicU64,
    seqs: Mutex<HashMap<usize, SeqObs>>,
    global: Mutex<AcceptStats>,
}

impl Exl3DflashObserver {
    fn build(parts: Exl3DflashParts, config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<Self> {
        let Exl3DflashParts {
            head,
            drafter_store,
            k,
        } = parts;
        let shift: i64 = match std::env::var(ENV_LAYER_SHIFT) {
            Ok(v) => v
                .trim()
                .parse()
                .with_context(|| format!("{ENV_LAYER_SHIFT}={v:?} is not an integer"))?,
            Err(_) => 0,
        };
        let capture_layers =
            shifted_layers(&head.target_layer_ids, shift, config.num_hidden_layers)?;
        let mode = parse_capture(std::env::var(ENV_CAPTURE).ok().as_deref())?;
        let hidden = config.hidden_size;
        ensure!(
            head.target_hidden_size == hidden,
            "EXL3 DFlash: drafter target hidden {} != target {hidden}",
            head.target_hidden_size
        );
        let slot_bytes = capture_layers.len() * hidden * 2;
        // Resolved here, never first inside a decode graph capture.
        let hc_elem = crate::layers::ops::hc_elem_bytes(&config.model_type);
        let contract = match mode {
            CaptureMode::Fp32Mean => gpu.kernel(
                "hyper_connection",
                &crate::layers::ops::hc_kernel_name(&config.model_type, "hc_contract_strided"),
            )?,
            CaptureMode::Bf16Mean => {
                ensure!(
                    hc_elem == 4,
                    "{ENV_CAPTURE}=bf16mean needs the FP32 highway (unset ATLAS_GLM_HC_BF16)"
                );
                gpu.kernel("exl3_dflash_capture", "exl3_dflash_contract_bf16_streams")?
            }
        };
        tracing::info!(
            "EXL3 DFlash capture: mode {mode:?}, layer shift {shift}, layers {capture_layers:?}"
        );
        let save = gpu.alloc(slot_bytes)?;
        gpu.memset(save, 0, slot_bytes)?;
        let verify_save = if verify_rows() > 0 {
            let bytes = (k + 1) * slot_bytes;
            let ptr = gpu.alloc(bytes)?;
            gpu.memset(ptr, 0, bytes)?;
            ptr
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            head,
            _drafter_store: drafter_store,
            capture_layers,
            k,
            save,
            slot_bytes,
            contract,
            hidden,
            hc_mult: config.hc_mult,
            hc_elem,
            rank: config.tp_rank,
            log_every: parse_log_every(std::env::var(ENV_LOG_EVERY).ok().as_deref()),
            trace: std::env::var(ENV_TRACE).as_deref() == Ok("1"),
            profile: std::env::var(ENV_PROFILE).as_deref() == Ok("1"),
            probe_every: parse_probe(std::env::var(ENV_VERIFY_PROBE).ok().as_deref()),
            spec: spec_enabled(),
            force_accept: match std::env::var(ENV_FORCE_ACCEPT) {
                Ok(v) => Some(
                    v.trim()
                        .parse::<usize>()
                        .ok()
                        .filter(|&j| j <= k)
                        .with_context(|| format!("{ENV_FORCE_ACCEPT}={v:?} outside 0..={k}"))?,
                ),
                Err(_) => None,
            },
            token_log: std::env::var(ENV_TOKEN_LOG).as_deref() == Ok("1"),
            rollback_check_every: parse_probe(std::env::var(ENV_ROLLBACK_CHECK).ok().as_deref()),
            verify_save,
            generation: AtomicU64::new(1),
            seqs: Mutex::new(HashMap::new()),
            global: Mutex::new(AcceptStats::new(k)),
        })
    }

    /// Stream-mean of `rows` highway rows from `row0` into BF16 rows
    /// `dst_stride` bytes apart.
    #[allow(clippy::too_many_arguments)]
    fn contract_rows(
        &self,
        gpu: &dyn GpuBackend,
        highway: DevicePtr,
        row0: usize,
        rows: usize,
        dst: DevicePtr,
        dst_stride: usize,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.contract)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(highway.offset(row0 * self.hc_mult * self.hidden * self.hc_elem))
            .arg_ptr(dst)
            .arg_u32(self.hidden as u32)
            .arg_u32(self.hc_mult as u32)
            .arg_u32((dst_stride / 2) as u32)
            .launch(stream)
    }

    /// Make room for `need` rows, sliding the window (keep the newest rows,
    /// never more than half the accumulator, src/dst never overlapping).
    fn make_room(
        &self,
        gpu: &dyn GpuBackend,
        d: &mut DflashProposerState,
        need: usize,
        stream: u64,
    ) -> Result<()> {
        if d.ctx_len + need <= d.max_ctx_len {
            return Ok(());
        }
        ensure!(
            need <= d.max_ctx_len,
            "EXL3 DFlash: {need} rows exceed the context window"
        );
        let keep = (d.max_ctx_len / 2)
            .min(d.max_ctx_len - need)
            .min(d.ctx_len / 2);
        let drop_n = d.ctx_len - keep;
        if keep > 0 {
            gpu.copy_d2d_async(
                d.ctx_hidden_acc.offset(drop_n * self.slot_bytes),
                d.ctx_hidden_acc,
                keep * self.slot_bytes,
                stream,
            )?;
        }
        d.ctx_positions.drain(..drop_n);
        d.ctx_len = keep;
        d.ctx_committed = 0;
        Ok(())
    }

    fn log_line(&self, scope: &str, request: &str, slot: usize, stats: &AcceptStats) {
        tracing::info!(
            "DFLASH_ACCEPT rank={} scope={scope} req={request} slot={slot} {}",
            self.rank,
            stats.summary()
        );
    }

    /// Finish a sequence's scorer, log it, fold it into the global totals and
    /// release its drafter state.
    fn parity_line(&self, scope: &str, request: &str, slot: usize, parity: &VerifyParity) {
        if self.probe_every > 0 {
            tracing::info!(
                "DFLASH_VERIFY_PARITY rank={} scope={scope} req={request} slot={slot} {}",
                self.rank,
                parity.summary()
            );
        }
    }

    fn retire(&self, gpu: &dyn GpuBackend, slot: usize, mut obs: SeqObs, request: &str) {
        obs.scorer.finish();
        self.log_line("final", request, slot, obs.scorer.stats());
        self.parity_line("final", request, slot, &obs.parity);
        if self.spec {
            let (launches, unique, routes) = crate::layers::glm5_exl3_ffn::verify_stats::take();
            if launches > 0 {
                tracing::info!(
                    "DFLASH_EXPERTS rank={} req={request} slot={slot} multirow_launches={launches} \
                     unique_per_launch={:.2} routes_per_launch={:.2}",
                    self.rank,
                    unique as f64 / launches as f64,
                    routes as f64 / launches as f64
                );
            }
            tracing::info!(
                "DFLASH_SPEC rank={} scope=final req={request} slot={slot} spec_steps={} \
                 emitted={} tokens_per_step={:.3} rollback_checks={} state_identical={} \
                 argmax_identical={}",
                self.rank,
                obs.spec_steps,
                obs.spec_emitted,
                obs.spec_emitted as f64 / obs.spec_steps.max(1) as f64,
                obs.rollback_checks.0,
                obs.rollback_checks.1,
                obs.rollback_checks.2
            );
            // Unconditional engagement proof for the thinking-budget hard
            // gate: `fired=0` on a budget-crossing request means the
            // boundary landed on a step end (the gate had nothing to drop).
            tracing::info!(
                "DFLASH_THINKBUDGET rank={} scope=final req={request} slot={slot} \
                 fired={} kept={} dropped={}",
                self.rank,
                obs.think_budget_fired,
                obs.think_budget_kept,
                obs.think_budget_dropped
            );
            if self.profile && obs.spec_steps > 0 {
                let n = obs.spec_steps as f64;
                tracing::info!(
                    "DFLASH_SPEC_PROFILE rank={} req={request} slot={slot} draft_ms={:.2} \
                     verify_ms={:.2} commit_ms={:.2} (synchronized, per spec step)",
                    self.rank,
                    obs.spec_ms[0] / n,
                    obs.spec_ms[1] / n,
                    obs.spec_ms[2] / n
                );
            }
        }
        {
            let mut global = self.global.lock();
            global.merge(obs.scorer.stats());
            self.log_line("global", "-", slot, &global);
        }
        if let Some(reason) = &obs.broken {
            tracing::warn!(
                "DFLASH_ACCEPT rank={} req={request} slot={slot}: drafting stopped early: {reason}",
                self.rank
            );
        }
        if let Err(error) = self
            .head
            .free_state(gpu, Some(obs.owner), obs.state.as_mut())
        {
            tracing::warn!("EXL3 DFlash: drafter state release failed: {error:#}");
        }
    }
}

fn request_label(seq: &SequenceState) -> String {
    seq.request_id
        .as_deref()
        .map_or_else(|| "-".to_string(), str::to_string)
}

impl TransformerModel {
    /// Factory: install the observer built by [`load`].
    pub fn attach_exl3_dflash(&mut self, parts: Exl3DflashParts) -> Result<()> {
        ensure!(
            self.exl3_dflash.is_none(),
            "EXL3 DFlash observer already attached"
        );
        ensure!(
            self.proposer.is_none(),
            "EXL3 DFlash observer cannot coexist with a speculative proposer"
        );
        let observer = Exl3DflashObserver::build(parts, &self.config, self.gpu.as_ref())?;
        tracing::info!(
            "EXL3 DFlash draft-only observer ACTIVE on rank {} (k={}, capture after layers {:?}, \
             DFLASH_ACCEPT every {} steps)",
            observer.rank,
            observer.k,
            observer.capture_layers,
            observer.log_every
        );
        self.exl3_dflash = Some(Box::new(observer));
        Ok(())
    }

    /// New sequence in `seq.slot_idx`: fresh drafter state (any previous
    /// occupant is retired first). Never fails the allocation: on error the
    /// sequence simply runs without drafts.
    pub(crate) fn exl3_dflash_on_alloc(&self, seq: &SequenceState) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        if let Err(error) = self.exl3_dflash_on_alloc_inner(obs, seq) {
            tracing::warn!(
                "EXL3 DFlash rank {} slot {}: no drafter state for this sequence: {error:#}",
                obs.rank,
                seq.slot_idx
            );
        }
    }

    fn exl3_dflash_on_alloc_inner(
        &self,
        obs: &Exl3DflashObserver,
        seq: &SequenceState,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();
        if let Some(old) = obs.seqs.lock().remove(&seq.slot_idx) {
            obs.retire(gpu, seq.slot_idx, old, "-");
        }
        let generation = obs.generation.fetch_add(1, Ordering::Relaxed);
        let owner = SequenceGeneration::new(seq.slot_idx, generation)?;
        let mut state = obs.head.alloc_state(gpu)?;
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .context("EXL3 DFlash: drafter state type")?;
        dstate.lifecycle = Some(CaptureDescriptor::bind(
            owner,
            0,
            0,
            obs.head.gamma,
            dstate.ctx_slot_bytes,
        )?);
        obs.seqs.lock().insert(
            seq.slot_idx,
            SeqObs {
                owner,
                state,
                scorer: DraftScorer::new(obs.k),
                pending_capture: None,
                chunk: None,
                broken: None,
                steps: 0,
                parity: VerifyParity::default(),
                observed_through: None,
                spec_steps: 0,
                spec_emitted: 0,
                rollback_checks: (0, 0, 0),
                think_budget_fired: 0,
                think_budget_kept: 0,
                think_budget_dropped: 0,
                spec_ms: [0.0; 3],
            },
        );
        Ok(())
    }

    /// Sequence released: final `DFLASH_ACCEPT` line and drafter cleanup.
    pub(crate) fn exl3_dflash_on_free(&self, seq: &SequenceState) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        if obs.token_log && obs.rank == 0 && seq.tokens.len() > seq.prompt_len {
            tracing::info!(
                "DFLASH_TOKENS rank=0 req={} n={} tokens={:?}",
                request_label(seq),
                seq.tokens.len() - seq.prompt_len,
                &seq.tokens[seq.prompt_len..]
            );
        }
        if let Some(old) = obs.seqs.lock().remove(&seq.slot_idx) {
            obs.retire(self.gpu.as_ref(), seq.slot_idx, old, &request_label(seq));
        }
    }

    /// Prefill capture after layer `layer_idx` over `rows` highway rows at
    /// absolute positions `pos0..pos0 + rows`.
    /// Never fails the prefill: an error stops drafting for this sequence.
    pub(crate) fn exl3_dflash_prefill_layer(
        &self,
        seq: &SequenceState,
        layer_idx: usize,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        if let Err(error) =
            self.exl3_dflash_prefill_layer_inner(obs, seq, layer_idx, pos0, rows, stream)
        {
            tracing::warn!(
                "EXL3 DFlash rank {} slot {}: prefill capture failed at layer {layer_idx}: {error:#}",
                obs.rank,
                seq.slot_idx
            );
            if let Some(so) = obs.seqs.lock().get_mut(&seq.slot_idx) {
                so.broken = Some(format!("prefill capture: {error:#}"));
            }
        }
        Ok(())
    }

    fn exl3_dflash_prefill_layer_inner(
        &self,
        obs: &Exl3DflashObserver,
        seq: &SequenceState,
        layer_idx: usize,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(slot) = obs.capture_layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        let gpu = self.gpu.as_ref();
        let mut seqs = obs.seqs.lock();
        let Some(so) = seqs.get_mut(&seq.slot_idx) else {
            return Ok(());
        };
        if so.broken.is_some() {
            return Ok(());
        }
        if slot == 0 {
            so.chunk = None;
            let d = so.dstate();
            let end = pos0 + rows;
            let window_start = seq.tokens.len().max(end).saturating_sub(d.max_ctx_len);
            let first = pos0.max(window_start);
            if first >= end {
                return Ok(());
            }
            if let Some(&last) = d.ctx_positions.last()
                && last as usize >= first
            {
                so.broken = Some(format!(
                    "prefill row {first} does not follow context position {last}"
                ));
                return Ok(());
            }
            obs.make_room(gpu, d, end - first, stream)?;
            let acc_row = d.ctx_len;
            so.chunk = Some(ChunkRows {
                first,
                row0: first - pos0,
                rows: end - first,
                acc_row,
            });
        }
        let Some(chunk) = so.chunk else {
            return Ok(());
        };
        let acc = so.dstate().ctx_hidden_acc;
        obs.contract_rows(
            gpu,
            self.buffers.hc_streams(),
            chunk.row0,
            chunk.rows,
            acc.offset(chunk.acc_row * obs.slot_bytes + slot * obs.hidden * 2),
            obs.slot_bytes,
            stream,
        )?;
        if slot + 1 == obs.capture_layers.len() {
            let d = so.dstate();
            d.ctx_positions
                .extend((chunk.first..chunk.first + chunk.rows).map(|p| p as i32));
            d.ctx_len += chunk.rows;
            so.chunk = None;
            so.pending_capture = None;
        }
        Ok(())
    }

    /// Decode capture after layer `layer_idx` (row 0 of the highway) into the
    /// observer's save row. Stream-ordered and graph-capturable.
    pub(crate) fn exl3_dflash_decode_capture(&self, layer_idx: usize, stream: u64) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        let Some(slot) = obs.capture_layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        obs.contract_rows(
            self.gpu.as_ref(),
            self.buffers.hc_streams(),
            0,
            1,
            obs.save.offset(slot * obs.hidden * 2),
            obs.slot_bytes,
            stream,
        )
    }

    /// Before the serial decode of `token` at `seq.seq_len`: score earlier
    /// drafts against `token`, append the previous position's context row and
    /// propose `k` drafts after `token`. Never fails the decode: a drafter
    /// error stops drafting for this sequence.
    ///
    /// Only a failed verify-probe rollback is returned as an error: the
    /// target state could then differ from the serial one.
    pub(crate) fn exl3_dflash_before_decode(
        &self,
        token: u32,
        seq: &mut SequenceState,
    ) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        let mut seqs = obs.seqs.lock();
        let Some(so) = seqs.get_mut(&seq.slot_idx) else {
            return Ok(());
        };
        if so.broken.is_some() {
            return Ok(());
        }
        let pos = seq.seq_len;
        so.observe(pos, token);
        let drafts = match self.exl3_dflash_step(obs, so, token, pos, seq) {
            Ok(drafts) => drafts,
            Err(error) => {
                tracing::warn!(
                    "EXL3 DFlash rank {} slot {}: drafting stopped at position {pos}: {error:#}",
                    obs.rank,
                    seq.slot_idx
                );
                so.broken = Some(format!("{error:#}"));
                return Ok(());
            }
        };
        if obs.probe_every > 0 && so.steps.is_multiple_of(obs.probe_every) && drafts.len() == obs.k
        {
            let mut rows = Vec::with_capacity(drafts.len() + 1);
            rows.push(token);
            rows.extend_from_slice(&drafts);
            match self.exl3_dflash_verify_probe(&rows, seq) {
                Ok(argmax) => so.parity.record(pos, &drafts, &argmax),
                Err(error) => {
                    // The probe restored the target state before returning a
                    // probe error; a restore failure is the `?` inside.
                    tracing::warn!(
                        "EXL3 DFlash rank {} slot {}: verify probe disabled at {pos}: {error:#}",
                        obs.rank,
                        seq.slot_idx
                    );
                    so.broken = Some(format!("verify probe: {error:#}"));
                }
            }
            if obs.log_every > 0 && so.steps.is_multiple_of(obs.log_every) {
                obs.parity_line("periodic", &request_label(seq), seq.slot_idx, &so.parity);
            }
        }
        Ok(())
    }

    /// One `rows.len()`-row verify forward of `rows` at `seq.seq_len`
    /// (the production K=gamma verify, EXL3 arms), then a full restore:
    /// SSM state from the pre-verify checkpoint, per-layer aux state, and
    /// `seq_len`/`tokens`. MLA K/V and index rows past `seq_len` are
    /// position-addressed and rewritten by the serial decode. Returns the
    /// per-row argmax. Restore errors propagate (outer `Result`); a verify
    /// error is returned only after the restore succeeded.
    fn exl3_dflash_verify_probe(&self, rows: &[u32], seq: &mut SequenceState) -> Result<Vec<u32>> {
        let len = seq.seq_len;
        let tokens = seq.tokens.len();
        use crate::traits::Model;
        self.checkpoint_ssm_states(seq)?;
        let verified = self.decode_verify_graphed_kgamma(rows, seq, 0);
        seq.seq_len = len;
        seq.tokens.truncate(tokens);
        let stream = self.gpu.default_stream();
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            self.layers[i].rollback_aux_verify(
                state.as_mut(),
                0,
                rows.len(),
                self.gpu.as_ref(),
                stream,
            )?;
        }
        self.rollback_ssm_states(seq, 0)?;
        verified
    }

    fn exl3_dflash_step(
        &self,
        obs: &Exl3DflashObserver,
        so: &mut SeqObs,
        token: u32,
        pos: usize,
        seq: &SequenceState,
    ) -> Result<Vec<u32>> {
        let gpu = self.gpu.as_ref();
        let stream = gpu.default_stream();
        let profile_now =
            obs.profile && obs.log_every > 0 && (so.steps + 1).is_multiple_of(obs.log_every);
        let clock = |sync: bool| -> Result<std::time::Instant> {
            if sync {
                gpu.synchronize(stream)?;
            }
            Ok(std::time::Instant::now())
        };
        let t_start = clock(profile_now)?;
        match so.pending_capture.take() {
            Some(prev) => {
                ensure!(
                    prev + 1 == pos,
                    "captured position {prev} does not precede decode position {pos}"
                );
                let d = so.dstate();
                obs.make_room(gpu, d, 1, stream)?;
                gpu.copy_d2d_async(
                    obs.save,
                    d.ctx_hidden_acc.offset(d.ctx_len * obs.slot_bytes),
                    obs.slot_bytes,
                    stream,
                )?;
                d.ctx_positions.push(prev as i32);
                d.ctx_len += 1;
            }
            None => {
                let d = so.dstate();
                ensure!(
                    pos > 0 && d.ctx_positions.last().copied() == Some((pos - 1) as i32),
                    "context ends at {:?}, decode position {pos} (prefill capture missing)",
                    d.ctx_positions.last()
                );
            }
        }
        // propose() must not append again: every row is ours.
        so.dstate().skip_next_decode_append = false;
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
            request_id: seq.request_id.clone(),
        };
        let owner = so.owner;
        let t_appended = clock(profile_now)?;
        let propose = |state: &mut dyn ProposerState| {
            obs.head.propose(
                token,
                DevicePtr::NULL,
                pos,
                obs.k,
                state,
                Some(owner),
                &ctx,
                stream,
                None,
                None,
                None,
            )
        };
        let drafts = propose(so.state.as_mut())?;
        if profile_now {
            let t_proposed = clock(true)?;
            // Same block again: context precompute is already committed, so
            // this times fc-free block forward + head + selector + readback.
            let again = propose(so.state.as_mut())?;
            let t_repeat = clock(true)?;
            let (head_ms, selector_ms) = obs.head.profile_tail_ms(gpu, 5)?;
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
            let propose_ms = ms(t_appended, t_proposed);
            let block_ms = ms(t_proposed, t_repeat);
            tracing::info!(
                "DFLASH_PROFILE rank={} step={} pos={pos} ctx={} append_ms={:.3} propose_ms={:.3} \
                 block_repeat_ms={:.3} precompute_est_ms={:.3} head_ms={:.3} selector_ms={:.3} \
                 layers_embed_readback_est_ms={:.3} repeat_identical={}",
                obs.rank,
                so.steps + 1,
                so.dstate().ctx_len,
                ms(t_start, t_appended),
                propose_ms,
                block_ms,
                propose_ms - block_ms,
                head_ms,
                selector_ms,
                block_ms - head_ms - selector_ms,
                again == drafts
            );
        }
        so.scorer.record(pos, &drafts);
        so.steps += 1;
        if obs.trace {
            tracing::info!(
                "DFLASH_DRAFT rank={} req={} slot={} pos={pos} bonus={token} ctx={} drafts={:?}",
                obs.rank,
                request_label(seq),
                seq.slot_idx,
                so.dstate().ctx_len,
                drafts
            );
        }
        if obs.log_every > 0 && so.steps.is_multiple_of(obs.log_every) {
            obs.log_line(
                "periodic",
                &request_label(seq),
                seq.slot_idx,
                so.scorer.stats(),
            );
        }
        Ok(drafts)
    }

    /// After a successful serial decode of position `pos`: its context row
    /// sits in the save buffer for the next step.
    pub(crate) fn exl3_dflash_after_decode(&self, slot: usize, pos: usize) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        if let Some(so) = obs.seqs.lock().get_mut(&slot)
            && so.broken.is_none()
        {
            so.pending_capture = Some(pos);
        }
    }

    /// Batched (n > 1) decode does not capture: stop drafting for those
    /// sequences (phase 1 measures C=1).
    pub(crate) fn exl3_dflash_batched_decode(&self, slots: &[usize]) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        let mut seqs = obs.seqs.lock();
        for slot in slots {
            if let Some(so) = seqs.get_mut(slot)
                && so.broken.is_none()
            {
                so.broken = Some("batched decode (phase 1 drafts at C=1 only)".to_string());
            }
        }
    }
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
