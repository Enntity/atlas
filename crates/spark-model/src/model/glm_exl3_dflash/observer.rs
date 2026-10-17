// SPDX-License-Identifier: AGPL-3.0-only
//! The replicated drafter observer and its per-sequence state.

use super::*;

/// One prefill chunk's context rows, reserved at the first capture layer and
/// committed at the last.
#[derive(Clone, Copy)]
pub(super) struct ChunkRows {
    /// Absolute position of the first kept row.
    pub(super) first: usize,
    /// Highway row of `first`.
    pub(super) row0: usize,
    pub(super) rows: usize,
    /// Accumulator row of `first`.
    pub(super) acc_row: usize,
}

pub(super) struct SeqObs {
    pub(super) owner: SequenceGeneration,
    pub(super) state: Box<dyn ProposerState>,
    pub(super) scorer: DraftScorer,
    /// Position whose hiddens the last decode forward left in `save`.
    pub(super) pending_capture: Option<usize>,
    pub(super) chunk: Option<ChunkRows>,
    pub(super) broken: Option<String>,
    pub(super) steps: u64,
    pub(super) parity: VerifyParity,
    /// Last position whose emitted token the scorers have seen.
    pub(super) observed_through: Option<usize>,
    /// Speculative-lane steps and the tokens they emitted.
    pub(super) spec_steps: u64,
    pub(super) spec_emitted: u64,
    /// Rollback checks: (run, state identical, accepted argmax identical).
    pub(super) rollback_checks: (u64, u64, u64),
    /// Thinking-budget hard-gate engagement (`DFLASH_THINKBUDGET`):
    /// truncation events, committed emissions and dropped governed
    /// emissions. Logged unconditionally at retire so a plain run proves
    /// whether the gate engaged.
    pub(super) think_budget_fired: u64,
    pub(super) think_budget_kept: u64,
    pub(super) think_budget_dropped: u64,
    /// Summed synchronized ms of (draft, verify, commit) under PROFILE.
    pub(super) spec_ms: [f64; 3],
}

impl SeqObs {
    /// Feed the token emitted at `pos` to the scorers once.
    pub(super) fn observe(&mut self, pos: usize, token: u32) {
        if self.observed_through.is_some_and(|p| pos <= p) {
            return;
        }
        self.scorer.observe(pos, token);
        self.parity.observe(pos, token);
        self.observed_through = Some(pos);
    }

    pub(super) fn dstate(&mut self) -> &mut DflashProposerState {
        self.state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .expect("EXL3 DFlash state is a DflashProposerState")
    }
}

pub struct Exl3DflashObserver {
    pub(super) head: BlockDiffusionDraftHead,
    pub(super) _drafter_store: WeightStore,
    pub(super) capture_layers: Vec<usize>,
    pub(super) k: usize,
    /// `[n_capture * hidden]` BF16: the last decoded position's context row.
    pub(super) save: DevicePtr,
    pub(super) slot_bytes: usize,
    pub(super) contract: KernelHandle,
    pub(super) hidden: usize,
    pub(super) hc_mult: usize,
    pub(super) hc_elem: usize,
    pub(super) rank: usize,
    pub(super) log_every: u64,
    pub(super) trace: bool,
    pub(super) profile: bool,
    pub(super) probe_every: u64,
    /// Speculative lane on (`ATLAS_GLM_EXL3_DFLASH_SPEC=1`).
    pub(super) spec: bool,
    /// `ATLAS_GLM_EXL3_DFLASH_FORCE_ACCEPT` cap.
    pub(super) force_accept: Option<usize>,
    pub(super) token_log: bool,
    pub(super) rollback_check_every: u64,
    /// `[k + 1, n_capture * hidden]` BF16 context rows of the last verify
    /// (NULL unless the probe or the speculative lane is on).
    pub(super) verify_save: DevicePtr,
    pub(super) generation: AtomicU64,
    pub(super) seqs: Mutex<HashMap<usize, SeqObs>>,
    pub(super) global: Mutex<AcceptStats>,
}

impl Exl3DflashObserver {
    pub(super) fn build(
        parts: Exl3DflashParts,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
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
    pub(super) fn contract_rows(
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
    pub(super) fn make_room(
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

    pub(super) fn log_line(&self, scope: &str, request: &str, slot: usize, stats: &AcceptStats) {
        tracing::info!(
            "DFLASH_ACCEPT rank={} scope={scope} req={request} slot={slot} {}",
            self.rank,
            stats.summary()
        );
    }

    /// Finish a sequence's scorer, log it, fold it into the global totals and
    /// release its drafter state.
    pub(super) fn parity_line(
        &self,
        scope: &str,
        request: &str,
        slot: usize,
        parity: &VerifyParity,
    ) {
        if self.probe_every > 0 {
            tracing::info!(
                "DFLASH_VERIFY_PARITY rank={} scope={scope} req={request} slot={slot} {}",
                self.rank,
                parity.summary()
            );
        }
    }

    pub(super) fn retire(&self, gpu: &dyn GpuBackend, slot: usize, mut obs: SeqObs, request: &str) {
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

pub(super) fn request_label(seq: &SequenceState) -> String {
    seq.request_id
        .as_deref()
        .map_or_else(|| "-".to_string(), str::to_string)
}
