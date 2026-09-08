// SPDX-License-Identifier: AGPL-3.0-only

//! Bind the host pair plan to the existing C1 capture, target rows and request.
//! The head validates before E1; the worker independently binds its local pool.

use super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::speculative::glm_repair::{RepairInput, RepairSpan};
use crate::speculative::glm_repair_policy as policy;
use crate::traits::{Model, SequenceState};
use anyhow::{Context, Result, ensure};
use std::sync::atomic::Ordering;

impl TransformerModel {
    pub(super) fn glm_repair_context(&self) -> ForwardContext<'_> {
        ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        }
    }

    fn glm_repair_input<'a>(
        &self,
        seq: &'a SequenceState,
        token: u32,
        position: usize,
        drafts: usize,
        hidden_row: usize,
    ) -> Result<RepairInput<'a>> {
        ensure!(
            self.config.model_type == "glm5_next"
                && self.config.tp_world_size == 2
                && self.config.ep_world_size == 2
                && self.levers.max_decode_seqs == 1
                && self.levers.drafter.prefill
                && !self.levers.drafter.carry,
            "GLM repair requires the resolved C1 TP2/EP2 prefill-only lane"
        );
        ensure!(
            self.comm_ref().is_some(),
            "GLM repair requires the live EP communicator"
        );
        {
            let cache = self.kv_cache.lock();
            let config = cache.config();
            ensure!(
                config.dtype == spark_runtime::kv_cache::KvCacheDtype::Bf16
                    && config
                        .layer_dtypes
                        .iter()
                        .all(|d| *d == spark_runtime::kv_cache::KvCacheDtype::Bf16),
                "GLM repair requires actual BF16 target cache on every layer"
            );
        }
        ensure!(
            seq.seq_len == position && self.mtp_slot_draft_capacity(seq.slot_idx) >= 4,
            "GLM repair live target position/verify capacity mismatch"
        );
        let row = self
            .config
            .hidden_size
            .checked_mul(2)
            .context("GLM repair hidden bytes overflow")?;
        let capture_bytes = self
            .mtp_prefill_capacity
            .checked_mul(row)
            .context("GLM capture bytes overflow")?;
        Ok(RepairInput {
            token,
            tokens: &seq.tokens,
            prompt_len: seq.prompt_len,
            position,
            drafts,
            generation: seq.mtp_capture_gen,
            capture_generation: self.mtp_prefill_capture_gen.load(Ordering::Relaxed),
            captured_rows: self.mtp_prefill_capture_len.load(Ordering::Relaxed),
            context_tokens: self.mtp_prefill_capacity,
            capture: RepairSpan {
                ptr: self.mtp_prefill_hidden,
                bytes: capture_bytes,
            },
            normalized: RepairSpan {
                ptr: self.buffers.norm_output(),
                bytes: self.buffers.sizes().norm_output,
            },
            bonus: RepairSpan {
                ptr: self.mtp_hidden_save,
                bytes: row,
            },
            hidden_row,
        })
    }

    pub(super) fn validate_glm_mtp_repair(
        &self,
        seq: &SequenceState,
        token: u32,
        position: usize,
        drafts: usize,
        hidden_row: usize,
        grammar: bool,
    ) -> Result<()> {
        if !policy::enabled() {
            return Ok(());
        }
        ensure!(
            !grammar,
            "GLM repair does not support grammar-constrained requests"
        );
        policy::validate_environment()?;
        let input = self.glm_repair_input(seq, token, position, drafts, hidden_row)?;
        let state = seq
            .proposer_state
            .as_ref()
            .context("GLM repair missing proposer state")?;
        let repair = self
            .proposer
            .as_ref()
            .and_then(|p| p.glm_pair_repair())
            .context("GLM repair missing proposer capability")?;
        repair.validate_prepare(&input, state.as_ref(), &self.glm_repair_context())
    }

    pub(super) fn prepare_glm_mtp_repair(
        &self,
        seq: &mut SequenceState,
        token: u32,
        position: usize,
        drafts: usize,
        stream: u64,
    ) -> Result<()> {
        if !policy::enabled() {
            return Ok(());
        }
        // Temporarily take the state to borrow complete token/capture metadata
        // independently. Always restore it, including a Failed phase on error.
        let mut state = seq
            .proposer_state
            .take()
            .context("GLM repair missing state")?;
        let result = (|| {
            let input = self.glm_repair_input(
                seq,
                token,
                position,
                drafts,
                self.last_mtp_hidden_idx.load(Ordering::Relaxed),
            )?;
            let repair = self
                .proposer
                .as_ref()
                .and_then(|p| p.glm_pair_repair())
                .context("GLM repair missing capability")?;
            repair.prepare(&input, state.as_mut(), &self.glm_repair_context(), stream)
        })();
        seq.proposer_state = Some(state);
        if result.is_ok() {
            // The owned prompt has been consumed. It is now accepted-row
            // staging and must never be reinterpreted by the legacy primer.
            self.mtp_prefill_capture_len.store(0, Ordering::Relaxed);
            *self.mtp_store_range.lock() = (0, 0);
            if self.stats.dumped.keyed("glm_mtp_pair_repair") {
                tracing::info!(
                    "GLM accepted-pair repair engaged: request generation={}, position={}, drafts={}",
                    seq.mtp_capture_gen,
                    position,
                    drafts
                );
            }
        }
        result
    }

    pub(super) fn record_glm_mtp_verified_impl(
        &self,
        seq: &mut SequenceState,
        base: usize,
        tokens: &[u32],
        accepted: usize,
    ) -> Result<()> {
        ensure!(
            self.paired_handoff().is_none(),
            "paired verdict staging is not available in Gate1"
        );
        if !policy::enabled() {
            return Ok(());
        }
        ensure!(
            tokens.len() == 5 && accepted <= 4,
            "GLM repair verdict requires K5"
        );
        let end = base
            .checked_add(accepted + 1)
            .context("GLM verdict position overflow")?;
        ensure!(
            seq.seq_len == end
                && seq.tokens.len() == end
                && seq.tokens.get(base..end) == Some(&tokens[..accepted + 1]),
            "GLM verdict does not match committed target tokens"
        );
        let generation = seq.mtp_capture_gen;
        let capture_generation = self.mtp_prefill_capture_gen.load(Ordering::Relaxed);
        let rows = self.buffers.sizes().norm_output
            / self
                .config
                .hidden_size
                .checked_mul(2)
                .filter(|n| *n > 0)
                .context("GLM verdict hidden geometry")?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("GLM verdict missing state")?
            .as_any_mut()
            .downcast_mut::<crate::layers::Glm5MtpProposerState>()
            .context("GLM verdict has foreign state")?;
        state.record_verified(
            generation,
            capture_generation,
            base,
            tokens,
            accepted,
            end,
            rows,
        )
    }
}
