// SPDX-License-Identifier: AGPL-3.0-only
//! Producer receipts are minted only at actual model/request boundaries.
use super::types::TransformerModel;
use crate::speculative::glm_repair::{GlmPairedHandoff, RepairSpan};
use crate::traits::SequenceState;
use anyhow::{Context, Result, ensure};
use std::sync::atomic::Ordering;
#[path = "glm_c2_handoff_decode.rs"]
mod decode;
#[path = "glm_c2_predispatch.rs"]
mod predispatch;
#[path = "glm_c2_ssm.rs"]
mod ssm;
#[path = "glm_c2_transport.rs"]
mod transport;
#[path = "glm_c2_verification.rs"]
mod verification;

/// Not constructible from an arbitrary pointer/row bundle outside this module.
pub struct GlmPairedInput<'a> {
    data: RequestData<'a>,
}
pub(crate) struct RequestData<'a> {
    pub tokens: &'a [u32],
    pub slot: usize,
    pub prompt: usize,
    pub position: usize,
    pub capture_generation: u64,
    pub capture: RepairSpan,
    pub normalized: RepairSpan,
    pub secondary_event: u64,
}
impl GlmPairedInput<'_> {
    pub(crate) fn data(&self) -> &RequestData<'_> {
        &self.data
    }
}

impl TransformerModel {
    pub(super) fn reject_paired_batch_producer(&self) -> Result<()> {
        ensure!(
            self.paired_handoff().is_none(),
            "paired handoff does not support this legacy producer/state mutation"
        );
        Ok(())
    }
    pub(super) fn paired_prefill_result(
        &self,
        seq: &mut SequenceState,
        result: Result<spark_runtime::gpu::DevicePtr>,
    ) -> Result<spark_runtime::gpu::DevicePtr> {
        let Some(capability) = self.paired_handoff() else {
            return result;
        };
        match result {
            Ok(ptr) => Ok(ptr),
            Err(error) => {
                let quarantined = seq
                    .proposer_state
                    .as_mut()
                    .context("paired failed producer state missing")
                    .and_then(|state| capability.quarantine(state.as_mut(), self.gpu.as_ref()));
                Err(error).context(format!("paired prefill failed; quarantine={quarantined:?}"))
            }
        }
    }
    pub(super) fn try_glm_paired_propose(
        &self,
        seq: &mut SequenceState,
        token: u32,
        position: usize,
        drafts: usize,
        grammar: bool,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<Option<Vec<u32>>> {
        let Some(capability) = self.paired_handoff() else {
            return Ok(None);
        };
        self.paired_validate_propose(seq, token, position, drafts, grammar)?;
        let mut state = seq
            .proposer_state
            .take()
            .context("paired proposal state missing")?;
        let result = (|| {
            let input = self.paired_input(seq)?;
            capability.propose_owned(&input, token, state.as_mut(), ctx, stream)
        })();
        seq.proposer_state = Some(state);
        result.map(Some)
    }

    pub(super) fn paired_handoff(&self) -> Option<&dyn GlmPairedHandoff> {
        self.proposer.as_ref()?.glm_pair_repair()?.paired_handoff()
    }

    pub(in crate::model) fn paired_owner_capacity(&self) -> Result<usize> {
        let capacity = self
            .paired_handoff()
            .context("paired capacity capability missing")?
            .owner_capacity(self.gpu.as_ref())?;
        ensure!(
            capacity == self.levers.max_decode_seqs as usize && capacity == self.ssm_pool.max_slots,
            "paired private/target/admitted owner capacities differ"
        );
        Ok(capacity)
    }

    fn paired_profile(&self, seq: &SequenceState) -> Result<()> {
        let capacity = self.paired_owner_capacity()?;
        ensure!(
            self.config.model_type == "glm5_next"
                && self.config.hidden_size == 4096
                && self.config.tp_world_size == 2
                && self.config.ep_world_size == 2
                && self.levers.drafter.prefill
                && !self.levers.drafter.carry
                && !self.prefix_cache.is_active()
                && seq.slot_idx < capacity
                && seq.adapter_id == 0
                && seq.adapter_slot < 0,
            "paired producer requires the base GLM bounded-owner prefill-only profile"
        );
        let comm = self
            .comm
            .as_ref()
            .context("paired producer requires actual TP2 communicator")?;
        ensure!(
            comm.world_size() == 2
                && comm.rank() == self.config.ep_rank
                && self.config.tp_rank == self.config.ep_rank,
            "paired producer communicator differs from actual model rank"
        );
        ensure!(
            !self.gpu.stream_is_capturing(self.gpu.default_stream()),
            "paired producer cannot run inside active stream capture"
        );
        self.hidden_trace_adapter_ownership().ensure_absent()?;
        let cache = self.kv_cache.lock();
        ensure!(
            cache.config().dtype == spark_runtime::kv_cache::KvCacheDtype::Bf16
                && cache
                    .config()
                    .layer_dtypes
                    .iter()
                    .all(|d| *d == spark_runtime::kv_cache::KvCacheDtype::Bf16),
            "paired producer requires actual BF16 target cache"
        );
        Ok(())
    }

    pub(super) fn paired_prefill_preflight(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        start: usize,
        rows: usize,
        last: bool,
        stream: u64,
    ) -> Result<()> {
        let Some(capability) = self.paired_handoff() else {
            return Ok(());
        };
        self.paired_profile(seq)?;
        ensure!(
            !super::trait_impl::drafter_prefill::eager_drafter_disabled()
                && stream == self.gpu.default_stream()
                && start == 0
                && rows == tokens.len()
                && last
                && rows >= 2
                && rows <= self.buffers.max_batch_tokens()
                && rows <= self.mtp_prefill_capacity
                && tokens
                    .iter()
                    .all(|&token| (token as usize) < self.config.vocab_size)
                && seq.seq_len == 0
                && seq.tokens.is_empty()
                && (seq.prompt_len == 0 || seq.prompt_len == rows)
                && seq.cached_prefix_tokens == 0
                && seq.marconi_skip_to == 0
                && !seq.prefix_lookup_applied
                && seq.block_table.is_empty()
                && seq.mtp_capture_gen == 0,
            "paired prefill requires a cold complete single chunk within actual arena"
        );
        capability.validate_cold(
            seq.proposer_state
                .as_ref()
                .context("paired preflight state missing")?
                .as_ref(),
            seq.slot_idx,
            rows,
            &self.glm_repair_context(),
        )
    }

    pub(in crate::model) fn paired_input<'a>(
        &self,
        seq: &'a SequenceState,
    ) -> Result<GlmPairedInput<'a>> {
        self.paired_profile(seq)?;
        ensure!(
            seq.tokens.len() == seq.seq_len
                && seq.prompt_len > 0
                && seq.prompt_len <= seq.seq_len
                && seq.mtp_capture_gen != 0,
            "paired producer request metadata is incomplete"
        );
        Ok(GlmPairedInput {
            data: RequestData {
                tokens: &seq.tokens,
                slot: seq.slot_idx,
                prompt: seq.prompt_len,
                position: seq.seq_len,
                capture_generation: seq.mtp_capture_gen,
                capture: RepairSpan {
                    ptr: self.mtp_prefill_hidden,
                    bytes: self
                        .mtp_prefill_capacity
                        .checked_mul(8192)
                        .context("paired capture extent overflow")?,
                },
                normalized: RepairSpan {
                    ptr: self.buffers.norm_output(),
                    bytes: self.buffers.sizes().norm_output,
                },
                secondary_event: self.secondary_event,
            },
        })
    }

    pub(super) fn try_glm_paired_eager(
        &self,
        seq: &mut SequenceState,
        is_last: bool,
        stream: u64,
    ) -> Result<bool> {
        let Some(capability) = self.paired_handoff() else {
            return Ok(false);
        };
        if !is_last {
            return Ok(true);
        }
        ensure!(
            !super::trait_impl::drafter_prefill::eager_drafter_disabled(),
            "paired handoff requires eager primer"
        );
        ensure!(
            seq.mtp_capture_gen == self.mtp_prefill_capture_gen.load(Ordering::Relaxed)
                && self.mtp_prefill_capture_len.load(Ordering::Relaxed) == seq.prompt_len,
            "paired primer does not own complete prompt capture"
        );
        let mut state = seq
            .proposer_state
            .take()
            .context("paired primer state missing")?;
        let result = (|| {
            let input = self.paired_input(seq)?;
            capability.prime(&input, state.as_mut(), &self.glm_repair_context(), stream)
        })();
        seq.proposer_state = Some(state);
        result?;
        Ok(true)
    }
}
