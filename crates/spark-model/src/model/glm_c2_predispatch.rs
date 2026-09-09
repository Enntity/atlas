// SPDX-License-Identifier: AGPL-3.0-only
//! Shared immutable actual-model validation before selected command dispatch.
use super::*;
use crate::speculative::glm_paired_execution::{GlmPairedExecution, sealed};
use crate::traits::Model;
use spark_runtime::gpu::DevicePtr;

impl sealed::Sealed for TransformerModel {}
impl GlmPairedExecution for TransformerModel {
    fn validate_bootstrap(&self, seq: &SequenceState, token: u32) -> Result<()> {
        self.paired_wire_profile(0)?;
        self.paired_validate_bootstrap(seq, token)
    }
    fn bootstrap(&self, seq: &mut SequenceState, token: u32) -> Result<DevicePtr> {
        self.paired_send_bootstrap(seq, token)
    }
    fn validate_verify(&self, seq: &SequenceState, tokens: &[u32]) -> Result<()> {
        self.paired_validate_verify(seq, tokens)
    }
    fn validate_propose(
        &self,
        seq: &SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: Option<&[i32]>,
    ) -> Result<()> {
        self.paired_validate_propose(seq, seed, position, drafts, grammar.is_some())
            .map(|_| ())
    }
    fn verify(&self, seq: &mut SequenceState, tokens: &[u32]) -> Result<Vec<u32>> {
        self.paired_send_verify(seq, tokens)
    }
    fn propose(
        &self,
        seq: &mut SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.paired_send_propose(seq, seed, position, drafts, grammar)
    }
}

impl TransformerModel {
    pub(in crate::model) fn paired_target_budget(
        &self,
        seq: &SequenceState,
        cache: &spark_runtime::kv_cache::PagedKvCache,
    ) -> Result<(usize, usize)> {
        self.paired_target_row_budget(seq, cache, 5)
    }

    fn paired_target_row_budget(
        &self,
        seq: &SequenceState,
        cache: &spark_runtime::kv_cache::PagedKvCache,
        rows: usize,
    ) -> Result<(usize, usize)> {
        ensure!(
            rows == 1 || rows == 5,
            "paired target budget requires scalar or K5"
        );
        ensure!(
            !self.prefix_cache.is_active() && cache.config().cache_blocks_per_seq.is_none(),
            "paired target budget requires inactive prefix cache and no HSS"
        );
        self.paired_target_map(seq, cache, seq.seq_len)?;
        let end = seq
            .seq_len
            .checked_add(rows)
            .context("paired target end overflow")?;
        let needed = end.div_ceil(cache.block_size());
        let additional = needed.saturating_sub(seq.block_table.len());
        ensure!(
            end <= 2048
                && needed <= self.max_blocks_per_seq as usize
                && additional <= cache.num_free_blocks(),
            "paired next target exceeds actual table or free-block budget"
        );
        Ok((end, needed))
    }

    pub(in crate::model) fn paired_target_preflight(
        &self,
        seq: &SequenceState,
        rows: usize,
    ) -> Result<()> {
        self.paired_profile(seq)?;
        self.paired_ssm_bindings(seq)?;
        let sizes = self.buffers.sizes();
        let metadata_end = (self.max_blocks_per_seq as usize)
            .checked_mul(20)
            .and_then(|n| n.checked_add(32768 + 768))
            .context("paired K5 metadata overflow")?;
        ensure!(
            self.mtp_slot_draft_capacity(seq.slot_idx) >= 4
                && self.buffers.max_batch_tokens() >= 5
                && sizes.hidden_states >= 5 * 8192
                && sizes.norm_output >= 5 * 8192
                && sizes.logits >= 5 * self.config.vocab_size * 2
                && sizes.scratch >= metadata_end
                && seq.disk_block_ids.is_empty()
                && seq.hss_window_start() == 0,
            "paired K5 requires actual dense target rows/arena/slot capacity"
        );
        {
            let cache = self.kv_cache.lock();
            ensure!(
                cache.config().cache_blocks_per_seq.is_none(),
                "paired K5 does not support HSS target cache"
            );
            self.paired_target_row_budget(seq, &cache, rows)?;
        }
        let scratch = self.buffers.scratch().0;
        let scratch_end = scratch
            .checked_add(sizes.scratch as u64)
            .context("paired scratch end overflow")?;
        ensure!(scratch != 0, "paired scratch owner is null");
        for (pointer, bytes) in [
            (self.buffers.hidden_states(), sizes.hidden_states),
            (self.buffers.norm_output(), sizes.norm_output),
        ] {
            let end = pointer
                .0
                .checked_add(bytes as u64)
                .context("paired target span overflow")?;
            ensure!(
                !pointer.is_null() && (end <= scratch || scratch_end <= pointer.0),
                "paired target scratch aliases actual hidden owner"
            );
        }
        Ok(())
    }

    pub(in crate::model) fn paired_validate_verify(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
    ) -> Result<()> {
        let capability = self
            .paired_handoff()
            .context("paired verification capability missing")?;
        self.paired_target_preflight(seq, 5)?;
        ensure!(
            tokens.len() == 5
                && tokens
                    .iter()
                    .all(|&t| (t as usize) < self.config.vocab_size),
            "paired verify requires five valid issued tokens"
        );
        let state = seq
            .proposer_state
            .as_ref()
            .context("paired K5 state missing")?;
        let input = self.paired_input(seq)?;
        capability.validate_verify(&input, tokens, state.as_ref(), &self.glm_repair_context())
    }

    pub(in crate::model) fn paired_validate_propose(
        &self,
        seq: &SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: bool,
    ) -> Result<(u64, u64)> {
        let capability = self
            .paired_handoff()
            .context("paired proposal capability missing")?;
        self.paired_target_preflight(seq, 5)?;
        ensure!(
            !grammar && drafts == 4 && position == seq.seq_len,
            "paired proposal requires fixed four drafts at actual request position"
        );
        let state = seq
            .proposer_state
            .as_ref()
            .context("paired proposal state missing")?;
        let input = self.paired_input(seq)?;
        capability.validate_propose(
            &input,
            seed,
            state.as_ref(),
            &self.glm_repair_context(),
            self.gpu.default_stream(),
        )
    }
}
