// SPDX-License-Identifier: AGPL-3.0-only
//! Actual single-owner K5 wrapper, with one-time verdict detachment.
use super::*;
use crate::traits::Model;

impl TransformerModel {
    // Selected allocation precedes target work; absent capability performs no I/O.
    pub(in crate::model) fn paired_allocate_target(
        &self,
        seq: &mut SequenceState,
        cache: &mut spark_runtime::kv_cache::PagedKvCache,
        width: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.paired_handoff().is_none() {
            return Ok(false);
        }
        let end = seq
            .seq_len
            .checked_add(width)
            .context("paired verify end overflow")?;
        let needed = end.div_ceil(cache.block_size());
        crate::model::block_mgmt::ensure_blocks_through_decode(
            seq,
            needed - 1,
            cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;
        self.paired_target_map(seq, cache, end)?;
        self.gpu.stream_wait_event(stream, self.secondary_event)?;
        Ok(true)
    }

    pub(in crate::model) fn paired_physical_block(
        &self,
        seq: &SequenceState,
        index: usize,
        paired: bool,
    ) -> Result<u32> {
        if paired {
            seq.physical_block_for(index)
                .context("paired checked physical block missing")
        } else {
            Ok(seq.physical_block_for(index).unwrap_or(0))
        }
    }

    pub(in crate::model) fn paired_commit_check(
        &self,
        seq: &mut SequenceState,
        committed: usize,
        width: usize,
        completed: bool,
    ) -> Result<()> {
        self.paired_ssm_bindings(seq)?;
        let capability = self
            .paired_handoff()
            .context("paired commit capability missing")?;
        let mut state = seq
            .proposer_state
            .take()
            .context("paired commit state missing")?;
        let result = (|| {
            let input = self.paired_input(seq)?;
            capability.commit_target(
                &input,
                committed,
                width,
                completed,
                state.as_mut(),
                &self.glm_repair_context(),
            )
        })();
        seq.proposer_state = Some(state);
        result
    }

    pub(in crate::model) fn paired_failed_transaction(
        &self,
        seq: &mut SequenceState,
        error: anyhow::Error,
    ) -> anyhow::Error {
        if let Some(capability) = self.paired_handoff() {
            let quarantined = seq
                .proposer_state
                .as_mut()
                .context("paired failed state missing")
                .and_then(|state| capability.quarantine(state.as_mut(), self.gpu.as_ref()));
            return error.context(format!(
                "paired transaction failed; quarantine={quarantined:?}"
            ));
        }
        error
    }

    pub(in crate::model) fn paired_before_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
    ) -> Result<bool> {
        let Some(capability) = self.paired_handoff() else {
            return Ok(false);
        };
        self.paired_profile(seq)?;
        self.paired_ssm_bindings(seq)?;
        let end = seq
            .seq_len
            .checked_add(5)
            .context("paired target K5 end overflow")?;
        let sizes = self.buffers.sizes();
        let metadata_end = (self.max_blocks_per_seq as usize)
            .checked_mul(20)
            .and_then(|n| n.checked_add(32768 + 768))
            .context("paired K5 metadata overflow")?;
        ensure!(
            tokens.len() == 5
                && end <= 2048
                && tokens
                    .iter()
                    .all(|&t| (t as usize) < self.config.vocab_size)
                && self.mtp_slot_draft_capacity(seq.slot_idx) >= 4
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
            self.paired_target_map(seq, &cache, seq.seq_len)?;
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
        let mut state = seq
            .proposer_state
            .take()
            .context("paired K5 state missing")?;
        let result = (|| {
            let input = self.paired_input(seq)?;
            capability.begin_verify(&input, tokens, state.as_mut(), &self.glm_repair_context())
        })();
        seq.proposer_state = Some(state);
        result?;
        Ok(true)
    }

    pub(in crate::model) fn paired_target_map(
        &self,
        seq: &SequenceState,
        cache: &spark_runtime::kv_cache::PagedKvCache,
        rows: usize,
    ) -> Result<()> {
        let bs = cache.block_size();
        ensure!(
            bs == 16 && rows > 0 && rows <= 2048,
            "paired target map requires dense16 actual rows"
        );
        let needed = rows.div_ceil(bs);
        ensure!(
            seq.block_table.len() >= needed
                && seq.block_table.len() <= self.max_blocks_per_seq as usize
                && seq.disk_block_ids.is_empty()
                && seq.hss_window_start() == 0,
            "paired target historical map missing or exceeds metadata capacity"
        );
        for (index, &block) in seq.block_table.iter().enumerate() {
            ensure!(
                (block as usize) < cache.num_blocks()
                    && block <= i32::MAX as u32
                    && !seq.block_table[..index].contains(&block),
                "paired target historical map is invalid or aliases itself"
            );
        }
        Ok(())
    }

    pub(in crate::model) fn paired_after_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        produced: Result<Vec<u32>>,
    ) -> Result<Vec<u32>> {
        let capability = self
            .paired_handoff()
            .context("paired K5 capability missing")?;
        let mut state = seq
            .proposer_state
            .take()
            .context("paired K5 result state missing")?;
        let result: Result<Vec<u32>> = (|| {
            let predictions = produced?;
            let input = self.paired_input(seq)?;
            capability.publish_verify(
                &input,
                tokens,
                &predictions,
                state.as_mut(),
                &self.glm_repair_context(),
            )?;
            Ok(predictions)
        })();
        let result = match result {
            Ok(predictions) => Ok(predictions),
            Err(error) => {
                self.gpu.abort_capture_if_active(self.gpu.default_stream());
                let quarantine = capability.quarantine(state.as_mut(), self.gpu.as_ref());
                Err(error).context(format!("paired K5 failed; quarantine={quarantine:?}"))
            }
        };
        seq.proposer_state = Some(state);
        result
    }

    pub(in crate::model) fn paired_record_verified(
        &self,
        seq: &mut SequenceState,
        base: usize,
        tokens: &[u32],
        accepted: usize,
    ) -> Result<()> {
        let capability = self
            .paired_handoff()
            .context("paired verdict capability missing")?;
        let mut state = seq
            .proposer_state
            .take()
            .context("paired verdict state missing")?;
        let result = (|| {
            let input = self.paired_input(seq)?;
            capability.record_verify(
                &input,
                base,
                tokens,
                accepted,
                state.as_mut(),
                &self.glm_repair_context(),
            )
        })();
        seq.proposer_state = Some(state);
        result
    }
}
