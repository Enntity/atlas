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
        ensure!(width == 5, "paired target allocation requires K5");
        let (end, needed) = self.paired_target_budget(seq, cache)?;
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
        self.paired_validate_verify(seq, tokens)?;
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
