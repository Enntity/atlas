// SPDX-License-Identifier: AGPL-3.0-only
//! One exclusive producer for canonical slots 0/1 and original normalized rows 0/5.
use super::*;

#[path = "paired_joint_verdict.rs"]
mod verdict;

impl Pool {
    fn pair_candidate(
        &self,
        inputs: &[crate::model::GlmPairedInput<'_>; 2],
        tokens: &[[u32; 5]; 2],
        states: [&Glm5MtpProposerState; 2],
        ctx: &ForwardContext,
    ) -> Result<PairVerification> {
        ensure!(
            self.capacity.owners() == 2,
            "fixed paired compute requires capacity2 until physical group mapping is admitted"
        );
        let records = [
            self.verify_candidate(&inputs[0], &tokens[0], states[0], ctx)?,
            self.verify_candidate(&inputs[1], &tokens[1], states[1], ctx)?,
        ];
        ensure!(
            records[0].slot == 0 && records[1].slot == 1 && !ctx.graph_capture,
            "fixed pair requires canonical live slots and eager execution"
        );
        let normalized_bytes = ctx.buffers.sizes().norm_output;
        ensure!(
            normalized_bytes >= 10 * ROW_BYTES,
            "pair normalized arena lacks ten rows"
        );
        kv_rows_plan::DeviceSpan {
            ptr: ctx.buffers.norm_output(),
            bytes: normalized_bytes,
        }
        .end()?;
        Ok(PairVerification {
            records,
            normalized_bytes,
            detached: [false; 2],
            committed: [false; 2],
        })
    }

    pub(super) fn pair_owner(&self, ctx: &ForwardContext) -> Result<&PairVerification> {
        ensure!(
            !self.producer_failed,
            "paired verification terminally failed"
        );
        let Some(Producer::Pair(pair)) = &self.verification else {
            anyhow::bail!("actual Pair verification receipt missing");
        };
        ensure!(
            !ctx.graph_capture && pair.normalized_bytes == ctx.buffers.sizes().norm_output,
            "pair normalized arena/capture changed"
        );
        for index in 0..2 {
            let record = &pair.records[index];
            let slot = &self.slots[index];
            ensure!(
                record.slot == index
                    && record.generation == slot.generation
                    && slot.active
                    && !slot.failed
                    && !slot.retiring
                    && record.normalized == ctx.buffers.norm_output()
                    && slot.issued
                        == if pair.detached[index] {
                            None
                        } else {
                            Some(record.issued)
                        },
                "Pair owner, issued attempt, or original arena changed"
            );
        }
        Ok(pair)
    }
}

impl Glm5MtpHead {
    pub(super) fn paired_validate_verify_pair(
        &self,
        inputs: &[crate::model::GlmPairedInput<'_>; 2],
        tokens: &[[u32; 5]; 2],
        states: [&dyn ProposerState; 2],
        ctx: &ForwardContext,
    ) -> Result<[(u64, u64); 2]> {
        let states = [
            states[0]
                .as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("pair state0 missing")?,
            states[1]
                .as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("pair state1 missing")?,
        ];
        for state in states {
            self.validate_paired_live(state, ctx.gpu)?;
        }
        let pool = self.paired.as_ref().context("paired pool missing")?.lock();
        let candidate = pool.pair_candidate(inputs, tokens, states, ctx)?;
        Ok(candidate
            .records
            .map(|record| (record.generation, record.issued.attempt)))
    }

    pub(super) fn paired_begin_verify_pair(
        &self,
        inputs: &[crate::model::GlmPairedInput<'_>; 2],
        tokens: &[[u32; 5]; 2],
        states: [&mut dyn ProposerState; 2],
        ctx: &ForwardContext,
    ) -> Result<()> {
        let [a, b] = states;
        let states = [
            a.as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("pair state0 missing")?,
            b.as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("pair state1 missing")?,
        ];
        for state in states {
            self.validate_paired_live(state, ctx.gpu)?;
        }
        let mut pool = self.paired.as_ref().context("paired pool missing")?.lock();
        let candidate = pool.pair_candidate(inputs, tokens, states, ctx)?;
        pool.verification = Some(Producer::Pair(candidate));
        Ok(())
    }

    pub(super) fn paired_publish_verify_pair(
        &self,
        inputs: &[crate::model::GlmPairedInput<'_>; 2],
        tokens: &[[u32; 5]; 2],
        predictions: &[[u32; 5]; 2],
        states: [&mut dyn ProposerState; 2],
        ctx: &ForwardContext,
    ) -> Result<()> {
        let [a, b] = states;
        let states = [
            a.as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("pair state0 missing")?,
            b.as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("pair state1 missing")?,
        ];
        for state in states {
            self.validate_paired_live(state, ctx.gpu)?;
        }
        let mut pool = self.paired.as_ref().context("paired pool missing")?.lock();
        let pair = pool.pair_owner(ctx)?;
        for index in 0..2 {
            ensure!(
                pool.matches_request(states[index], &inputs[index], ctx)? == index,
                "pair publish slot order changed"
            );
            let record = &pair.records[index];
            let end = record
                .issued
                .base
                .checked_add(5)
                .context("pair verify end overflow")?;
            let data = inputs[index].data();
            ensure!(
                !record.produced
                    && !pair.detached[index]
                    && tokens[index] == record.issued.tokens
                    && data.position == end
                    && data.tokens.get(record.issued.base..end) == Some(tokens[index].as_slice())
                    && data.tokens.get(..record.issued.base)
                        == Some(pool.slots[index].issued_prefix.as_slice())
                    && predictions[index]
                        .iter()
                        .all(|&p| (p as usize) < ctx.config.vocab_size),
                "Pair actual producer return mismatch"
            );
        }
        if let Some(Producer::Pair(pair)) = pool.verification.as_mut() {
            for record in &mut pair.records {
                record.produced = true;
            }
        }
        Ok(())
    }
}
