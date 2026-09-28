// SPDX-License-Identifier: AGPL-3.0-only
//! One actual K5 producer owns normalized scratch until verdict detachment.
use super::*;

impl Pool {
    pub(super) fn next_attempt(&self, index: usize) -> Result<u64> {
        let slot = &self.slots[index];
        ensure!(slot.issued.is_none(), "paired proposal already issued");
        slot.attempt
            .checked_add(1)
            .context("paired attempt exhausted")
    }
    pub(super) fn scratch_idle(&self) -> Result<()> {
        ensure!(
            !self.producer_failed && self.verification.is_none(),
            "paired verification producer is active or terminally failed"
        );
        Ok(())
    }

    pub(super) fn issue(
        &mut self,
        index: usize,
        base: usize,
        seed: u32,
        drafts: &[u32],
        prefix: &[u32],
    ) -> Result<()> {
        self.scratch_idle()?;
        ensure!(
            drafts.len() == 4,
            "paired proposal did not return four drafts"
        );
        ensure!(
            prefix.len() == base && base <= self.context,
            "paired issued prefix exceeds actual context"
        );
        let attempt = self.next_attempt(index)?;
        let slot = &mut self.slots[index];
        let mut tokens = [seed; 5];
        tokens[1..].copy_from_slice(drafts);
        slot.attempt = attempt;
        slot.issued = Some(IssuedProposal {
            attempt,
            base,
            tokens,
        });
        slot.issued_prefix.clear();
        slot.issued_prefix.extend_from_slice(prefix);
        Ok(())
    }

    pub(super) fn verification_owner(&self, index: usize, tokens: &[u32]) -> Result<&Verification> {
        ensure!(
            !self.producer_failed,
            "paired verification terminally failed"
        );
        let Producer::Single(record) = self
            .verification
            .as_ref()
            .context("paired actual verification receipt missing")?
        else {
            anyhow::bail!("Single verification cannot consume Pair producer");
        };
        ensure!(
            record.slot == index
                && record.generation == self.slots[index].generation
                && Some(record.issued) == self.slots[index].issued
                && tokens == record.issued.tokens,
            "paired verification owner/issued tokens changed"
        );
        Ok(record)
    }
}

impl Pool {
    pub(super) fn verify_candidate(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &Glm5MtpProposerState,
        ctx: &ForwardContext,
    ) -> Result<Verification> {
        self.scratch_idle()?;
        let index = self.matches_request(state, input, ctx)?;
        let slab = kv_rows_plan::DeviceSpan {
            ptr: self.slab,
            bytes: self.capacity.slab_bytes(),
        };
        for span in [
            kv_rows_plan::DeviceSpan {
                ptr: ctx.buffers.scratch(),
                bytes: ctx.buffers.sizes().scratch,
            },
            kv_rows_plan::DeviceSpan {
                ptr: input.data().normalized.ptr,
                bytes: input.data().normalized.bytes,
            },
        ] {
            ensure!(
                !span.overlaps(slab)?,
                "paired K5 actual scratch aliases owned slab"
            );
        }
        let slot = &self.slots[index];
        let issued = slot.issued.context("paired K5 has no issued proposal")?;
        let repair_state::RepairPhase::Proposed(plan) = state.repair else {
            anyhow::bail!("paired K5 requires outstanding proposal");
        };
        ensure!(
            tokens == issued.tokens
                && input.data().tokens == slot.issued_prefix
                && input.data().position == issued.base
                && plan.position() == issued.base
                && plan.generation() == slot.generation
                && state.seq_len == plan.speculative_cache_end()
                && state.last_num_drafted == 4
                && input.data().normalized.bytes >= 5 * ROW_BYTES
                && !slot.proposing,
            "paired K5 does not match actual issued proposal"
        );
        Ok(Verification {
            slot: index,
            generation: slot.generation,
            issued,
            normalized: input.data().normalized.ptr,
            produced: false,
        })
    }
}

impl Glm5MtpHead {
    pub(super) fn paired_validate_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("paired K5 state missing")?;
        self.validate_paired_live(state, ctx.gpu)?;
        self.paired
            .as_ref()
            .context("paired pool missing")?
            .lock()
            .verify_candidate(input, tokens, state, ctx)?;
        Ok(())
    }

    pub(super) fn paired_begin_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired K5 state missing")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let mut pool = self.paired.as_ref().context("paired pool missing")?.lock();
        let candidate = pool.verify_candidate(input, tokens, state, ctx)?;
        pool.verification = Some(Producer::Single(candidate));
        Ok(())
    }

    pub(super) fn paired_publish_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        predictions: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired K5 state missing")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let mut pool = self.paired.as_ref().context("paired pool missing")?.lock();
        let index = pool.matches_request(state, input, ctx)?;
        let record = pool.verification_owner(index, tokens)?;
        let end = record
            .issued
            .base
            .checked_add(5)
            .context("paired verify end overflow")?;
        ensure!(
            !record.produced
                && input.data().position == end
                && input.data().tokens.get(record.issued.base..end) == Some(tokens)
                && input.data().tokens.get(..record.issued.base)
                    == Some(pool.slots[index].issued_prefix.as_slice())
                && input.data().normalized.ptr == record.normalized
                && predictions.len() == 5
                && predictions
                    .iter()
                    .all(|&p| (p as usize) < ctx.config.vocab_size),
            "paired K5 actual producer return mismatch"
        );
        if let Some(Producer::Single(record)) = pool.verification.as_mut() {
            record.produced = true;
        }
        Ok(())
    }
}
