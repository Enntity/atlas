// SPDX-License-Identifier: AGPL-3.0-only
//! Distinct bounded producer for three through eight actual owners; no allocation.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;

#[path = "paired_owners_verdict.rs"]
mod verdict;

#[derive(Clone, Copy)]
enum Records {
    Three([Verification; 3]),
    Four([Verification; 4]),
    Five([Verification; 5]),
    Six([Verification; 6]),
    Seven([Verification; 7]),
    Eight([Verification; 8]),
}

#[derive(Clone, Copy)]
pub(super) struct OwnerVerification {
    records: Records,
    normalized_bytes: usize,
    pub(super) detached: [bool; 8],
    pub(super) committed: [bool; 8],
}

impl OwnerVerification {
    pub(super) fn records(&self) -> &[Verification] {
        match &self.records {
            Records::Three(v) => v,
            Records::Four(v) => v,
            Records::Five(v) => v,
            Records::Six(v) => v,
            Records::Seven(v) => v,
            Records::Eight(v) => v,
        }
    }
    fn records_mut(&mut self) -> &mut [Verification] {
        match &mut self.records {
            Records::Three(v) => v,
            Records::Four(v) => v,
            Records::Five(v) => v,
            Records::Six(v) => v,
            Records::Seven(v) => v,
            Records::Eight(v) => v,
        }
    }
    pub(super) fn ordinal(&self, slot: usize) -> Result<usize> {
        self.records()
            .iter()
            .position(|r| r.slot == slot)
            .context("physical slot is not an active owner-batch producer member")
    }
    pub(super) fn all_detached(&self) -> bool {
        self.detached[..self.records().len()].iter().all(|&v| v)
    }
    pub(super) fn all_committed(&self) -> bool {
        self.committed[..self.records().len()].iter().all(|&v| v)
    }
}

impl Pool {
    fn owner_candidate(
        &self,
        shape: GlmOwnerBatchShape,
        inputs: &[crate::model::GlmPairedInput<'_>],
        tokens: &[[u32; 5]],
        states: &[&dyn ProposerState],
        ctx: &ForwardContext,
    ) -> Result<OwnerVerification> {
        let n = shape.owners();
        ensure!(
            inputs.len() == n && tokens.len() == n && states.len() == n,
            "owner candidate input/token/state count mismatch"
        );
        ensure!(
            n <= self.capacity.owners() && !ctx.graph_capture,
            "owner candidate exceeds capacity or is capturing"
        );
        let record = |i: usize| {
            let state = states[i]
                .as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .context("owner candidate state type")?;
            self.verify_candidate(&inputs[i], &tokens[i], state, ctx)
        };
        let records = match n {
            3 => Records::Three([record(0)?, record(1)?, record(2)?]),
            4 => Records::Four([record(0)?, record(1)?, record(2)?, record(3)?]),
            5 => Records::Five([record(0)?, record(1)?, record(2)?, record(3)?, record(4)?]),
            6 => Records::Six([
                record(0)?,
                record(1)?,
                record(2)?,
                record(3)?,
                record(4)?,
                record(5)?,
            ]),
            7 => Records::Seven([
                record(0)?,
                record(1)?,
                record(2)?,
                record(3)?,
                record(4)?,
                record(5)?,
                record(6)?,
            ]),
            8 => Records::Eight([
                record(0)?,
                record(1)?,
                record(2)?,
                record(3)?,
                record(4)?,
                record(5)?,
                record(6)?,
                record(7)?,
            ]),
            _ => anyhow::bail!("owner candidate shape changed"),
        };
        let normalized_bytes = ctx.buffers.sizes().norm_output;
        ensure!(
            normalized_bytes >= shape.rows() * ROW_BYTES,
            "owner normalized arena lacks packed rows"
        );
        kv_rows_plan::DeviceSpan {
            ptr: ctx.buffers.norm_output(),
            bytes: normalized_bytes,
        }
        .end()?;
        let result = OwnerVerification {
            records,
            normalized_bytes,
            detached: [false; 8],
            committed: [false; 8],
        };
        ensure!(
            result.records().windows(2).all(|v| v[0].slot < v[1].slot),
            "owner producer requires distinct increasing physical slots"
        );
        Ok(result)
    }

    pub(super) fn owners_owner(&self, ctx: &ForwardContext) -> Result<&OwnerVerification> {
        ensure!(!self.producer_failed, "owner producer terminally failed");
        let Some(Producer::Owners(group)) = &self.verification else {
            anyhow::bail!("actual owner-batch producer receipt missing");
        };
        ensure!(
            !ctx.graph_capture && group.normalized_bytes == ctx.buffers.sizes().norm_output,
            "owner producer normalized arena/capture changed"
        );
        ensure!(
            group.records().len() <= self.capacity.owners()
                && group.records().windows(2).all(|v| v[0].slot < v[1].slot),
            "owner producer physical order/capacity changed"
        );
        for (ordinal, record) in group.records().iter().enumerate() {
            ensure!(
                self.capacity.contains(record.slot),
                "owner producer physical slot exceeds capacity"
            );
            let slot = &self.slots[record.slot];
            ensure!(
                record.generation == slot.generation
                    && slot.active
                    && !slot.failed
                    && !slot.retiring
                    && record.normalized == ctx.buffers.norm_output()
                    && slot.issued
                        == if group.detached[ordinal] {
                            None
                        } else {
                            Some(record.issued)
                        },
                "owner producer generation/attempt/original arena changed"
            );
        }
        Ok(group)
    }
}

impl Glm5MtpHead {
    fn owner_state<'s>(
        &self,
        state: &'s dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<&'s Glm5MtpProposerState> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("owner producer state type")?;
        self.validate_paired_live(state, ctx.gpu)?;
        Ok(state)
    }

    pub(super) fn paired_validate_verify_owners(
        &self,
        shape: GlmOwnerBatchShape,
        inputs: &[crate::model::GlmPairedInput<'_>],
        tokens: &[[u32; 5]],
        states: &[&dyn ProposerState],
        ctx: &ForwardContext,
    ) -> Result<[(u64, u64); 8]> {
        ensure!(states.len() == shape.owners(), "owner state count changed");
        for state in states {
            self.owner_state(*state, ctx)?;
        }
        let pool = self
            .paired
            .as_ref()
            .context("owner producer pool missing")?
            .lock();
        let candidate = pool.owner_candidate(shape, inputs, tokens, states, ctx)?;
        let mut facts = [(0, 0); 8];
        for (i, record) in candidate.records().iter().enumerate() {
            facts[i] = (record.generation, record.issued.attempt);
        }
        Ok(facts)
    }

    pub(super) fn paired_begin_verify_owners(
        &self,
        shape: GlmOwnerBatchShape,
        inputs: &[crate::model::GlmPairedInput<'_>],
        tokens: &[[u32; 5]],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
    ) -> Result<()> {
        ensure!(states.len() == shape.owners(), "owner state count changed");
        // Only the active prefix is passed. These are borrowed references, not
        // new owners or allocations; every actual state is independently checked.
        let mut refs: [&dyn ProposerState; 8] = [&*states[0]; 8];
        for (i, state) in states.iter().enumerate() {
            self.owner_state(&**state, ctx)?;
            refs[i] = &**state;
        }
        let mut pool = self
            .paired
            .as_ref()
            .context("owner producer pool missing")?
            .lock();
        let candidate =
            pool.owner_candidate(shape, inputs, tokens, &refs[..shape.owners()], ctx)?;
        pool.verification = Some(Producer::Owners(candidate));
        Ok(())
    }

    pub(super) fn paired_publish_verify_owners(
        &self,
        shape: GlmOwnerBatchShape,
        inputs: &[crate::model::GlmPairedInput<'_>],
        tokens: &[[u32; 5]],
        predictions: &[[u32; 5]],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
    ) -> Result<()> {
        let n = shape.owners();
        ensure!(
            inputs.len() == n && tokens.len() == n && predictions.len() == n && states.len() == n,
            "owner publish input/token/prediction/state count mismatch"
        );
        for state in states.iter() {
            self.owner_state(&**state, ctx)?;
        }
        let mut pool = self
            .paired
            .as_ref()
            .context("owner producer pool missing")?
            .lock();
        let group = pool.owners_owner(ctx)?;
        ensure!(
            group.records().len() == n,
            "owner publish shape differs from actual receipt"
        );
        for (i, record) in group.records().iter().enumerate() {
            let state = states[i]
                .as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .expect("validated owner state");
            ensure!(
                pool.matches_request(state, &inputs[i], ctx)? == record.slot,
                "owner publish physical order changed"
            );
            let end = record
                .issued
                .base
                .checked_add(5)
                .context("owner verify end overflow")?;
            let data = inputs[i].data();
            ensure!(
                !record.produced
                    && !group.detached[i]
                    && tokens[i] == record.issued.tokens
                    && data.position == end
                    && data.tokens.get(record.issued.base..end) == Some(tokens[i].as_slice())
                    && data.tokens.get(..record.issued.base)
                        == Some(pool.slots[record.slot].issued_prefix.as_slice())
                    && predictions[i]
                        .iter()
                        .all(|&p| (p as usize) < ctx.config.vocab_size),
                "owner actual producer return mismatch"
            );
        }
        if let Some(Producer::Owners(group)) = pool.verification.as_mut() {
            for record in group.records_mut() {
                record.produced = true;
            }
        }
        Ok(())
    }
}
