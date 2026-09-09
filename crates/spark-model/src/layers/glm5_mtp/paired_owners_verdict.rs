// SPDX-License-Identifier: AGPL-3.0-only
//! All verdicts validate before the first detach; receipt lives through all commits.
use super::*;

impl Glm5MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::glm5_mtp::paired) fn paired_record_verify_owners(
        &self,
        shape: GlmOwnerBatchShape,
        inputs: &[crate::model::GlmPairedInput<'_>],
        bases: &[usize],
        tokens: &[[u32; 5]],
        accepted: &[usize],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
    ) -> Result<()> {
        let n = shape.owners();
        ensure!(
            inputs.len() == n
                && bases.len() == n
                && tokens.len() == n
                && accepted.len() == n
                && states.len() == n,
            "owner verdict slice count mismatch"
        );
        for state in states.iter() {
            self.owner_state(&**state, ctx)?;
        }
        let owner = self.paired.as_ref().context("owner verdict pool missing")?;
        let (group, pending, slab) = {
            let mut pool = owner.lock();
            let group = *pool.owners_owner(ctx)?;
            ensure!(
                group.records().len() == n
                    && group.detached == [false; 4]
                    && group.committed == [false; 4],
                "owner verdict shape changed or already detached/committed"
            );
            let mut pending = [repair_state::RepairPhase::Failed; 4];
            for (i, record) in group.records().iter().enumerate() {
                let state = states[i]
                    .as_any()
                    .downcast_ref::<Glm5MtpProposerState>()
                    .expect("validated owner state");
                ensure!(
                    pool.matches_request(state, &inputs[i], ctx)? == record.slot,
                    "owner verdict physical order changed"
                );
                let count = accepted[i];
                ensure!(count <= 4, "owner accepted count exceeds four");
                let end = bases[i]
                    .checked_add(count + 1)
                    .context("owner verdict end overflow")?;
                let data = inputs[i].data();
                ensure!(
                    record.produced
                        && record.issued.base == bases[i]
                        && record.issued.tokens == tokens[i]
                        && data.position == end
                        && data.tokens.get(bases[i]..end) == Some(&tokens[i][..count + 1])
                        && data.tokens.get(..bases[i])
                            == Some(pool.slots[record.slot].issued_prefix.as_slice()),
                    "owner verdict differs from actual verified committed prefix"
                );
                pending[i] = state.repair;
                pending[i].record(
                    record.generation,
                    record.generation,
                    bases[i],
                    &tokens[i],
                    count,
                    end,
                    state.seq_len,
                    5,
                )?;
            }
            // Pending phases are copies until ALL actual owner verdicts pass.
            for record in group.records() {
                pool.slots[record.slot].writing = true;
            }
            (group, pending, pool.slab)
        };
        let result = (|| {
            for (i, record) in group.records().iter().enumerate() {
                // Dense ordinal selects packed normalized rows; immutable physical
                // slot selects private storage, including a noncontiguous3 cohort.
                Self::paired_detach_rows(
                    ctx,
                    record.normalized.offset(i * 5 * ROW_BYTES),
                    slab.offset(record.slot * SLOT_BYTES),
                    accepted[i],
                )?;
                let state = states[i]
                    .as_any_mut()
                    .downcast_mut::<Glm5MtpProposerState>()
                    .expect("validated owner state");
                let mut pool = owner.lock();
                pool.matches_request(state, &inputs[i], ctx)?;
                pool.owners_owner(ctx)?;
                let slot = &mut pool.slots[record.slot];
                slot.bonus = Some(HiddenView {
                    generation: slot.generation,
                    position: bases[i] + accepted[i],
                    row: 5,
                    rows: 1,
                });
                slot.writing = false;
                slot.issued = None;
                slot.commit_queued = false;
                state.repair = pending[i];
                if let Some(Producer::Owners(group)) = pool.verification.as_mut() {
                    group.detached[i] = true;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            let mut pool = owner.lock();
            pool.producer_failed = true;
            for (i, record) in group.records().iter().enumerate() {
                pool.slots[record.slot].failed = true;
                states[i]
                    .as_any_mut()
                    .downcast_mut::<Glm5MtpProposerState>()
                    .expect("validated owner state")
                    .repair = repair_state::RepairPhase::Failed;
            }
        }
        result
    }
}
