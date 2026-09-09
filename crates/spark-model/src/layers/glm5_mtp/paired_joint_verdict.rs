// SPDX-License-Identifier: AGPL-3.0-only
//! Validate both verdicts before either detach; retain the producer until both commits.
use super::*;

impl Glm5MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::glm5_mtp::paired) fn paired_record_verify_pair(
        &self,
        inputs: &[crate::model::GlmPairedInput<'_>; 2],
        bases: [usize; 2],
        tokens: &[[u32; 5]; 2],
        accepted: [usize; 2],
        states: [&mut dyn ProposerState; 2],
        ctx: &ForwardContext,
    ) -> Result<()> {
        let [a, b] = states;
        let mut states = [
            a.as_any_mut()
                .downcast_mut::<Glm5MtpProposerState>()
                .context("pair verdict state0 missing")?,
            b.as_any_mut()
                .downcast_mut::<Glm5MtpProposerState>()
                .context("pair verdict state1 missing")?,
        ];
        for state in &states {
            self.validate_paired_live(state, ctx.gpu)?;
        }
        let owner = self.paired.as_ref().context("paired pool missing")?;
        let (pair, pending, slab) = {
            let mut pool = owner.lock();
            let pair = *pool.pair_owner(ctx)?;
            ensure!(
                pair.detached == [false; 2] && pair.committed == [false; 2],
                "Pair verdict already detached or committed"
            );
            let mut pending = [states[0].repair, states[1].repair];
            for index in 0..2 {
                ensure!(
                    pool.matches_request(states[index], &inputs[index], ctx)? == index,
                    "pair verdict slot order changed"
                );
                let record = pair.records[index];
                let base = bases[index];
                let count = accepted[index];
                ensure!(count <= 4, "pair accepted count exceeds four");
                let end = base
                    .checked_add(count + 1)
                    .context("pair verdict end overflow")?;
                let data = inputs[index].data();
                ensure!(
                    record.produced
                        && record.issued.base == base
                        && record.issued.tokens == tokens[index]
                        && data.position == end
                        && data.tokens.get(base..end) == Some(&tokens[index][..count + 1])
                        && data.tokens.get(..base)
                            == Some(pool.slots[index].issued_prefix.as_slice()),
                    "Pair verdict differs from actual verified committed prefix"
                );
                pending[index].record(
                    record.generation,
                    record.generation,
                    base,
                    &tokens[index],
                    count,
                    end,
                    states[index].seq_len,
                    5,
                )?;
            }
            // No copy or phase change above: a bad second verdict cannot detach the first.
            for slot in &mut pool.slots {
                slot.writing = true;
            }
            (pair, pending, pool.slab)
        };
        let result = (|| {
            for index in 0..2 {
                // The sealed input still names original row zero. Only this actual
                // producer derives the canonical owner's source row within that arena.
                let source = pair.records[index].normalized.offset(index * 5 * ROW_BYTES);
                Self::paired_detach_rows(
                    ctx,
                    source,
                    slab.offset(index * SLOT_BYTES),
                    accepted[index],
                )?;
                let mut pool = owner.lock();
                pool.matches_request(states[index], &inputs[index], ctx)?;
                pool.pair_owner(ctx)?;
                let slot = &mut pool.slots[index];
                slot.bonus = Some(HiddenView {
                    generation: slot.generation,
                    position: bases[index] + accepted[index],
                    row: 5,
                    rows: 1,
                });
                slot.writing = false;
                slot.issued = None;
                slot.commit_queued = false;
                states[index].repair = pending[index];
                if let Some(Producer::Pair(pair)) = pool.verification.as_mut() {
                    pair.detached[index] = true;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            let mut pool = owner.lock();
            pool.producer_failed = true;
            for index in 0..2 {
                pool.slots[index].failed = true;
                states[index].repair = repair_state::RepairPhase::Failed;
            }
        }
        result
    }
}
