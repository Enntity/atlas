// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed paired verification: both selections/commits precede either next E1.
#[cfg(test)]
#[path = "glm_c2_pair_step_tests.rs"]
mod tests;
use super::{ActiveSeq, emit_step, glm_c2_serial, verify_pipeline_helper};
use super::{logit_processors::LogitsContext, sched_ctx::SchedCtx};
use anyhow::{Context, Result, ensure};
use spark_model::speculative::glm_paired_execution::GlmPairedExecution;
use spark_model::traits::Model;

pub(super) fn try_step_pair(
    model: &dyn Model,
    capability: &dyn GlmPairedExecution,
    active: &mut [ActiveSeq],
    owners: [Option<usize>; 2],
    sched: &SchedCtx,
    verify_ctx: &LogitsContext,
) -> Result<bool> {
    let [Some(i0), Some(i1)] = owners else {
        return Ok(false);
    };
    ensure!(
        i0 != i1 && i0 < active.len() && i1 < active.len(),
        "paired scheduler owner mapping"
    );
    // Canonical physical owner order, independent of the scheduler vector order.
    let (a0, a1) = if i0 < i1 {
        let (left, right) = active.split_at_mut(i1);
        (&mut left[i0], &mut right[0])
    } else {
        let (left, right) = active.split_at_mut(i0);
        (&mut right[0], &mut left[i1])
    };
    if glm_c2_serial::stopped(a0, sched, true)
        || glm_c2_serial::stopped(a1, sched, true)
        || a0.pending_drafts.is_empty()
        || a1.pending_drafts.is_empty()
    {
        return Ok(false); // Only before pair claim/header; cold or draining remains serial.
    }
    let tokens = [
        glm_c2_serial::issued(a0, model.vocab_size())?,
        glm_c2_serial::issued(a1, model.vocab_size())?,
    ];
    let bases = [a0.seq.seq_len, a1.seq.seq_len];
    capability.validate_verify_pair([&a0.seq, &a1.seq], &tokens)?;
    capability.check_communication_health()?;
    let raw = capability.verify_pair([&mut a0.seq, &mut a1.seq], &tokens)?;
    for (owner, a) in [&*a0, &*a1].into_iter().enumerate() {
        let end = bases[owner]
            .checked_add(5)
            .context("paired scheduler position overflow")?;
        ensure!(
            a.seq.seq_len == end
                && a.seq.tokens.len() == end
                && a.seq.tokens.get(bases[owner]..) == Some(tokens[owner].as_slice())
                && raw[owner]
                    .iter()
                    .all(|&t| (t as usize) < model.vocab_size()),
            "paired scheduler actual target append/result mismatch"
        );
    }
    let selected = [
        verify_pipeline_helper::verify_pick_all_with_pipeline_checked(
            model, &raw[0], a0, verify_ctx, 0,
        )?,
        verify_pipeline_helper::verify_pick_all_with_pipeline_checked(
            model, &raw[1], a1, verify_ctx, 5,
        )?,
    ];
    ensure!(
        selected
            .iter()
            .all(|rows| rows.len() == 5 && rows.iter().all(|&t| (t as usize) < model.vocab_size())),
        "paired scheduler checked selection malformed"
    );
    let accepted: [usize; 2] = std::array::from_fn(|owner| {
        let matched = (0..4)
            .take_while(|&row| tokens[owner][row + 1] == selected[owner][row])
            .count();
        glm_c2_serial::accepted_before_boundary(
            [&*a0, &*a1][owner],
            &selected[owner],
            matched,
            verify_ctx.glm_tool_boundary,
        )
    });
    capability.finish_verify_pair([&mut a0.seq, &mut a1.seq], &tokens, accepted)?;
    sched.stats.glm_c2.pair_committed(accepted);
    a0.pending_drafts.clear();
    a1.pending_drafts.clear();
    // Both selected vectors and both commits are complete. E1 for owner0 may
    // now reuse shared logits/norm without changing owner1's pending selection.
    for (owner, a) in [a0, a1].into_iter().enumerate() {
        for row in 0..=accepted[owner] {
            if glm_c2_serial::stopped(a, sched, false) {
                break;
            }
            let token = if row < accepted[owner] {
                tokens[owner][row + 1]
            } else {
                selected[owner][accepted[owner]]
            };
            capability.check_communication_health()?;
            emit_step::emit_token_at_position(a, token, None, sched, bases[owner] + row + 1);
            a.last_token = token;
        }
        glm_c2_serial::propose(capability, model, a, sched)?;
    }
    Ok(true)
}
