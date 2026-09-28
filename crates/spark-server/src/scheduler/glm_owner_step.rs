// SPDX-License-Identifier: AGPL-3.0-only
//! Wider cohort dispatch ahead of the preserved pair/scalar drain path.
use super::{
    ActiveSeq, emit_step, glm_c2_serial, logit_processors::LogitsContext, sched_ctx::SchedCtx,
    verify_pipeline_helper,
};
use anyhow::{Context, Result, ensure};
use spark_model::layer::glm_owner_verify::GlmOwnerBatchShape;
use spark_model::{speculative::glm_paired_execution::GlmPairedExecution, traits::Model};

pub(super) fn try_step_owners(
    model: &dyn Model,
    capability: &dyn GlmPairedExecution,
    active: &mut [ActiveSeq],
    sched: &SchedCtx,
    verify_ctx: &LogitsContext,
) -> Result<bool> {
    let capacity = capability.owner_capacity()?;
    ensure!(
        (2..=8).contains(&capacity),
        "owner scheduler capacity changed"
    );
    let mut physical: [Option<&mut ActiveSeq>; 8] = std::array::from_fn(|_| None);
    for a in active {
        if glm_c2_serial::stopped(a, sched, true) {
            continue;
        }
        if a.pending_drafts.is_empty() {
            // A cold live owner must still execute this tick. Preserve the
            // complete pair/scalar path rather than skip a partial cohort.
            return Ok(false);
        }
        let slot = a.seq.slot_idx;
        ensure!(
            slot < capacity && physical[slot].is_none(),
            "owner scheduler physical mapping changed"
        );
        physical[slot] = Some(a);
    }
    let mut cohort = std::array::from_fn::<_, 8, _>(|_| None);
    let mut count = 0;
    for (ordinal, owner) in physical.into_iter().flatten().enumerate() {
        cohort[ordinal] = Some(owner);
        count += 1;
    }
    match count {
        3 => step_cohort::<3>(cohort, model, capability, sched, verify_ctx),
        4 => step_cohort::<4>(cohort, model, capability, sched, verify_ctx),
        5 => step_cohort::<5>(cohort, model, capability, sched, verify_ctx),
        6 => step_cohort::<6>(cohort, model, capability, sched, verify_ctx),
        7 => step_cohort::<7>(cohort, model, capability, sched, verify_ctx),
        8 => step_cohort::<8>(cohort, model, capability, sched, verify_ctx),
        _ => Ok(false),
    }
}

fn step_cohort<const N: usize>(
    mut cohort: [Option<&mut ActiveSeq>; 8],
    model: &dyn Model,
    capability: &dyn GlmPairedExecution,
    sched: &SchedCtx,
    verify_ctx: &LogitsContext,
) -> Result<bool> {
    ensure!(
        cohort[..N].iter().all(Option::is_some) && cohort[N..].iter().all(Option::is_none),
        "owner scheduler cohort coverage changed"
    );
    let owners = std::array::from_fn(|i| cohort[i].take().expect("complete distinct cohort"));
    step::<N>(owners, model, capability, sched, verify_ctx)
}

fn step<const N: usize>(
    mut owners: [&mut ActiveSeq; N],
    model: &dyn Model,
    capability: &dyn GlmPairedExecution,
    sched: &SchedCtx,
    verify_ctx: &LogitsContext,
) -> Result<bool> {
    let shape = GlmOwnerBatchShape::new(N)?;
    let bases = owners.each_ref().map(|a| a.seq.seq_len);
    let mut tokens = [[0; 5]; N];
    for (ordinal, a) in owners.iter().enumerate() {
        tokens[ordinal] = glm_c2_serial::issued(a, model.vocab_size())?;
    }
    capability.validate_verify_owners(shape, &owners.each_ref().map(|a| &a.seq), &tokens)?;
    capability.check_communication_health()?;
    let raw =
        capability.verify_owners(shape, &mut owners.each_mut().map(|a| &mut a.seq), &tokens)?;
    ensure!(
        raw[N..].iter().all(|row| *row == [0; 5]),
        "owner scheduler inactive result tail changed"
    );
    for (ordinal, a) in owners.iter().enumerate() {
        let end = bases[ordinal]
            .checked_add(5)
            .context("owner scheduler position overflow")?;
        ensure!(
            a.seq.seq_len == end
                && a.seq.tokens.len() == end
                && a.seq.tokens.get(bases[ordinal]..) == Some(tokens[ordinal].as_slice())
                && raw[ordinal]
                    .iter()
                    .all(|&t| (t as usize) < model.vocab_size()),
            "owner scheduler actual target append/result mismatch"
        );
    }
    let mut selected = [[0; 5]; N];
    for (ordinal, a) in owners.iter_mut().enumerate() {
        let rows = verify_pipeline_helper::verify_pick_all_with_pipeline_checked(
            model,
            &raw[ordinal],
            a,
            verify_ctx,
            ordinal * 5,
        )?;
        ensure!(
            rows.len() == 5 && rows.iter().all(|&t| (t as usize) < model.vocab_size()),
            "owner scheduler checked selection malformed"
        );
        selected[ordinal].copy_from_slice(&rows);
    }
    let accepted: [usize; N] = std::array::from_fn(|ordinal| {
        let matched = (0..4)
            .take_while(|&row| tokens[ordinal][row + 1] == selected[ordinal][row])
            .count();
        glm_c2_serial::accepted_before_boundary(
            owners[ordinal],
            &selected[ordinal],
            matched,
            verify_ctx.glm_tool_boundary,
        )
    });
    capability.finish_verify_owners(
        shape,
        &mut owners.each_mut().map(|a| &mut a.seq),
        &tokens,
        &accepted,
    )?;
    sched.stats.glm_c2.owners_committed(&accepted);
    for a in &mut owners {
        a.pending_drafts.clear();
    }
    // Every selection and every actual detach/commit completed before the
    // first visible token or scratch-reusing E1 for any physical owner.
    for (ordinal, a) in owners.into_iter().enumerate() {
        for row in 0..=accepted[ordinal] {
            if glm_c2_serial::stopped(a, sched, false) {
                break;
            }
            let token = if row < accepted[ordinal] {
                tokens[ordinal][row + 1]
            } else {
                selected[ordinal][accepted[ordinal]]
            };
            capability.check_communication_health()?;
            emit_step::emit_token_at_position(a, token, None, sched, bases[ordinal] + row + 1);
            a.last_token = token;
        }
        glm_c2_serial::propose(capability, model, a, sched)?;
    }
    Ok(true)
}

#[cfg(test)]
#[path = "glm_owner_step_tests.rs"]
mod tests;
