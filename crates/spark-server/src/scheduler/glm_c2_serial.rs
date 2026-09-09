// SPDX-License-Identifier: AGPL-3.0-only
//! Unactivated paired serial control; serving still requires admission and T2/T3.
use super::{ActiveSeq, logit_processors::LogitsContext, sched_ctx::SchedCtx};
use super::{emit_step, helpers, mod_helpers, sample_step, verify_pipeline_helper};
use anyhow::Result;
use spark_model::speculative::glm_paired_execution::GlmPairedExecution;
use spark_model::traits::Model;

/// A complete owner transaction before moving to its peer. No serving caller;
/// any error must enter the future armed T2 boundary, not ordinary retirement.
#[allow(dead_code)]
pub(super) fn step_selected_serial(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &SchedCtx,
    verify_ctx: &LogitsContext,
) -> Result<()> {
    let capability = model
        .glm_paired_execution()
        .ok_or_else(|| anyhow::anyhow!("paired serial capability unavailable"))?;
    anyhow::ensure!(
        (1..=2).contains(&active.len()),
        "paired serial occupancy must be 1..2"
    );
    let mut owners = [None; 2];
    for (index, a) in active.iter_mut().enumerate() {
        let slot = a.seq.slot_idx;
        anyhow::ensure!(
            slot < 2 && owners[slot].is_none(),
            "paired serial owner slots must be distinct 0/1"
        );
        owners[slot] = Some(index);
        if !a.finished {
            anyhow::ensure!(
                a.seq.seq_len.checked_add(1).is_some() && a.seq.tokens.len() == a.seq.seq_len,
                "paired serial canonical token length or position overflow"
            );
            anyhow::ensure!(!a.disable_mtp, "paired serial request opted out of MTP");
        }
        if stopped(a, sched, true) {
            continue;
        }
        anyhow::ensure!(
            a.temperature == 0.0 && a.grammar_state.is_none() && a.top_logprobs.is_none(),
            "paired serial requires greedy grammarless sampling without logprobs"
        );
        anyhow::ensure!(
            a.last_token < model.vocab_size() as u32,
            "paired serial host token identity mismatch"
        );
        if a.pending_drafts.is_empty() {
            anyhow::ensure!(
                a.seq.seq_len == a.seq.prompt_len,
                "paired serial missing steady-state drafts"
            );
            capability.validate_bootstrap(&a.seq, a.last_token)?;
        } else {
            let issued = issued(a, model.vocab_size())?;
            capability.validate_verify(&a.seq, &issued)?;
        }
    }
    for index in owners.into_iter().flatten() {
        let a = &mut active[index];
        if stopped(a, sched, true) {
            continue;
        }
        if a.pending_drafts.is_empty() {
            bootstrap(model, capability, a, sched)?;
        } else {
            verdict(model, capability, a, sched, verify_ctx)?;
        }
    }
    Ok(())
}

fn stopped(a: &mut ActiveSeq, sched: &SchedCtx, ceiling: bool) -> bool {
    mod_helpers::enforce_request_deadlines(std::slice::from_mut(a));
    if a.finished || emit_step::retire_if_cancelled(a) {
        return true;
    }
    if a.remaining == 0
        || (ceiling && helpers::seqlen_force_stop(a.seq.seq_len, sched.limits.max_seq_len))
    {
        a.finished = true;
    }
    a.finished
}

fn issued(a: &ActiveSeq, vocab: usize) -> Result<[u32; 5]> {
    anyhow::ensure!(
        a.pending_drafts.len() == 4 && a.pending_drafts.iter().all(|t| (*t as usize) < vocab),
        "paired serial needs exactly four valid drafts"
    );
    Ok([
        a.last_token,
        a.pending_drafts[0],
        a.pending_drafts[1],
        a.pending_drafts[2],
        a.pending_drafts[3],
    ])
}

fn propose(
    capability: &dyn GlmPairedExecution,
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &SchedCtx,
) -> Result<()> {
    if stopped(a, sched, true) {
        return Ok(());
    }
    let position = a.seq.seq_len;
    capability.validate_propose(&a.seq, a.last_token, position, 4, None)?;
    let drafts = capability.propose(&mut a.seq, a.last_token, position, 4, None)?;
    anyhow::ensure!(
        drafts.len() == 4 && drafts.iter().all(|t| (*t as usize) < model.vocab_size()),
        "paired serial proposal must return four valid drafts"
    );
    a.pending_drafts = drafts;
    Ok(())
}

fn bootstrap(
    model: &dyn Model,
    capability: &dyn GlmPairedExecution,
    a: &mut ActiveSeq,
    sched: &SchedCtx,
) -> Result<()> {
    let logits = capability.bootstrap(&mut a.seq, a.last_token)?;
    let penalties = sample_step::penalty_params_for(
        a,
        sample_step::PositionKind::Verify,
        0.0,
        None,
        Vec::new(),
    );
    let history = sample_step::penalty_history_scope(&a.output_tokens, a.tool_call_end_token);
    let token = sample_step::sample_token_with_grammar_checked(
        model,
        logits,
        0.0,
        a.top_k,
        a.top_p,
        &[],
        None,
        &penalties,
        history,
        &sched.levers.sampling(),
    )?;
    anyhow::ensure!(
        (token as usize) < model.vocab_size(),
        "paired serial invalid selected seed"
    );
    if stopped(a, sched, false) {
        return Ok(());
    }
    emit_step::emit_token_at_position(a, token, None, sched, a.seq.seq_len);
    a.last_token = token;
    propose(capability, model, a, sched)
}

fn verdict(
    model: &dyn Model,
    capability: &dyn GlmPairedExecution,
    a: &mut ActiveSeq,
    sched: &SchedCtx,
    verify_ctx: &LogitsContext,
) -> Result<()> {
    let tokens = issued(a, model.vocab_size())?;
    let base = a.seq.seq_len;
    let end = base
        .checked_add(5)
        .ok_or_else(|| anyhow::anyhow!("paired serial position overflow"))?;
    let raw = capability.verify(&mut a.seq, &tokens)?;
    anyhow::ensure!(
        raw.len() == 5
            && raw.iter().all(|t| (*t as usize) < model.vocab_size())
            && a.seq.seq_len == end
            && a.seq.tokens.len() == end
            && a.seq.tokens[base..] == tokens,
        "paired serial invalid target verification append"
    );
    let selected = verify_pipeline_helper::verify_pick_all_with_pipeline_checked(
        model, &raw, a, verify_ctx, 0,
    )?;
    anyhow::ensure!(
        selected.len() == 5 && selected.iter().all(|t| (*t as usize) < model.vocab_size()),
        "paired serial invalid selected verdict"
    );
    let accepted = (0..4).take_while(|i| tokens[i + 1] == selected[*i]).count();
    model.ep_broadcast_cmd(accepted as u32)?;
    a.seq.seq_len = base + accepted + 1;
    a.seq.tokens.truncate(a.seq.seq_len);
    model.record_glm_mtp_verified(&mut a.seq, base, &tokens, accepted)?;
    model.trim_proposer_state(&mut a.seq, accepted, 0)?;
    model.commit_accepted_prefix(&mut a.seq, accepted + 1, 5)?;
    a.pending_drafts.clear();
    // Every model acknowledgement above precedes even a cancelled/finished emit.
    for i in 0..=accepted {
        if stopped(a, sched, false) {
            break;
        }
        let token = if i < accepted {
            tokens[i + 1]
        } else {
            selected[accepted]
        };
        emit_step::emit_token_at_position(a, token, None, sched, base + i + 1);
        a.last_token = token;
    }
    propose(capability, model, a, sched)
}

#[cfg(test)]
#[path = "glm_c2_serial_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm_c2_serial_round_tests.rs"]
mod round_tests;
