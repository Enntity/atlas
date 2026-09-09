// SPDX-License-Identifier: AGPL-3.0-only
//! Selected cold producer. Its caller retains the request owner across errors.
use super::{PrefillInProgress, sample_step, sched_ctx::SchedCtx};
use anyhow::Result;
use spark_model::traits::Model;

pub(super) fn cold(
    model: &dyn Model,
    prefill: &mut PrefillInProgress,
    sched: &SchedCtx,
) -> Result<u32> {
    let capability = model
        .glm_paired_execution()
        .ok_or_else(|| anyhow::anyhow!("selected cold Model capability missing"))?;
    let logits = capability.cold_prefill(&mut prefill.seq, &prefill.prompt_tokens)?;
    // The existing grammarless first-token path already propagates every
    // argmax/read error. Keep its neutral penalties, EOS mask and tie behavior.
    let first = sample_step::sample_first_token(
        model,
        logits,
        prefill.temperature,
        prefill.top_k,
        prefill.top_p,
        prefill.min_p,
        &prefill.eos_tokens,
        None,
        &sched.levers.sampling(),
    )?;
    anyhow::ensure!(
        (first as usize) < model.vocab_size(),
        "invalid selected first token"
    );
    capability.check_communication_health()?;
    Ok(first)
}

pub(super) fn promote(
    p: PrefillInProgress,
    first: u32,
    tokens: &super::glm_c2_selected::Tokens,
    ring_slots: usize,
) -> super::ActiveSeq {
    let spontaneous = !p.enable_thinking && tokens.think_start == Some(first);
    let legacy_tool = p.require_tool_call && tokens.tool_start.is_some();
    let immediate =
        p.max_tokens == 0 || (!spontaneous && (p.eos_tokens.contains(&first) || p.max_tokens <= 1));
    let mut active = super::phase_promote_prefills::build_active_seq_from_prefill(
        p,
        first,
        spontaneous,
        legacy_tool,
        0,
        immediate,
        std::time::Instant::now(),
        tokens.think_end,
        tokens.think_start,
        tokens.tool_start,
        tokens.tool_end,
        ring_slots,
    );
    // This first token was selected before ActiveSeq existed and is published
    // directly by admission. Mirror only its tool bookkeeping; do not emit or
    // charge the token twice, or treat a reasoning token as a tool opener.
    if !active.output_tokens.is_empty() && !active.inside_thinking {
        if active.require_tool_call && active.tool_call_start_token == Some(first) {
            active.require_tool_call = false;
            active.tool_call_opened = true;
        }
        super::emit_step::update_tool_param_state(&mut active, first);
        if active.tool_call_end_token == Some(first) {
            active.tool_call_completed = true;
        }
    }
    active
}

#[cfg(test)]
#[path = "glm_c2_selected_prefill_tests.rs"]
mod tests;
