// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-only DFlash target + arriving-prompt co-dispatch.
//!
//! Stateful attention remains on each lane's native kernel. The rows join at
//! the stateless FFN boundary so the routed expert weights are swept once.

use super::*;

fn fused_width(remaining_prompt: usize, capacity: usize) -> Option<usize> {
    let width = remaining_prompt.min(capacity);
    (width > 0).then_some(width)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_step_dflash_prefill(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    dflash_verify_raw_argmax: bool,
) -> bool {
    if !model.supports_dflash_prefill_fusion()
        || active.len() != 1
        || prefilling.is_empty()
        || active[0].pending_drafts.is_empty()
        || active[0].finished
        || prefilling[0].seq.collect_prompt_logprobs.is_some()
    {
        return false;
    }

    let p = &prefilling[0];
    if p.chunk_offset != p.seq.seq_len || p.chunk_offset >= p.prompt_tokens.len() {
        tracing::error!(
            "GLM DFlash/prefill state mismatch: chunk_offset={} seq_len={} prompt_len={}",
            p.chunk_offset,
            p.seq.seq_len,
            p.prompt_tokens.len()
        );
        return false;
    }
    let remaining = p.prompt_tokens.len() - p.chunk_offset;
    let k = active[0].pending_drafts.len() + 1;
    let Some(prefill_k) = fused_width(remaining, model.dflash_prefill_capacity(k)) else {
        return false;
    };
    if !(2..=spark_runtime::buffers::GLM53_VERIFY_MAX_ROWS).contains(&k) {
        return false;
    }

    if let Err(error) = model.sync_secondary() {
        tracing::error!("GLM DFlash/prefill sync_secondary: {error:#}");
        active[0].finished = true;
        completed_indices.push((0, None));
        return true;
    }

    let drafts = std::mem::take(&mut active[0].pending_drafts);
    active[0].pending_draft_conf.clear();
    let offered = &drafts[..];
    let mut target_tokens = Vec::with_capacity(k);
    target_tokens.push(active[0].last_token);
    target_tokens.extend_from_slice(offered);

    let chunk_start = prefilling[0].chunk_offset;
    let chunk_end = chunk_start + prefill_k;
    let prompt_block = prefilling[0].prompt_tokens[chunk_start..chunk_end].to_vec();
    if let Err(error) = model.ep_broadcast_dflash_prefill(
        active[0].seq.slot_idx as u32,
        prefilling[0].seq.slot_idx as u32,
        &target_tokens,
        &prompt_block,
        prefilling[0].prompt_tokens.len(),
    ) {
        tracing::error!("broadcast GLM DFlash/prefill: {error:#}");
        active[0].pending_drafts = drafts;
        return false;
    }

    let started = Instant::now();
    let result = match model.decode_verify_dflash_with_prefill(
        &target_tokens,
        &mut active[0].seq,
        &prompt_block,
        &mut prefilling[0].seq,
        prefilling[0].prompt_tokens.len(),
        model.default_stream(),
    ) {
        Ok(result) => result,
        Err(error) => {
            tracing::error!("GLM DFlash/prefill target pass: {error:#}");
            // The worker is waiting for the active lane's verdict. Release it
            // before failing the head request so the rank pair cannot deadlock.
            let _ = model.ep_broadcast_tokens(&[0]);
            active[0].finished = true;
            completed_indices.push((0, None));
            return true;
        }
    };
    let target_ms = started.elapsed().as_secs_f64() * 1000.0;
    prefilling[0].chunk_offset = chunk_end;
    let prompt_finished = chunk_end == prefilling[0].prompt_tokens.len();

    // Sample the arriving request before the active request's proposer reuses
    // the shared logits/scratch arena.
    if prompt_finished {
        match sample_first_token(
            model,
            result.prefill_logits,
            prefilling[0].temperature,
            prefilling[0].top_k,
            prefilling[0].top_p,
            prefilling[0].min_p,
            &prefilling[0].eos_tokens,
            prefilling[0].grammar_state.as_mut(),
            &sched.levers.sampling(),
        ) {
            Ok(first) => completed_indices.push((0, Some(first))),
            Err(error) => {
                tracing::error!("GLM fused prefill first-token sample: {error:#}");
                completed_indices.push((0, None));
            }
        }
    }

    let verify_ctx = crate::scheduler::logit_processors::LogitsContext {
        watchdog: sched.watchdog,
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        boundary_mask: sched.masks.boundary.clone(),
        mid_word_mask: sched.masks.mid_word.clone(),
        sampling: sched.levers.sampling(),
        timing: sched.timing.clone(),
    };
    let verified = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
        result.target_argmax
    } else {
        crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &result.target_argmax,
            &mut active[0],
            &verify_ctx,
            0,
        )
    };
    let num_accepted = offered
        .iter()
        .zip(&verified)
        .take_while(|(draft, target)| draft == target)
        .count();

    if let Err(error) = model.ep_broadcast_tokens(&[num_accepted as u32]) {
        tracing::error!("broadcast GLM DFlash/prefill verdict: {error:#}");
        active[0].finished = true;
        completed_indices.push((0, None));
        return true;
    }

    let a = &mut active[0];
    a.last_token_time = Instant::now();
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);
    let pre_verify_len = a.seq.seq_len.saturating_sub(k);
    let target_seq_len = pre_verify_len + num_accepted + 1;
    a.seq.seq_len = target_seq_len;
    a.seq.tokens.truncate(target_seq_len);
    if let Err(error) = model.commit_ctx_from_row(&mut a.seq, 0, num_accepted + 1, pre_verify_len) {
        tracing::error!("GLM fused commit_ctx_from_row: {error:#}");
        a.finished = true;
    }
    if let Err(error) = model.commit_accepted_prefix(&mut a.seq, num_accepted + 1, k) {
        tracing::error!("GLM fused commit_accepted_prefix: {error:#}");
        a.finished = true;
    }

    if !a.finished {
        for &token in offered.iter().take(num_accepted) {
            emit_token(a, token, None, sched);
            if a.finished {
                break;
            }
        }
    }
    if !a.finished
        && let Some(&bonus) = verified.get(num_accepted)
    {
        emit_token(a, bonus, None, sched);
        a.last_token = bonus;
    }
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash_prefill",
            if num_accepted == offered.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();
    if let Err(error) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("GLM fused trim_proposer_state: {error:#}");
    }

    if !a.finished && crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
        let next_num_drafts =
            crate::scheduler::adaptive_spec::configured_dflash_depth_limit(a, num_drafts);
        if let Err(error) = model.configure_dflash_sampling(&mut a.seq, a.temperature, a.seed) {
            tracing::error!("configure fused DFlash sampling: {error:#}");
        }
        let grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            next_num_drafts,
            &mut a.seq,
            0,
            grammar_mask.as_deref(),
        ) {
            Ok(next) if !next.is_empty() => a.pending_drafts = next,
            Ok(_) => {}
            Err(error) => tracing::error!("GLM fused DFlash re-propose: {error:#}"),
        }
    }

    tracing::info!(
        "GLM DFlash/prefill fused: target_K={} prompt_rows={} prompt={}/{} target_ms={:.1} accepted={}/{}",
        k,
        prefill_k,
        chunk_end,
        prefilling[0].prompt_tokens.len(),
        target_ms,
        num_accepted,
        offered.len(),
    );
    true
}

#[cfg(test)]
mod tests {
    use super::fused_width;

    #[test]
    fn width_is_bounded_by_prompt_and_available_target_block() {
        assert_eq!(fused_width(27, 504), Some(27));
        assert_eq!(fused_width(600, 504), Some(504));
        assert_eq!(fused_width(1, 504), Some(1));
        assert_eq!(fused_width(8, 0), None);
    }
}
