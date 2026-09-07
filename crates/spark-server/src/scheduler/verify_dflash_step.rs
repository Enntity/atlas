// SPDX-License-Identifier: AGPL-3.0-only

//! Width-generic drafted-token verification (DFlash and MTP K>=5).

use super::*;

#[path = "verify_dflash_ledger.rs"]
pub(super) mod ledger;

#[cfg(test)]
#[path = "verify_dflash_repair_tests.rs"]
mod repair_tests;

/// Width-generic γ-token verify with accept-prefix.
///
/// Routes `[last_token, drafts...]` through Atlas's width-generic target
/// verifier and finds the first index where draft ≠ verified argmax. Tokens
/// before the first mismatch are accepted; the target token at the mismatch
/// becomes the bonus token and subsequent drafts are dropped.
///
/// Remaining DFlash-specific work:
///   * Per-position logprobs extraction.
///   * Sliding-window state rollback for sliding-attention layers
///     (Gemma-4-style; not used by Qwen3.6 targets).
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    let ledger_enabled = ledger::enabled_for(&a.seq);
    step_verify_dflash_inner(
        model,
        a,
        sched,
        drafts,
        num_drafts,
        verify_ctx,
        dflash_verify_raw_argmax,
        ledger_enabled,
    );
}

// Keep diagnostic eligibility outside the inference body: tests can exercise
// the actual host-only K5 path without changing process-wide environment.
#[allow(clippy::too_many_arguments)]
fn step_verify_dflash_inner(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
    ledger_enabled: bool,
) {
    let _step_timer = crate::scheduler::mtp_timing::StepTimer::new(&sched.timing, a.seq.seq_len);
    let ledger_position = a.seq.seq_len;

    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // tokens = [last_verified, draft_0, draft_1, ..., draft_{γ-1}]
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);

    // EP rank 1 must execute the same K-row target forward in NCCL lockstep.
    // F5 is width-generic: K, then K tokens, followed after verification by
    // the accepted-draft count. Fixed K=2/3/4 retain their established wire
    // commands and never pass through here.
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF5) {
        tracing::error!("EP broadcast generic verify cmd: {e:#}");
        a.finished = true;
        return;
    }
    if let Err(e) = model.ep_broadcast_cmd(tokens.len() as u32) {
        tracing::error!("EP broadcast generic verify width: {e:#}");
        a.finished = true;
        return;
    }
    if let Err(e) = model.ep_broadcast_tokens(&tokens) {
        tracing::error!("EP broadcast generic verify tokens: {e:#}");
        a.finished = true;
        return;
    }

    // STEP-TIMING (ATLAS_DFLASH_STEP_TIMING=1): split the ~0.88s/step into
    // verify (target M=1+γ forward) vs propose (drafter forward, tail below).
    // The ledger never had this split — it guessed "FFN + double sweep". This
    // measures it. Gated so the hot path pays nothing when the env is unset.
    let step_timing = std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1");
    let t_verify = std::time::Instant::now();
    let verified_argmax = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            a.finished = true;
            return;
        }
    };
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::VerifyForward, t_verify);
    let verify_ms = if step_timing {
        t_verify.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    a.last_token_time = Instant::now();

    // Preserve only the diagnostic's fixed five IDs before the existing
    // selection branch may move the Vec. Disabled/exhausted paths copy none.
    let ledger_capture = if ledger_enabled {
        a.mtp_acct.glm_k5_ledger.prepare(
            true,
            ledger_position,
            a.last_token,
            drafts,
            &verified_argmax,
            model.vocab_size(),
        )
    } else {
        None
    };

    // DFlash drafter proposes on raw argmax; when dflash_verify_raw_argmax is set
    // (process-wide DFlash mode), skip the rep_pen/DRY pipeline so verifier and
    // drafter judge on the SAME (GOLD) basis. For non-DFlash callers (unreachable
    // today since step_verify_dflash is only dispatched at drafts.len()>=4 which
    // only DFlash produces), apply the full pre-sample pipeline as in K=2/3/4.
    let verified = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
        verified_argmax
    } else {
        crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &verified_argmax,
            a,
            verify_ctx,
            0,
        )
    };

    // `decode_verify` already advanced `seq.seq_len` by `tokens.len()` and
    // pushed all γ+1 tokens into `seq.tokens`. The accept-prefix logic below
    // determines how many to keep — the rest must be rolled back so the
    // KV cache, SSM state, and emitted token sequence stay consistent.

    // Accept-prefix: drafts[i] is "accepted" iff drafts[i] == verified[i].
    // verified[i] is the target's argmax at position i (i.e. its
    // prediction for what should follow `tokens[i]`). drafts[i] was the
    // proposer's guess for the same slot. First mismatch terminates the
    // accepted prefix; verified[first_mismatch] becomes the bonus token.
    let mut num_accepted = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    if let Some(record) = a
        .mtp_acct
        .glm_k5_ledger
        .finish(ledger_capture, &verified, num_accepted)
    {
        ledger::emit(a.seq.slot_idx, &record);
    }

    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast generic verify result: {e:#}");
        a.finished = true;
        return;
    }
    crate::scheduler::mtp_accept_debug::record(
        1,
        drafts.len(),
        drafts.first() == verified.first(),
        num_accepted,
    );
    if !dflash_verify_raw_argmax {
        a.mtp_acct.record_depth_verify(
            drafts.len(),
            num_accepted,
            sched.levers.mtp_single_depth_adapt,
        );
    }

    // Adaptive speculation (ATLAS_DFLASH_ADAPTIVE=1): feed the rolling
    // accept window; may suspend this seq's speculation (see adaptive_spec).
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);

    // Roll back the over-extended `seq_len` and `seq.tokens`. The verify
    // advanced both by `tokens.len() = γ+1` (all γ drafts + the prefix
    // bonus slot). We keep the original prefix + `num_accepted` drafts +
    // 1 bonus position. So the post-rollback target is
    // `pre_verify_len + num_accepted + 1` — note we do NOT push the bonus
    // again via emit_token's path (emit_token only updates the user-facing
    // output buffer, not seq.tokens), so the bonus stays in seq.tokens
    // exactly where decode_verify put it.
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens.len());
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    if let Err(e) = model.record_glm_mtp_verified(&mut a.seq, pre_verify_len, &tokens, num_accepted)
    {
        tracing::error!("GLM verified-pair record: {e:#}");
        a.finished = true;
        return;
    }

    // EAGLE-fix (ATLAS_DFLASH_EAGLE_FIX=1): append one ctx slot per committed
    // position (rows 0..=num_accepted at N..=N+num_accepted), with the bonus
    // generator (row num_accepted) freshest. Fixes the ctx-undercount (was 1
    // slot/step regardless of num_accepted) and the EAGLE conditioning shift.
    // Sets skip_next_decode_append so the propose below does NOT re-append row 0.
    // Unified ctx commit (ATLAS_DFLASH_UNIFIED_CTX=1): ONE unconditional
    // commit at the K=gamma point — rows 0..=num_accepted at RoPE base
    // pre_verify_len. Structural replacement for dflash_eagle_kgamma_append.
    if sched.levers.dflash_unified_ctx {
        if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len) {
            tracing::error!("commit_ctx (kgamma): {e:#}");
        }
    } else {
        let eagle_fix = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
        if eagle_fix
            && let Err(e) =
                model.dflash_eagle_kgamma_append(&mut a.seq, num_accepted, pre_verify_len)
        {
            tracing::error!("dflash_eagle_kgamma_append: {e:#}");
        }
    }

    // Emit accepted drafts.
    for i in 0..num_accepted {
        emit_token(a, drafts[i], None, sched);
        if a.finished {
            return;
        }
    }

    // Bonus token = verified[num_accepted] (the one that "corrected" the draft
    // at the first mismatch, or the next-prediction past the full-accept case).
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_token(a, bonus, None, sched);
        if a.finished {
            return;
        }
        a.last_token = bonus;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            if dflash_verify_raw_argmax {
                "dflash"
            } else {
                "mtp"
            },
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();

    tracing::debug!(
        "K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={}",
        drafts.len(),
        num_accepted,
        drafts.len(),
        100.0 * (num_accepted as f64) / (drafts.len() as f64),
        a.seq.seq_len,
    );

    // Item #2 (STree-style in-place verify commit). h_state is canonical:
    //  - num_accepted == k_verify (full accept): no-op (h_state already correct)
    //  - 0 < num_accepted < k_verify (partial): intermediate[total_accepted-1] → h_state
    // No checkpoint write needed — the next start_checkpoint_async syncs.
    //
    // k_verify = drafts.len() + 1 (the prefix bonus position is also verified).
    let k_verify = drafts.len() + 1;
    let total_accepted = num_accepted + 1; // bonus is always "accepted"
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, total_accepted, k_verify) {
        tracing::error!("commit_accepted_prefix (dflash): {e:#}");
        a.finished = true;
        return;
    }

    // DFlash hidden is captured per-layer inside the verify graph
    // (verify_d.rs try_dflash_capture at position k-1), mirroring verify_b.rs.
    // No post-loop save needed; calling save_dflash_hidden_for_propose here
    // would overwrite the correct per-layer intermediates with a repeated
    // final-layer hidden, collapsing all 5 slots to the same value.
    let bonus_token_idx = total_accepted.saturating_sub(1);
    if let Err(e) = model.save_hidden_for_mtp(bonus_token_idx, 0) {
        tracing::error!("save_hidden_for_mtp (dflash): {e:#}");
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
            return;
        }
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
            return;
        }
    }

    // Re-propose for next step — unless adaptive speculation just suspended
    // this seq (no drafts → the scheduler serial-decodes it via bootstrap).
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    let t_propose = std::time::Instant::now();
    if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
        let next_num_drafts = if dflash_verify_raw_argmax {
            num_drafts
        } else {
            a.mtp_acct
                .depth_drafts(num_drafts, sched.levers.mtp_single_depth_adapt)
        };
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            next_num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {
                if spark_model::speculative::glm_repair_policy::enabled() {
                    a.finished = true;
                }
            }
            Err(e) => {
                tracing::error!("run_mtp_propose_multi (dflash): {e:#}");
                if spark_model::speculative::glm_repair_policy::enabled() {
                    a.finished = true;
                }
            }
        }
    }
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::Propose, t_propose);
    if step_timing {
        let propose_ms = t_propose.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            "DFLASH STEP_TIMING: verify={:.1}ms propose={:.1}ms (K={}, accepted={})",
            verify_ms,
            propose_ms,
            tokens.len(),
            num_accepted,
        );
    }
}
