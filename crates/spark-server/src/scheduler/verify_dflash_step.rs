// SPDX-License-Identifier: AGPL-3.0-only

//! Width-generic drafted-token verification (DFlash, MTP K>=5, and repaired GLM K2/K3).

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
        a.engine_error = Some(format!("{e:#}"));
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
    // commands, except repaired GLM K2/K3 which needs the explicit verdict hook.
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
            a.engine_error = Some(format!("{e:#}"));
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
    verify_dflash_tail(
        model,
        a,
        sched,
        drafts,
        num_drafts,
        verify_ctx,
        dflash_verify_raw_argmax,
        ledger_enabled,
        ledger_position,
        &tokens,
        verified_argmax,
        step_timing,
        verify_ms,
        false,
    );
}

/// Everything after the target forward: verdict, EP verdict word, rollback,
/// repair record, emission, commit and re-propose for ONE sequence. Shared by
/// the per-sequence verify and each owner of the owner-batched GLM verify.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_dflash_tail(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
    ledger_enabled: bool,
    ledger_position: usize,
    tokens: &[u32],
    verified_argmax: Vec<u32>,
    step_timing: bool,
    verify_ms: f64,
    defer_propose: bool,
) -> Option<usize> {
    let raw_trace = if std::env::var("ATLAS_LIGHTNING_VERIFY_TOKEN_TRACE").as_deref() == Ok("1") {
        Some(verified_argmax.clone())
    } else {
        None
    };

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
    let verified = if crate::scheduler::helpers::dflash_seq_uses_raw_argmax(
        dflash_verify_raw_argmax,
        sched.levers.dflash_masked_verify,
        model.is_lightning_dspark_product(),
        a,
    ) {
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
    if let Some(raw) = raw_trace {
        tracing::info!(
            "LIGHTNING VERIFY TOKEN TRACE slot={} last={} drafts={:?} raw={:?} processed={:?} accepted={}",
            a.seq.slot_idx,
            a.last_token,
            drafts,
            raw,
            verified,
            num_accepted
        );
    }
    if std::env::var("ATLAS_DFLASH_VERIFY_TRACE").ok().as_deref() == Some("1") {
        let n = drafts.len().min(verified.len()).min(4);
        tracing::info!(
            "DFLASH CMP last={} drafts[0..{n}]={:?} verified[0..{n}]={:?} accepted={}",
            a.last_token,
            &drafts[..n],
            &verified[..n],
            num_accepted,
        );
    }

    // Logprobs for the tokens this step emits: the accepted prefix plus the
    // bonus, which is exactly `verified[0..=num_accepted]` because an accepted
    // draft is by definition equal to `verified` at that index. Extracted here,
    // before `commit_ctx` reuses the logits buffer — the same ordering the
    // verify_k2/k3/k4 paths use.
    //
    // Without this the DFlash path emitted every token with `logprobs: None`,
    // so a request that asked for logprobs got the OpenAI four-array shape back
    // filled entirely with nulls: structurally valid, informationally empty,
    // and indistinguishable from "this model has no logprobs". The MTP verify
    // paths have always extracted them; only this one did not. Gated on the
    // request, so an ordinary run copies no logits and pays nothing.
    let verify_lps = if let Some(k_logprobs) = a.top_logprobs {
        let upto = (num_accepted + 1).min(verified.len());
        extract_verify_logprobs(model, &verified[..upto], k_logprobs, 0)
    } else {
        Vec::new()
    };

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
        return None;
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
    a.spec_adapt.survival.record(drafts.len(), num_accepted);

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
        return None;
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
        emit_token(a, drafts[i], verify_lps.get(i).cloned(), sched);
        if a.finished {
            return None;
        }
    }

    // Bonus token = verified[num_accepted] (the one that "corrected" the draft
    // at the first mismatch, or the next-prediction past the full-accept case).
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_token(a, bonus, verify_lps.get(bonus_idx).cloned(), sched);
        if a.finished {
            return None;
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
        a.engine_error = Some(format!("{e:#}"));
        a.finished = true;
        return None;
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
            return None;
        }
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
            return None;
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
        // Owner-batched callers propose every owner in one drafter pass.
        if defer_propose && _mtp_grammar_mask.is_none() {
            return Some(next_num_drafts);
        }
        let proposal: anyhow::Result<Vec<u32>> =
            if model.mtp_propose_batch_min() == 1 && _mtp_grammar_mask.is_none() {
                let one_token = [a.last_token];
                let one_position = [a.seq.seq_len];
                let one_stash = [bonus_token_idx];
                let mut one_seq = [&mut a.seq];
                match model.run_mtp_propose_batched(
                    &one_token,
                    &one_position,
                    &one_stash,
                    next_num_drafts,
                    &mut one_seq,
                    0,
                    None,
                    // B1 arm is entered only when this seq's mask is None.
                    None,
                ) {
                    Ok(Some(mut all)) if all.len() == 1 => Ok(all.remove(0)),
                    Ok(Some(all)) => Err(anyhow::anyhow!(
                        "DFlash B1 parity proposer returned {} sequence rows",
                        all.len()
                    )),
                    Ok(None) => Err(anyhow::anyhow!("DFlash B1 parity proposer declined")),
                    Err(error) => Err(error),
                }
            } else {
                model.run_mtp_propose_multi(
                    a.last_token,
                    a.seq.seq_len,
                    next_num_drafts,
                    &mut a.seq,
                    0,
                    _mtp_grammar_mask.as_deref(),
                )
            };
        match proposal {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {
                // Lightning product fail-closed boundary: an empty
                // re-propose is an admission violation, not a silent
                // serial-decode fallback on the next bootstrap.
                crate::scheduler::helpers::handle_dspark_repropose_failure(
                    model,
                    a,
                    crate::scheduler::helpers::ProposalOutcome::Empty,
                );
                if spark_model::speculative::glm_repair_policy::enabled() {
                    a.finished = true;
                }
            }
            Err(e) => {
                tracing::error!("run_mtp_propose_multi (dflash): {e:#}");
                crate::scheduler::helpers::handle_dspark_repropose_failure(
                    model,
                    a,
                    crate::scheduler::helpers::ProposalOutcome::Error,
                );
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
    None
}

/// Owner-batched GLM verify (repaired K3 or DFlash block): one target
/// traversal for every owner, then each owner's ordinary tail in order, each
/// preceded by restoring that owner's verify rows on both ranks.
pub fn step_verify_glm_long_batched(
    model: &dyn Model,
    group: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    step_verify_glm_long_with(
        model,
        group,
        sched,
        num_drafts,
        verify_ctx,
        dflash_verify_raw_argmax,
        &mut |rows, tokens, seqs| model.decode_verify_glm_long_owner_rows(rows, tokens, seqs),
    );
}

/// [`step_verify_glm_long_batched`] with the owners' target traversal
/// supplied: `traverse(rows, owner-major tokens, seqs)` must leave each owner
/// advanced by `rows` rows and its final rows staged for
/// `begin_glm_long_owner_tail`, returning the owner-major argmax IDs (as
/// `decode_verify_glm_long_owner_rows` does, or a prefill chunk carrying the
/// owners).
#[allow(clippy::too_many_arguments)]
pub fn step_verify_glm_long_with(
    model: &dyn Model,
    group: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
    traverse: &mut dyn FnMut(usize, &[u32], &mut [&mut SequenceState]) -> anyhow::Result<Vec<u32>>,
) {
    let fail_all = |group: &mut [&mut ActiveSeq]| group.iter_mut().for_each(|a| a.finished = true);
    let t_step = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        fail_all(group);
        return;
    }
    let sync_ms = t_step.elapsed().as_secs_f64() * 1000.0;
    let drafts: Vec<Vec<u32>> = group
        .iter_mut()
        .map(|a| {
            a.pending_draft_conf.clear();
            std::mem::take(&mut a.pending_drafts)
        })
        .collect();
    let rows = drafts[0].len() + 1;
    let tokens: Vec<Vec<u32>> = group
        .iter()
        .zip(&drafts)
        .map(|(a, d)| {
            std::iter::once(a.last_token)
                .chain(d.iter().copied())
                .collect()
        })
        .collect();
    let flat: Vec<u32> = tokens.concat();
    let positions: Vec<usize> = group.iter().map(|a| a.seq.seq_len).collect();
    let step_timing = std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1");
    let t_verify = std::time::Instant::now();
    let verified = {
        let mut seqs: Vec<&mut SequenceState> = group.iter_mut().map(|a| &mut a.seq).collect();
        traverse(rows, &flat, &mut seqs)
    };
    let verified = match verified {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("GLM owner traversal (n={} rows={rows}): {e:#}", group.len());
            fail_all(group);
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
    // DFlash captures each owner's verify rows into its stable hidden-save
    // slot. A tail's commit_ctx reads the front slot, so each owner's region is
    // packed there first; slot 0 IS the front, so its owner runs last, after
    // the preserved front is restored.
    let n = group.len();
    let save_slots: Vec<Option<usize>> = group
        .iter()
        .map(|a| {
            dflash_verify_raw_argmax
                .then(|| a.seq.dflash_hidden_save_slot().ok())
                .flatten()
        })
        .collect();
    let front_owner = (0..n).find(|&o| save_slots[o] == Some(0));
    let order: Vec<usize> = (0..n)
        .filter(|&o| Some(o) != front_owner)
        .chain(front_owner)
        .collect();
    if front_owner.is_some()
        && let Err(e) = model.preserve_dflash_save_front(rows, 0)
    {
        tracing::error!("preserve_dflash_save_front: {e:#}");
        fail_all(group);
        return;
    }
    let mut deferred: Vec<(usize, usize)> = Vec::new();
    for (done, &owner) in order.iter().enumerate() {
        let a = &mut *group[owner];
        let _step_timer =
            crate::scheduler::mtp_timing::StepTimer::new(&sched.timing, positions[owner]);
        let begun = model
            .begin_glm_long_owner_tail(a.seq.slot_idx as u32, owner, &tokens[owner])
            .and_then(|()| front_save_slot(model, save_slots[owner], rows));
        if let Err(e) = begun {
            tracing::error!("begin_glm_long_owner_tail (owner={owner}): {e:#}");
            for &o in &order[done..] {
                group[o].finished = true;
            }
            return;
        }
        a.last_token_time = Instant::now();
        let ledger_enabled = ledger::enabled_for(&a.seq);
        if let Some(next) = verify_dflash_tail(
            model,
            a,
            sched,
            &drafts[owner],
            num_drafts,
            verify_ctx,
            dflash_verify_raw_argmax,
            ledger_enabled,
            positions[owner],
            &tokens[owner],
            verified[owner * rows..(owner + 1) * rows].to_vec(),
            step_timing,
            verify_ms,
            dflash_verify_raw_argmax,
        ) {
            deferred.push((owner, next));
        }
    }
    propose_owner_batch(model, group, &deferred, &save_slots, rows, sched, step_timing);
    if step_timing {
        tracing::info!(
            "GLM OWNER STEP_TIMING: owners={} rows={rows} sync={sync_ms:.1}ms verify={verify_ms:.1}ms total={:.1}ms",
            group.len(),
            t_step.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Put `slot`'s hidden-save region at the front the DFlash commit and
/// per-sequence propose read (restoring the preserved front for slot 0).
fn front_save_slot(model: &dyn Model, slot: Option<usize>, rows: usize) -> anyhow::Result<()> {
    match slot {
        Some(0) => model.restore_dflash_save_front(rows, 0),
        Some(slot) => model.pack_dflash_save_seq(slot, rows, 0),
        None => Ok(()),
    }
}

/// One drafter pass for every owner whose tail deferred its re-propose
/// (`(owner, drafts)`), falling back to per-sequence proposals when the
/// proposer declines or the draft counts differ.
fn propose_owner_batch(
    model: &dyn Model,
    group: &mut [&mut ActiveSeq],
    deferred: &[(usize, usize)],
    save_slots: &[Option<usize>],
    rows: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    step_timing: bool,
) {
    if deferred.is_empty() {
        return;
    }
    let t_propose = Instant::now();
    let num_drafts = deferred[0].1;
    let mut owners: Vec<usize> = deferred.iter().map(|&(o, _)| o).collect();
    owners.sort_unstable();
    let batched = if owners.len() >= 2 && deferred.iter().all(|&(_, d)| d == num_drafts) {
        let tokens: Vec<u32> = owners.iter().map(|&o| group[o].last_token).collect();
        let positions: Vec<usize> = owners.iter().map(|&o| group[o].seq.seq_len).collect();
        let stash = vec![0usize; owners.len()];
        let mut seqs: Vec<&mut SequenceState> = group
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| owners.binary_search(i).is_ok())
            .map(|(_, a)| &mut a.seq)
            .collect();
        model.run_mtp_propose_batched(&tokens, &positions, &stash, num_drafts, &mut seqs, 0, None)
    } else {
        Ok(None)
    };
    let fail = |a: &mut ActiveSeq, outcome| {
        crate::scheduler::helpers::handle_dspark_repropose_failure(model, a, outcome);
        if spark_model::speculative::glm_repair_policy::enabled() {
            a.finished = true;
        }
    };
    match batched {
        Ok(Some(all)) if all.len() == owners.len() => {
            for (&o, d) in owners.iter().zip(all) {
                if d.is_empty() {
                    fail(group[o], crate::scheduler::helpers::ProposalOutcome::Empty);
                } else {
                    group[o].pending_drafts = d;
                }
            }
        }
        other => {
            if let Err(e) = other {
                tracing::error!("run_mtp_propose_batched (owners={}): {e:#}", owners.len());
            }
            // Slot 0 last: its region is the (restored) front.
            let front = deferred.iter().position(|&(o, _)| save_slots[o] == Some(0));
            let order = (0..deferred.len())
                .filter(|&i| Some(i) != front)
                .chain(front);
            for i in order {
                let (o, drafts) = deferred[i];
                let a = &mut *group[o];
                let proposal = front_save_slot(model, save_slots[o], rows).and_then(|()| {
                    model.run_mtp_propose_multi(a.last_token, a.seq.seq_len, drafts, &mut a.seq, 0, None)
                });
                match proposal {
                    Ok(d) if !d.is_empty() => a.pending_drafts = d,
                    Ok(_) => fail(a, crate::scheduler::helpers::ProposalOutcome::Empty),
                    Err(e) => {
                        tracing::error!("run_mtp_propose_multi (owner {o}): {e:#}");
                        fail(a, crate::scheduler::helpers::ProposalOutcome::Error);
                    }
                }
            }
        }
    }
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::Propose, t_propose);
    if step_timing {
        tracing::info!(
            "GLM OWNER PROPOSE: owners={} {:.1}ms",
            deferred.len(),
            t_propose.elapsed().as_secs_f64() * 1000.0
        );
    }
}
