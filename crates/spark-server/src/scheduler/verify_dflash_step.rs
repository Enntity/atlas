// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash-based verify step (drafted token verification).

use super::*;

/// Verify several equal-width GLM DFlash blocks in one target weight sweep.
/// Rows stay sequence-major so each KDA/DSA lane advances causally while the
/// 288-expert EXL3 weights are streamed only once.
pub fn step_verify_dflash_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    let n = batch.len();
    if n < 2 {
        return;
    }
    let drafts = batch
        .iter_mut()
        .map(|active| {
            active.pending_draft_conf.clear();
            std::mem::take(&mut active.pending_drafts)
        })
        .collect::<Vec<_>>();
    let draft_count = drafts[0].len();
    let k = draft_count + 1;
    if !drafts.iter().all(|tokens| tokens.len() == draft_count)
        || !model.can_batch_verify_dflash(n, k)
    {
        tracing::error!("invalid DFlash verify batch shape n={n} k={k}");
        for (active, pending) in batch.iter_mut().zip(drafts) {
            active.pending_drafts = pending;
        }
        return;
    }
    if let Err(error) = model.sync_secondary() {
        tracing::error!("batched DFlash sync_secondary: {error:#}");
        for active in batch.iter_mut() {
            active.finished = true;
        }
        return;
    }
    let mut tokens = Vec::with_capacity(n * k);
    for (active, pending) in batch.iter().zip(&drafts) {
        tokens.push(active.last_token);
        tokens.extend_from_slice(pending);
    }
    let seq_ids = batch
        .iter()
        .map(|active| active.seq.slot_idx as u32)
        .collect::<Vec<_>>();
    if let Err(error) = model.ep_broadcast_dflash_verify_batch(&seq_ids, &tokens, k) {
        tracing::error!("broadcast batched DFlash verify: {error:#}");
        for active in batch.iter_mut() {
            active.finished = true;
        }
        return;
    }
    let started = std::time::Instant::now();
    let verdicts = {
        let mut seqs = batch
            .iter_mut()
            .map(|active| &mut active.seq)
            .collect::<Vec<_>>();
        match model.decode_verify_dflash_batched(&tokens, k, &mut seqs, 0) {
            Ok(verdicts) => verdicts,
            Err(error) => {
                tracing::error!("decode_verify_dflash_batched: {error:#}");
                for active in batch.iter_mut() {
                    active.finished = true;
                }
                return;
            }
        }
    };
    let verify_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut accepted = Vec::with_capacity(n);
    let mut picked = Vec::with_capacity(n);
    for (sequence, active) in batch.iter_mut().enumerate() {
        active.last_token_time = Instant::now();
        let raw = &verdicts[sequence * k..(sequence + 1) * k];
        let verified = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
            raw.to_vec()
        } else {
            crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
                model, raw, active, verify_ctx, 0,
            )
        };
        let count = drafts[sequence]
            .iter()
            .zip(&verified)
            .take_while(|(draft, target)| draft == target)
            .count();
        accepted.push(count);
        picked.push(verified);
    }
    let accepted_wire = accepted
        .iter()
        .map(|&value| value as u32)
        .collect::<Vec<_>>();
    if let Err(error) = model.ep_broadcast_tokens(&accepted_wire) {
        tracing::error!("broadcast batched DFlash verdicts: {error:#}");
        for active in batch.iter_mut() {
            active.finished = true;
        }
        return;
    }

    // Commit every sequence's recurrent rollback and DFlash context before
    // any proposer runs. The shared capture remains sequence-major.
    let mut propose_indexes = Vec::new();
    for sequence in 0..n {
        let active = &mut batch[sequence];
        let num_accepted = accepted[sequence];
        crate::scheduler::adaptive_spec::record_verify(active, num_accepted, sched);
        let pre_verify_len = active.seq.seq_len.saturating_sub(k);
        let target_seq_len = pre_verify_len + num_accepted + 1;
        active.seq.seq_len = target_seq_len;
        active.seq.tokens.truncate(target_seq_len);
        if let Err(error) = model.commit_ctx_from_row(
            &mut active.seq,
            sequence * k,
            num_accepted + 1,
            pre_verify_len,
        ) {
            tracing::error!("commit_ctx_from_row: {error:#}");
            active.finished = true;
            continue;
        }
        if let Err(error) = model.commit_accepted_prefix(&mut active.seq, num_accepted + 1, k) {
            tracing::error!("commit_accepted_prefix (batched DFlash): {error:#}");
            active.finished = true;
        }
    }

    for sequence in 0..n {
        let active = &mut batch[sequence];
        if active.finished {
            continue;
        }
        let num_accepted = accepted[sequence];
        for &token in drafts[sequence].iter().take(num_accepted) {
            emit_token(active, token, None, sched);
            if active.finished {
                break;
            }
        }
        if active.finished {
            continue;
        }
        if let Some(&bonus) = picked[sequence].get(num_accepted) {
            emit_token(active, bonus, None, sched);
            active.last_token = bonus;
        }
        crate::metrics::SPEC_DECODE_VERIFY
            .with_label_values(&[
                "dflash_batched",
                if num_accepted == draft_count {
                    "accept_all"
                } else {
                    "accept_partial"
                },
            ])
            .inc();
        if let Err(error) = model.trim_proposer_state(&mut active.seq, num_accepted, 0) {
            tracing::error!("trim_proposer_state (batched DFlash): {error:#}");
        }
        if crate::scheduler::adaptive_spec::spec_allowed(active, sched) {
            propose_indexes.push(sequence);
        }
    }
    if propose_indexes.len() >= 2 && model.mtp_propose_batch_max() >= propose_indexes.len() {
        let propose_started = std::time::Instant::now();
        let propose_count = propose_indexes.len();
        let tokens = propose_indexes
            .iter()
            .map(|&index| batch[index].last_token)
            .collect::<Vec<_>>();
        let positions = propose_indexes
            .iter()
            .map(|&index| batch[index].seq.seq_len)
            .collect::<Vec<_>>();
        let stash = vec![0usize; propose_indexes.len()];
        let mut wanted = propose_indexes.iter().copied().peekable();
        let mut seqs = Vec::with_capacity(propose_indexes.len());
        for (index, active) in batch.iter_mut().enumerate() {
            if wanted.peek().copied() == Some(index) {
                seqs.push(&mut active.seq);
                wanted.next();
            }
        }
        match model
            .run_mtp_propose_batched(&tokens, &positions, &stash, num_drafts, &mut seqs, 0, None)
        {
            Ok(Some(next)) => {
                for (&index, drafts) in propose_indexes.iter().zip(next) {
                    batch[index].pending_drafts = drafts;
                }
                propose_indexes.clear();
            }
            Ok(None) => {}
            Err(error) => tracing::error!("DFlash batch propose: {error:#}"),
        }
        tracing::info!(
            "DFLASH BATCH propose: n={} gamma={} proposer_ms={:.1}",
            propose_count,
            num_drafts,
            propose_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    for index in propose_indexes {
        let active = &mut batch[index];
        let grammar_mask = mtp_grammar_mask_for(active);
        match model.run_mtp_propose_multi(
            active.last_token,
            active.seq.seq_len,
            num_drafts,
            &mut active.seq,
            0,
            grammar_mask.as_deref(),
        ) {
            Ok(next) if !next.is_empty() => active.pending_drafts = next,
            Ok(_) => {}
            Err(error) => tracing::error!("DFlash serial re-propose fallback: {error:#}"),
        }
    }
    tracing::info!(
        "DFLASH BATCH verify: n={} K={} target_ms={:.1} accepted={:?}",
        n,
        k,
        verify_ms,
        accepted,
    );
}

/// DFlash γ-token verify with accept-prefix.
/// Single-sequence fallback: routes `[last_token, drafts...]` through the
/// fixed-width verifier. The GLM TP2 path commits KV, KDA, and sparse-index
/// rollback state atomically.
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // tokens = [last_verified, draft_0, draft_1, ..., draft_{γ-1}]
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);

    // EP/TP2: the worker must enter the same K-row GLM target forward before
    // rank 0 reaches the first MoE all-reduce. K is checkpoint-defined (γ=8
    // for GLM DFlash2), so carry it explicitly instead of inventing another
    // fixed-width command.
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF5) {
        tracing::error!("EP broadcast DFlash verify cmd: {e:#}");
        a.finished = true;
        return;
    }
    if let Err(e) = model.ep_broadcast_cmd(tokens.len() as u32) {
        tracing::error!("EP broadcast DFlash verify width: {e:#}");
        a.finished = true;
        return;
    }
    for &token in &tokens {
        if let Err(e) = model.ep_broadcast_cmd(token) {
            tracing::error!("EP broadcast DFlash verify token: {e:#}");
            a.finished = true;
            return;
        }
    }

    // Optionally split step timing into target verify and draft proposal.
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
    let verify_ms = if step_timing {
        t_verify.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    a.last_token_time = Instant::now();

    // Keep verifier and drafter on the same raw-argmax basis in DFlash mode.
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

    let standard_verdict = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
        model
            .dflash_sparse_distribution(&a.seq)
            .and_then(|distribution| {
                match crate::scheduler::dflash_rejection::standard_rejection(
                    model,
                    a,
                    drafts,
                    &distribution,
                ) {
                    Ok(verdict) => verdict,
                    Err(error) => {
                        tracing::warn!(
                            "DFlash2 standard rejection reference failed; using raw equality: {error:#}"
                        );
                        None
                    }
                }
            })
    } else {
        None
    };

    // Accept-prefix: drafts[i] is "accepted" iff drafts[i] == verified[i].
    // verified[i] is the target's argmax at position i (i.e. its
    // prediction for what should follow `tokens[i]`). drafts[i] was the
    // proposer's guess for the same slot. First mismatch terminates the
    // accepted prefix; verified[first_mismatch] becomes the bonus token.
    let num_accepted = if let Some(verdict) = standard_verdict.as_ref() {
        verdict.accepted
    } else {
        let mut accepted = 0usize;
        for i in 0..drafts.len() {
            if i + 1 >= verified.len() {
                break;
            }
            if drafts[i] == verified[i] {
                accepted += 1;
            } else {
                break;
            }
        }
        accepted
    };
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast DFlash accepted prefix: {e:#}");
        a.finished = true;
        return;
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

    // Commit every target-confirmed row in EAGLE order: accepted drafts plus
    // the bonus generator. The verify graph captured all K rows per target
    // layer, so partial rejection never conditions the next proposal on a
    // rejected tail row.
    if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len) {
        tracing::error!("commit_ctx (kgamma): {e:#}");
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
        let bonus = standard_verdict
            .as_ref()
            .map_or(verified[bonus_idx], |verdict| verdict.bonus);
        emit_token(a, bonus, None, sched);
        if a.finished {
            return;
        }
        a.last_token = bonus;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash",
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();

    tracing::info!(
        "DFLASH K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={}",
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

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }

    // Re-propose for next step — unless adaptive speculation just suspended
    // this seq (no drafts → the scheduler serial-decodes it via bootstrap).
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    let t_propose = std::time::Instant::now();
    if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
        let next_num_drafts =
            crate::scheduler::adaptive_spec::configured_dflash_depth_limit(a, num_drafts);
        if let Err(error) = model.configure_dflash_sampling(&mut a.seq, a.temperature, a.seed) {
            tracing::error!("configure DFlash sampling: {error:#}");
        }
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            next_num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => tracing::error!("run_mtp_propose_multi (dflash): {e:#}"),
        }
    }
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
