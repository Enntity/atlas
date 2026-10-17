// SPDX-License-Identifier: AGPL-3.0-only

//! step_decode_only: batched decode for active sequences (no MTP).

use super::*;

/// Decode-only step: batched decode for all active sequences (no MTP).
pub fn step_decode_only(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    mut spill: Option<&mut KvSpillManager>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) {
    let t0 = std::time::Instant::now();
    let n = active.len();
    // GLM long context has no plain multi-sequence decode: its multi-seq
    // MLA path is the verify lane, and a batch of independent rows fails
    // its checks (fatal on the EP worker). Decode each sequence alone on the
    // proven single-slot path, which also keeps the DFlash drafter context
    // appended per sequence (n == 1 below).
    if n > 1 && spark_model::speculative::glm_repair_policy::long_context_enabled() {
        let mut done = Vec::with_capacity(n);
        for a in active.drain(..) {
            let mut one = vec![a];
            step_decode_only(
                model,
                &mut one,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                adaptive_sampling,
                sched,
                spill.as_deref_mut(),
                swapped,
                preempted,
            );
            done.append(&mut one);
        }
        *active = done;
        return;
    }
    // Batched decode (CUDA-graph replay + batched-recurrent SSM) requires the
    // active sequences in SSM-pool-slot order, so batch position i maps to a
    // contiguous state address (pool_base + i*stride). The pool assigns
    // consecutive slots but the active list is in reverse-arrival order
    // ([7,6,..,0] for 8 seqs), which fails the contiguity check in
    // ssm_batched_recurrent.rs and the graph-capture slot==i assumption,
    // forcing the eager per-seq loop (no concurrency scaling). Sort ascending
    // by SSM slot (falling back to KV slot for non-SSM models) so the
    // contiguous-slot invariant holds and the batched paths engage. The whole
    // ActiveSeq is reordered, so the post-decode position->seq mapping stays
    // consistent.
    if n > 1 {
        active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
    }

    // CONCURRENT-DECODE DIAG: per-step batch state (slot, seq_len, etc).
    // Demoted to debug after the 2026-04-22 stride+graph fixes shipped —
    // it was a hot per-decode log line that drowned production traces.
    // Re-enable with `RUST_LOG=spark_server::scheduler=debug`.
    if n > 1 && tracing::enabled!(tracing::Level::DEBUG) {
        let diag: Vec<String> = active
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let bt0 = a.seq.block_table.first().copied().unwrap_or(u32::MAX);
                let btn = a.seq.block_table.len();
                format!(
                    "[{i}: slot={} seq_len={} bt={}/{} last={} out_n={}]",
                    a.seq.slot_idx,
                    a.seq.seq_len,
                    bt0,
                    btn,
                    a.last_token,
                    a.output_tokens.len(),
                )
            })
            .collect();
        tracing::debug!("CONC_DIAG n={n}: {}", diag.join(" "));
    }

    // EP broadcasts (seq_id preamble + cmd per active seq) are emitted
    // inside `decode_batch_dispatch` itself, interleaved with each per-seq
    // `decode()` call. Batching them up-front here would diverge the head's
    // comm-stream op order ([B,B,...,B,AR,AR,...]) from the worker's
    // ([B,AR,...,AR,B,AR,...,AR,...]) and deadlock NCCL — observed
    // empirically as a 51s broadcast timeout on the worker followed by
    // stale comm data reads. See decode_a2.rs for the full rationale.

    // F4 native EXL3 DFlash2 speculative lane (ATLAS_GLM_EXL3_DFLASH_SPEC=1):
    // a greedy, processor-free single sequence verifies the drafter's
    // proposal and emits accepted drafts + bonus. `None` from the model
    // (lane off, drafter unhealthy, capacity) falls through to serial decode.
    if n == 1 && exl3_spec_eligible(&active[0]) && exl3_spec_processors_idle(&active[0], sched) {
        let a = &mut active[0];
        let budget = a.think_budget_state();
        match model.exl3_dflash_spec_step(a.last_token, &mut a.seq, &budget) {
            Ok(Some(tokens)) => {
                a.last_token_time = Instant::now();
                for tok in tokens {
                    emit_token(a, tok, None, sched);
                    a.last_token = tok;
                    if a.finished {
                        break;
                    }
                }
                return;
            }
            Ok(None) => {}
            Err(e) => {
                tracing::error!("EXL3 DFlash speculative step: {e:#}");
                for mut a in active.drain(..) {
                    send_error(model, &mut a, &format!("{e:#}"));
                }
                return;
            }
        }
    }

    // Decode, PREEMPTING on KV exhaustion instead of failing the whole batch.
    //
    // Sequences grow one block per `block_size` tokens as they decode; when
    // admission overcommitted the pool (see `admission`), decode is where
    // the collision lands. `decode_batch_with_preemption` spills or requeues
    // ONE victim per retry — for later RESUME, never a kill — so the rest of
    // the batch makes progress and the victim's stream continues once blocks
    // free up. See the `preempt` module docs for the policy and the
    // measured C=128 evidence that motivated it.
    let Some(logits) =
        super::preempt::decode_batch_with_preemption(model, active, spill, swapped, preempted)
    else {
        return;
    };
    // Preemption may have shrunk the batch; `n` gates the n==1 paths below.
    let n = active.len();
    if n == 0 {
        return;
    }

    // Ctx-holes fix (ATLAS_DFLASH_SERIAL_APPEND=1): think-gated stretches
    // route HERE (mod.rs sends `inside_thinking` seqs to step_decode_only,
    // never the mtp bootstrap), so their captured target hiddens were
    // overwritten and permanently lost — the dominant ctx hole: a 270-token
    // think stretch leaves the drafter conditioned on the prompt alone
    // (observed GAP≈290 at first propose, accept ≤6%). Append each decoded
    // token's capture. n==1 only: `try_dflash_capture` stores row 0, which
    // is ambiguous in a multi-seq batch (fine here — DFlash runs
    // --max-batch-size 1).
    if n == 1 {
        if sched.levers.dflash_unified_ctx {
            // Unified ctx commit: serial token at RoPE position seq_len-1
            // (decode() advanced seq_len past the token just processed).
            let base_pos = active[0].seq.seq_len.saturating_sub(1);
            if let Err(e) = model.commit_ctx(&mut active[0].seq, 1, base_pos) {
                tracing::error!("commit_ctx (decode_only serial): {e:#}");
            }
        } else if sched.levers.dflash_serial_append
            && let Err(e) = model.dflash_serial_ctx_append(&mut active[0].seq)
        {
            tracing::error!("dflash_serial_ctx_append (decode_only): {e:#}");
        }
    }

    process_decode_logits(
        model,
        active,
        logits,
        t0,
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        adaptive_sampling,
        sched,
    );
}

impl ActiveSeq {
    /// Snapshot of the thinking-budget accounting for one F4 speculative
    /// step (`ThinkBudgetState`). SSOT: read at the single dispatch point so
    /// the model-side commit replay and the serial governor see the same
    /// per-sequence state; copied onto the EP wire for rank symmetry.
    pub fn think_budget_state(&self) -> spark_model::traits::ThinkBudgetState {
        spark_model::traits::ThinkBudgetState {
            inside_thinking: self.inside_thinking,
            thinking_tokens: self.thinking_tokens,
            thinking_budget: self.thinking_budget,
            force_end_thinking: self.force_end_thinking,
            spontaneous_think_budget: self.spontaneous_think_budget,
            think_watchdog_fires: self.think_watchdog_fires,
            think_start_token: self.think_start_token,
            think_end_token: self.think_end_token,
            tool_call_start_token: self.tool_call_start_token,
        }
    }
}

/// Confidence observes logits and can arm a forced close even when the raw
/// pick is ordinary text. Leave such thinking steps to the canonical pipeline.
/// Structural picks are separately stopped at the model's commit boundary.
fn exl3_spec_processors_idle(a: &ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    !sched.levers.adadec_diagnostic
        && !(a.inside_thinking
            && !sched.levers.disable_watchdogs
            && sched.watchdog.confidence_early_stop)
        // Unlimited thinking still has the serial hard override; avoid a
        // speculative block crossing that otherwise untracked boundary.
        && !(a.inside_thinking && a.thinking_budget.is_none())
}

/// The F4 EXL3 speculative lane emits raw verify argmaxes, so it only takes
/// steps whose serial sampling would also be a plain argmax: greedy, no
/// grammar, logprobs, penalties or bias, and no pending thinking-end /
/// tool-call forcing. Everything else decodes serially.
fn exl3_spec_eligible(a: &ActiveSeq) -> bool {
    a.temperature == 0.0
        && a.grammar_state.is_none()
        && a.top_logprobs.is_none()
        && a.repetition_penalty == 1.0
        && a.presence_penalty == 0.0
        && a.frequency_penalty == 0.0
        && a.lz_penalty == 0.0
        && a.dry_multiplier == 0.0
        && a.logit_bias.is_empty()
        && !a.force_end_thinking
        && !a.think_just_ended
        && !a.require_tool_call
        // Thinking-budget gate (draft side): when the budget is already
        // exhausted the step could emit at most the single arming-trip token
        // before the serial governor takes over — defer to serial, which
        // emits that token and arms exactly like this path would. A step
        // that reaches the boundary MID-STREAM is truncated at commit
        // (`ThinkBudgetState::plain_emission_len`, the hard gate).
        && (!a.inside_thinking
            || a.thinking_budget.is_none_or(|b| a.thinking_tokens < b))
}

#[cfg(test)]
#[path = "decode_step_tests.rs"]
mod decode_step_tests;
