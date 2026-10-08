// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MULTI=1` scheduling: short prompts admitted in
//! the same tick (or already waiting) prefill in ONE model forward
//! (`Model::prefill_multi`) instead of one forward each.
//!
//! Admission (`phase_start_prefills`): when at least two eligible prompts
//! arrive in a tick, or eligible ones already wait, each eligible prompt is
//! set up as usual but deferred at chunk 0 WITHOUT the per-request prefill
//! command to the worker ranks (`prefill_a_step`): the model's multi pass
//! sends the worker all of them at once. The next phase
//! (`phase_continue_prefills`) runs every waiting eligible prompt, up to
//! [`max_rows`] rows a pass, before any other prefill path, and samples each
//! first token from its own logits.
//!
//! Eligibility is a pure function of the request, read alike at both sites
//! ([`eligible`]): a text prompt of at most `MULTI_MAX_PROMPT` tokens that
//! fits one chunk, without prompt logprobs or a per-request adapter. So on a
//! multi-rank model a prompt the admission deferred is always one this phase
//! takes: nothing else may prefill it, since its worker never got a command.

use std::time::Instant;

use spark_model::layers::ops::qwen4exp_rowinv::MULTI_MAX_PROMPT;
use spark_model::traits::{Model, PrefillSlice};
use spark_runtime::gpu::DevicePtr;

use super::sample_first_token;
use super::types::PrefillInProgress;

/// Whether a prompt takes the multi-sequence prefill.
pub(super) fn eligible(
    model: &dyn Model,
    prompt_len: usize,
    one_chunk: bool,
    prompt_logprobs: bool,
    adapter_slot: i32,
) -> bool {
    model.supports_prefill_multi()
        && (1..=MULTI_MAX_PROMPT).contains(&prompt_len)
        && one_chunk
        && !prompt_logprobs
        && adapter_slot < 0
}

/// A waiting prefill this phase takes.
fn takes(model: &dyn Model, p: &PrefillInProgress) -> bool {
    p.chunk_offset == 0
        && eligible(
            model,
            p.prompt_tokens.len(),
            true,
            p.seq.collect_prompt_logprobs.is_some(),
            p.seq.adapter_slot,
        )
}

/// Rows one pass takes (`ATLAS_QWEN4EXP_PREFILL_MULTI_MAX_TOKENS`, default
/// 8192). Read once.
pub(super) fn max_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_PREFILL_MULTI_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= MULTI_MAX_PROMPT)
            .unwrap_or(8192)
    })
}

/// Whether eligible prompts already wait for a pass.
pub(super) fn pending(model: &dyn Model, prefilling: &[PrefillInProgress]) -> bool {
    model.supports_prefill_multi() && prefilling.iter().any(|p| takes(model, p))
}

/// Run every waiting eligible prompt, in passes of at most [`max_rows`]
/// rows. Returns whether anything ran; `completed` gets `(index, first
/// token)` per prompt (`None`: failed, freed by the promotion).
pub(super) fn run(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    prefilling: &mut [PrefillInProgress],
    completed: &mut Vec<(usize, Option<u32>)>,
    prefill_stream: u64,
    prefill_event: u64,
) -> bool {
    if !model.supports_prefill_multi() {
        return false;
    }
    let mut waiting: Vec<usize> = (0..prefilling.len())
        .filter(|&i| takes(model, &prefilling[i]))
        .collect();
    if waiting.is_empty() {
        return false;
    }
    while !waiting.is_empty() {
        let mut rows = 0usize;
        let take = waiting
            .iter()
            .take_while(|&&i| {
                rows += prefilling[i].prompt_tokens.len();
                rows <= max_rows()
            })
            .count()
            .max(1);
        let pass: Vec<usize> = waiting.drain(..take).collect();
        run_pass(
            model,
            sched,
            prefilling,
            &pass,
            completed,
            prefill_stream,
            prefill_event,
        );
    }
    true
}

fn run_pass(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    prefilling: &mut [PrefillInProgress],
    pass: &[usize],
    completed: &mut Vec<(usize, Option<u32>)>,
    prefill_stream: u64,
    prefill_event: u64,
) {
    let t0 = Instant::now();
    let mut in_pass = vec![false; prefilling.len()];
    pass.iter().for_each(|&i| in_pass[i] = true);
    let mut slices: Vec<PrefillSlice<'_>> = prefilling
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| in_pass[*i])
        .map(|(_, p)| PrefillSlice {
            prompt_tokens: &p.prompt_tokens,
            seq: &mut p.seq,
            chunk_start: 0,
            chunk_len: p.prompt_tokens.len(),
            is_last_chunk: true,
        })
        .collect();
    let logits = model.prefill_multi(&mut slices, prefill_stream);
    drop(slices);
    let logits: Vec<DevicePtr> = match logits {
        Ok(l) if l.len() == pass.len() => l,
        Ok(l) => {
            tracing::error!(
                "prefill_multi returned {} logits for {} prompts",
                l.len(),
                pass.len()
            );
            completed.extend(pass.iter().map(|&i| (i, None)));
            return;
        }
        Err(e) => {
            tracing::error!("prefill_multi of {} prompts failed: {e:#}", pass.len());
            completed.extend(pass.iter().map(|&i| (i, None)));
            return;
        }
    };
    let _ = model.record_event(prefill_event, prefill_stream);
    let _ = model.stream_wait_event(model.default_stream(), prefill_event);
    for (k, &i) in pass.iter().enumerate() {
        let p = &mut prefilling[i];
        p.chunk_offset = p.prompt_tokens.len();
        match sample_first_token(
            model,
            logits[k],
            p.temperature,
            p.top_k,
            p.top_p,
            p.min_p,
            &crate::scheduler::min_tokens_ban::first_token_suppress(&p.eos_tokens, p.min_tokens),
            p.grammar_state.as_mut(),
            &sched.levers.sampling(),
        ) {
            Ok(first) => completed.push((i, Some(first))),
            Err(e) => {
                tracing::error!("prefill_multi[{k}] sampling: {e:#}");
                completed.push((i, None));
            }
        }
    }
    tracing::info!(
        "Multi-sequence prefill: {} prompts, {} rows, {:.1} ms",
        pass.len(),
        pass.iter()
            .map(|&i| prefilling[i].prompt_tokens.len())
            .sum::<usize>(),
        t0.elapsed().as_secs_f64() * 1e3
    );
}
