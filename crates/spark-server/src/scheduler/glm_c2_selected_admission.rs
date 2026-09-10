// SPDX-License-Identifier: AGPL-3.0-only
//! Request validation precedes allocation; every issued failure is nonreturning.
use super::{ActiveSeq, InferenceRequest, Model, SchedCtx, Tokens};
use crate::glm_terminal_session::SelectedOperation;
use crate::scheduler::{PrefillInProgress, ResponseSink, StreamEvent};
use anyhow::{Result, ensure};

pub(super) fn validate(req: &InferenceRequest, model: &dyn Model) -> Result<()> {
    let cancelled = matches!(req, InferenceRequest::Streaming { cancel_flag, .. }
        if cancel_flag.load(std::sync::atomic::Ordering::Acquire));
    ensure!(!cancelled, "request cancelled before selected admission");
    ensure!(
        !req.timeout_at()
            .is_some_and(|at| std::time::Instant::now() >= at),
        "request deadline elapsed before selected admission"
    );
    let (temperature, grammar) = match req {
        InferenceRequest::Streaming {
            temperature,
            grammar_spec,
            ..
        }
        | InferenceRequest::Blocking {
            temperature,
            grammar_spec,
            ..
        } => (*temperature, grammar_spec.is_some()),
    };
    ensure!(
        temperature == 0.0 && !grammar && !req.disable_mtp(),
        "selected paired serving requires greedy unconstrained requests with MTP enabled"
    );
    ensure!(
        !req.has_image_pixels()
            && req.adapter_slot() < 0
            && req.num_beams() == 1
            && req.src_lang_id() == 0
            && req.tgt_lang_id() == 0,
        "selected paired serving requires base-model text without beam or language routing"
    );
    ensure!(
        req.top_logprobs().is_none() && req.prompt_logprobs().is_none(),
        "selected paired serving does not support logprobs"
    );
    let prompt = req.prompt_tokens_arc();
    ensure!(
        (2..=1024).contains(&prompt.len())
            && prompt.iter().all(|t| (*t as usize) < model.vocab_size()),
        "selected paired serving requires 2..1024 valid cold prompt tokens"
    );
    ensure!(
        req.max_tokens().checked_add(prompt.len()).is_some(),
        "request length overflow"
    );
    Ok(())
}

pub(super) fn reject(req: InferenceRequest, message: &str) {
    let mut sink = match req {
        InferenceRequest::Streaming { token_tx, .. } => ResponseSink::Streaming(token_tx),
        InferenceRequest::Blocking { response_tx, .. } => ResponseSink::Blocking(Some(response_tx)),
    };
    crate::scheduler::lifecycle::send_error_to_sink(&mut sink, message);
}

pub(super) fn admit(
    operation: &SelectedOperation<'_>,
    model: &dyn Model,
    req: InferenceRequest,
    tokens: &Tokens,
    sched: &SchedCtx,
) -> ActiveSeq {
    let mut p = prepare(operation, model, req, tokens);
    let first = operation.require(crate::scheduler::glm_c2_selected_prefill::cold(
        model, &mut p, sched,
    ));
    let spontaneous = !p.enable_thinking && tokens.think_start == Some(first);
    let mut a = crate::scheduler::glm_c2_selected_prefill::promote(
        p,
        first,
        tokens,
        model.decode_rollback_ring_slots(),
        sched.limits.glm_tool_boundary,
    );
    crate::scheduler::mod_helpers::enforce_request_deadlines(std::slice::from_mut(&mut a));
    let cancelled = crate::scheduler::emit_step::retire_if_cancelled(&mut a);
    // Completed cold ownership and health precede even the first streamed token.
    operation.require(
        model
            .glm_paired_execution()
            .unwrap()
            .check_communication_health(),
    );
    if !cancelled
        && !crate::tui::shutdown::requested()
        && a.guard_stop.is_none()
        && !spontaneous
        && !a.output_tokens.is_empty()
        && !a.eos_tokens.contains(&first)
        && let ResponseSink::Streaming(ref tx) = a.sink
        && !crate::scheduler::mod_helpers::bounded_stream_send(
            tx,
            StreamEvent::Token(first),
            "selected first token",
        )
    {
        a.finished = true;
    }
    a
}

fn prepare(
    operation: &SelectedOperation<'_>,
    model: &dyn Model,
    mut req: InferenceRequest,
    tokens: &Tokens,
) -> PrefillInProgress {
    let request_start = std::time::Instant::now();
    let mut eos = tokens.eos.clone();
    eos.extend(req.take_stop_tokens());
    eos.sort_unstable();
    eos.dedup();
    let top_k = req.top_k();
    let top_p = req.top_p();
    let top_n_sigma = req.top_n_sigma();
    let min_p = req.min_p();
    let rep = req.repetition_penalty();
    let presence = req.presence_penalty();
    let frequency = req.frequency_penalty();
    let lz = req.lz_penalty();
    let dry = req.dry_multiplier();
    let dry_base = req.dry_base();
    let dry_allowed = req.dry_allowed_length();
    let bias = req.logit_bias().to_vec();
    let min_tokens = req.min_tokens();
    let session = req.session_hash();
    let thinking = req.enable_thinking();
    let budget = req.thinking_budget();
    let repetition = req.repetition_detection();
    let require_tool = req.require_tool_call();
    let tools = req.tools_present();
    let suppress_tool = req.suppress_tool_call();
    let disabled = req.disable_mtp();
    let seed = req.seed();
    let timeout = req.timeout_at();
    let (prompt, max, sink, cancel) = match req {
        InferenceRequest::Streaming {
            prompt_tokens,
            max_tokens,
            token_tx,
            cancel_flag,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Streaming(token_tx),
            Some(cancel_flag),
        ),
        InferenceRequest::Blocking {
            prompt_tokens,
            max_tokens,
            response_tx,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Blocking(Some(response_tx)),
            None,
        ),
    };
    let mut seq = operation.require(model.alloc_sequence());
    seq.session_hash = session;
    crate::scheduler::prefill_a_step_params::build_prefill_in_progress(
        prompt,
        session,
        seq,
        0,
        max,
        min_tokens,
        eos,
        sink,
        cancel,
        request_start,
        0.0,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        rep,
        presence,
        frequency,
        lz,
        dry,
        dry_base,
        dry_allowed,
        bias,
        thinking,
        budget,
        repetition,
        tokens.spontaneous_budget,
        require_tool,
        tools,
        suppress_tool,
        disabled,
        None,
        seed,
        None,
        timeout,
    )
}
