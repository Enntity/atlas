// SPDX-License-Identifier: AGPL-3.0-only

//! Content-phase delivery after token accounting and reasoning boundaries.

use super::{
    DeltaVec, StreamCtx, StreamState, apply_stop_string_holdback, strip_bare_role_literal,
};
use crate::api::chat_stream::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};
use crate::api::sanitizer::sanitize_content_chunk_with_literal_state;
use crate::api::stream_guards::check_loop_watchdog;
use crate::{ir::StreamDelta, tool_parser};

pub(super) fn handle_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    mut deltas: DeltaVec,
) -> DeltaVec {
    // ── Content phase: full-decode + slice (matches reasoning path) ──
    //
    // Previously this path used the HF `tokenizers` crate's
    // `DecodeStream` (`decoder.step(tok)`). That incremental decoder
    // drops the leading metaspace byte at certain BPE-token boundaries
    // for byte-level tokenizers like Qwen's GPT-2-style BPE — verified
    // live 2026-05-25 against the FP8 Qwen3.6 model, opencode session
    // `ses_1a0e59bc7ffeFKSvtvWqoswsll`: tool-call `<parameter=content>`
    // for a Cargo.toml emitted `name = test-rust-axum-v32version =
    // 0.1.0edition = 2021` (no newlines between fields, no quotes
    // around values). Non-streaming `tokenizer.decode(&all_toks)`
    // for the same tokens produces the correct multi-line TOML.
    //
    // The fix: mirror the reasoning path — keep `state.all_toks`
    // populated with content tokens by handle_token_inner, decode
    // the cumulative list, and emit the byte slice that's stable past
    // `state.emitted`. `trim_end_matches('\u{FFFD}')` defers any
    // incomplete UTF-8 multi-byte sequence at the tail until the next
    // token completes it. `state.all_toks` and `state.emitted` are
    // reset by the reasoning-boundary handler, so this slice references the
    // post-thinking content only.
    // Incrementally extend the stable decoded text instead of re-decoding the
    // whole `all_toks` list every token (O(n²)). `content_decoded` stays
    // byte-identical to the previous `decode(&all_toks)` trimmed of any
    // trailing incomplete-multibyte token.
    let delta_stable = ctx.state.tokenizer.incremental_decode(
        &state.all_toks,
        &mut state.detok_prefix_offset,
        &mut state.detok_read_offset,
    );
    state.content_decoded.push_str(&delta_stable);
    let stable_end = state.content_decoded.len();
    let mut delta = if stable_end > state.emitted {
        let raw = state.content_decoded[state.emitted..stable_end].to_string();
        state.emitted = stable_end;
        raw
    } else {
        return deltas;
    };
    // Retire the lazy `content_decoder` field — kept in StreamState
    // only to avoid a wider state-struct migration. The HF decoder is
    // no longer the source of truth.
    let _ = &state.content_decoder;

    // Strip residual think tags from content after thinking is done.
    if state.thinking_done {
        // SSOT with the blocking path (api/chat_blocking.rs), which had no
        // equivalent scrub and therefore leaked '</think>' into content.
        delta = crate::api::strip::scrub_think_markers(&delta);
        // If model re-opens <think>, suppress content from <think> onward.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
            state.content_decoded.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
        }
        // Orphan tool-call markup in content when NO tools are defined: the
        // model emits a spurious `<tool_call>…` block after a normal answer and
        // the tool detector (which would strip it) is not running. Cut content
        // at the opener — SSOT with the blocking path's strip_orphan_tool_markup.
        // When tools ARE active the detector owns tool markup, so leave it.
        let tools_active_request =
            !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
        if !tools_active_request {
            delta = crate::api::strip::strip_orphan_tool_markup_with_literal_state(
                &delta,
                &state.literal_xml_state,
            );
        }
    }

    // Bare role-literal leak (Qwen3.5/3.6) — companion to the
    // scheduler-side <|im_start|> hard-stop. Suppressed mid tool-call
    // body: there a standalone `tool` token is the leading BPE fragment
    // of a `tool_*` NAME (issue #222) being reassembled, not a role leak.
    // ChatML formats only: elsewhere `user`, `assistant` and `tool` are
    // ordinary words, and GLM streams " tool" as one token, so the strip
    // deleted it from prose ("I don't have a that can ...").
    if ctx.role_literal_leaks {
        let inside_tool_call = state
            .detector
            .as_ref()
            .is_some_and(|d| d.inside_tool_call());
        strip_bare_role_literal(&mut delta, inside_tool_call);
    }

    if delta.is_empty() {
        return deltas;
    }

    // Multi-token stop sequences via string matching, with a vLLM-style
    // hold-back buffer (see `vllm/v1/engine/detokenizer.py`
    // `IncrementalDetokenizer.update`). All the state mutation lives in
    // `apply_stop_string_holdback` so the algorithm can be unit-tested
    // without spinning up a full `StreamCtx`.
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        delta = apply_stop_string_holdback(
            &delta,
            &ctx.stop_strings,
            ctx.stop_string_buffer_len,
            &mut state.accumulated_content,
            &mut state.stop_string_emitted_len,
            &mut state.stop_string_triggered,
        );
        // Entry was gated on `!triggered`, so a flip here is a GENUINE
        // client stop-sequence match (the watchdog paths set the flag
        // elsewhere): record it for the wire finish_reason ("stop", per
        // OpenAI) and cancel generation.
        if state.stop_string_triggered {
            state.note_stop_string_match();
        }
        if delta.is_empty() {
            // Either everything is sitting in the hold-back window
            // (waiting for the next chunk / stream close) or a match
            // already truncated the emittable bytes to nothing.
            return deltas;
        }
    }

    if state.stop_string_triggered {
        // Only the current token's genuine client stop match can reach this
        // branch. Emit its pre-stop prefix once, preserving markup sanitization.
        // Tokens queued AFTER a guard or stop match are rejected at handle_token
        // entry, before detokenization, content accumulation or token-ID tracking.
        if !delta.is_empty() {
            let cleaned = sanitize_content_chunk_with_literal_state(
                &delta,
                &mut state.tag_scan_buf,
                &mut state.suppressing_param_leak,
                &mut state.inside_envelope,
                &mut state.literal_xml_state,
                &ctx.leak_markers,
            );
            if !cleaned.is_empty() {
                deltas.push(StreamDelta::Content {
                    text: cleaned,
                    token_ids: state.take_ids_if(ctx.req_return_token_ids),
                });
            }
        }
        return deltas;
    }

    // Fork: detector-active vs pure-content path.
    if state.detector.is_some() {
        // Drain the detector outputs into a local Vec so we can drop
        // the &mut borrow on `state.detector` before the helpers below
        // (which take other &mut state fields) run.
        if ctx.state.tool_guard.enabled
            && state.detector.as_ref().is_some_and(|d| {
                d.pending_tool_bytes() > 0
                    && d.pending_tool_bytes().saturating_add(delta.len())
                        > ctx.state.tool_guard.max_bytes
            })
        {
            crate::api::chat_stream::tool_security::reject(
                state,
                atlas_tool_guard::BlockReason::ArgumentsTooLarge,
            );
            return deltas;
        }
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        deltas.extend(events_out);
                        // A normal content event is not a stop. One decoded
                        // token can yield text before, inside and after a
                        // literal marker; drain every event unless a guard
                        // actually cancelled this response.
                        if state.stop_string_triggered {
                            return deltas;
                        }
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx, &mut deltas);
                }
            }
            if state.guard_stop == Some(atlas_tool_guard::GUARD_STOP) {
                deltas.clear();
                return deltas;
            }
        }
    } else {
        let sanitized = sanitize_content_chunk_with_literal_state(
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &mut state.literal_xml_state,
            &ctx.leak_markers,
        );
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            deltas.extend(events_out);
            return deltas;
        }
        // process_detector_content does NOT pre-sanitize when called
        // from the no-detector branch — but the sanitizer was already
        // run above, so the helper's branch handling matches.
    }

    deltas
}

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `Content` delta. Returns `Some(deltas)` when the watchdog
/// fired (caller must short-circuit), else `None` (caller continues).
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<DeltaVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized = sanitized_or_raw;

    // F4 SimHash guard — fence-aware sentence-loop state machine
    // (`simhash_step` owns the rules; this call site only threads the
    // per-stream state through). Armed by the model's
    // `enable_stream_loop_guards`, past the min_tokens floor.
    let guards_armed = ctx.state.chat.stream_loop_guards
        && crate::api::stream_guards::watchdog_floor_reached(
            state.generated_tokens,
            ctx.min_tokens,
        );
    let semantic_trip = if guards_armed && !state.loop_watchdog_triggered {
        crate::loop_simhash::simhash_step(
            &mut state.simhash_pending,
            &mut state.simhash_in_fence,
            &mut state.simhash_fence_scan,
            &mut state.simhash_guard,
            sanitized,
        )
    } else {
        false
    };

    let token_trip = guards_armed
        && check_loop_watchdog(
            sanitized,
            &mut state.loop_scan_buf,
            state.loop_watchdog_triggered,
        );

    if semantic_trip || token_trip {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
        }
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;
        state.guard_stop = Some(if semantic_trip {
            "simhash_semantic_loop"
        } else {
            "token_loop_watchdog"
        });
        state
            .cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);

        // Watchdog fired: short-circuit the stream with no further
        // content. The model emitted a degenerate loop; we end the
        // response here rather than salvaging a synthetic tool call.
        return Some(DeltaVec::new());
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let out: DeltaVec = vec![StreamDelta::Content {
            text: sanitized.to_string(),
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        }];
        return Some(out);
    }
    None
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
fn detector_content_arm(state: &mut StreamState, ctx: &StreamCtx, text: &str) -> Option<DeltaVec> {
    let sanitized = sanitize_content_chunk_with_literal_state(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &mut state.literal_xml_state,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}
