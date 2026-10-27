// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Token` / `StreamEvent::TokenWithLogprobs` arm of the
// streaming `flat_map` closure (originally ~672 LoC at the top of the
// `chat_stream::chat_completions_stream` body).
//
// Returns the neutral stream deltas produced for this single token.
// The caller encodes them into wire SSE events at the surface seam
// (`openai::delta_to_chunk_events`).

use crate::ir::StreamDelta;
use crate::literal_xml::{first_marker_outside_literal, state_after};

use super::super::sanitizer::sanitize_content_chunk_with_literal_state;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::strip::{
    maybe_log_decode_trace, strip_all_preserving_boundary, strip_preserving_boundary,
};

type DeltaVec = Vec<StreamDelta>;

#[path = "content_token.rs"]
mod content_token;
#[path = "role_literals.rs"]
mod role_literals;
#[path = "stop_strings.rs"]
mod stop_strings;
#[path = "terminal_token.rs"]
mod terminal_token;

pub(super) use role_literals::strip_bare_role_literal;
pub(super) use stop_strings::apply_stop_string_holdback;

/// Maximum consecutive tokens the stream may spend with
/// `state.suppressing_param_leak == true` (sanitizer holding content
/// because of an orphan `<parameter=` / `<tool_call>` opener without
/// a matching close). When the model degenerates into a doom-loop of
/// partial-envelope leakage — observed 2026-05-24 on
/// opencode-hotfix.jsonl seq=10: 8192 tokens emitted after Atlas
/// rejected a `write({})` call, all suppressed by the sanitizer, no
/// content-loop watchdog fire (the period exceeded 64) — this
/// threshold ends the stream cleanly instead of burning to
/// `max_tokens=8192`. 256 tokens is enough headroom for legitimately
/// long tool-call bodies that take many tokens to close (long
/// `content` field on a `write` call) while bounding worst-case
/// wasted decode at ~10s @ 30 tok/s.
const MAX_SUPPRESS_STREAK_TOKENS: u32 = 256;

/// Process one token. Returns the stream deltas to forward to the
/// client (empty `Vec` is valid).
///
/// Thin wrapper around [`handle_token_inner`] that runs the
/// orphan-suppression streak watchdog after every token regardless
/// of which early-return branch fired in the body. The watchdog
/// can't live inside `handle_token_inner` because that function has
/// many early returns (one per emission path) — putting the check
/// at the end of the body would only fire when the natural fall-
/// through is taken, leaving the doom-loop case (long suppressed
/// stream of orphan `<tool_call>` openers) uncaught.
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    // Count generated tokens without reopening a terminal parser/stream.
    crate::metrics::DECODED_TOKENS_TOTAL.inc();
    terminal_token::while_open(state, |state| handle_open_token(state, ctx, tok))
}

fn handle_open_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let result = handle_token_inner(state, ctx, tok);

    // Orphan-suppression streak watchdog. The sanitizer flips
    // `suppressing_param_leak=true` when it sees an orphan
    // `<tool_call>` / `<parameter=` opener without a matching close.
    // Suppressing forever (until max_tokens) burns the user's
    // patience and decode budget — observed live as an 8192-token
    // doom loop. If the streak exceeds the bound, end the stream.
    // `--reference-mode` keeps every stream-side cut below off: a reference
    // server ends a response only on EOS / stop / max_tokens.
    if state.suppressing_param_leak && !state.stop_string_triggered && !ctx.state.reference_mode {
        state.suppress_streak_tokens = state.suppress_streak_tokens.saturating_add(1);
        if state.suppress_streak_tokens > MAX_SUPPRESS_STREAK_TOKENS {
            tracing::warn!(
                streak = state.suppress_streak_tokens,
                "orphan tool-call suppression streak exceeded {MAX_SUPPRESS_STREAK_TOKENS} tokens; ending stream",
            );
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            // ★ Name the guard so the wire reason comes out "length".
            // Without this the scheduler sees a bare `cancel_flag` with
            // budget left, falls to its early-finalize rule and reports
            // "stop" — telling the client the model finished, when in
            // fact we truncated it mid-doom-loop.
            //
            // This was the THIRD such path. The scheduler guards and the
            // simhash/token-loop watchdogs were fixed first; this one
            // survived both and cost the agentic gate runs 0 and 7 on
            // three consecutive shas, because the harness's
            // `was_cut_off()` only nudges on "length".
            //
            // ★ Note the harness's OTHER recovery route cannot cover it:
            // `emitted_unparsed_call` scans the text for `<tool_call>` /
            // `<function=`, and the sanitizer has already suppressed
            // exactly those bytes — which is why this streak fired at all.
            state.guard_stop = Some("suppress_streak");
            state
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
        }
    } else if !state.suppressing_param_leak {
        state.suppress_streak_tokens = 0;
    }

    result
}

fn handle_token_inner(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let mut deltas: DeltaVec = Vec::new();
    state.generated_tokens = state.generated_tokens.saturating_add(1);
    state.all_toks.push(tok);
    // One push per call == one sampled token == one increment of
    // `usage.completion_tokens`. Drained onto the next client-visible
    // chunk (return_token_ids); a no-op clone-free take when opted out.
    if ctx.req_return_token_ids {
        state.pending_token_ids.push(tok);
    }

    // GLM's native opener ends reasoning and must also reach the existing
    // content/tool parser. The startup carrier is model-scoped, not inferred
    // from the display name or from tool-shaped reasoning text.
    let native_marker_is_literal = ctx.state.glm_tool_boundary == Some(tok)
        && state.detector.is_some()
        && state.all_toks.len() > 1
        && ctx
            .state
            .tokenizer
            .decode(&state.all_toks[..state.all_toks.len() - 1])
            .ok()
            .is_some_and(|prefix| state_after(&prefix).is_literal());
    let implicit_tool = !state.thinking_done
        && state.detector.is_some()
        && ctx.state.glm_tool_boundary == Some(tok)
        && !native_marker_is_literal;
    // ── Thinking-phase: token-ID based boundary detection ──────────
    if !state.thinking_done && (ctx.state.think_end_token_id == Some(tok) || implicit_tool) {
        // Keep the opener out of a residual reasoning delta's token IDs;
        // it will be restored once, with the retained content token below.
        if implicit_tool && ctx.req_return_token_ids {
            state.pending_token_ids.pop();
        }
        state.thinking_done = true;
        // Emit only the residual reasoning delta not yet sent
        // by incremental streaming (e.g. trailing bytes held
        // back due to incomplete UTF-8 at prior token boundary).
        // The full reasoning has already been streamed
        // incrementally via reasoning_chunk deltas above —
        // re-emitting the full text here would double it.
        if ctx.enable_thinking && state.all_toks.len() > 1 {
            let full = ctx
                .state
                .tokenizer
                .decode(&state.all_toks[..state.all_toks.len() - 1])
                .unwrap_or_default();
            let stable = full.trim_end_matches('\u{FFFD}');
            if stable.len() > state.emitted {
                let residual = &stable[state.emitted..];
                // Same fix as the in-loop emit: whitespace-only residuals
                // are legitimate `\n   ` indents that the model emitted;
                // dropping them would lose chars permanently.
                if !residual.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: residual.to_string(),
                        token_ids: state.take_ids_if(ctx.req_return_token_ids),
                    });
                }
            }
        }
        // Flush the reasoning sanitizer's tail buffer. Without this, up to
        // ~18 trailing bytes of the final thinking block (or anything held
        // back for partial-tag fusion) are silently dropped. Skip when
        // suppression is active (no close arrived during thinking) — those
        // bytes are intentionally not surfaced.
        if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
            let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
            // Whitespace-only tail can be a real trailing `\n   ` indent
            // — emit anything non-empty so byte boundaries align.
            if !tail.is_empty() {
                deltas.push(StreamDelta::Reasoning {
                    text: tail,
                    token_ids: Vec::new(),
                });
            }
        }
        // Reset tool detector to clear any thinking-era tag fragments.
        if let Some(ref mut det) = state.detector {
            det.reset();
        }
        state.emitted = 0; // Reset — next decode will be content-only
        state.all_toks.clear(); // Clear thinking tokens from accumulator
        state.content_decoded.clear();
        state.detok_prefix_offset = 0;
        state.detok_read_offset = 0;
        if !implicit_tool {
            return deltas;
        }
        state.all_toks.push(tok);
        if ctx.req_return_token_ids {
            state.pending_token_ids.push(tok);
        }
    }
    if !state.thinking_done {
        // Still in thinking — accumulate but don't emit as content
        if ctx.enable_thinking {
            // Layer-A one-shot guard: after the in-think tool-call leak
            // scanner has fired, suppress all subsequent reasoning
            // deltas for this stream. The scheduler's `cancel_flag`
            // (set when the scanner fired) finalises the sequence
            // within one token via `emit_step::emit_token`; this
            // guard catches the in-flight token race so the next
            // opener never reaches the client.
            if state.reasoning_xml_leak_detected {
                return deltas;
            }
            // Open thinking: emit as reasoning_content. Incrementally extend
            // the stable decoded text instead of re-decoding all_toks (O(n²)).
            let delta_stable = ctx.state.tokenizer.incremental_decode(
                &state.all_toks,
                &mut state.detok_prefix_offset,
                &mut state.detok_read_offset,
            );
            state.content_decoded.push_str(&delta_stable);
            let stable_end = state.content_decoded.len();
            if stable_end > state.emitted {
                let raw = state.content_decoded[state.emitted..stable_end].to_string();
                let mut cleaned = raw.clone();
                let reasoning_literal_before = state.reasoning_literal_xml_state;
                state.emitted = stable_end;
                // Strip format tokens that shouldn't appear in thinking.
                // `<think>` only fires at the literal opener (always
                // whitespace-adjacent in the prompt), so a plain replace
                // is safe here.
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                // GLM and Qwen4 can spell tool XML with ordinary text tokens
                // during reasoning. Preserve those bytes just as blocking does.
                // GLM's native opener was handled by token ID above; its tool
                // transition and the legacy models' leak guards stay separate.
                if ctx.state.tokenizer.glm_tool_markup_as_text {
                    if !state.reasoning_suppressing_leak && !cleaned.is_empty() {
                        deltas.push(StreamDelta::Reasoning {
                            text: cleaned,
                            token_ids: state.take_ids_if(ctx.req_return_token_ids),
                        });
                    }
                    return deltas;
                }
                // Boundary-preserving strip: see `strip_preserving_boundary`
                // doc — prevents `the<tool_call>...</tool_call>project`
                // from collapsing to `theproject`.
                while let Some((start, _)) = first_marker_outside_literal(
                    &cleaned,
                    reasoning_literal_before,
                    &["<tool_call>"],
                ) {
                    if let Some((end_rel, end_len)) = first_marker_outside_literal(
                        &cleaned[start..],
                        reasoning_literal_before,
                        &["</tool_call>"],
                    ) {
                        let end = start + end_rel + end_len;
                        cleaned = strip_preserving_boundary(&cleaned, start, end);
                    } else {
                        cleaned = cleaned[..start].to_string();
                        break;
                    }
                }
                if let Some((start, _)) = first_marker_outside_literal(
                    &cleaned,
                    reasoning_literal_before,
                    &["<function="],
                ) {
                    cleaned = cleaned[..start].to_string();
                }
                // Strip leaked tool-call closing tags from reasoning
                // (observed pattern: `</parameter></function>` right
                // before a role-word repetition loop). Route through
                // `strip_all_preserving_boundary` (2026-05-23 sweep)
                // to avoid gluing words when a closing tag straddles
                // two reasoning sentences.
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    while let Some((start, len)) =
                        first_marker_outside_literal(&cleaned, reasoning_literal_before, &[*tag])
                    {
                        cleaned = strip_preserving_boundary(&cleaned, start, start + len);
                    }
                }
                // Collapse role-word repetition loops (Qwen3.5/3.6
                // post-tool-call hallucination): `userX...userX` →
                // "" until no adjacent pairs remain, then strip
                // line-bounded standalones (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    cleaned = strip_all_preserving_boundary(&cleaned, &pair);
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                maybe_log_decode_trace(&raw, &cleaned, stable_end, stable_end - raw.len());
                // Layer-A in-think tool-call leak scanner. The per-
                // delta strippers above can miss boundary splits
                // (e.g. `<too` in delta N + `l_call>` in delta N+1)
                // and even when they strip, the model keeps emitting
                // the next repetition because its own KV already
                // contains the literal opener. This sliding-window
                // detector across deltas catches the opener on
                // arrival, drops the delta, sets the loop-cap flag
                // (→ finish_reason="length" via the PR #87 override)
                // and flips the scheduler cancel_flag so generation
                // terminates within one token via PR #89.
                let tools_active_request =
                    !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
                if tools_active_request {
                    state.reasoning_xml_scan_buf.push_str(&cleaned);
                    if state.reasoning_xml_scan_buf.len() > 256 {
                        let drop_to = state.reasoning_xml_scan_buf.len() - 256;
                        let cut = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .find(|&(i, _)| i >= drop_to)
                            .map(|(i, _)| i)
                            .unwrap_or(state.reasoning_xml_scan_buf.len());
                        let dropped = state.reasoning_xml_scan_buf[..cut].to_string();
                        state.reasoning_xml_scan_state.feed(&dropped);
                        state.reasoning_xml_scan_buf.drain(..cut);
                    }
                    let opener = first_marker_outside_literal(
                        &state.reasoning_xml_scan_buf,
                        state.reasoning_xml_scan_state,
                        &["<tool_call>", "<function=", "<parameter=", "<invoke "],
                    )
                    .map(|(at, len)| {
                        let marker = state.reasoning_xml_scan_buf[at..at + len].to_string();
                        (at, marker)
                    });
                    if let Some((at, op)) = opener {
                        // Snapshot the log tail BEFORE consuming the match,
                        // so the WARN still shows the opener in context.
                        let tail_start = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .rev()
                            .nth(63)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let tail = state.reasoning_xml_scan_buf[tail_start..].to_string();
                        // Consume the buffer through this opener so ONE
                        // occurrence is counted once, not on every
                        // subsequent delta the rolling tail still contains
                        // it in.
                        let consumed = state.reasoning_xml_scan_buf[..at + op.len()].to_string();
                        state.reasoning_xml_scan_state.feed(&consumed);
                        state.reasoning_xml_scan_buf.drain(..at + op.len());
                        state.reasoning_xml_opener_hits =
                            state.reasoning_xml_opener_hits.saturating_add(1);
                        let threshold = ctx.state.chat.in_think_leak_openers;
                        if threshold != 0 && state.reasoning_xml_opener_hits >= threshold {
                            state.reasoning_xml_leak_detected = true;
                            // Name the cut for the --dump body / Done event;
                            // `tool_loop_capped` wires finish_reason
                            // "length" (the shipped doom-loop contract).
                            state.guard_stop = Some("in_think_tool_leak");
                            state.tool_loop_capped = true;
                            state.stop_string_triggered = true;
                            state
                                .cancel_flag
                                .store(true, std::sync::atomic::Ordering::Release);
                            tracing::warn!(
                                model = %ctx.model,
                                request_id = %ctx.id,
                                opener = op,
                                hits = state.reasoning_xml_opener_hits,
                                threshold,
                                tail = %tail,
                                "in-think tool-call leak: opener threshold reached; cancelling \
                                 sequence (finish_reason \"length\", guard in_think_tool_leak). \
                                 Raise/disable via ATLAS_INTHINK_TOOL_LEAK_OPENERS (0 = strip-only)"
                            );
                            return deltas;
                        }
                        // Below threshold: keep streaming. A model whose
                        // native tool syntax surfaces in legitimate
                        // reasoning (qwen3_coder family) lands here instead
                        // of losing the whole turn. Same-delta openers were
                        // stripped by the cleanups above; an opener SPLIT
                        // across deltas may have partially reached the
                        // client's reasoning channel — cosmetic, and true
                        // of the pre-knob code's one free delta too.
                        tracing::warn!(
                            model = %ctx.model,
                            request_id = %ctx.id,
                            opener = op,
                            hits = state.reasoning_xml_opener_hits,
                            threshold,
                            "in-think tool-call opener observed in reasoning; below \
                             ATLAS_INTHINK_TOOL_LEAK_OPENERS threshold, not cancelling"
                        );
                    }
                }
                // F19: final structured sanitisation pass catches
                // any leak markers the hand-rolled cleanups missed.
                let cleaned = sanitize_content_chunk_with_literal_state(
                    &cleaned,
                    &mut state.reasoning_tag_scan_buf,
                    &mut state.reasoning_suppressing_leak,
                    &mut state.reasoning_inside_envelope,
                    &mut state.reasoning_literal_xml_state,
                    &ctx.leak_markers,
                );
                // Emit whitespace-only chunks too. The `sanitize_content_chunk`
                // holdback can roll out runs of `\n   ` (newline + indent) as
                // a single committed chunk when the suffix exceeds tag_max
                // chars; dropping those via `trim().is_empty()` permanently
                // loses byte boundaries because `state.emitted` already
                // advanced past them. Symptom: streamed reasoning has
                // `**\n -Calculate` where the model actually emitted
                // `**\n   - Calculate` — verified byte-for-byte against the
                // non-streaming response on temp=0 seed=42 (live A/B
                // 2026-05-25). Drop only TRULY empty chunks.
                if !cleaned.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: cleaned,
                        token_ids: state.take_ids_if(ctx.req_return_token_ids),
                    });
                }
            }
        }
        return deltas;
    }

    content_token::handle_content(state, ctx, deltas)
}
