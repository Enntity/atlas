// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::ir::Usage;
use std::sync::{Arc, atomic::AtomicBool};

fn state() -> StreamState {
    let mut state = StreamState::new(false, false, Arc::new(AtomicBool::new(false)), vec![]);
    state.tag_scan_buf = "The lantern".into();
    state.reasoning_tag_scan_buf = "buffered reasoning".into();
    state.pending_token_ids = vec![7, 8];
    state
}

fn usage() -> Usage {
    Usage {
        prompt_tokens: 28,
        completion_tokens: 63,
        cached_prompt_tokens: 4,
        reasoning_tokens: 2,
        accepted_prediction_tokens: 3,
        time_to_first_token_ms: 321.0,
        response_tokens_per_second: 22.0,
    }
}

fn flush_sanitizer(state: &mut StreamState) -> DeltaVec {
    let text = crate::api::stream_guards::flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &crate::tool_parser::LeakMarkers::EMPTY,
    );
    vec![StreamDelta::Content {
        text,
        token_ids: state.take_ids_if(true),
    }]
}

fn assert_finish_only(deltas: &[StreamDelta], expected: &str) {
    assert!(
        matches!(deltas, [StreamDelta::Finish { reason, usage: actual, token_ids, .. }]
            if reason.as_wire() == expected && *actual == usage() && token_ids.is_empty()),
        "guard must emit metadata/finish only, got {deltas:?}"
    );
}

#[test]
fn terminal_done_semantic_guard_does_not_flush_lantern_or_refusal() {
    let mut s = state();
    s.guard_stop = Some("simhash_semantic_loop");
    s.refusal_scan_buf = "I cannot help with that request.".into();
    let (deltas, reason) = finalize_done(&mut s, &ctx(), "stop", usage(), true, flush_sanitizer);
    assert_eq!(reason, "length");
    assert_finish_only(&deltas, "length");
    assert_eq!(
        s.tag_scan_buf, "The lantern",
        "guarded flush body must not run"
    );
    assert!(s.pending_token_ids.is_empty());
}

#[test]
fn terminal_done_tool_or_generic_guard_skips_buffered_fragments() {
    for tool_cap in [false, true] {
        let mut s = state();
        s.tool_loop_capped = tool_cap;
        s.stop_string_triggered = !tool_cap;
        s.detector = Some(crate::tool_parser::StreamingToolDetector::new_with_tools(
            vec![],
        ));
        s.detector
            .as_mut()
            .unwrap()
            .process("<tool_call><function=test>");
        let (deltas, _) = finalize_done(&mut s, &ctx(), "length", usage(), true, |_| {
            panic!("terminal guard must not flush the tool detector")
        });
        assert_finish_only(&deltas, "length");
    }
}

#[test]
fn terminal_done_guard_during_flush_discards_all_generated_delta_kinds_and_ids() {
    let mut s = state();
    let (deltas, _) = finalize_done(&mut s, &ctx(), "stop", usage(), true, |s| {
        let mut deltas = flush_sanitizer(s);
        s.guard_stop = Some("token_loop_watchdog");
        s.pending_token_ids.push(99);
        deltas.extend([
            StreamDelta::Reasoning {
                text: std::mem::take(&mut s.reasoning_tag_scan_buf),
                token_ids: vec![9],
            },
            StreamDelta::ToolCallStart {
                index: 0,
                id: "call-1".into(),
                name: "test".into(),
            },
            StreamDelta::ToolCallArgs {
                index: 0,
                fragment: "{\"late\":true}".into(),
                token_ids: vec![10],
            },
            StreamDelta::Refusal {
                text: "buffered refusal".into(),
            },
        ]);
        deltas
    });
    assert_finish_only(&deltas, "length");
    assert!(s.pending_token_ids.is_empty());
}

#[test]
fn terminal_done_normal_eos_and_explicit_stop_keep_safe_sanitizer_prefix() {
    for explicit in [false, true] {
        let mut s = state();
        s.stop_string_triggered = explicit;
        s.stop_string_matched = explicit;
        s.tag_scan_buf = if explicit {
            "safe pre-stop prefix"
        } else {
            "The lantern"
        }
        .into();
        let expected = s.tag_scan_buf.clone();
        let (deltas, reason) = finalize_done(&mut s, &ctx(), "stop", usage(), true, flush_sanitizer);
        assert_eq!(reason, "stop");
        assert!(
            matches!(&deltas[..], [StreamDelta::Content { text, token_ids }, StreamDelta::Finish { usage: actual, .. }]
            if text == &expected && token_ids == &[7,8] && *actual == usage())
        );
    }
}

#[test]
fn terminal_done_keeps_existing_finish_precedence_and_tool_history() {
    for (scheduler, tool_cap, salvaged, matched, expected) in [
        ("timeout", true, true, true, "timeout"),
        ("stop", true, true, true, "length"),
        ("stop", false, true, false, "tool_calls"),
        ("length", false, false, true, "stop"),
    ] {
        let mut s = state();
        s.guard_stop = Some("simhash_semantic_loop");
        s.tool_loop_capped = tool_cap;
        s.salvaged_tool_call = salvaged;
        s.stop_string_matched = matched;
        let (deltas, reason) = finalize_done(&mut s, &ctx(), scheduler, usage(), true, flush_sanitizer);
        assert_eq!(reason, expected);
        assert_eq!(s.salvaged_tool_call, salvaged);
        assert_finish_only(&deltas, expected);
    }
}

#[test]
fn terminal_done_refusal_annotation_preserves_full_content() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let text = "I cannot help with that request. Here is a safe alternative.";
    let mut s = state();
    s.tag_scan_buf = text.into();
    s.refusal_scan_buf = text.into();
    let (deltas, reason) = finalize_done(&mut s, &ctx(), "stop", usage(), false, flush_sanitizer);
    assert_eq!(reason, "stop");
    assert!(matches!(&deltas[..], [
        StreamDelta::Content { text: content, .. },
        StreamDelta::Refusal { text: refusal },
        StreamDelta::Finish { reason: crate::ir::FinishReason::Stop, .. }
    ] if content == text && refusal == "I cannot help with that request."));
}

fn fix_tokenizer() -> crate::tokenizer::ChatTokenizer {
    let dir = tempfile::tempdir().unwrap();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab({ let mut v = std::collections::HashMap::with_hasher(ahash::RandomState::default()); v.insert("[UNK]".to_string(), 0u32); ahash::AHashMap::from_iter(v) })
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let inner = tokenizers::Tokenizer::new(model);
    inner.save(dir.path().join("tokenizer.json"), false).unwrap();
    crate::tokenizer::ChatTokenizer::from_model_dir(dir.path(), 0, true, "glm5_next", "", None, false).unwrap()
}

/// Minimal ctx for the new signature: finalize_done reads only
/// `session_hash` (#55 registry) plus the explicit args.
fn ctx() -> StreamCtx {
    StreamCtx {
        _active_guard: crate::metrics::ActiveRequestGuard::new(),
        state: Arc::new(crate::api::test_app_state::app_state(fix_tokenizer())),
        model: "fixture".into(),
        id: "fixture".into(),
        prompt_len: 0,
        session_hash: 0,
        enable_thinking: false,
        tool_defs_for_backfill: vec![],
        cwd_for_normalize: None,
        stop_strings: vec![],
        tool_retry_enabled: false,
        prompt_tokens: Arc::new(vec![]),
        prompt_vocab: Arc::default(),
        grammar_spec: None,
        max_tokens: 64,
        timeout_at: None,
        stop_string_buffer_len: 0,
        leak_markers: crate::tool_parser::LeakMarkers::EMPTY,
        role_literal_leaks: false,
        wants_typed_arguments: false,
        max_tool_calls_per_response: 8,
        min_tokens: 0,
        req_return_token_ids: false,
        req_ctx: None,
        dump_seq: None,
    }
}
