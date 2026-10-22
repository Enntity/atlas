// SPDX-License-Identifier: AGPL-3.0-only
//! Token -> Done regressions for the Qwen XML failures found on the pair.
use super::*;
use crate::api::chat_stream::handle_done::handle_done;

fn qwen() -> Arc<AppState> {
    app_for_model("qwen4_exp", tool_parser::ToolCallFormat::Qwen3Coder)
}

fn run(tools: bool, tokens: Vec<u32>) -> Vec<ir::StreamDelta> {
    let count = tokens.len();
    let (ctx, mut stream, mut deltas) = stream_for_app(qwen(), tools, tokens);
    deltas.extend(handle_done(
        &mut stream,
        &ctx,
        "stop".into(),
        count,
        1.0,
        1.0,
        0,
        0,
        0,
    ));
    assert!(!stream.cancel_flag.load(Ordering::Acquire));
    deltas
}

#[test]
fn qwen_literal_xml_preserves_split_role_word_with_and_without_tools() {
    for tools in [false, true] {
        for (open, marker) in [(24, "</tool_call>"), (30, "<tool_call>")] {
            let deltas = run(tools, vec![END, 23, open, 25, 26, 27]);
            let content: String = deltas
                .iter()
                .filter_map(|d| match d {
                    ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                content,
                format!("The tag is `{marker}` and this is prose. END."),
                "{deltas:?}"
            );
            assert!(
                !deltas
                    .iter()
                    .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
            );
        }
    }
}

#[test]
fn qwen_literal_xml_plain_opener_survives_detector_done_flush() {
    let deltas = run(true, vec![END, 28, TOOL, 29]);
    let content: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        content, "The delimiter <tool_call> opens a tool invocation. END.",
        "{deltas:?}"
    );
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
    );
}

#[test]
fn qwen_literal_xml_fix_keeps_real_function_parameter_calls() {
    let deltas = run(true, vec![END, TOOL, 31, 11]);
    let names: Vec<_> = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["get_weather"], "{deltas:?}");
    let args: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&args).unwrap(),
        json!({"city":"Oslo"})
    );
    assert!(deltas.iter().any(
        |d| matches!(d, ir::StreamDelta::Finish { reason, .. } if reason.as_wire() == "tool_calls")
    ));
}

#[test]
fn qwen_literal_xml_does_not_disable_legacy_chatml_filtering() {
    assert!(!qwen().tokenizer.filters_bare_role_literals());
    assert!(!app(true).tokenizer.filters_bare_role_literals());
    assert!(app(false).tokenizer.filters_bare_role_literals());
    for word in ["tool", "user", "assistant"] {
        let mut text = word.to_owned();
        super::super::handle_token::strip_bare_role_literal(&mut text, false);
        assert!(text.is_empty());
    }
}

#[test]
fn qwen_literal_xml_quoted_fenced_and_malformed_examples_stay_content() {
    // Existing fixture tokens contain complete literal examples; the split
    // token regression above covers the separate role-word emission.
    for token in [17, 18, 20, 32] {
        let expected = qwen().tokenizer.decode(&[token]).unwrap();
        let deltas = run(true, vec![END, token]);
        let content: String = deltas
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, expected, "{deltas:?}");
        assert!(
            !deltas
                .iter()
                .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
        );
    }
}

#[test]
fn qwen_literal_xml_unsupported_closed_and_unclosed_envelopes_are_content() {
    for token in [33, 34] {
        let expected = qwen().tokenizer.decode(&[token]).unwrap();
        let deltas = run(true, vec![END, token]);
        let content: String = deltas
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, expected, "{deltas:?}");
        assert!(
            !deltas
                .iter()
                .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
        );
    }
}
