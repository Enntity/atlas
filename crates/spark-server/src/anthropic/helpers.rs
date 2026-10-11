// SPDX-License-Identifier: AGPL-3.0-only

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::tool_parser;

use super::types::*;

pub(super) fn anthropic_error(status: StatusCode, error_type: &str, message: String) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}

// ── Conversion helpers ──

impl From<&AnthropicTool> for tool_parser::ToolDefinition {
    /// Convert an Anthropic tool to an OpenAI-compatible tool definition.
    fn from(t: &AnthropicTool) -> Self {
        tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
            },
        }
    }
}

impl From<&AnthropicToolChoice> for tool_parser::ToolChoice {
    /// Convert an Anthropic tool_choice to an OpenAI-compatible tool_choice.
    fn from(tc: &AnthropicToolChoice) -> Self {
        match tc.choice_type.as_str() {
            "any" => tool_parser::ToolChoice::Mode("required".to_string()),
            "auto" => tool_parser::ToolChoice::Mode("auto".to_string()),
            "none" => tool_parser::ToolChoice::Mode("none".to_string()),
            "tool" => {
                if let Some(ref name) = tc.name {
                    tool_parser::ToolChoice::Specific {
                        function: tool_parser::ToolChoiceFunction { name: name.clone() },
                    }
                } else {
                    tool_parser::ToolChoice::Mode("auto".to_string())
                }
            }
            _ => tool_parser::ToolChoice::Mode("auto".to_string()),
        }
    }
}

/// Convert an OpenAI finish reason to Anthropic's stop_reason string.
pub(super) fn convert_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        // Safety-filtered output maps to Anthropic's dedicated refusal
        // stop reason (2025-05 API), not a normal end_turn — clients
        // branch on this to avoid retrying verbatim.
        "content_filter" => "refusal",
        // The Anthropic vocabulary has no deadline reason. "max_tokens" is
        // its only "the output is truncated" slot — mapping a deadline cut
        // to the default "end_turn" would tell the client the turn is
        // COMPLETE, which is the exact silent-truncation bug this reason
        // exists to prevent.
        crate::ir::FINISH_REASON_TIMEOUT => "max_tokens",
        // Not an Anthropic stop reason on purpose: an engine abort
        // must not read as end_turn (the truncation this exists for).
        "error" => "error",
        _ => "end_turn",
    }
}

/// Anthropic `(stop_reason, stop_sequence)` for a finished choice — the one
/// mapping shared by the non-streaming response (`translate.rs`) and the
/// streaming `message_delta` (`translator.rs`). A client stop sequence gets
/// the dedicated `"stop_sequence"` reason and is echoed ONLY on a `Stop`
/// finish; every other finish (tool_use, max_tokens, timeout, refusal)
/// reports `stop_sequence: null` even if a stop string was recorded.
pub(super) fn anthropic_stop(
    reason: &crate::ir::FinishReason,
    matched_stop: Option<String>,
) -> (&'static str, Option<String>) {
    match (reason, matched_stop) {
        (crate::ir::FinishReason::Stop, Some(s)) => ("stop_sequence", Some(s)),
        (r, _) => (convert_stop_reason(r.as_wire()), None),
    }
}

#[cfg(test)]
mod anthropic_stop_tests {
    use super::anthropic_stop;
    use crate::ir::{FINISH_REASON_TIMEOUT, FinishReason};

    #[test]
    fn maps_every_finish_reason() {
        let m = || Some("END".to_string());
        assert_eq!(
            anthropic_stop(&FinishReason::Stop, m()),
            ("stop_sequence", m())
        );
        assert_eq!(
            anthropic_stop(&FinishReason::Stop, None),
            ("end_turn", None)
        );
        // Old non-streaming path echoed the stop here (translate.rs).
        assert_eq!(
            anthropic_stop(&FinishReason::ToolCalls, m()),
            ("tool_use", None)
        );
        assert_eq!(
            anthropic_stop(&FinishReason::Length, m()),
            ("max_tokens", None)
        );
        assert_eq!(
            anthropic_stop(&FinishReason::ContentFilter, None),
            ("refusal", None)
        );
        // Old non-streaming path mapped every `Other` to end_turn, telling the
        // client a deadline-truncated turn was complete.
        assert_eq!(
            anthropic_stop(&FinishReason::Other(FINISH_REASON_TIMEOUT.into()), m()),
            ("max_tokens", None)
        );
        // Engine abort: never end_turn, never echoes a stop.
        assert_eq!(anthropic_stop(&FinishReason::Error, m()), ("error", None));
        assert_eq!(
            anthropic_stop(&FinishReason::Other("x".into()), None),
            ("end_turn", None)
        );
    }
}
