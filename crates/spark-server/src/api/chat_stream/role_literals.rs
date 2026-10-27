// SPDX-License-Identifier: AGPL-3.0-only

/// Drop a delta that is nothing but a bare role literal (`user` /
/// `assistant` / `tool`) — a Qwen3.5/3.6 hallucination leak, companion to
/// the scheduler-side `<|im_start|>` hard-stop.
///
/// `inside_tool_call` MUST be true whenever the streaming tool-call
/// detector is mid-body (between `<tool_call>` and its close). Issue #222:
/// for a `tool_*`-prefixed tool name (`tool_search`, `tool_call`,
/// `tool_describe`) the byte-level BPE tokenizer emits a standalone `tool`
/// token as the leading fragment of the NAME. Without the guard this strip
/// clears that fragment, and the detector reassembles the name from the
/// remainder (`_search`), truncating the streamed tool-call name by exactly
/// `len("tool") == 4` chars. Non-streaming was unaffected because it parses
/// the whole buffer at once. The guard confines the strip to genuine
/// content leaks (no tool call in flight).
pub(in crate::api::chat_stream) fn strip_bare_role_literal(
    delta: &mut String,
    inside_tool_call: bool,
) {
    if inside_tool_call {
        return;
    }
    let trimmed = delta.trim();
    if delta.len() < 20 && matches!(trimmed, "user" | "assistant" | "tool") {
        tracing::debug!("role-literal strip: dropped bare '{trimmed}' delta");
        delta.clear();
    }
}

#[cfg(test)]
mod role_literal_strip_tests {
    use super::strip_bare_role_literal;
    use crate::tool_parser::{DetectorOutput, StreamingToolDetector};

    /// Faithfully mirror the `handle_token` content-phase pipeline for the
    /// two steps under test: the bare role-literal strip (guarded by the
    /// detector's `inside_tool_call()`) feeding the surviving delta into the
    /// streaming tool-call detector. Returns every `ToolCallStart` name.
    fn stream_names(chunks: &[&str]) -> Vec<String> {
        let mut det = StreamingToolDetector::new();
        let mut names = Vec::new();
        for &c in chunks {
            let mut delta = c.to_string();
            // Same call the real content phase makes, in the same order:
            // read the detector's in-body flag BEFORE feeding this delta.
            let inside_tool_call = det.inside_tool_call();
            strip_bare_role_literal(&mut delta, inside_tool_call);
            if delta.is_empty() {
                continue;
            }
            for o in det.process(&delta) {
                if let DetectorOutput::ToolCallStart { name, .. } = o {
                    names.push(name);
                }
            }
        }
        for o in det.flush() {
            if let DetectorOutput::ToolCallStart { name, .. } = o {
                names.push(name);
            }
        }
        names
    }

    /// Issue #222: a `tool_search` NAME whose leading `tool` arrives as a
    /// standalone BPE fragment must stream intact, not truncate to `_search`.
    #[test]
    fn tool_search_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool", // standalone BPE fragment of the NAME
            "_search\", \"arguments\": {\"query\": \"CRM\"}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_search".to_string()]);
    }

    /// `tool_call` — the name that collides most directly with the markup.
    #[test]
    fn tool_call_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool",
            "_call\", \"arguments\": {}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_call".to_string()]);
    }

    /// `tool_describe` — same class, different suffix.
    #[test]
    fn tool_describe_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool",
            "_describe\", \"arguments\": {\"id\": 7}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_describe".to_string()]);
    }

    /// Non-`tool_*` name fed in fragments must remain intact (control).
    #[test]
    fn ordinary_name_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"get",
            "_weather\", \"arguments\": {\"city\": \"NYC\"}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["get_weather".to_string()]);
    }

    /// The #204/#205/#206 leak suppression the guard must NOT regress: a bare
    /// role literal in genuine content (no tool call in flight) is still
    /// cleared. Verified for all three literals and for a `tool` fragment
    /// that is only a real leak when `inside_tool_call == false`.
    #[test]
    fn bare_role_literal_still_stripped_outside_tool_call() {
        for lit in ["user", "assistant", "tool", "  tool  "] {
            let mut d = lit.to_string();
            strip_bare_role_literal(&mut d, false);
            assert!(
                d.is_empty(),
                "bare role literal {lit:?} must be stripped in content"
            );
        }
    }

    /// Inside a tool-call body the same fragments must survive untouched.
    #[test]
    fn bare_role_literal_preserved_inside_tool_call() {
        for lit in ["user", "assistant", "tool"] {
            let mut d = lit.to_string();
            strip_bare_role_literal(&mut d, true);
            assert_eq!(d, lit, "fragment {lit:?} must survive inside a tool call");
        }
    }

    #[test]
    fn dsml_split_opener_preserves_tool_fragment_and_streams_call() {
        let mut det = StreamingToolDetector::new();
        let chunks = [
            "\n\n",
            "<｜DSM",
            "L",
            "｜",
            "tool",
            "_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ];
        let mut outputs = Vec::new();
        for chunk in chunks {
            let mut delta = chunk.to_string();
            strip_bare_role_literal(&mut delta, det.inside_tool_call());
            if !delta.is_empty() {
                outputs.extend(det.process(&delta));
            }
        }
        outputs.extend(det.flush());

        let mut content = String::new();
        let mut calls = Vec::new();
        for output in outputs {
            match output {
                DetectorOutput::Content(text) => content.push_str(&text),
                DetectorOutput::ToolCall(call, index) => calls.push((index, call)),
                _ => {}
            }
        }
        assert!(content.is_empty(), "DSML envelope leaked: {content:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1.function.name, "get_weather");
        assert_eq!(calls[0].1.function.arguments, r#"{"city":"Paris"}"#);
    }

    /// Ordinary content (not a bare role literal) is never touched, in or out
    /// of a tool call.
    #[test]
    fn ordinary_content_untouched() {
        for inside in [false, true] {
            let mut d = "the tool ran".to_string();
            strip_bare_role_literal(&mut d, inside);
            assert_eq!(d, "the tool ran");
        }
    }
}
