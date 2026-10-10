// SPDX-License-Identifier: AGPL-3.0-only

/// Remove reasoning blocks from legacy completion output.
pub(crate) fn strip_thinking_tags(text: &str) -> String {
    // Plain completions retain token-decoded whitespace, including code
    // indentation and standalone whitespace tokens.
    if !text.contains("<think>") && !text.contains("</think>") {
        return text.to_owned();
    }
    let parser = crate::reasoning_parser::ReasoningFormat::Qwen.into_parser();
    parser.extract_thinking(text, false).1
}

#[cfg(test)]
mod tests {
    use super::strip_thinking_tags;

    #[test]
    fn preserves_completion_bytes_without_complete_reasoning_markers() {
        for text in [
            " Paris",
            " ",
            "\n",
            "\t answer \t\n",
            "    return 42\n",
            "<thin",
            "partial </thin",
            "",
        ] {
            assert_eq!(strip_thinking_tags(text), text, "completion {text:?}");
        }
    }

    #[test]
    fn removes_closed_and_prompt_opened_reasoning() {
        for text in ["<think>reason</think> answer ", "reason</think> answer "] {
            assert_eq!(strip_thinking_tags(text), "answer");
        }
    }

    #[test]
    fn hides_unfinished_reasoning_block() {
        assert_eq!(strip_thinking_tags("<think>unfinished"), "");
    }
}
