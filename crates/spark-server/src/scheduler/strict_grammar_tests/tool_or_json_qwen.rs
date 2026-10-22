// SPDX-License-Identifier: AGPL-3.0-only

//! The tool-call-or-JSON union in Qwen's format through the scheduler.
//! Qwen closes reasoning with `</think>` before a call (it has no native
//! opener that ends reasoning, unlike GLM), so the union starts at the
//! first token after `</think>` and the model's mass picks the branch there.

use super::*;
use crate::scheduler::test_support::{GRAMMAR_TOOL_CALL as TOOL_CALL, THINK_END};
use crate::tool_parser::{FunctionDefinition, Qwen3CoderParser, ToolDefinition};

const CALL_BODY: &str = "\n<function=get_weather>\n<parameter=location>\nBoston\n</parameter>\n</function>\n</tool_call>";

fn qwen_union(opens_in_thinking: bool) -> GrammarState {
    let spec = GrammarSpec::ToolCallOrJson {
        tools: vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
            },
        }],
        parser: std::sync::Arc::new(Qwen3CoderParser),
        schema: Some(SCHEMA.into()),
    };
    let mut sink = ResponseSink::Blocking(None);
    // No native opener: `glm_tool_boundary` is GLM's alone.
    compile_grammar_state(
        &mut engine(),
        &Some(spec),
        &[EOS],
        opens_in_thinking,
        None,
        None,
        &mut sink,
    )
    .unwrap()
    .expect("qwen3_coder has a tool grammar")
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// A thinking turn: reasoning, `</think>`, then the output under the union.
fn after_think(output: &[u32]) -> crate::scheduler::types::ActiveSeq {
    let sched = SchedCtx::for_test();
    let (mut a, _rx) = test_seq(vec![], 400, None, 8);
    a.finished = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.tools_present = true;
    a.tool_call_start_token = Some(TOOL_CALL);
    a.grammar_state = Some(qwen_union(true));
    for &t in bytes("calls it.")
        .iter()
        .chain([THINK_END].iter())
        .chain(output)
    {
        emit_token(&mut a, t, None, &sched);
    }
    a
}

#[test]
fn a_qwen_call_after_think_completes_the_tool_branch() {
    let mut output = vec![TOOL_CALL];
    output.extend(bytes(CALL_BODY));
    let mut a = after_think(&output);
    assert!(!a.inside_thinking);
    assert!(
        !a.finished && a.engine_error.is_none(),
        "{:?}",
        a.engine_error
    );
    assert!(
        a.grammar_state.as_mut().unwrap().stop_legal(&[EOS]),
        "the call is complete"
    );
}

#[test]
fn a_qwen_json_answer_after_think_streams_and_a_fence_fails_loud() {
    let a = after_think(&bytes(r#"{"bridge":"Tower"}"#));
    assert!(
        !a.finished && a.engine_error.is_none(),
        "{:?}",
        a.engine_error
    );

    let a = after_think(&bytes("```"));
    assert!(a.finished);
    assert!(
        a.engine_error
            .as_deref()
            .is_some_and(|e| e.contains("response_format"))
    );
}

/// Inside `<think>` the opener is reasoning for Qwen: the union neither
/// consumes it nor takes it as output.
#[test]
fn qwen_reasoning_is_not_the_unions_first_token() {
    let gs = qwen_union(true);
    assert!(gs.is_strict());
    assert!(!gs.masks_first_token(), "reasoning first");
    assert!(!gs.opens_output(TOOL_CALL), "no native opener for Qwen");
    assert!(qwen_union(false).masks_first_token());
}
