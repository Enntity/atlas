// SPDX-License-Identifier: AGPL-3.0-only
//! A forced tool grammar must not force a tool opener into Qwen reasoning.
use super::*;
use crate::scheduler::test_support::{GRAMMAR_TOOL_CALL as TOOL_CALL, THINK_END};
use crate::tool_parser::{FunctionDefinition, Qwen3CoderParser, ToolDefinition};

fn required(thinking: bool, native: Option<u32>) -> GrammarState {
    let spec = GrammarSpec::ToolCall {
        tools: vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: Some(serde_json::json!({"type":"object",
                    "properties":{"city":{"type":"string"}},"required":["city"]})),
            },
        }],
        parser: std::sync::Arc::new(Qwen3CoderParser),
        use_triggers: false,
    };
    compile_grammar_state(
        &mut engine(),
        &Some(spec),
        &[EOS],
        thinking,
        native,
        Some(TOOL_CALL),
        &mut ResponseSink::Blocking(None),
    )
    .unwrap()
    .unwrap()
}

#[test]
fn required_qwen_defers_first_token_mask_until_reasoning_ends() {
    let gs = required(true, None);
    assert!(!gs.masks_first_token());
    assert!(!gs.opens_output(TOOL_CALL));
    assert!(!gs.is_strict(), "preserve existing tool refusal policy");
    assert!(required(false, None).masks_first_token());
    assert!(
        required(true, Some(TOOL_CALL)).masks_first_token(),
        "native GLM boundary keeps its established first-token behavior"
    );
}

#[test]
fn required_qwen_reasoning_leaves_the_tool_matcher_at_its_start() {
    let sched = SchedCtx::for_test();
    let (mut a, _rx) = test_seq(vec![], 400, None, 8);
    a.finished = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.tools_present = true;
    a.tool_call_start_token = Some(TOOL_CALL);
    a.grammar_state = Some(required(true, None));
    for token in "Reason about <tool_call> as text.".bytes().map(u32::from) {
        emit_token(&mut a, token, None, &sched);
    }
    assert!(a.inside_thinking && a.grammar_state.is_some());
    assert!(!a.grammar_state.as_mut().unwrap().stop_legal(&[EOS]));
    emit_token(&mut a, THINK_END, None, &sched);
    emit_token(&mut a, TOOL_CALL, None, &sched);
    for token in
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
            .bytes()
            .map(u32::from)
    {
        emit_token(&mut a, token, None, &sched);
    }
    assert!(!a.inside_thinking && a.engine_error.is_none());
    assert!(a.grammar_state.as_mut().unwrap().stop_legal(&[EOS]));
}

#[test]
fn required_qwen_first_pick_cannot_be_a_raw_tool_opener() {
    use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
    let mut row = vec![0.0; 2048];
    row[EOS as usize] = 30.0;
    row[TOOL_CALL as usize] = 20.0;
    row[b'p' as usize] = 10.0;
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: vec![row],
            cursor: 0,
        })
    });
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    let mut gs = required(true, None);
    let picked = crate::scheduler::sample_step::sample_first_token(
        &model,
        spark_runtime::gpu::DevicePtr::NULL,
        0.0,
        0,
        1.0,
        0.0,
        &[EOS],
        Some(&mut gs),
        &SchedCtx::for_test().levers.sampling(),
    );
    SCRIPT.with(|s| *s.borrow_mut() = None);
    assert_eq!(
        picked.unwrap(),
        b'p' as u32,
        "suppress both EOS and the reasoning-only tool opener"
    );
    assert!(
        !gs.stop_legal(&[EOS]),
        "the output matcher has not advanced"
    );
}
