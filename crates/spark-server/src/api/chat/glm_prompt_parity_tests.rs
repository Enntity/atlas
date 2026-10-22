// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash prompts are byte-identical to the reference engine's.
//!
//! `test_data/chat_templates/glm-5.3-flash-fp4-local.cases.json` holds, per
//! request form (no thinking flag, `enable_thinking` true/false, every
//! `reasoning_effort`, `chat_template_kwargs`, tools, `response_format`,
//! single and multi-turn with past reasoning and out-of-order parallel tool
//! results), the prompt vLLM renders from the checkpoint's own
//! `chat_template.jinja` (a byte copy sits beside it; regenerate the cases
//! with the `.cases.py` there). Each request runs the production path here —
//! wire request → IR → [`prepare_chat_prompt`] → tokens — through a
//! byte-level tokenizer, so decoding the prompt tokens is exact.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::prepare_chat_prompt;
use crate::AppState;

/// Every special token the template emits, so the prompt-tail reasoning
/// detection sees `<think>` / `</think>` as single tokens, as in the real
/// GLM tokenizer.
const SPECIAL: &[&str] = &[
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
];

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// GLM's MODEL.toml `[behavior]`, for the keys the prompt path reads.
fn glm_behavior() -> atlas_kernels::ModelBehavior {
    let b = crate::api::test_app_state::build_behavior::parse_behavior(&manifest_path(
        "../../kernels/gb10/glm-5.3-flash",
    ));
    assert_eq!(
        b.tool_call_parser, "poolside_v1",
        "actual GLM metadata read"
    );
    atlas_kernels::ModelBehavior {
        thinking_default: b.thinking_default,
        thinking_in_tools: b.thinking_in_tools,
        max_thinking_budget: b.max_thinking_budget,
        cap_thinking_at_max_tokens: b.cap_thinking_at_max_tokens,
        verbatim_reasoning_effort: b.verbatim_reasoning_effort,
        disable_cwd_hint_injection: b.disable_cwd_hint_injection,
        ..Default::default()
    }
}

/// A byte-level BPE with no merges plus the GLM special tokens: every
/// string round-trips through encode/decode unchanged.
fn byte_level_tokenizer(dir: &Path) -> crate::tokenizer::ChatTokenizer {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let vocab: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
    inner.with_decoder(Some(ByteLevel::default()));
    let special: Vec<_> = SPECIAL
        .iter()
        .map(|t| tokenizers::AddedToken::from(*t, true))
        .collect();
    inner.add_special_tokens(special).unwrap();
    inner.save(dir.join("tokenizer.json"), false).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    std::fs::copy(
        manifest_path("test_data/chat_templates/glm-5.3-flash-fp4-local.jinja"),
        dir.join("chat_template.jinja"),
    )
    .unwrap();
    // The repo root is searched for a template override, exactly as serve
    // does: GLM must find none and render the checkpoint's own template.
    // Tool JSON as the SparkGLM profiles serve it (ATLAS_USE_HF_REF_JSON_DUMPS=1).
    crate::tokenizer::ChatTokenizer::from_model_dir_with(
        dir,
        0,
        true,
        "glm5_next",
        "",
        Some(&manifest_path("../..")),
        false,
        crate::tokenizer::jinja_helpers::ToolJsonStyle::HfSpaced,
    )
    .unwrap()
}

fn glm_state(dir: &Path) -> Arc<AppState> {
    let tokenizer = byte_level_tokenizer(dir);
    let id = |t: &str| {
        let ids = tokenizer.encode(t).unwrap();
        assert_eq!(ids.len(), 1, "{t} is one token");
        ids[0]
    };
    Arc::new(AppState {
        think_start_token_id: Some(id("<think>")),
        think_end_token_id: Some(id("</think>")),
        tool_call_start_token_id: Some(id("<tool_call>")),
        tool_call_parser: Some(Arc::from(
            crate::tool_parser::ToolCallFormat::PoolsideV1.into_parser(),
        )),
        behavior: glm_behavior(),
        ..crate::api::test_app_state::app_state(tokenizer)
    })
}

#[test]
fn every_request_form_renders_the_reference_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state(dir.path());
    let cases: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(manifest_path(
            "test_data/chat_templates/glm-5.3-flash-fp4-local.cases.json",
        ))
        .unwrap(),
    )
    .unwrap();
    assert!(cases.len() >= 40, "case table present");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let wire: crate::openai::ChatCompletionRequest =
            serde_json::from_value(case["request"].clone()).unwrap();
        wire.validate_reasoning_effort().unwrap();
        let mut req: crate::ir::ChatRequest = wire.into();
        let Ok(prepared) = prepare_chat_prompt(&state, &mut req) else {
            panic!("{name}: prepare failed");
        };
        let prompt = state
            .tokenizer
            .decode_with_special(&prepared.prompt_tokens)
            .unwrap();
        assert_eq!(prompt, case["prompt"].as_str().unwrap(), "{name}");
        // The template opens reasoning on every request, and no server
        // budget closes it: the model ends it, or max_tokens does.
        assert!(prepared.enable_thinking, "{name}: reasoning is parsed");
        assert_eq!(prepared.thinking_budget, None, "{name}: no server budget");
    }
}

#[test]
fn an_explicit_client_budget_is_the_only_thinking_cap() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state(dir.path());
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm", "max_tokens": 4096, "thinking_token_budget": 300,
        "messages": [{"role": "user", "content": "Hi"}]
    }))
    .unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    let Ok(prepared) = prepare_chat_prompt(&state, &mut req) else {
        panic!("prepare failed");
    };
    assert!(prepared.enable_thinking);
    assert_eq!(prepared.thinking_budget, Some(300));
}

#[test]
fn literal_glm_tool_markup_stays_text_while_structured_call_stays_native() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state(dir.path());
    let tool_open = state.tokenizer.encode("<tool_call>").unwrap();
    assert_eq!(tool_open.len(), 1, "GLM opener must be one native token");
    let native_tool_open = tool_open[0];
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": "Quote <tool_call> and </tool_call> literally."
            },
            {
                "role": "assistant",
                "content": "I will check.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Oslo\"}"
                    }
                }]
            }
        ]
    }))
    .unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    let prepared = prepare_chat_prompt(&state, &mut req).expect("GLM prompt prepares");
    let prompt = state
        .tokenizer
        .decode_with_special(&prepared.prompt_tokens)
        .unwrap();
    assert!(
        prompt.contains("Quote <tool_call> and </tool_call> literally."),
        "literal tool markup must survive prompt decoding: {prompt:?}"
    );
    assert_eq!(
        prepared
            .prompt_tokens
            .iter()
            .filter(|&&id| id == native_tool_open)
            .count(),
        1,
        "only the structured historical call may use the native opener"
    );
}
