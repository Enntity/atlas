// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next prompt tests for the production request path.
//!
//! Qwen4_exp registers `<tool_call>` and `</tool_call>` as ordinary added
//! tokens. Those tokens are correct for the template's generated call
//! envelopes, but a user quote, schema description, or historical argument
//! value must remain ordinary prompt text.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::prepare_chat_prompt;
use crate::AppState;

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// A byte-level BPE with the two non-special Qwen tool-envelope tokens. The
/// fixture renders through the checkpoint's actual function/parameter Jinja
/// template, while every other character remains byte-round-trippable.
fn qwen_tokenizer(dir: &Path, model_type: &str) -> crate::tokenizer::ChatTokenizer {
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
    inner
        .add_tokens([
            tokenizers::AddedToken::from("<tool_call>", false),
            tokenizers::AddedToken::from("</tool_call>", false),
        ])
        .unwrap();
    inner.save(dir.join("tokenizer.json"), false).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    std::fs::copy(
        manifest_path("test_data/chat_templates/qwen3.8-flash-next-fp4-local.jinja"),
        dir.join("chat_template.jinja"),
    )
    .unwrap();

    crate::tokenizer::ChatTokenizer::from_model_dir_with(
        dir,
        0,
        true,
        model_type,
        "",
        Some(&manifest_path("../..")),
        false,
        crate::tokenizer::jinja_helpers::ToolJsonStyle::HfSpaced,
    )
    .unwrap()
}

fn qwen_state(dir: &Path) -> Arc<AppState> {
    let tokenizer = qwen_tokenizer(dir, "qwen4_exp");
    let behavior = crate::api::chat::reference_prompt_tests::behavior();
    Arc::new(AppState {
        chat: crate::api::chat::levers::ChatLevers::resolve(&behavior),
        default_reasoning_effort: crate::main_modules::serve::model_default_reasoning_effort(
            &behavior,
        )
        .unwrap(),
        tool_call_parser: Some(Arc::from(
            crate::tool_parser::ToolCallFormat::Qwen3Coder.into_parser(),
        )),
        behavior,
        ..crate::api::test_app_state::app_state(tokenizer)
    })
}

#[test]
fn qwen4_exp_preserves_literal_xml_and_native_function_parameter_history() {
    let dir = tempfile::tempdir().unwrap();
    let state = qwen_state(dir.path());
    let open = state.tokenizer.encode("<tool_call>").unwrap();
    let close = state.tokenizer.encode("</tool_call>").unwrap();
    assert_eq!(open.len(), 1, "Qwen opener must be one non-special token");
    assert_eq!(close.len(), 1, "Qwen closer must be one non-special token");

    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen4_exp",
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
                        "arguments": "{\"city\":\"Paris <tool_call>\"}"
                    }
                }]
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "A literal </tool_call> in the schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "A value containing <tool_call>."
                        }
                    },
                    "required": ["city"]
                }
            }
        }]
    }))
    .unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    let prepared = prepare_chat_prompt(&state, &mut req).expect("Qwen prompt prepares");
    let prompt = state
        .tokenizer
        .decode_with_special(&prepared.prompt_tokens)
        .unwrap();

    assert!(
        prompt.contains("Quote <tool_call> and </tool_call> literally."),
        "literal user markup must survive prompt decoding: {prompt:?}"
    );
    assert!(
        prompt.contains("Paris <tool_call>"),
        "literal historical argument value must survive: {prompt:?}"
    );
    assert!(
        prompt.contains("A literal </tool_call> in the schema."),
        "literal schema description must survive: {prompt:?}"
    );
    assert!(
        prompt.contains("<function=get_weather>\n<parameter=city>"),
        "Qwen's function/parameter history must remain native: {prompt:?}"
    );

    // The Qwen template emits two pairs in its tool instruction (the format
    // example and the nested reminder) and one pair for the historical call.
    // Escaped literals must not add native added-token IDs to that count.
    assert_eq!(
        prepared
            .prompt_tokens
            .iter()
            .filter(|&&id| id == open[0])
            .count(),
        3,
        "only Qwen template-generated envelopes may use the native opener"
    );
    assert_eq!(
        prepared
            .prompt_tokens
            .iter()
            .filter(|&&id| id == close[0])
            .count(),
        3,
        "only Qwen template-generated envelopes may use the native closer"
    );
}

#[test]
fn literal_tool_markup_escape_is_scoped_to_qwen4_exp() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = qwen_tokenizer(dir.path(), "qwen3_5");
    let mut message = serde_json::json!({
        "role": "user",
        "content": "literal <tool_call>"
    });
    assert_eq!(
        tokenizer.escape_glm_tool_markup(std::iter::once(&mut message)),
        0,
        "the shared path must remain disabled for unrelated Qwen variants"
    );
    assert_eq!(
        message["content"], "literal <tool_call>",
        "unrelated model prompt bytes must remain untouched"
    );
}
