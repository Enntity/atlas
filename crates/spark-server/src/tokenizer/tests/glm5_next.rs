// SPDX-License-Identifier: AGPL-3.0-only

//! Regression coverage for the GLM-5.3 OpenAI template override.

use serde_json::json;

fn render(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    let raw = include_str!("../../../../../jinja-templates/openai/glm5_next.jinja");
    let converted = crate::tokenizer::jinja_helpers::convert_python_jinja_to_minijinja(raw);
    let env = crate::tokenizer::jinja_helpers::build_jinja_env(&converted)
        .expect("GLM-5.3 OpenAI template compiles");
    crate::tokenizer::chat_render::render_chat(
        &env,
        messages,
        tools,
        crate::tokenizer::chat_render::RenderFlags {
            enable_thinking,
            ..Default::default()
        },
    )
    .expect("GLM-5.3 OpenAI template renders")
}

#[test]
fn glm5_tools_respect_resolved_thinking_before_generation() {
    let tools = [json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];
    let messages = [json!({"role": "user", "content": "What is the weather in Paris?"})];
    // The API resolver supplies false for silent clients under the GLM tool
    // default, but explicit enable wins that default (thinking.rs tests).
    for enabled in [false, true] {
        let rendered = render(&messages, Some(&tools), enabled);
        let suffix = if enabled {
            "<|assistant|><think>"
        } else {
            "<|assistant|><think></think>"
        };
        assert!(rendered.ends_with(suffix), "resolved thinking={enabled}");
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("get_weather"));
    }
}

#[test]
fn glm5_without_tools_preserves_stock_reasoning_prompt() {
    let messages = [json!({"role": "user", "content": "What is the weather in Paris?"})];
    for enabled in [false, true] {
        let rendered = render(&messages, None, enabled);
        assert!(rendered.ends_with("<|assistant|><think>"));
    }
}
