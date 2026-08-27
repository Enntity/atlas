// SPDX-License-Identifier: AGPL-3.0-only

//! Regression coverage for the GLM-5.3 OpenAI template override.

use serde_json::json;

fn render(messages: &[serde_json::Value], tools: Option<&[serde_json::Value]>) -> String {
    let raw = include_str!("../../../../../jinja-templates/openai/glm5_next.jinja");
    let converted = crate::tokenizer::jinja_helpers::convert_python_jinja_to_minijinja(raw);
    let env = crate::tokenizer::jinja_helpers::build_jinja_env(&converted)
        .expect("GLM-5.3 OpenAI template compiles");
    crate::tokenizer::chat_render::render_chat(
        &env,
        messages,
        tools,
        crate::tokenizer::chat_render::RenderFlags::default(),
    )
    .expect("GLM-5.3 OpenAI template renders")
}

#[test]
fn glm5_tools_close_reasoning_before_generation() {
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
    let rendered = render(&messages, Some(&tools));
    assert!(rendered.ends_with("<|assistant|><think></think>"));
    assert!(rendered.contains("<tools>"));
    assert!(rendered.contains("get_weather"));
}

#[test]
fn glm5_without_tools_preserves_stock_reasoning_prompt() {
    let messages = [json!({"role": "user", "content": "What is the weather in Paris?"})];
    let rendered = render(&messages, None);
    assert!(rendered.ends_with("<|assistant|><think>"));
}
