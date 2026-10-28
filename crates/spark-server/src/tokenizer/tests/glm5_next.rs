// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 media markers through the checkpoint's own template. Prompt parity
//! with the reference engine for every request form is pinned end to end in
//! `api/chat/glm_prompt_parity_tests.rs`.

use serde_json::json;

fn render(messages: &[serde_json::Value]) -> String {
    super::qwen_dense::render_fixture(
        "glm-5.3-flash-fp4-local",
        messages,
        None,
        crate::tokenizer::chat_render::RenderFlags {
            enable_thinking: true,
            ..Default::default()
        },
    )
    .expect("GLM-5.3 checkpoint template renders")
}

fn render_with(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: crate::tokenizer::chat_render::RenderFlags<'_>,
) -> String {
    super::qwen_dense::render_fixture("glm-5.3-flash-fp4-local", messages, tools, flags)
        .expect("GLM-5.3 checkpoint template renders")
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
    for enabled in [false, true] {
        let rendered = render_with(
            &messages,
            Some(&tools),
            crate::tokenizer::chat_render::RenderFlags {
                enable_thinking: enabled,
                ..Default::default()
            },
        );
        let suffix = if enabled {
            "<|assistant|><think>"
        } else {
            "<|assistant|><think></think>"
        };
        assert!(rendered.ends_with(suffix), "resolved thinking={enabled}");
        assert_eq!(rendered.contains("Reasoning Effort: Max"), enabled);
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("get_weather"));
    }
}

#[test]
fn glm5_without_tools_opens_think_even_when_thinking_is_off() {
    let messages = [json!({"role": "user", "content": "What is the weather in Paris?"})];
    let off = render_with(
        &messages,
        None,
        crate::tokenizer::chat_render::RenderFlags {
            enable_thinking: false,
            ..Default::default()
        },
    );
    assert!(off.ends_with("<|assistant|><think>"));
    assert!(off.contains("Reasoning Effort: Low"));
    let on = render_with(
        &messages,
        None,
        crate::tokenizer::chat_render::RenderFlags {
            enable_thinking: true,
            ..Default::default()
        },
    );
    assert!(on.ends_with("<|assistant|><think>"));
    assert!(on.contains("Reasoning Effort: Max"));
}

#[test]
fn glm5_response_format_thinking_off_renders_the_empty_think_block() {
    let messages = [json!({"role": "user", "content": "Name one bridge."})];
    let flags = |enable_thinking| crate::tokenizer::chat_render::RenderFlags {
        enable_thinking,
        empty_think_off: true,
        ..Default::default()
    };
    let off = render_with(&messages, None, flags(false));
    assert!(off.ends_with("<|assistant|><think></think>"), "{off}");
    assert!(!off.contains("Reasoning Effort: Low"));
    assert_eq!(render_with(&messages, None, flags(true)), render(&messages));
}

#[test]
fn glm5_rendered_prompt_keeps_two_image_markers_in_order() {
    let messages = [json!({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "compare"},
            {"type": "image"}
        ]
    })];
    let rendered = render(&messages);
    let marker = "<|begin_of_image|><|image|><|end_of_image|>";
    assert_eq!(rendered.matches(marker).count(), 2, "{rendered}");
    assert!(!rendered.contains("<|vision_start|>"));
    assert!(!rendered.contains("<|image_pad|>"));
}

#[test]
fn glm5_multiturn_media_markers_are_not_grouped_or_dropped() {
    let messages = [
        json!({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "first"}]}),
        json!({"role": "assistant", "content": "noted"}),
        json!({"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image"}]}),
    ];
    let rendered = render(&messages);
    let marker = "<|begin_of_image|><|image|><|end_of_image|>";
    assert_eq!(rendered.matches(marker).count(), 2, "{rendered}");
    let first = rendered.find(marker).expect("first image marker");
    let second = rendered[first + marker.len()..]
        .find(marker)
        .map(|offset| first + marker.len() + offset)
        .expect("second image marker");
    assert!(first < second);
}

#[test]
fn glm5_video_template_emits_one_canonical_compact_marker() {
    // Temporal grouping and odd-frame repair happen in the processor; the
    // shipped chat template must contribute exactly one video triple for both
    // a two-frame clip and a clip whose sampled frames are repaired to a pair.
    for content in [
        json!([{"type": "video"}, {"type": "text", "text": "two frames"}]),
        json!([{"type": "video"}, {"type": "text", "text": "odd source"}]),
    ] {
        let rendered = render(&[json!({"role": "user", "content": content})]);
        assert_eq!(
            rendered
                .matches("<|begin_of_video|><|video|><|end_of_video|>")
                .count(),
            1,
            "{rendered}"
        );
        assert!(!rendered.contains("<|vision_start|>"));
    }
}
