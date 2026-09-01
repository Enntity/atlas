// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 chat-template compatibility.
//!
//! The upstream template unconditionally emits both a `Reasoning Effort`
//! system prefix and an open `<think>` block.  That makes the otherwise
//! standard `chat_template_kwargs.enable_thinking=false` request impossible
//! to honor and, more subtly, feeds seven extra tokens to the target and
//! DFlash drafter compared with the Mia/vLLM recipe.  Derive a narrowly-scoped
//! OpenAI variant from the checkpoint template so direct mode suppresses the
//! reasoning prefix and starts after a closed, empty reasoning block.

use anyhow::{Result, bail};

const GENERATION_ANCHOR: &str = "<|assistant|>{{- '<think>' -}}";
const OPENAI_GENERATION_BLOCK: &str = "<|assistant|>{%- if enable_thinking -%}\
{{- '<think>' -}}{%- else -%}{{- '<think></think>' -}}{%- endif -%}";
const REASONING_ANCHOR: &str = "{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}";
const OPENAI_REASONING_BLOCK: &str = "{%- if enable_thinking and effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}";

pub(super) fn derive_openai_template(checkpoint_template: &str) -> Result<String> {
    let generation_anchors = checkpoint_template.matches(GENERATION_ANCHOR).count();
    if generation_anchors != 1 {
        bail!(
            "GLM-5.3 template contract changed: expected exactly one generation anchor, found {generation_anchors}"
        );
    }
    let reasoning_anchors = checkpoint_template.matches(REASONING_ANCHOR).count();
    if reasoning_anchors != 1 {
        bail!(
            "GLM-5.3 template contract changed: expected exactly one reasoning anchor, found {reasoning_anchors}"
        );
    }
    Ok(checkpoint_template
        .replacen(REASONING_ANCHOR, OPENAI_REASONING_BLOCK, 1)
        .replacen(GENERATION_ANCHOR, OPENAI_GENERATION_BLOCK, 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(template: &str, enable_thinking: bool) -> String {
        let env = super::super::jinja_helpers::build_jinja_env(template).unwrap();
        env.get_template("chat")
            .unwrap()
            .render(minijinja::context! {
                messages => Vec::<serde_json::Value>::new(),
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
            })
            .unwrap()
    }

    #[test]
    fn direct_mode_suppresses_reasoning_prefix_and_closes_thinking() {
        let upstream = format!(
            "{{%- set effective_reasoning_effort = 'max' -%}}{REASONING_ANCHOR}{{% if add_generation_prompt %}}{GENERATION_ANCHOR}{{% endif %}}"
        );
        let adapted = derive_openai_template(&upstream).unwrap();
        assert_eq!(
            render(&adapted, true),
            "<|system|>Reasoning Effort: Max<|assistant|><think>"
        );
        assert_eq!(render(&adapted, false), "<|assistant|><think></think>");
    }

    #[test]
    fn changed_upstream_contract_fails_closed() {
        let err = derive_openai_template("<|assistant|>").unwrap_err();
        assert!(
            err.to_string()
                .contains("expected exactly one generation anchor")
        );
    }

    #[test]
    fn changed_reasoning_contract_fails_closed() {
        let err = derive_openai_template(GENERATION_ANCHOR).unwrap_err();
        assert!(
            err.to_string()
                .contains("expected exactly one reasoning anchor")
        );
    }
}
