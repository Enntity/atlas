// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `serve.rs`.
//!
//! A sibling file rather than an inline `mod tests`, matching
//! `serve_load_tests.rs` and the rest of this directory: `serve.rs` is not on
//! the file-size-cap allow-list, and the vision-bound cases alone (since
//! moved to `processor_pixels_tests.rs` with their reader) would have
//! carried it past 500 lines.

use super::*;

// canonicalize_model_quant is exercised via integration through
// the server boot path; unit-testing it requires building
// ModelConfig which has no `Default` impl (it's intentionally
// bound to a loaded model). The pair-compatibility table is a
// pure function and worth a unit test.

#[test]
fn compat_self_pair() {
    assert!(quant_pair_compatible("nvfp4", "nvfp4"));
    assert!(quant_pair_compatible("fp8", "fp8"));
    assert!(quant_pair_compatible("bf16", "bf16"));
}

#[test]
fn compat_nvfp4_handles_fp8_and_bf16() {
    assert!(quant_pair_compatible("nvfp4", "fp8"));
    assert!(quant_pair_compatible("nvfp4", "bf16"));
}

// EXL3 kernels ship only in the exl3-labeled bundle
// (kernels/gb10/qwen3.8-flash-next/exl3/, composed on nvfp4/), so an exl3
// checkpoint needs that build: an nvfp4-only or BF16 build must refuse it at
// the pairing check rather than fail later on a missing `exl3` module.
#[test]
fn compat_exl3_model_needs_exl3_bundle() {
    assert!(quant_pair_compatible("exl3", "exl3"));
    assert!(!quant_pair_compatible("nvfp4", "exl3"));
    assert!(!quant_pair_compatible("bf16", "exl3"));
    assert!(!quant_pair_compatible("exl3", "nvfp4"));
}

// A `*` build compiles qwen3.8-flash-next as two variants, exl3 sorting first.
#[test]
fn pick_quant_variant_prefers_exact_then_compatible() {
    let built = ["exl3", "nvfp4"];
    assert_eq!(pick_quant_variant(&built, "exl3"), Some(0));
    assert_eq!(pick_quant_variant(&built, "nvfp4"), Some(1));
    assert_eq!(pick_quant_variant(&built, "fp8"), Some(1));
    assert_eq!(pick_quant_variant(&built, "bf16"), Some(1));
    assert_eq!(pick_quant_variant(&["nvfp4"], "exl3"), None);
    assert_eq!(pick_quant_variant(&["exl3"], "exl3"), Some(0));
}

#[test]
fn incompat_unknown_rejected() {
    assert!(!quant_pair_compatible("nvfp4", "gptq-4bit"));
    assert!(!quant_pair_compatible("fp8", "nvfp4"));
}

// ── --default-chat-template-kwargs parsing ─────────────────────────────
// The operator knob for the served default reasoning_effort (2026-08-15).
// Fail-fast contract: bad JSON, unknown keys, and unknown effort values
// abort startup — a typo'd operator default must never boot a server
// that silently serves a different tier (the pre-change parser warned
// and IGNORED).

#[test]
fn default_kwargs_reasoning_effort_sets_both_halves() {
    use crate::ir::{EffortLevel, ReasoningEffort, ThinkingDirective};
    let kw = parse_default_chat_template_kwargs(r#"{"reasoning_effort":"xhigh"}"#).unwrap();
    // The template string AND the budget directive come from one parse,
    // so effort-silent requests get the same tier on both paths.
    assert_eq!(kw.reasoning_effort, Some(ReasoningEffort::Max));
    assert_eq!(kw.thinking, ThinkingDirective::OnEffort(EffortLevel::XHigh));
    assert_eq!(kw.preserve_thinking, None);

    // "none" as the server default = thinking off by default.
    let kw = parse_default_chat_template_kwargs(r#"{"reasoning_effort":"none"}"#).unwrap();
    assert_eq!(kw.reasoning_effort, None);
    assert_eq!(kw.thinking, ThinkingDirective::Off);
}

#[test]
fn model_default_effort_is_template_only_and_fails_fast() {
    use crate::ir::ReasoningEffort;
    let with = |e: &'static str| atlas_kernels::ModelBehavior {
        default_reasoning_effort: e,
        ..Default::default()
    };
    // Unset: the template's own default applies.
    assert_eq!(model_default_reasoning_effort(&with("")).unwrap(), None);
    assert_eq!(
        model_default_reasoning_effort(&with("medium")).unwrap(),
        Some(ReasoningEffort::Medium)
    );
    assert_eq!(
        model_default_reasoning_effort(&with("max")).unwrap(),
        Some(ReasoningEffort::Max)
    );
    // Not a thinking-on tier, or not a tier at all: refuse to boot.
    assert!(model_default_reasoning_effort(&with("none")).is_err());
    assert!(model_default_reasoning_effort(&with("hihg")).is_err());
}

#[test]
fn default_kwargs_explicit_thinking_keys_outrank_effort_directive() {
    use crate::ir::{ReasoningEffort, ThinkingDirective};
    let kw =
        parse_default_chat_template_kwargs(r#"{"thinking_budget":512,"reasoning_effort":"low"}"#)
            .unwrap();
    // Budget rung wins for the directive; the effort string still sets
    // the template default.
    assert_eq!(kw.thinking, ThinkingDirective::On { budget: Some(512) });
    assert_eq!(kw.reasoning_effort, Some(ReasoningEffort::Low));
}

#[test]
fn default_kwargs_legacy_shapes_still_parse() {
    use crate::ir::ThinkingDirective;
    let kw = parse_default_chat_template_kwargs(r#"{"enable_thinking":true}"#).unwrap();
    assert_eq!(kw.thinking, ThinkingDirective::On { budget: None });
    let kw = parse_default_chat_template_kwargs(r#"{"enable_thinking":false}"#).unwrap();
    assert_eq!(kw.thinking, ThinkingDirective::Off);
    let kw = parse_default_chat_template_kwargs("").unwrap();
    assert_eq!(kw, DefaultChatTemplateKwargs::default());
    let kw = parse_default_chat_template_kwargs(r#"{"preserve_thinking":false}"#).unwrap();
    assert_eq!(kw.preserve_thinking, Some(false));
    let kw = parse_default_chat_template_kwargs(r#"'{"reasoning_effort":"low"}'"#).unwrap();
    assert_eq!(kw.reasoning_effort, Some(crate::ir::ReasoningEffort::Low));
}

#[test]
fn default_kwargs_fail_fast_on_typos() {
    // Unknown effort value.
    assert!(
        parse_default_chat_template_kwargs(r#"{"reasoning_effort":"hgih"}"#)
            .unwrap_err()
            .to_string()
            .contains("hgih")
    );
    // Unknown key (deny_unknown_fields): the old parser silently ignored
    // it, which is exactly how "--default-chat-template-kwargs
    // reasoning_effort=..." appeared to work while doing nothing.
    // ("reasoning_efforts" — plural — is the unknown-key stand-in; a true
    // misspelling here trips the typos CI lint.)
    assert!(parse_default_chat_template_kwargs(r#"{"reasoning_efforts":"low"}"#).is_err());
    // Invalid JSON.
    assert!(parse_default_chat_template_kwargs("not json").is_err());
}
