// SPDX-License-Identifier: AGPL-3.0-only

use super::reconcile_prompt_thinking;
use crate::tokenizer::ChatTokenizer;
use serde_json::json;
use tokenizers::{AddedToken, Tokenizer, models::wordlevel::WordLevel};

const START: u32 = 1;
const END: u32 = 2;

#[test]
fn rendered_glm_generation_tail_reconciles_requested_thinking() {
    // A tiny tokenizer preserves the real template's special-token boundary;
    // rendering and encoding run through the same public path as the API.
    let dir = tempfile::tempdir().unwrap();
    let model = WordLevel::builder()
        .vocab([("[UNK]".to_owned(), 0)].into_iter().collect())
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut inner = Tokenizer::new(model);
    inner
        .add_special_tokens([
            AddedToken::from("<think>", true),
            AddedToken::from("</think>", true),
            AddedToken::from("<|assistant|>", true),
        ])
        .unwrap();
    inner
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    let tokenizer = ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "glm5_next",
        Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .as_path(),
        ),
        false,
    )
    .unwrap();
    assert_eq!(tokenizer.encode("<think></think>").unwrap(), [START, END]);
    let messages = [json!({"role":"user","content":"Use get_weather for Oslo."})];
    let tools = [json!({"type":"function","function":{
        "name":"get_weather","parameters":{"type":"object","properties":{
            "city":{"type":"string"}},"required":["city"]}
    }})];
    for requested in [true, false] {
        let closed = tokenizer
            .apply_chat_template_openai(&messages, Some(&tools), requested, false)
            .unwrap();
        assert!(closed.ends_with(&[3, START, END]));
        assert_eq!(
            reconcile_prompt_thinking(&closed, Some(START), Some(END), requested, Some(32), 128),
            (false, None),
            "closed GLM tools tail must start generation outside reasoning"
        );
        let open = tokenizer
            .apply_chat_template_openai(&messages, None, requested, false)
            .unwrap();
        assert!(open.ends_with(&[3, START]));
        assert_eq!(
            reconcile_prompt_thinking(&open, Some(START), Some(END), requested, Some(32), 128),
            (true, Some(if requested { 32 } else { 128 }))
        );
    }
}

#[test]
fn unrelated_or_missing_prompt_markers_preserve_policy() {
    for enabled in [false, true] {
        for tokens in [&[9, 10][..], &[START, END, 3][..], &[END][..]] {
            assert_eq!(
                reconcile_prompt_thinking(tokens, Some(START), Some(END), enabled, Some(32), 128),
                (enabled, Some(32))
            );
        }
        assert_eq!(
            reconcile_prompt_thinking(&[START, END], None, Some(END), enabled, Some(32), 128),
            (enabled, Some(32))
        );
    }
    assert_eq!(
        reconcile_prompt_thinking(&[9, START], Some(START), None, false, None, 128),
        (true, Some(128))
    );
}
