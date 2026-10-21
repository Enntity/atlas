// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise both completion handlers through the real scheduler request
//! boundary: speculation policy, original token IDs and early rejection.

use super::*;
use crate::api::inference_types::{InferenceRequest, InferenceResponse};
use crate::main_modules::model_host::CurrentModel;
use serde_json::json;

fn tokenizer() -> crate::tokenizer::ChatTokenizer {
    let dir = tempfile::tempdir().unwrap();
    let vocab = [
        "[UNK]", "answer", "<think>", "secret", "</think>", "STOP", "prompt",
    ]
    .iter()
    .enumerate()
    .map(|(i, w)| (w.to_string(), i as u32))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
    inner
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "glm5_next",
        "",
        None,
        false,
    )
    .unwrap()
}

#[tokio::test]
async fn completion_disable_mtp_reaches_scheduler_for_blocking_and_streaming() {
    for stream in [false, true] {
        // Omitted is the default-on speculative behavior; true is an
        // explicit request-level opt-out and must reach both request kinds.
        for disabled in [false, true] {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let state = Arc::new(AppState {
                request_tx: tx,
                ..crate::api::test_app_state::app_state(tokenizer())
            });
            let mut body = json!({"model": "fixture", "prompt": [1],
                                  "max_tokens": 1, "stream": stream,
                                  "temperature": 0.7, "top_k": 20, "top_p": 0.95});
            if disabled {
                body["disable_mtp"] = json!(true);
            }
            let req: CompletionRequest = serde_json::from_value(body).unwrap();
            let scheduler = tokio::spawn(async move {
                let request: InferenceRequest = rx.recv().await.unwrap();
                assert_eq!(request.disable_mtp(), disabled);
                match request {
                    InferenceRequest::Blocking {
                        temperature,
                        top_k,
                        top_p,
                        response_tx,
                        ..
                    } => {
                        assert!(!stream);
                        assert_eq!((temperature, top_k, top_p), (0.7, 20, 0.95));
                        let result = response_tx.send(Ok(InferenceResponse {
                            output_tokens: vec![1],
                            finish_reason: "length".into(),
                            time_to_first_token_ms: 0.0,
                            decode_time_ms: 0.0,
                            logprobs: vec![],
                            reasoning_tokens: 0,
                            cached_prompt_tokens: 0,
                            accepted_prediction_tokens: 0,
                            prompt_logprobs: vec![],
                        }));
                        assert!(result.is_ok());
                    }
                    InferenceRequest::Streaming {
                        temperature,
                        top_k,
                        top_p,
                        ..
                    } => {
                        assert!(stream);
                        assert_eq!((temperature, top_k, top_p), (0.7, 20, 0.95));
                    }
                }
            });
            let response =
                crate::api::completions::completions(CurrentModel(state), Ok(axum::Json(req)))
                    .await;
            assert_eq!(response.status(), StatusCode::OK);
            scheduler.await.unwrap();
        }
    }
}

#[tokio::test]
async fn completion_token_ids_handler_preserves_ids_usage_and_default_omission() {
    for requested in [false, true] {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let state = Arc::new(AppState {
            request_tx: tx,
            ..crate::api::test_app_state::app_state(tokenizer())
        });
        let mut body = json!({"model": "fixture", "prompt": [6], "echo": true,
                              "max_tokens": 5, "stop": ["STOP"]});
        if requested {
            body["return_token_ids"] = json!(true);
        }
        let req: CompletionRequest = serde_json::from_value(body).unwrap();
        let scheduler = tokio::spawn(async move {
            let InferenceRequest::Blocking {
                prompt_tokens,
                echo,
                response_tx,
                ..
            } = rx.recv().await.unwrap()
            else {
                panic!("expected a blocking scheduler request");
            };
            assert_eq!(prompt_tokens.as_slice(), &[6]);
            assert!(echo);
            assert!(
                response_tx
                    .send(Ok(InferenceResponse {
                        // Decodes to <think>secret</think>answerSTOP. Text processing
                        // removes reasoning and STOP, while IDs retain all five tokens.
                        output_tokens: vec![2, 3, 4, 1, 5],
                        finish_reason: "stop".into(),
                        time_to_first_token_ms: 0.0,
                        decode_time_ms: 0.0,
                        logprobs: vec![],
                        reasoning_tokens: 3,
                        cached_prompt_tokens: 0,
                        accepted_prediction_tokens: 0,
                        prompt_logprobs: vec![],
                    }))
                    .is_ok()
            );
        });
        let response =
            crate::api::completions::completions(CurrentModel(state), Ok(axum::Json(req))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(wire["choices"][0]["text"], "promptanswer");
        assert_eq!(wire["choices"][0]["finish_reason"], "stop");
        assert_eq!(wire["choices"][0].get("token_ids").is_some(), requested);
        if requested {
            assert_eq!(wire["choices"][0]["token_ids"], json!([2, 3, 4, 1, 5]));
        }
        assert_eq!(wire["usage"]["prompt_tokens"], 1);
        assert_eq!(wire["usage"]["completion_tokens"], 5);
        assert_eq!(wire["usage"]["total_tokens"], 6);
        assert_eq!(
            wire["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );
        scheduler.await.unwrap();
    }
}

#[tokio::test]
async fn completion_token_ids_stream_is_rejected_before_scheduler_enqueue() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(AppState {
        request_tx: tx,
        ..crate::api::test_app_state::app_state(tokenizer())
    });
    let req: CompletionRequest = serde_json::from_value(json!({
        "model": "fixture", "prompt": [1], "max_tokens": 1,
        "stream": true, "return_token_ids": true
    }))
    .unwrap();
    let response =
        crate::api::completions::completions(CurrentModel(state.clone()), Ok(axum::Json(req)))
            .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    let bytes = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        wire["error"]["message"],
        "return_token_ids requires stream=false on /v1/completions"
    );
}
