// SPDX-License-Identifier: AGPL-3.0-only

//! Route request shapes before EP allocation or any prefill.
use crate::api::{InferenceRequest, StreamEvent};
use crate::scheduler::types::ResponseSink;

#[derive(Default)]
struct Shape {
    grammar: bool,
    tools: bool,
    disabled: bool,
    suppress: bool,
    long_context: bool,
    adapter: bool,
    vision: bool,
    beam: bool,
}

fn allowed(s: &Shape) -> bool {
    !fallback_needed(s) && !s.adapter && !s.beam
}

/// Shapes that remain valid requests but must use the native serial decoder.
/// They cannot enter the repaired GLM MTP collective because grammar/tool
/// masks, vision rows, request-local MTP opt-outs, and budgets beyond the
/// repair verifier's indexed 32K domain all change the command or state
/// contract of that collective.
fn fallback_needed(s: &Shape) -> bool {
    s.grammar || s.tools || s.disabled || s.suppress || s.long_context || s.vision
}

pub(super) fn filter(requests: Vec<InferenceRequest>, enabled: bool) -> Vec<InferenceRequest> {
    if !enabled {
        return requests;
    }
    requests
        .into_iter()
        .filter_map(|req| {
            let shape = Shape {
                grammar: req.has_grammar_spec(),
                tools: req.tools_present() || req.require_tool_call(),
                disabled: req.disable_mtp(),
                suppress: req.suppress_tool_call(),
                long_context: req.prompt_len().saturating_add(req.max_tokens())
                    > spark_model::speculative::glm_repair_policy::MAX_LONG_CONTEXT,
                adapter: req.adapter_slot() >= 0,
                vision: req.has_image_pixels(),
                beam: req.num_beams() > 1,
            };
            if allowed(&shape) {
                return Some(req);
            }
            if fallback_needed(&shape) && !shape.adapter && !shape.beam {
                let mut req = req;
                req.disable_mtp_for_fallback();
                tracing::debug!(
                    grammar = shape.grammar,
                    tools = shape.tools,
                    explicit_disable = shape.disabled,
                    suppress = shape.suppress,
                    long_context = shape.long_context,
                    vision = shape.vision,
                    "GLM repair request routed through native decode fallback"
                );
                return Some(req);
            }
            reject(match req {
                InferenceRequest::Streaming { token_tx, .. } => ResponseSink::Streaming(token_tx),
                InferenceRequest::Blocking { response_tx, .. } => {
                    ResponseSink::Blocking(Some(response_tx))
                }
            });
            None
        })
        .collect()
}

fn reject(sink: ResponseSink) {
    let error = "GLM MTP repair rejects adapter and beam requests; grammar, tools, vision, and explicit MTP opt-outs use native decode fallback";
    match sink {
        ResponseSink::Streaming(tx) => {
            let _ = tx.blocking_send(StreamEvent::Error(error.into()));
        }
        ResponseSink::Blocking(Some(tx)) => {
            let _ = tx.send(Err(anyhow::anyhow!(error)));
        }
        ResponseSink::Blocking(None) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocking_tools_request() -> InferenceRequest {
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        InferenceRequest::Blocking {
            prompt_tokens: std::sync::Arc::new(vec![1, 2, 3]),
            session_hash: 0,
            adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            image_pixels: vec![],
            max_tokens: 4,
            min_tokens: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            logit_bias: vec![],
            stop_tokens: vec![],
            enable_thinking: false,
            thinking_budget: None,
            repetition_detection: None,
            require_tool_call: false,
            tools_present: true,
            suppress_tool_call: false,
            disable_mtp: false,
            grammar_spec: None,
            seed: Some(1),
            top_logprobs: None,
            prompt_logprobs: None,
            echo: false,
            timeout_at: None,
            response_tx,
        }
    }

    #[test]
    fn plain_shape_enters_repair_lane() {
        assert!(allowed(&Shape::default()));
    }

    #[test]
    fn fallback_marks_request_for_native_decode() {
        let filtered = filter(vec![blocking_tools_request()], true);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].disable_mtp());
    }

    #[test]
    fn grammar_tools_disable_suppress_and_vision_need_native_fallback() {
        for shape in [
            Shape {
                grammar: true,
                ..Shape::default()
            },
            Shape {
                tools: true,
                ..Shape::default()
            },
            Shape {
                disabled: true,
                ..Shape::default()
            },
            Shape {
                suppress: true,
                ..Shape::default()
            },
            Shape {
                long_context: true,
                ..Shape::default()
            },
            Shape {
                vision: true,
                ..Shape::default()
            },
        ] {
            assert!(fallback_needed(&shape));
            assert!(!shape.adapter && !shape.beam);
            assert!(!allowed(&shape));
        }
    }

    #[test]
    fn adapter_and_beam_remain_rejected() {
        for shape in [
            Shape {
                adapter: true,
                ..Shape::default()
            },
            Shape {
                beam: true,
                ..Shape::default()
            },
        ] {
            assert!(!fallback_needed(&shape));
            assert!(!allowed(&shape));
        }
    }
    #[test]
    fn rejection_is_delivered_once_to_both_real_sinks() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        reject(ResponseSink::Streaming(tx));
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Error(_))));
        assert!(rx.try_recv().is_err());
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        reject(ResponseSink::Blocking(Some(tx)));
        assert!(rx.try_recv().unwrap().is_err());
    }
}
