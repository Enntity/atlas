// SPDX-License-Identifier: AGPL-3.0-only

//! Reject unsupported repair requests before EP allocation or any prefill.
use crate::api::{InferenceRequest, StreamEvent};
use crate::scheduler::types::ResponseSink;

#[derive(Default)]
struct Shape {
    grammar: bool,
    tools: bool,
    disabled: bool,
    suppress: bool,
    adapter: bool,
    vision: bool,
    beam: bool,
}

fn allowed(s: &Shape) -> bool {
    !(s.grammar || s.tools || s.disabled || s.suppress || s.adapter || s.vision || s.beam)
}

pub(super) fn filter(requests: Vec<InferenceRequest>, enabled: bool) -> Vec<InferenceRequest> {
    if !enabled {
        return requests;
    }
    requests
        .into_iter()
        .filter_map(|req| {
            let grammar = match &req {
                InferenceRequest::Streaming { grammar_spec, .. }
                | InferenceRequest::Blocking { grammar_spec, .. } => grammar_spec.is_some(),
            };
            let shape = Shape {
                grammar,
                tools: req.tools_present() || req.require_tool_call(),
                disabled: req.disable_mtp(),
                suppress: req.suppress_tool_call(),
                adapter: req.adapter_slot() >= 0,
                vision: req.has_image_pixels(),
                beam: req.num_beams() > 1,
            };
            if allowed(&shape) {
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
    let error = "GLM MTP repair accepts only plain text, unconstrained, base-model requests with MTP enabled";
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
    #[test]
    fn unsupported_request_shapes_rejected_before_any_model_call() {
        assert!(allowed(&Shape::default()));
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
                adapter: true,
                ..Shape::default()
            },
            Shape {
                vision: true,
                ..Shape::default()
            },
            Shape {
                beam: true,
                ..Shape::default()
            },
        ] {
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
