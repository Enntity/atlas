// SPDX-License-Identifier: AGPL-3.0-only

//! Admission check for vision input (Atlas-Inf/atlas#180).
//!
//! A request whose images the engine cannot serve used to be admitted, fail
//! at prefill start, and reach the client as a generic error; agent clients
//! then resent the identical history forever. The preprocessing already knows
//! each item's vision tokens (its pad count), so the request is refused here,
//! before the template render and the queue, with a 400 the client can act
//! on. Every endpoint that takes images goes through `prepare_chat_prompt`
//! (chat completions, Responses, Anthropic messages), streaming or not, so
//! this one call covers them all.

use axum::http::StatusCode;
use axum::response::Response;
use spark_model::vision_limits::VisionRowLimits;

use super::super::compact::openai_error_response_with_param;

/// `error.code` of the refusal, for clients that branch on it.
pub(crate) const VISION_TOO_LARGE_CODE: &str = "vision_input_too_large";

/// `Err` (a 400 `invalid_request_error`) when the items, `pad_counts` vision
/// tokens each in prompt order, cannot be served under `limits`.
#[allow(clippy::result_large_err)]
pub(super) fn check(
    limits: Option<&VisionRowLimits>,
    pad_counts: &[usize],
) -> Result<(), Response> {
    let (Some(limits), false) = (limits, pad_counts.is_empty()) else {
        return Ok(());
    };
    limits.check(pad_counts).map_err(|message| {
        tracing::warn!(
            items = pad_counts.len(),
            rows = pad_counts.iter().sum::<usize>(),
            "vision request refused at admission: {message}"
        );
        openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            message,
            Some("messages"),
            Some(VISION_TOO_LARGE_CODE),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> VisionRowLimits {
        VisionRowLimits::resolve(
            spark_model::vision_limits::window_rows(16_384, 2, 16_388),
            16_388,
            16_384,
            262_144,
        )
    }

    async fn body(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn an_over_limit_request_is_a_400_naming_counts_and_the_limit() {
        // The reporter's 16680-row history against a 16384-row request limit.
        let resp = check(Some(&limits()), &[4096, 4096, 4096, 4096, 296]).unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], VISION_TOO_LARGE_CODE);
        assert_eq!(v["error"]["param"], "messages");
        let msg = v["error"]["message"].as_str().unwrap();
        for part in ["5 images", "16680", "16384"] {
            assert!(msg.contains(part), "{part} missing from {msg}");
        }
    }

    #[test]
    fn servable_and_text_requests_pass() {
        assert!(check(Some(&limits()), &[4096; 4]).is_ok());
        assert!(check(Some(&limits()), &[]).is_ok(), "text request");
        assert!(check(None, &[]).is_ok(), "text-only model");
    }
}
