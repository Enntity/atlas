// SPDX-License-Identifier: AGPL-3.0-only

//! Test fixtures: a neutral [`AppState`] for API tests, and the build-time
//! MODEL.toml parser so tests read a target's real `[behavior]` (the CI
//! runner builds without kernel targets, so none is compiled in).

use std::sync::Arc;
use std::time::Duration;

use crate::AppState;

// Its other returned policy fields are unused by any one test; no duplicate
// metadata parser/default.
#[allow(dead_code)]
#[path = "../../../atlas-kernels/build_parse_behavior.rs"]
pub(crate) mod build_behavior;

/// A text-only, tool-free [`AppState`] with no reasoning markers. A suite
/// overrides the fields it exercises with struct-update syntax
/// (`AppState { behavior, ..app_state(tokenizer) }`).
pub(crate) fn app_state(tokenizer: crate::tokenizer::ChatTokenizer) -> AppState {
    let (request_tx, _) = tokio::sync::mpsc::channel(1);
    AppState {
        tokenizer,
        model_name: "fixture".into(),
        adapter_name: None,
        adapter_names: vec![],
        active_adapter: Arc::default(),
        max_seq_len: 4096,
        request_tx,
        rotation_tx: None,
        vision_config: None,
        vision_max_pixels: None,
        remote_image_policy: Default::default(),
        video_ffmpeg: Default::default(),
        video_fps: 1.0,
        default_temperature: 0.0,
        default_top_k: 0,
        default_top_p: 1.0,
        default_top_n_sigma: 0.0,
        default_min_p: 0.0,
        tool_call_parser: None,
        chat: Default::default(),
        reasoning_parser: None,
        think_end_token_id: None,
        think_start_token_id: None,
        tool_max_tokens: 192,
        sampling_presets: Default::default(),
        tool_call_start_token_id: None,
        glm_tool_boundary: None,
        auto_compact_threshold: None,
        request_timeout: 0,
        effective_context: 0,
        behavior: Default::default(),
        disable_thinking: false,
        default_thinking: Default::default(),
        default_reasoning_effort: None,
        response_store: crate::response_store::ResponseStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        conversation_store: crate::conversation_store::ConversationStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        rate_limiter: crate::rate_limiter::RateLimiter::with_config(
            crate::rate_limiter::RateLimitConfig {
                rpm: 0,
                tpm: 0,
                burst_rpm: 1,
                burst_tpm: 1,
            },
        ),
        dump_writer: None,
        lora_stageable: Default::default(),
        lora_peer_addr: None,
        promotion: None,
        promoted_slots: Arc::default(),
        lora_disk_stageable: Default::default(),
    }
}
