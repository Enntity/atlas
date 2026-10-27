// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/models` handlers: list + retrieve served model ids, plus the
//! embeddings auto-probe stub. Split out of `completions.rs` when the M2
//! LoRA advertise path (resident adapters ARE served models) grew it past
//! the 500-LoC cap.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::openai::{ModelInfo, ModelListResponse};

use super::compact::openai_error_response;

/// GET /v1/models
pub async fn list_models(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
) -> Json<ModelListResponse> {
    // An empty list is the honest answer before a model is chosen — better than
    // 503, because the endpoint itself is working and the answer is "none".
    let Some(state) = host.current() else {
        return Json(ModelListResponse {
            object: "list".to_string(),
            data: Vec::new(),
        });
    };
    // #22 hardening: bound the advertised set so the pre-sized allocation can
    // never be driven large. adapter_names is startup-bounded by --max-loras
    // today, but capping keeps the allocation independent of that (and clears the
    // CodeQL "allocation size from a user-provided value" alert). A real pool far
    // below the cap is unaffected.
    const MAX_ADVERTISED_MODELS: usize = 1024;
    let advertised = state.adapter_names.len().min(MAX_ADVERTISED_MODELS);
    let mut data = Vec::with_capacity(advertised.saturating_add(1));
    // The resident adapters ARE served models — advertise them first (slot
    // order; data[0] is the default route) so clients can pick a fine-tune.
    for adapter in state.adapter_names.iter().take(MAX_ADVERTISED_MODELS) {
        data.push(ModelInfo::advertise(adapter.clone(), state.max_seq_len));
    }
    // Cold STAGEABLE names (peer- and disk-backed) are selectable via `model`
    // and fault in on first use — advertise them too (before the base model),
    // bounded by the same cap. They are NOT in adapter_names (the resident
    // routing index), so this only affects what /v1/models lists.
    for name in state
        .lora_stageable
        .keys()
        .chain(state.lora_disk_stageable.keys())
        .take(MAX_ADVERTISED_MODELS.saturating_sub(data.len()))
    {
        data.push(ModelInfo::advertise(name.clone(), state.max_seq_len));
    }
    data.push(ModelInfo::advertise(
        state.model_name.clone(),
        state.max_seq_len,
    ));
    for entry in &mut data {
        entry.atlas_image_processor = image_processor(state.vision_config.as_ref());
    }
    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

/// GET /v1/models/{model_id} — retrieve a single model (OpenAI SDK `client.models.retrieve()`).
pub async fn get_model(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    // An empty list is the honest answer before a model is chosen — better than
    // 503, because the endpoint itself is working and the answer is "none".
    let Some(state) = host.current() else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": crate::error_hints::message_with_hint(
                        "no model is loaded",
                        "model_not_loaded",
                    ),
                    "type": "model_not_loaded",
                    "hint": crate::error_hints::hint_for("model_not_loaded"),
                }
            })),
        )
            .into_response();
    };
    // Any resident adapter is a routable model id (M2): `models.retrieve(name)`
    // must succeed for every adapter advertised by /v1/models, not just slot 0.
    let known = model_id == state.model_name
        || state.adapter_names.iter().any(|n| n == &model_id)
        || state.is_stageable_name(&model_id);
    if known {
        let mut entry = ModelInfo::advertise(model_id, state.max_seq_len);
        entry.atlas_image_processor = image_processor(state.vision_config.as_ref());
        Json(entry).into_response()
    } else {
        openai_error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{model_id}' does not exist"),
        )
    }
}

/// POST /v1/embeddings — stub for clients that probe this endpoint during auto-detection.
pub async fn embeddings_stub() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        "Embeddings are not supported by this model. Atlas serves generative (chat/completion) models only.".into(),
    )
}

/// Report only the bounded Qwen still-image policy supported by the reference
/// benchmark. Video has separate bounds and must not consume this profile.
fn image_processor(config: Option<&atlas_core::config::VisionConfig>) -> Option<serde_json::Value> {
    let v = config?;
    if v.is_glm5_next {
        return None;
    }
    let grid = v.patch_size.checked_mul(v.spatial_merge_size)?;
    let cell_area = grid.checked_mul(grid).filter(|&n| n > 0)?;
    let max = v.max_pixels.filter(|&n| n >= cell_area)?;
    let patches =
        spark_model::layers::vision_encoder::derive_max_patches(v.max_pixels, v.patch_size).0;
    Some(serde_json::json!({
        "algorithm": "qwen-smart-resize-v1",
        "patch_size": v.patch_size,
        "spatial_merge_size": v.spatial_merge_size,
        "min_pixels": v.min_pixels.unwrap_or(0),
        "max_pixels": max,
        "encoder_capacity_pixels": patches * v.patch_size.max(1) * v.patch_size.max(1),
    }))
}

#[cfg(test)]
mod image_profile_tests {
    use super::*;
    #[test]
    fn image_profile_uses_resolved_bounds_and_encoder_capacity() {
        let mut config = spark_model::vision_preprocess::fixtures::qwen_config();
        config.max_pixels = Some(16_777_216);
        config.min_pixels = Some(65_536);
        let profile = image_processor(Some(&config)).unwrap();
        assert_eq!(profile["max_pixels"], 16_777_216);
        assert_eq!(profile["min_pixels"], 65_536);
        assert_eq!(profile["encoder_capacity_pixels"], 4_194_304);
        config.max_pixels = Some(262_144);
        let profile = image_processor(Some(&config)).unwrap();
        assert_eq!(profile["max_pixels"], 262_144);
        assert_eq!(profile["encoder_capacity_pixels"], 262_144);
        config.max_pixels = Some(1);
        assert!(image_processor(Some(&config)).is_none());
        config.is_glm5_next = true;
        assert!(image_processor(Some(&config)).is_none());
        config.is_glm5_next = false;
        config.max_pixels = None;
        assert!(image_processor(Some(&config)).is_none());
        assert!(image_processor(None).is_none());
    }
}
