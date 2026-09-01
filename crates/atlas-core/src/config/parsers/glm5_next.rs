// SPDX-License-Identifier: AGPL-3.0-only

//! Strict parser for the official `zai-org/GLM-5.3-Flash` checkpoint.
//!
//! KDA and DSA have persistent state that Atlas's GDN and ordinary MLA paths
//! do not represent. This parser therefore accepts the exact architecture we
//! are implementing and fails closed on a superficially similar variant.

use anyhow::{Context, Result, bail};

use super::super::{
    LayerType, ModelConfig, finalize_config, parse_quantization_config, parse_vision_config,
};

const MODEL: &str = "glm5_next";

pub(crate) fn parse_glm5_next(raw: &serde_json::Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("glm5_next config missing required `text_config`")?;
    let mut normalized = text.clone();

    let layer_types = string_array(text, "layer_types")?
        .iter()
        .map(|kind| match kind.as_str() {
            "linear_attention" => Ok(LayerType::LinearAttention),
            "deepseek_sparse_attention" => Ok(LayerType::FullAttention),
            other => bail!("{MODEL} unsupported layer type `{other}`"),
        })
        .collect::<Result<Vec<_>>>()?;
    normalized["layer_types"] = serde_json::Value::Array(
        layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::LinearAttention => serde_json::json!("linear_attention"),
                LayerType::FullAttention => serde_json::json!("full_attention"),
                _ => unreachable!("GLM parser emits only KDA or DSA"),
            })
            .collect(),
    );

    let eos = text
        .get("eos_token_id")
        .and_then(serde_json::Value::as_array)
        .context("glm5_next text_config.eos_token_id must be a non-empty array")?;
    let first_eos = eos
        .first()
        .and_then(serde_json::Value::as_u64)
        .context("glm5_next text_config.eos_token_id[0] must be an integer")?;
    let eos_token_ids = eos
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .context("glm5_next eos_token_id entries must fit in u32")
        })
        .collect::<Result<Vec<_>>>()?;
    normalized["eos_token_id"] = serde_json::json!(first_eos);

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("failed to parse glm5_next text_config")?;
    config.model_type = MODEL.to_string();
    config.layer_types = layer_types;
    config.nested_config = true;
    config.attn_gated = false;
    config.weight_prefix = "model.language_model".to_string();
    config.eos_token_ids = eos_token_ids;

    if config.num_experts == 0 {
        config.num_experts = required_usize(text, "n_routed_experts")?;
    }
    let shared = required_usize(text, "n_shared_experts")?;
    config.shared_expert_intermediate_size = shared
        .checked_mul(config.moe_intermediate_size)
        .context("glm5_next shared expert width overflow")?;
    config.mlp_only_layers = dense_mlp_layers(text, config.num_hidden_layers)?;
    config.use_routing_bias = required_str(text, "topk_method")? == "noaux_tc";

    let linear = text
        .get("linear_attn_config")
        .context("glm5_next missing linear_attn_config")?;
    config.linear_num_key_heads = required_usize(linear, "num_heads")?;
    config.linear_num_value_heads = config.linear_num_key_heads;
    config.linear_key_head_dim = required_usize(linear, "head_dim")?;
    config.linear_value_head_dim = config.linear_key_head_dim;
    config.linear_conv_kernel_dim = required_usize(linear, "short_conv_kernel_size")?;
    config.kda_gate_lower_bound = required_f64(linear, "gate_lower_bound")? as f32;

    // HF leaves text_config.head_dim at zero; DSA declares the actual QK
    // width separately as qk_head_dim.
    config.head_dim = required_usize(text, "qk_head_dim")?;
    config.index_kpool = required_usize(text, "index_kpool")?;
    config.index_kpool_always_select_tail = required_bool(text, "index_kpool_always_select_tail")?;
    config.index_kpool_compress = required_bool(text, "index_kpool_compress")?;
    config.index_share_for_mtp_iteration = required_bool(text, "index_share_for_mtp_iteration")?;
    config.indexer_rope_interleave = required_bool(text, "indexer_rope_interleave")?;
    config.indexer_types = string_array(text, "indexer_types")?;

    config.hc_mult = required_usize(text, "hc_mult")?;
    config.hc_sinkhorn_iters = required_usize(text, "hc_sinkhorn_iters")?;
    config.hc_eps = required_f64(text, "hc_eps")? as f32;
    config.mtp_num_hidden_layers = required_usize(text, "num_nextn_predict_layers")?;
    config.num_mtp_modules = config.mtp_num_hidden_layers;
    config.mtp_transformer_layers = 1;
    config.vision = parse_vision_config(raw);
    config.quantization_config = parse_quantization_config(raw);

    validate_glm53_exl3_quant_contract(raw)?;
    validate_glm53_contract(&config, text, linear)?;
    finalize_config(&mut config, raw)?;
    Ok(config)
}

fn validate_glm53_exl3_quant_contract(raw: &serde_json::Value) -> Result<()> {
    let Some(quant) = raw.get("quantization_config") else {
        return Ok(());
    };
    if quant
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        != Some("exl3")
    {
        return Ok(());
    }
    if required_usize(quant, "bits")? != 4
        || required_usize(quant, "head_bits")? != 16
        || required_str(quant, "codebook")? != "mcg"
        || required_str(quant, "scope")? != "glm53_routed_experts_only"
        || required_str(quant, "non_routed_dtype_policy")? != "official_source_native"
        || required_str(quant, "version")? != "0.0.43"
    {
        bail!(
            "glm5_next EXL3 requires the pinned 4-bit MCG v0.0.43 routed-experts-only export with 16-bit heads and native non-routed tensors"
        );
    }
    Ok(())
}

fn validate_glm53_contract(
    config: &ModelConfig,
    text: &serde_json::Value,
    linear: &serde_json::Value,
) -> Result<()> {
    let kda = usize_array(linear, "kda_layers")?;
    let dsa = usize_array(linear, "full_attn_layers")?;
    if kda != layer_indices(config, LayerType::LinearAttention)
        || dsa != layer_indices(config, LayerType::FullAttention)
    {
        bail!("glm5_next linear_attn_config layer lists disagree with layer_types");
    }
    if config.indexer_types.len() != config.num_hidden_layers
        || config.indexer_types.iter().any(|kind| kind != "full")
    {
        bail!("glm5_next requires one `full` indexer type per decoder layer");
    }
    if config.index_topk == 0 || !config.index_topk.is_multiple_of(config.index_kpool) {
        bail!("glm5_next index_topk must be non-zero and divisible by index_kpool");
    }
    if config.qk_rope_head_dim != 0 || !required_bool(text, "mla_use_nope")? {
        bail!("glm5_next Flash contract requires NoPE DSA (qk_rope_head_dim=0)");
    }
    if required_str(text, "moe_router_dtype")? != "float32" {
        bail!("glm5_next Flash contract requires moe_router_dtype=float32");
    }
    let first_dense = required_usize(text, "first_k_dense_replace")?;
    if config.mlp_only_layers != (0..first_dense).collect::<Vec<_>>() {
        bail!("glm5_next dense MLP layers must exactly match first_k_dense_replace");
    }
    if config.num_hidden_layers != 45 || kda.len() != 34 || dsa.len() != 11 {
        bail!("glm5_next parser currently targets GLM-5.3-Flash's 45-layer 34-KDA/11-DSA topology");
    }
    Ok(())
}

fn layer_indices(config: &ModelConfig, kind: LayerType) -> Vec<usize> {
    config
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, actual)| (*actual == kind).then_some(i))
        .collect()
}

fn dense_mlp_layers(text: &serde_json::Value, expected: usize) -> Result<Vec<usize>> {
    let kinds = string_array(text, "mlp_layer_types")?;
    if kinds.len() != expected {
        bail!("glm5_next mlp_layer_types length must equal num_hidden_layers");
    }
    kinds
        .iter()
        .enumerate()
        .map(|(i, kind)| match kind.as_str() {
            "dense" => Ok(Some(i)),
            "sparse" => Ok(None),
            other => bail!("glm5_next unsupported MLP layer type `{other}`"),
        })
        .filter_map(Result::transpose)
        .collect()
}

fn required_usize(value: &serde_json::Value, key: &str) -> Result<usize> {
    let n = value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("{MODEL} missing unsigned integer `{key}`"))? as usize;
    if n == 0 {
        bail!("{MODEL} field `{key}` must be greater than zero");
    }
    Ok(n)
}

fn required_f64(value: &serde_json::Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .with_context(|| format!("{MODEL} missing number `{key}`"))
}

fn required_bool(value: &serde_json::Value, key: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .with_context(|| format!("{MODEL} missing boolean `{key}`"))
}

fn required_str<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .with_context(|| format!("{MODEL} missing string `{key}`"))
}

fn string_array(value: &serde_json::Value, key: &str) -> Result<Vec<String>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("{MODEL} missing array `{key}`"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .with_context(|| format!("{MODEL} `{key}` entries must be strings"))
        })
        .collect()
}

fn usize_array(value: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("{MODEL} missing array `{key}`"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .with_context(|| format!("{MODEL} `{key}` entries must be unsigned integers"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::capabilities::SsmArchitecture;
    use crate::config::{LayerType, parse_config};

    fn fixture() -> serde_json::Value {
        let dsa = (3..45).step_by(4).collect::<Vec<_>>();
        let kda = (0..45).filter(|i| !dsa.contains(i)).collect::<Vec<_>>();
        let layer_types = (0..45)
            .map(|i| {
                if dsa.contains(&i) {
                    "deepseek_sparse_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        let mlp_types = (0..45)
            .map(|i| if i < 3 { "dense" } else { "sparse" })
            .collect::<Vec<_>>();
        let mut text = serde_json::json!({
                "model_type": "glm5_next_text",
                "hidden_size": 4096,
                "num_hidden_layers": 45,
                "intermediate_size": 12288,
                "vocab_size": 154880,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "head_dim": 0,
                "eos_token_id": [154820, 154827, 154829],
                "pad_token_id": 154820,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 1048576,
                "layer_types": layer_types,
                "linear_attn_config": {
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                    "kda_layers": kda,
                    "full_attn_layers": dsa
                },
                "first_k_dense_replace": 3,
                "mlp_layer_types": mlp_types,
                "moe_intermediate_size": 2048,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "num_experts_per_tok": 8,
                "norm_topk_prob": true,
                "routed_scaling_factor": 2.5,
                "scoring_func": "sigmoid",
                "topk_method": "noaux_tc",
                "moe_router_dtype": "float32"
        });
        let attention = serde_json::json!({
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_head_dim": 256,
                "qk_nope_head_dim": 256,
                "qk_rope_head_dim": 0,
                "v_head_dim": 256,
                "mla_use_nope": true,
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_topk": 2048,
                "index_kpool": 4,
                "index_kpool_always_select_tail": true,
                "index_kpool_compress": true,
                "index_share_for_mtp_iteration": true,
                "indexer_rope_interleave": true,
                "indexer_types": vec!["full"; 45]
        });
        for (key, value) in attention.as_object().unwrap() {
            text[key] = value.clone();
        }
        let residual = serde_json::json!({
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 0.000001,
                "num_nextn_predict_layers": 1
        });
        for (key, value) in residual.as_object().unwrap() {
            text[key] = value.clone();
        }
        serde_json::json!({
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "image_token_id": 154854,
            "video_token_id": 154855,
            "text_config": text,
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "num_heads": 16,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "intermediate_size": 4096,
                "out_hidden_size": 4096
            },
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "modules_to_not_convert": ["lm_head", "model.layers.0.self_attn.A_log"]
            }
        })
    }

    #[test]
    fn official_flash_contract_maps_without_architecture_guessing() {
        let config = parse_config(&fixture().to_string()).expect("official GLM config must parse");
        assert_eq!(config.model_type, "glm5_next");
        assert_eq!(config.weight_prefix, "model.language_model");
        assert_eq!(config.num_ssm_layers(), 34);
        assert_eq!(config.num_attention_layers(), 11);
        assert_eq!(config.layer_type(3), LayerType::FullAttention);
        assert_eq!(config.mlp_only_layers, vec![0, 1, 2]);
        assert_eq!(config.linear_num_key_heads, 64);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.kda_gate_lower_bound, -5.0);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.index_topk, 2048);
        assert_eq!(config.index_kpool, 4);
        assert_eq!(config.num_experts, 288);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.eos_token_id, 154820);
        assert_eq!(config.eos_token_ids, vec![154820, 154827, 154829]);
        assert_eq!(config.mtp_num_hidden_layers, 1);
        assert_eq!(config.capabilities().ssm_architecture, SsmArchitecture::Kda);
        assert!(!config.kv_only_prefix_cache_is_safe());
        assert_eq!(config.vision.as_ref().unwrap().image_pad_token_id, 154854);
        let quant = config.quantization_config.as_ref().unwrap();
        assert_eq!(quant.quant_method, "fp8");
        assert!(quant.ignore_modules.iter().any(|name| name == "lm_head"));
    }

    #[test]
    fn malformed_index_pool_contract_fails_closed() {
        let mut raw = fixture();
        raw["text_config"]["index_topk"] = serde_json::json!(2047);
        let error = parse_config(&raw.to_string()).unwrap_err().to_string();
        assert!(error.contains("divisible by index_kpool"), "{error}");
    }

    #[test]
    fn router_dtype_must_be_present_and_float32() {
        for wrong in [serde_json::Value::Null, serde_json::json!("bfloat16")] {
            let mut raw = fixture();
            raw["text_config"]["moe_router_dtype"] = wrong;
            let error = parse_config(&raw.to_string()).unwrap_err().to_string();
            assert!(error.contains("moe_router_dtype"), "{error}");
        }
    }

    #[test]
    fn routed_expert_exl3_quantization_is_preserved() {
        let mut raw = fixture();
        raw["quantization_config"] = serde_json::json!({
            "bits": 4,
            "codebook": "mcg",
            "head_bits": 16,
            "non_routed_dtype_policy": "official_source_native",
            "quant_method": "exl3",
            "scope": "glm53_routed_experts_only",
            "version": "0.0.43"
        });
        let config = parse_config(&raw.to_string()).expect("EXL3 GLM config must parse");
        let quant = config.quantization_config.expect("EXL3 quant config");
        assert_eq!(quant.quant_method, "exl3");
    }

    #[test]
    fn incompatible_exl3_packing_fails_closed() {
        let mut raw = fixture();
        raw["quantization_config"] = serde_json::json!({
            "bits": 4,
            "codebook": "mcg",
            "head_bits": 16,
            "non_routed_dtype_policy": "all_quantized",
            "quant_method": "exl3",
            "scope": "glm53_routed_experts_only",
            "version": "0.0.43"
        });
        let error = parse_config(&raw.to_string()).unwrap_err().to_string();
        assert!(error.contains("routed-experts-only"), "{error}");
    }

    #[test]
    fn inconsistent_kda_layer_list_fails_closed() {
        let mut raw = fixture();
        raw["text_config"]["linear_attn_config"]["kda_layers"] = serde_json::json!([0, 1]);
        let error = parse_config(&raw.to_string()).unwrap_err().to_string();
        assert!(error.contains("disagree with layer_types"), "{error}");
    }

    #[test]
    fn two_spark_expert_partition_is_complete_and_disjoint() {
        let mut config = parse_config(&fixture().to_string()).unwrap();
        config.ep_world_size = 2;
        config.ep_rank = 0;
        assert_eq!(config.local_expert_range(), (0, 144));
        config.ep_rank = 1;
        assert_eq!(config.local_expert_range(), (144, 288));
    }
}
