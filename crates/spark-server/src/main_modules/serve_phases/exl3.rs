// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 checkpoint selection. Projection owners upload packed shards.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{
    gpu::GpuBackend,
    weights::{
        WeightStore,
        exl3::{
            Exl3CheckpointRouter,
            dense::{Exl3CheckpointDtype, Exl3DenseDtype},
        },
    },
};
use std::path::Path;

/// `None` leaves the established non-EXL3 topology policy with its caller.
pub(super) fn parallelism_supported(
    config: &ModelConfig,
    world: usize,
    tp: usize,
    ep: usize,
) -> Option<bool> {
    (crate::main_modules::serve::canonicalize_model_quant(config) == "exl3")
        .then_some(tp > 0 && world == tp && ep == 1)
}

pub(super) fn load(
    config: &ModelConfig,
    path: &Path,
    gpu: &dyn GpuBackend,
    reserve: usize,
) -> Result<WeightStore> {
    ensure!(
        config.model_type == "glm5_next" && config.ep_world_size == 1 && !config.expert_tp,
        "EXL3 validation requires GLM-5 with pure tensor parallelism (EP=1, expert_tp=false)"
    );
    let checkpoint = Exl3CheckpointRouter::open(path)?;
    let mut selected = Vec::new();
    for name in checkpoint
        .tensor_names()
        .filter(|name| is_companion(name, config.num_hidden_layers))
    {
        let source = checkpoint.tensor_dtype(name)?;
        let dtype = if source == Exl3CheckpointDtype::F32 || name.contains("hc_") {
            Exl3DenseDtype::F32
        } else {
            Exl3DenseDtype::Bf16
        };
        selected.push((name.to_string(), dtype));
    }
    WeightStore::from_exl3(checkpoint, &selected, gpu, reserve)
}

fn is_companion(name: &str, layers: usize) -> bool {
    if name.starts_with("model.visual.") || name.starts_with("visual.") {
        return false;
    }
    if let Some((_, rest)) = name.split_once(".layers.") {
        let Some((index, leaf)) = rest.split_once('.') else {
            return false;
        };
        if !index.parse::<usize>().is_ok_and(|index| index < layers) {
            return false;
        }
        return leaf == "input_layernorm.weight"
            || leaf == "post_attention_layernorm.weight"
            || leaf.starts_with("hc_attn_")
            || leaf.starts_with("hc_ffn_")
            || matches!(
                leaf,
                "self_attn.q_a_layernorm.weight"
                    | "self_attn.kv_a_layernorm.weight"
                    | "self_attn.indexer.wk.weight"
                    | "self_attn.indexer.weights_proj.weight"
                    | "self_attn.indexer.index_kpool_compress_gate"
                    | "self_attn.indexer.index_kpool_compress_ape"
                    | "self_attn.indexer.k_norm.weight"
                    | "self_attn.indexer.k_norm.bias"
            );
    }
    name.ends_with(".embed_tokens.weight")
        || name.ends_with(".norm.weight")
        || name.contains(".hc_head_")
}

#[cfg(test)]
mod tests {
    use super::{is_companion, parallelism_supported};
    #[test]
    fn native_topology_requires_explicit_pure_tp_and_matching_world() {
        let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        config.quantization_config = atlas_core::config::parse_quantization_config(
            &serde_json::json!({"quantization_config":{"quant_method":"exl3","quant_algo":"NVFP4"}}),
        );
        for (world, tp, ep, expected) in [
            (2, 2, 1, true),
            (1, 1, 1, true),
            (2, 1, 1, false),
            (1, 2, 1, false),
            (2, 2, 2, false),
            (0, 0, 1, false),
        ] {
            assert_eq!(
                parallelism_supported(&config, world, tp, ep),
                Some(expected)
            );
        }
    }

    #[test]
    fn non_exl3_topologies_remain_owned_by_the_legacy_policy() {
        let config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        assert_eq!(parallelism_supported(&config, 2, 2, 1), None);
        assert_eq!(parallelism_supported(&config, 2, 2, 2), None);
    }
    #[test]
    fn ordinary_selection_excludes_packed_shards_and_layer_owned_companions() {
        for leaf in [
            "input_layernorm.weight",
            "hc_attn_fn",
            "self_attn.indexer.wk.weight",
        ] {
            assert!(is_companion(
                &format!("model.language_model.layers.0.{leaf}"),
                45
            ));
        }
        for leaf in [
            "self_attn.kv_b_proj.weight",
            "linear_attn.b_proj.weight",
            "mlp.gate.weight",
            "mlp.experts.0.down_proj.trellis",
        ] {
            assert!(!is_companion(
                &format!("model.language_model.layers.0.{leaf}"),
                45
            ));
        }
        assert!(!is_companion(
            "model.language_model.layers.45.input_layernorm.weight",
            45
        ));
        assert!(!is_companion("model.visual.norm.weight", 45));
        assert!(is_companion("model.language_model.embed_tokens.weight", 45));
    }
}
