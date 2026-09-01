// SPDX-License-Identifier: AGPL-3.0-only

//! Exact checkpoint-name contract for GLM-5.3-Flash.
//!
//! The official checkpoint mixes KDA, sparse NoPE MLA, dense MLP, MoE, and
//! one physical MTP layer. Checking those seams before model construction
//! prevents a partial conversion or broken EP filter from becoming a much
//! less useful CUDA/NCCL failure.

use std::collections::HashSet;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::weights::WeightStore;

const ROOT: &str = "model.language_model";
const PROJECTIONS: [&str; 3] = ["down_proj", "gate_proj", "up_proj"];

pub(super) fn check_checkpoint_contract(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    if config.model_type != "glm5_next" {
        return Ok(());
    }

    let present = store.names().collect::<HashSet<_>>();
    for required in required_tensor_names(config) {
        if !present.contains(required.as_str()) {
            bail!("GLM-5.3-Flash checkpoint is missing required tensor `{required}`");
        }
    }

    let (local_start, local_end) = config.local_expert_range();
    for name in &present {
        let Some(expert) = expert_index(name) else {
            continue;
        };
        if expert < local_start || expert >= local_end {
            bail!(
                "GLM-5.3-Flash EP rank {}/{} loaded remote expert {expert} in `{name}`; \
                 expected only experts [{local_start}..{local_end})",
                config.ep_rank,
                config.ep_world_size,
            );
        }
    }

    tracing::info!(
        "GLM-5.3-Flash checkpoint contract passed: KDA={}, sparse DSA={}, \
         EP rank {}/{} experts=[{}..{}), physical MTP layer present but disabled",
        config.num_ssm_layers(),
        config.num_attention_layers(),
        config.ep_rank,
        config.ep_world_size,
        local_start,
        local_end,
    );
    Ok(())
}

fn required_tensor_names(config: &ModelConfig) -> Vec<String> {
    let mut names = vec![
        format!("{ROOT}.embed_tokens.weight"),
        format!("{ROOT}.norm.weight"),
        "lm_head.weight".to_string(),
    ];
    for layer in 0..config.num_hidden_layers {
        add_common_layer_names(&mut names, layer);
        match config.layer_type(layer) {
            LayerType::LinearAttention => add_kda_names(&mut names, layer),
            LayerType::FullAttention => add_dsa_names(&mut names, layer),
            other => unreachable!("strict GLM parser rejects layer type {other:?}"),
        }
        if config.mlp_only_layers.contains(&layer) {
            add_dense_mlp_names(&mut names, layer);
        } else {
            add_moe_names(&mut names, layer, config);
        }
    }

    // The checkpoint's one next-token-prediction block is physical layer 45.
    // Atlas deliberately disables it until GLM index sharing is implemented,
    // but these anchors distinguish the official artifact from a truncated one.
    let mtp = format!("{ROOT}.layers.{}", config.num_hidden_layers);
    for suffix in [
        "eh_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "shared_head.norm.weight",
    ] {
        names.push(format!("{mtp}.{suffix}"));
    }
    names
}

fn add_common_layer_names(names: &mut Vec<String>, layer: usize) {
    let root = format!("{ROOT}.layers.{layer}");
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ] {
        names.push(format!("{root}.{suffix}"));
    }
}

fn add_kda_names(names: &mut Vec<String>, layer: usize) {
    let root = format!("{ROOT}.layers.{layer}.self_attn");
    for suffix in [
        "A_log",
        "b_proj.weight",
        "dt_bias",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "k_conv1d.weight",
        "k_proj.weight",
        "o_norm.weight",
        "o_proj.weight",
        "q_conv1d.weight",
        "q_proj.weight",
        "v_conv1d.weight",
        "v_proj.weight",
    ] {
        names.push(format!("{root}.{suffix}"));
    }
}

fn add_dsa_names(names: &mut Vec<String>, layer: usize) {
    let root = format!("{ROOT}.layers.{layer}.self_attn");
    for suffix in [
        "indexer.index_kpool_compress_ape",
        "indexer.index_kpool_compress_gate",
        "indexer.k_norm.bias",
        "indexer.k_norm.weight",
        "indexer.weights_proj.weight",
        "indexer.wk.weight",
        "indexer.wq_b.weight",
        "kv_a_layernorm.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
        "q_a_layernorm.weight",
        "q_a_proj.weight",
        "q_b_proj.weight",
    ] {
        names.push(format!("{root}.{suffix}"));
    }
}

fn add_dense_mlp_names(names: &mut Vec<String>, layer: usize) {
    let root = format!("{ROOT}.layers.{layer}.mlp");
    for projection in PROJECTIONS {
        names.push(format!("{root}.{projection}.weight"));
    }
}

fn add_moe_names(names: &mut Vec<String>, layer: usize, config: &ModelConfig) {
    let root = format!("{ROOT}.layers.{layer}.mlp");
    names.push(format!("{root}.gate.weight"));
    names.push(format!("{root}.gate.e_score_correction_bias"));
    for projection in PROJECTIONS {
        names.push(format!("{root}.shared_experts.{projection}.weight"));
    }
    let (local_start, local_end) = config.local_expert_range();
    for expert in local_start..local_end {
        for projection in PROJECTIONS {
            let projection = format!("{root}.experts.{expert}.{projection}");
            for suffix in ["trellis", "suh", "svh", "mcg"] {
                names.push(format!("{projection}.{suffix}"));
            }
        }
    }
}

fn expert_index(name: &str) -> Option<usize> {
    let tail = name.split(".experts.").nth(1)?;
    tail.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{ROOT, check_checkpoint_contract, expert_index, required_tensor_names};
    use atlas_core::config::{LayerType, ModelConfig, QuantizationConfig};
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    fn config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".to_string();
        config.num_hidden_layers = 45;
        config.layer_types = (0usize..45)
            .map(|layer| {
                if (layer + 1).is_multiple_of(4) {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        config.mlp_only_layers = vec![0, 1, 2];
        config.num_experts = 288;
        config.ep_world_size = 1;
        config.ep_rank = 0;
        config.quantization_config = Some(QuantizationConfig {
            quant_method: "exl3".to_string(),
            quant_algo: String::new(),
            format: String::new(),
            ignore_modules: Vec::new(),
        });
        config
    }

    fn store_for_names(names: impl IntoIterator<Item = String>) -> WeightStore {
        WeightStore::from_map(
            names
                .into_iter()
                .map(|name| {
                    (
                        name,
                        WeightTensor {
                            ptr: DevicePtr::NULL,
                            shape: vec![1],
                            dtype: WeightDtype::BF16,
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        )
    }

    #[test]
    fn required_names_pin_architecture_and_all_experts() {
        let names = required_tensor_names(&config());
        assert!(names.contains(&format!("{ROOT}.layers.0.self_attn.A_log")));
        assert!(names.contains(&format!(
            "{ROOT}.layers.3.self_attn.indexer.weights_proj.weight"
        )));
        assert!(names.contains(&format!(
            "{ROOT}.layers.3.mlp.experts.144.gate_proj.trellis"
        )));
        assert!(names.iter().any(|name| name.contains(".experts.0.")));
        assert!(names.iter().any(|name| name.contains(".experts.287.")));
        assert!(names.contains(&format!("{ROOT}.layers.45.shared_head.norm.weight")));
    }

    #[test]
    fn expert_parser_ignores_shared_experts() {
        assert_eq!(
            expert_index("model.language_model.layers.3.mlp.experts.287.up_proj.weight"),
            Some(287)
        );
        assert_eq!(
            expert_index("model.language_model.layers.3.mlp.shared_experts.up_proj.weight"),
            None
        );
    }

    #[test]
    fn complete_rank_local_manifest_passes_and_missing_tensor_is_named() {
        let config = config();
        let mut names = required_tensor_names(&config);
        assert!(check_checkpoint_contract(&store_for_names(names.clone()), &config).is_ok());

        let missing = format!("{ROOT}.layers.3.self_attn.indexer.weights_proj.weight");
        names.retain(|name| name != &missing);
        let error = check_checkpoint_contract(&store_for_names(names), &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&missing), "{error}");
    }

    #[test]
    fn missing_last_tp_expert_fails_closed() {
        let config = config();
        let mut names = required_tensor_names(&config);
        let missing = format!("{ROOT}.layers.3.mlp.experts.287.gate_proj.trellis");
        names.retain(|name| name != &missing);
        let error = check_checkpoint_contract(&store_for_names(names), &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&missing), "{error}");
    }
}
