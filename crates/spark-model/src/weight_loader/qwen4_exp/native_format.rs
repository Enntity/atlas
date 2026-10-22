// SPDX-License-Identifier: AGPL-3.0-only
//! Reject target fallback conversions before constructing any Qwen layer.
use super::format::layer_components;
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::{WeightDtype, WeightStore};

fn require_tensor(
    store: &WeightStore,
    name: &str,
    dtype: WeightDtype,
    shape: &[usize],
) -> Result<()> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.dtype == dtype,
        "{name}: expected {dtype:?}, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape == shape,
        "{name}: expected shape {shape:?}, got {:?}",
        tensor.shape
    );
    Ok(())
}

fn require_bf16(store: &WeightStore, name: &str, shape: &[usize]) -> Result<()> {
    require_tensor(store, name, WeightDtype::BF16, shape)
}

fn require_nvfp4(store: &WeightStore, stem: &str, n: usize, k: usize) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "{stem}: invalid NVFP4 dimensions n={n} k={k}"
    );
    require_tensor(
        store,
        &format!("{stem}.weight"),
        WeightDtype::UInt8,
        &[n, k / 2],
    )?;
    require_tensor(
        store,
        &format!("{stem}.weight_scale"),
        WeightDtype::FP8E4M3,
        &[n, k / 16],
    )?;
    require_tensor(
        store,
        &format!("{stem}.weight_scale_2"),
        WeightDtype::FP32,
        &[],
    )
}

fn validate_mhc_site(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    with_inject: bool,
) -> Result<()> {
    let wide = config.hc_mult * config.hidden_size;
    require_bf16(store, &format!("{prefix}.hc_norm.weight"), &[wide])?;
    require_bf16(
        store,
        &format!("{prefix}.input_mix_weight_down.weight"),
        &[config.hc_lowrank, wide],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.input_mix_weight_up.weight"),
        &[wide, config.hc_lowrank],
    )?;
    if with_inject {
        require_bf16(
            store,
            &format!("{prefix}.block_inject_weight.weight"),
            &[config.hc_mult, wide],
        )?;
    }
    Ok(())
}

fn validate_mtp_attention(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    let h = config.hidden_size;
    let q = config.num_attention_heads * config.head_dim;
    let kv = config.num_key_value_heads * config.head_dim;
    let index = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    let prefix = "mtp.layers.0.self_attn";
    require_bf16(store, &format!("{prefix}.q_proj.weight"), &[q * 2, h])?;
    require_bf16(store, &format!("{prefix}.k_proj.weight"), &[kv, h])?;
    require_bf16(store, &format!("{prefix}.v_proj.weight"), &[kv, h])?;
    require_bf16(store, &format!("{prefix}.o_proj.weight"), &[h, q])?;
    require_bf16(
        store,
        &format!("{prefix}.q_norm.weight"),
        &[config.head_dim],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.k_norm.weight"),
        &[config.head_dim],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.indexer.index_qk_proj.weight"),
        &[index, h],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.indexer.q_layernorm.weight"),
        &[config.indexer_head_dim],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.indexer.k_layernorm.weight"),
        &[config.indexer_head_dim],
    )
}

fn validate_mtp_moe(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    let h = config.hidden_size;
    let mi = config.moe_intermediate_size;
    let si = config.shared_expert_intermediate_size;
    let prefix = "mtp.layers.0.mlp";

    require_bf16(
        store,
        &format!("{prefix}.gate.weight"),
        &[config.num_experts, h],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.shared_expert_gate.weight"),
        &[1, h],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.shared_expert.gate_proj.weight"),
        &[si, h],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.shared_expert.up_proj.weight"),
        &[si, h],
    )?;
    require_bf16(
        store,
        &format!("{prefix}.shared_expert.down_proj.weight"),
        &[h, si],
    )?;

    // Strict/Spark MTP is rank-0 and unsharded, so every routed expert must be
    // present in the store in the native ModelOpt NVFP4 layout.
    ensure!(
        !store.contains(&format!("{prefix}.experts.gate_up_proj"))
            && !store.contains(&format!("{prefix}.experts.down_proj")),
        "Qwen Strict/Spark requires native per-expert NVFP4 MTP weights; refusing stacked MTP experts"
    );
    for expert in 0..config.num_experts {
        require_nvfp4(
            store,
            &format!("{prefix}.experts.{expert}.gate_proj"),
            mi,
            h,
        )?;
        require_nvfp4(store, &format!("{prefix}.experts.{expert}.up_proj"), mi, h)?;
        require_nvfp4(
            store,
            &format!("{prefix}.experts.{expert}.down_proj"),
            h,
            mi,
        )?;
    }
    Ok(())
}

pub(super) fn validate_layer(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    index: usize,
) -> Result<()> {
    for component in layer_components(config.layer_types[index]) {
        for stem in component.stems(lp) {
            ensure!(
                crate::weight_map::mxfp8_geometry(store, &stem)?.is_some(),
                "Qwen serving mode requires native MXFP8 {stem}; refusing fallback conversion"
            );
        }
    }
    require_bf16(
        store,
        &format!("{lp}.mlp.gate.weight"),
        &[config.num_experts, config.hidden_size],
    )?;
    require_bf16(
        store,
        &format!("{lp}.mlp.shared_expert_gate.weight"),
        &[1, config.hidden_size],
    )?;
    for expert in (0..config.num_experts).filter(|&e| config.is_local_expert(e)) {
        for (projection, n, k) in [
            ("gate", config.moe_intermediate_size, config.hidden_size),
            ("up", config.moe_intermediate_size, config.hidden_size),
            ("down", config.hidden_size, config.moe_intermediate_size),
        ] {
            require_nvfp4(
                store,
                &format!("{lp}.mlp.experts.{expert}.{projection}_proj"),
                n,
                k,
            )?;
        }
    }
    Ok(())
}

pub(super) fn validate_target(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    for index in 0..config.num_hidden_layers {
        validate_layer(store, &config.layer_prefix(index), config, index)?;
    }
    // Serving caps the usable vocabulary to the tokenizer. The checkpoint
    // keeps its padded output rows; the projection reads only that vocabulary
    // prefix. Require enough rows, exact hidden width and native BF16 storage.
    let head = store.get("lm_head.weight")?;
    ensure!(
        head.dtype == WeightDtype::BF16,
        "lm_head.weight: expected BF16, got {:?}",
        head.dtype
    );
    ensure!(
        head.shape.len() == 2
            && head.shape[0] >= config.vocab_size
            && head.shape[1] == config.hidden_size,
        "lm_head.weight: expected at least {} rows of width {}, got {:?}",
        config.vocab_size,
        config.hidden_size,
        head.shape
    );
    // Vision remains the checkpoint's existing path; the decoder's per-layer
    // audit above includes the indexer, which must not fall back to NVFP4.
    Ok(())
}

/// Validate the complete native FP4-Local MTP namespace before any BF16 expert
/// or attention tensor can be converted during draft-module construction.
pub(super) fn validate_mtp(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    ensure!(
        config.mtp_num_hidden_layers == 1,
        "Qwen Strict/Spark supports exactly one native MTP layer, got {}",
        config.mtp_num_hidden_layers
    );
    let h = config.hidden_size;
    require_bf16(store, "mtp.fc_embedding.weight", &[h, h])?;
    require_bf16(store, "mtp.fc_hidden.weight", &[h, h])?;
    require_bf16(store, "mtp.pre_fc_norm_embedding.weight", &[h])?;
    require_bf16(
        store,
        "mtp.pre_fc_norm_hidden.weight",
        &[config.hc_mult * h],
    )?;
    validate_mhc_site(store, "mtp.layers.0.attn_hyper_connection", config, true)?;
    validate_mhc_site(store, "mtp.layers.0.mlp_hyper_connection", config, true)?;
    validate_mhc_site(store, "mtp.hyper_connection_mixer", config, false)?;
    validate_mtp_attention(store, config)?;
    validate_mtp_moe(store, config)
}
