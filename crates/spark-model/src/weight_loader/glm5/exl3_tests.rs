// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::LayerType;

fn config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        weight_prefix: "model.language_model".into(),
        tp_world_size: 2,
        tp_rank: 1,
        ep_world_size: 1,
        ep_rank: 0,
        num_hidden_layers: 4,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        mlp_only_layers: vec![0, 1, 2],
        hidden_size: 4096,
        intermediate_size: 12288,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_experts: 288,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        linear_num_key_heads: 32,
        linear_num_value_heads: 32,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        vocab_size: 154880,
        index_n_heads: 32,
        index_head_dim: 128,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

fn projection(name: &str) -> Exl3ProjectionSpec {
    Exl3ProjectionSpec::from_rank_local_config(&config(), name).unwrap()
}

#[test]
fn fused_kda_qkv_preserves_component_ownership() {
    let spec = projection("model.language_model.layers.0.self_attn.qkv_proj");
    assert_eq!((spec.in_features(), spec.out_features()), (4096, 24576));
    assert_eq!(spec.output_segments(), &[8192, 8192, 8192]);
    assert_eq!(spec.axis(), ShardAxis::Output);
}

#[test]
fn mla_reconstructs_full_heads_from_explicit_local_config() {
    for (name, k, n, axis) in [
        ("q_a_proj", 4096, 1536, ShardAxis::Replicated),
        ("kv_a_proj_with_mqa", 4096, 512, ShardAxis::Replicated),
        ("q_b_proj", 1536, 16384, ShardAxis::Output),
        ("kv_b_proj", 512, 32768, ShardAxis::Output),
        ("o_proj", 16384, 4096, ShardAxis::Input),
        ("indexer.wq_b", 1536, 4096, ShardAxis::Replicated),
    ] {
        let spec = projection(&format!("model.language_model.layers.3.self_attn.{name}"));
        assert_eq!(
            (spec.in_features(), spec.out_features(), spec.axis()),
            (k, n, axis)
        );
    }
}

#[test]
fn every_expert_is_tp_sharded_on_every_rank() {
    for rank in 0..2 {
        let mut cfg = config();
        cfg.tp_rank = rank;
        for expert in 0..cfg.num_experts {
            for (name, k, n, axis) in [
                ("gate_proj", 4096, 2048, ShardAxis::Output),
                ("up_proj", 4096, 2048, ShardAxis::Output),
                ("down_proj", 2048, 4096, ShardAxis::Input),
            ] {
                let name = format!("model.language_model.layers.3.mlp.experts.{expert}.{name}");
                let spec = Exl3ProjectionSpec::from_rank_local_config(&cfg, &name).unwrap();
                assert_eq!(
                    (spec.in_features(), spec.out_features(), spec.axis()),
                    (k, n, axis)
                );
            }
        }
    }
}

#[test]
fn shared_dense_and_head_dimensions_are_explicit() {
    for (name, k, n, axis) in [
        (
            "model.language_model.layers.0.mlp.gate_proj",
            4096,
            12288,
            ShardAxis::Output,
        ),
        (
            "model.language_model.layers.0.mlp.down_proj",
            12288,
            4096,
            ShardAxis::Input,
        ),
        (
            "model.language_model.layers.3.mlp.shared_experts.up_proj",
            4096,
            2048,
            ShardAxis::Output,
        ),
        (
            "model.language_model.layers.3.mlp.shared_experts.down_proj",
            2048,
            4096,
            ShardAxis::Input,
        ),
        ("lm_head", 4096, 154880, ShardAxis::Output),
    ] {
        let spec = projection(name);
        assert_eq!(
            (spec.in_features(), spec.out_features(), spec.axis()),
            (k, n, axis)
        );
    }
}

#[test]
fn unknown_or_incompatible_prefixes_fail_before_io() {
    for name in [
        "visual.blocks.0.attn.qkv",
        "model.language_model.layers.4.mlp.gate_proj",
        "model.language_model.layers.0.self_attn.q_b_proj",
        "model.language_model.layers.3.self_attn.qkv_proj",
        "model.language_model.layers.0.mlp.experts.0.up_proj",
        "model.language_model.layers.3.mlp.experts.288.up_proj",
        "model.language_model.layers.3.mlp.gate_proj",
        "model.language_model.layers.3.self_attn.unknown",
    ] {
        assert!(
            Exl3ProjectionSpec::from_rank_local_config(&config(), name).is_err(),
            "{name}"
        );
    }
}

#[test]
fn explicit_topology_and_positive_dimensions_are_required() {
    for field in 0..7 {
        let mut cfg = config();
        match field {
            0 => cfg.tp_world_size = 0,
            1 => cfg.tp_rank = 2,
            2 => cfg.ep_world_size = 2,
            3 => cfg.ep_rank = 1,
            4 => cfg.expert_tp = true,
            5 => cfg.model_type = "qwen3".into(),
            _ => cfg.hidden_size = 0,
        }
        assert!(Exl3ProjectionSpec::from_rank_local_config(&cfg, "lm_head").is_err());
    }
}

#[test]
fn checkpoint_router_uploads_only_segment_local_bytes_and_releases_buffers() {
    use atlas_core::scope::ModelResource;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let dir = Directory(std::env::temp_dir().join(format!(
        "atlas-glm-exl3-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    std::fs::create_dir(&dir.0).unwrap();
    let prefix = "model.language_model.layers.0.self_attn.qkv_proj";
    let k = 256;
    let n = 768;
    let trellis: Vec<u8> = (0..k / 16 * (n / 16) * 128)
        .map(|i: usize| (i.wrapping_mul(71) ^ (i >> 8)) as u8)
        .collect();
    let input: Vec<u8> = (0..k)
        .flat_map(|i| (0x3000 + i as u16).to_le_bytes())
        .collect();
    let output: Vec<u8> = (0..n)
        .flat_map(|i| (0x3400 + i as u16).to_le_bytes())
        .collect();
    let marker = 0x83dc_d12du32.to_le_bytes().to_vec();
    let mut header = serde_json::Map::new();
    let mut index = serde_json::Map::new();
    let mut data = Vec::new();
    for (suffix, dtype, shape, bytes) in [
        ("trellis", "I16", vec![16, 48, 64], &trellis),
        ("suh", "F16", vec![k], &input),
        ("svh", "F16", vec![n], &output),
        ("mul1", "I32", vec![], &marker),
    ] {
        let name = format!("{prefix}.{suffix}");
        header.insert(
            name.clone(),
            serde_json::json!({"dtype":dtype,
            "shape":shape,"data_offsets":[data.len(),data.len()+bytes.len()]}),
        );
        index.insert(name, serde_json::json!("weights.safetensors"));
        data.extend_from_slice(bytes);
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(data);
    std::fs::write(dir.0.join("weights.safetensors"), file).unwrap();
    std::fs::write(
        dir.0.join("model.safetensors.index.json"),
        serde_json::json!({"weight_map":index}).to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.0.join("config.json"),
        r#"{"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}"#,
    )
    .unwrap();
    let router = Exl3CheckpointRouter::open(&dir.0).unwrap();
    let mut cfg = config();
    cfg.hidden_size = k;
    cfg.linear_num_key_heads = 1;
    cfg.linear_num_value_heads = 1;
    let spec = Exl3ProjectionSpec::from_rank_local_config(&cfg, prefix).unwrap();
    let gpu = MockGpuBackend::new();
    let mut shard = spec.load(&router, &gpu).unwrap();
    assert_eq!(
        shard.plan().output_ranges(),
        &[128..256, 384..512, 640..768]
    );
    let buffers = shard.buffers().unwrap();
    assert_eq!(gpu.read_alloc(buffers[0]).unwrap().len(), trellis.len() / 2);
    assert_eq!(gpu.read_alloc(buffers[1]).unwrap(), input);
    let expected: Vec<u8> = [128..256, 384..512, 640..768]
        .iter()
        .flat_map(|range| output[range.start * 2..range.end * 2].iter().copied())
        .collect();
    assert_eq!(gpu.read_alloc(buffers[2]).unwrap(), expected);
    // Flat concatenation would select the wrong Q/K/V ownership boundaries.
    let expected: Vec<u8> = (0..16)
        .flat_map(|row| {
            let row_start = row * 48 * 128;
            let trellis = &trellis;
            [8..16, 24..32, 40..48].into_iter().flat_map(move |range| {
                trellis[row_start + range.start * 128..row_start + range.end * 128].to_vec()
            })
        })
        .collect();
    assert_eq!(gpu.read_alloc(buffers[0]).unwrap(), expected);
    shard.release(&gpu).unwrap();
    assert!(shard.buffers().is_err());
    assert!(
        buffers
            .iter()
            .all(|&buffer| gpu.read_alloc(buffer).is_none())
    );
}
