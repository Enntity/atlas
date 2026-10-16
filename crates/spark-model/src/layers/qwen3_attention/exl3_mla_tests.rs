// SPDX-License-Identifier: AGPL-3.0-only

use super::Qwen3AttentionLayer;
use crate::{
    layers::FfnComponent, weight_loader::glm5::exl3_mla::load_exl3_mla_layer_with_numerics,
    weight_map::DenseWeight,
};
use atlas_core::{
    config::{LayerType, ModelConfig},
    scope::ModelResource,
};
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend},
    kv_cache::KvCacheDtype,
    weights::{
        WeightStore,
        exl3::{Exl3CheckpointRouter, dense::Exl3DenseDtype},
    },
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> (Fixture, ModelConfig, Vec<(String, Exl3DenseDtype)>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "atlas-exl3-mla-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.weight_prefix = "model.language_model".into();
    config.hidden_size = 256;
    config.num_hidden_layers = 1;
    config.layer_types = vec![LayerType::FullAttention];
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = 128;
    config.qk_nope_head_dim = 128;
    config.qk_rope_head_dim = 0;
    config.v_head_dim = 128;
    config.q_lora_rank = 128;
    config.kv_lora_rank = 512;
    config.index_n_heads = 1;
    config.index_head_dim = 128;
    config.index_topk = 2048;
    config.index_kpool_always_select_tail = true;
    config.index_kpool = 4;
    config.tp_world_size = 2;
    config.ep_world_size = 1;
    let p = "model.language_model.layers.0.self_attn";
    let mut tensors = Vec::new();
    for (name, k, n) in [
        ("q_a_proj", 256, 128),
        ("q_b_proj", 128, 256),
        ("kv_a_proj_with_mqa", 256, 512),
        ("o_proj", 256, 256),
        ("indexer.wq_b", 128, 128),
    ] {
        tensors.push((
            format!("{p}.{name}.trellis"),
            Dtype::I16,
            vec![k / 16, n / 16, 64],
            vec![0u8; k * n / 2],
        ));
        tensors.push((
            format!("{p}.{name}.suh"),
            Dtype::F16,
            vec![k],
            0x3c00u16.to_le_bytes().repeat(k),
        ));
        tensors.push((
            format!("{p}.{name}.svh"),
            Dtype::F16,
            vec![n],
            0x3c00u16.to_le_bytes().repeat(n),
        ));
        tensors.push((
            format!("{p}.{name}.mul1"),
            Dtype::I32,
            vec![],
            0x83dc_d12du32.to_le_bytes().to_vec(),
        ));
    }
    let mut selected = Vec::new();
    for (name, shape) in [
        ("q_a_layernorm.weight", vec![128]),
        ("kv_a_layernorm.weight", vec![512]),
        ("indexer.wk.weight", vec![128, 256]),
        ("indexer.weights_proj.weight", vec![1, 256]),
        ("indexer.index_kpool_compress_gate", vec![128, 256]),
        ("indexer.index_kpool_compress_ape", vec![4, 128]),
        ("indexer.k_norm.weight", vec![128]),
        ("indexer.k_norm.bias", vec![128]),
    ] {
        let name = format!("{p}.{name}");
        let elements: usize = shape.iter().product();
        let ape = name.ends_with("index_kpool_compress_ape");
        tensors.push((
            name.clone(),
            if ape { Dtype::F32 } else { Dtype::F16 },
            shape,
            if ape {
                1.0f32.to_le_bytes().repeat(elements)
            } else {
                0x3c00u16.to_le_bytes().repeat(elements)
            },
        ));
        selected.push((
            name,
            if ape {
                Exl3DenseDtype::F32
            } else {
                Exl3DenseDtype::Bf16
            },
        ));
    }
    let kv: Vec<u8> = (0..512 * 512)
        .flat_map(|i| half::f16::from_f32((i / 512) as f32).to_le_bytes())
        .collect();
    tensors.push((
        format!("{p}.kv_b_proj.weight"),
        Dtype::F16,
        vec![512, 512],
        kv,
    ));
    safetensors::serialize_to_file(
        tensors.iter().map(|(name, dtype, shape, data)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        }),
        None,
        &root.join("a.safetensors"),
    )
    .unwrap();
    let weights: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(n, _, _, _)| (n.clone(), serde_json::json!("a.safetensors")))
        .collect();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({"weight_map":weights}).to_string(),
    )
    .unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}"#,
    )
    .unwrap();
    (Fixture(root), config, selected)
}

#[test]
fn actual_mla_loader_owns_rank_local_weights_and_dispatches_every_native_field() {
    let (f, mut config, selected) = fixture();
    for rank in 0..2 {
        config.tp_rank = rank;
        let gpu = MockGpuBackend::new();
        let router = Exl3CheckpointRouter::open(&f.0).unwrap();
        let mut store = WeightStore::from_exl3(router, &selected, &gpu, 0).unwrap();
        let norm = DenseWeight {
            weight: DevicePtr(8192),
        };
        let mut layer = load_exl3_mla_layer_with_numerics(
            &store,
            &config,
            0,
            0,
            norm,
            norm,
            FfnComponent::None,
            None,
            &gpu,
            KvCacheDtype::Bf16,
            3,
            77,
            fixture_numerics,
            None,
        )
        .unwrap();
        assert_eq!(
            gpu.kernel_requests_snapshot()
                .iter()
                .filter(|(module, name)| module == "exl3" && name == "exl3_input_hadamard_f16")
                .count(),
            10,
            "packed MLA projections require F16 input boundaries"
        );
        let layer = layer
            .as_any_mut()
            .unwrap()
            .downcast_mut::<Qwen3AttentionLayer>()
            .unwrap();
        let budget = layer.glm_index_budget(&config).unwrap();
        assert_eq!(budget.history_tokens(), 2044);
        assert_eq!(budget.dense_threshold(), 2047);
        assert_eq!(budget.physical_width(), 2051);
        let mut unsupported = config.clone();
        unsupported.index_kpool_always_select_tail = false;
        let launches = gpu.launch_count();
        assert!(
            layer
                .configure_exl3_index_budget(&gpu, &unsupported)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), launches);
        let native = layer.exl3_mla.as_ref().unwrap();
        let params = native.index_f32_params().unwrap();
        assert_eq!(gpu.read_alloc(params.norm_weight).unwrap().len(), 128 * 4);
        assert_eq!(gpu.read_alloc(params.norm_bias).unwrap().len(), 128 * 4);
        assert_eq!(
            params.ape,
            store
                .get("model.language_model.layers.0.self_attn.indexer.index_kpool_compress_ape")
                .unwrap()
                .ptr
        );
        native
            .index_exec()
            .unwrap()
            .validate(3, 1, 128, 256, 12, 77)
            .unwrap();
        let before = gpu.launch_count();
        native
            .index_exec()
            .unwrap()
            .query_and_weights(
                &gpu,
                DevicePtr(100),
                DevicePtr(200),
                DevicePtr(300),
                DevicePtr(400),
                2,
            )
            .unwrap();
        assert_eq!(gpu.launch_count() - before, 3);
        let mla = layer.mla.as_ref().unwrap();
        let kv = gpu.read_alloc(mla.wkv_b.weight).unwrap();
        assert_eq!(kv.len(), 256 * 512 * 2);
        assert_eq!(
            &kv[..2],
            &half::bf16::from_f32((rank * 256) as f32).to_le_bytes()
        );
        for (field, n, k) in [
            (&mla.wq_a, 128, 256),
            (&mla.wq_b, 128, 128),
            (&mla.wkv_a, 512, 256),
            (&mla.wo, 256, 128),
            (&mla.glm_indexer.as_ref().unwrap().wq_b, 128, 128),
        ] {
            let before = gpu.launch_count();
            assert!(
                layer
                    .try_exl3_mla_projection(
                        &gpu,
                        DevicePtr(100),
                        field,
                        DevicePtr(200),
                        2,
                        n,
                        k,
                        77
                    )
                    .unwrap()
            );
            assert_eq!(gpu.launch_count() - before, 3);
            assert!(
                layer
                    .try_exl3_mla_projection(
                        &gpu,
                        DevicePtr(100),
                        field,
                        DevicePtr(200),
                        4,
                        n,
                        k,
                        77
                    )
                    .is_err()
            );
            assert_eq!(gpu.launch_count() - before, 3);
        }
        let foreign = DenseWeight {
            weight: DevicePtr::NULL,
        };
        assert!(
            !layer
                .try_exl3_mla_projection(
                    &gpu,
                    DevicePtr(100),
                    &foreign,
                    DevicePtr(200),
                    2,
                    128,
                    128,
                    77
                )
                .unwrap()
        );
        assert!(layer.install_mla_nvfp4(&gpu, &[]).is_err());
        assert!(layer.install_mla_mxfp8(&gpu, &[]).is_err());
        layer.release_exl3_mla(&gpu).unwrap();
        layer.release_exl3_mla(&gpu).unwrap();
        store.release(&gpu).unwrap();
        assert_eq!(gpu.sweep_unreleased(), 0);
    }
}

fn fixture_numerics(
    _: usize,
    _: usize,
    _: usize,
    bits: u8,
    logical_n: usize,
    logical_offset: usize,
) -> anyhow::Result<spark_runtime::weights::exl3::native::Exl3LinearNumerics> {
    use spark_runtime::weights::exl3::native::{Exl3CooperativeProfile, Exl3LinearNumerics};
    Ok(Exl3LinearNumerics {
        logical_n,
        logical_offset,
        bits,
        reference_min_rows: 3,
        reference_pad: true,
        row_profiles: [
            None,
            Some(Exl3CooperativeProfile {
                tile_k: 32,
                tile_n: 128,
                slices: 48,
                half_partial: false,
            }),
            Some(Exl3CooperativeProfile {
                tile_k: 32,
                tile_n: 128,
                slices: 48,
                half_partial: false,
            }),
            Some(Exl3CooperativeProfile {
                tile_k: 32,
                tile_n: 128,
                slices: 48,
                half_partial: false,
            }),
        ],
        verify_rows_5_8: None,
    })
}
