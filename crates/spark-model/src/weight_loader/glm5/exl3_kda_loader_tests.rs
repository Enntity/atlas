// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::LayerType;
use safetensors::tensor::TensorView;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
impl Fixture {
    fn new(defect: Option<&str>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-glm-exl3-kda-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let p = "model.language_model.layers.0.self_attn";
        let mut tensors = Vec::new();
        for (name, k, n) in [("qkv_proj", 256, 768), ("o_proj", 256, 256)] {
            for (suffix, dtype, shape, data) in [
                (
                    "trellis",
                    Dtype::I16,
                    vec![k / 16, n / 16, 64],
                    (0..k * n / 2).map(|i| (i * 71 + i / 137) as u8).collect(),
                ),
                (
                    "suh",
                    Dtype::F16,
                    vec![k],
                    (0..k)
                        .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
                        .collect(),
                ),
                (
                    "svh",
                    Dtype::F16,
                    vec![n],
                    (0..n)
                        .flat_map(|i| half::f16::from_f32((i % 32) as f32 + 1.0).to_le_bytes())
                        .collect(),
                ),
                (
                    "mul1",
                    Dtype::I32,
                    vec![],
                    0x83dc_d12du32.to_le_bytes().to_vec(),
                ),
            ] {
                tensors.push((format!("{p}.{name}.{suffix}"), dtype, shape, data));
            }
        }
        for (name, mut dtype, shape) in [
            ("b_proj.weight", Dtype::F16, vec![2, 256]),
            ("f_a_proj.weight", Dtype::F16, vec![128, 256]),
            ("f_b_proj.weight", Dtype::F16, vec![256, 128]),
            ("g_a_proj.weight", Dtype::F16, vec![128, 256]),
            ("g_b_proj.weight", Dtype::F16, vec![256, 128]),
            ("conv1d.weight", Dtype::BF16, vec![768, 1, 4]),
            ("A_log", Dtype::F32, vec![2]),
            ("dt_bias", Dtype::F32, vec![256]),
            ("o_norm.weight", Dtype::BF16, vec![128]),
        ] {
            if defect == Some("conv_missing") && name == "conv1d.weight" {
                continue;
            }
            if defect == Some("wrong_f_b_dtype") && name == "f_b_proj.weight" {
                dtype = Dtype::BF16;
            }
            let n = shape.iter().product::<usize>();
            let data = match dtype {
                Dtype::F16 => (0..n)
                    .flat_map(|i| half::f16::from_f32((i % 257) as f32 / 16.0).to_le_bytes())
                    .collect(),
                Dtype::BF16 => (0..n)
                    .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
                    .collect(),
                Dtype::F32 => (0..n)
                    .flat_map(|i| (i as f32 / 32.0).to_le_bytes())
                    .collect(),
                _ => unreachable!(),
            };
            tensors.push((format!("{p}.{name}"), dtype, shape, data));
        }
        let views = tensors.iter().map(|(name, dtype, shape, data)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        });
        safetensors::serialize_to_file(views, None, &root.join("a.safetensors")).unwrap();
        let weight_map: serde_json::Map<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, _, _, _)| (name.clone(), serde_json::json!("a.safetensors")))
            .collect();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map":weight_map}).to_string(),
        )
        .unwrap();
        std::fs::write(
            root.join("config.json"),
            r#"{"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}"#,
        )
        .unwrap();
        Self(root)
    }
    fn config() -> ModelConfig {
        ModelConfig {
            model_type: "glm5_next".into(),
            weight_prefix: "model.language_model".into(),
            hidden_size: 256,
            num_hidden_layers: 1,
            layer_types: vec![LayerType::LinearAttention],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            tp_rank: 1,
            tp_world_size: 2,
            ep_rank: 0,
            ep_world_size: 1,
            expert_tp: false,
            ..ModelConfig::qwen3_next_80b_nvfp4()
        }
    }
}

#[test]
fn kda_load_splits_fused_qkv_conv_and_dense_auxiliaries_with_correct_dtypes() {
    let fixture = Fixture::new(None);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let mut resource = Glm5Exl3KdaResources::load_with_numerics(
        &router,
        &Fixture::config(),
        0,
        &gpu,
        3,
        77,
        fixture_numerics,
    )
    .unwrap();
    assert_eq!(
        gpu.kernel_requests_snapshot()
            .iter()
            .filter(|(module, name)| module == "exl3" && name == "exl3_input_hadamard_f16")
            .count(),
        8,
        "preflight and owned construction both resolve F16 Q/K/V/O boundaries",
    );
    for (projection, range) in [
        (&resource.q_proj, 128..256),
        (&resource.k_proj, 384..512),
        (&resource.v_proj, 640..768),
    ] {
        assert_eq!(projection.shard().plan().output_ranges(), &[range]);
        assert_eq!(
            (
                projection.shard().plan().in_features(),
                projection.shard().plan().out_features()
            ),
            (256, 128)
        );
    }
    let conv = resource.dense.tensor(Exl3KdaDenseSlot::Conv).unwrap();
    assert_eq!(conv.shape(), &[384, 1, 4]);
    let expected: Vec<u8> = [128..256, 384..512, 640..768]
        .into_iter()
        .flat_map(|range| {
            (range.start * 4..range.end * 4)
                .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
        })
        .collect();
    assert_eq!(gpu.read_alloc(conv.ptr().unwrap()).unwrap(), expected);
    for (slot, shape, dtype) in [
        (
            Exl3KdaDenseSlot::AuxMerged,
            vec![1 + 128 + 128, 256],
            Exl3DenseDtype::Bf16,
        ),
        (
            Exl3KdaDenseSlot::FBProj,
            vec![128, 128],
            Exl3DenseDtype::Bf16,
        ),
        (Exl3KdaDenseSlot::ALog, vec![1], Exl3DenseDtype::F32),
        (Exl3KdaDenseSlot::DtBias, vec![128], Exl3DenseDtype::F32),
        (Exl3KdaDenseSlot::ONorm, vec![128], Exl3DenseDtype::Bf16),
    ] {
        let tensor = resource.dense.tensor(slot).unwrap();
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), dtype);
    }
    let a_log = resource.dense.tensor(Exl3KdaDenseSlot::ALog).unwrap();
    assert_eq!(
        gpu.read_alloc(a_log.ptr().unwrap()).unwrap(),
        (1.0f32 / 32.0).to_le_bytes()
    );
    // B/FA/GA sources are retired; only interior views of the merged base remain.
    for slot in [
        Exl3KdaDenseSlot::BProj,
        Exl3KdaDenseSlot::FAProj,
        Exl3KdaDenseSlot::GAProj,
    ] {
        assert!(resource.dense.tensor(slot).is_err());
    }
    let merged = resource.dense.tensor(Exl3KdaDenseSlot::AuxMerged).unwrap();
    let base = merged.ptr().unwrap();
    let view = |slot| resource.dense.weight(slot).unwrap().weight;
    assert_eq!(view(Exl3KdaDenseSlot::BProj), base);
    assert_eq!(view(Exl3KdaDenseSlot::FAProj), base.offset(256 * 2));
    assert_eq!(view(Exl3KdaDenseSlot::GAProj), base.offset(129 * 256 * 2));
    let bytes = gpu.read_alloc(base).unwrap();
    let expected: Vec<u8> = (256..512)
        .flat_map(|i| {
            half::bf16::from_f32(half::f16::from_f32((i % 257) as f32 / 16.0).to_f32())
                .to_le_bytes()
        })
        .collect();
    assert_eq!(bytes[..256 * 2], expected[..]);
    assert_eq!(bytes.len(), 257 * 256 * 2);
    let fa_ga: Vec<u8> = (0..128 * 256)
        .flat_map(|i| {
            half::bf16::from_f32(half::f16::from_f32((i % 257) as f32 / 16.0).to_f32())
                .to_le_bytes()
        })
        .collect();
    assert_eq!(bytes[256 * 2..129 * 256 * 2], fa_ga[..]);
    assert_eq!(bytes[129 * 256 * 2..], fa_ga[..]);
    resource.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn malformed_kda_checkpoint_rolls_back_all_prior_device_allocations() {
    for defect in ["conv_missing", "wrong_f_b_dtype"] {
        let fixture = Fixture::new(Some(defect));
        let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
        let gpu = MockGpuBackend::new();
        assert!(
            Glm5Exl3KdaResources::load_with_numerics(
                &router,
                &Fixture::config(),
                0,
                &gpu,
                3,
                77,
                fixture_numerics
            )
            .is_err()
        );
        assert_eq!(gpu.alloc_count(), 0, "{defect}");
    }
}

#[test]
fn kda_loader_rejects_non_kda_layer_and_topology_before_uploading() {
    let fixture = Fixture::new(None);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let mut config = Fixture::config();
    config.layer_types[0] = LayerType::FullAttention;
    assert!(
        Glm5Exl3KdaResources::load_with_numerics(
            &router,
            &config,
            0,
            &gpu,
            3,
            77,
            fixture_numerics
        )
        .is_err()
    );
    let mut config = Fixture::config();
    config.ep_world_size = 2;
    assert!(
        Glm5Exl3KdaResources::load_with_numerics(
            &router,
            &config,
            0,
            &gpu,
            3,
            77,
            fixture_numerics
        )
        .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn kda_resources_transfer_into_real_native_layer_weights_with_owned_companions() {
    let fixture = Fixture::new(None);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let resources = Glm5Exl3KdaResources::load_with_numerics(
        &router,
        &Fixture::config(),
        0,
        &gpu,
        3,
        77,
        fixture_numerics,
    )
    .unwrap();
    let (mut weights, mut dense) = resources.into_weights(&gpu).unwrap();
    assert_eq!(
        weights.conv.weight,
        dense.weight(Exl3KdaDenseSlot::Conv).unwrap().weight
    );
    for projection in [
        &mut weights.q_proj,
        &mut weights.k_proj,
        &mut weights.v_proj,
        &mut weights.o_proj,
    ] {
        let Glm5Projection::Exl3(projection) = projection else {
            panic!("native EXL3 variant required")
        };
        projection.release(&gpu).unwrap();
    }
    dense.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn invalid_resource_transfer_releases_remaining_owners() {
    let fixture = Fixture::new(None);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let mut resources = Glm5Exl3KdaResources::load_with_numerics(
        &router,
        &Fixture::config(),
        0,
        &gpu,
        3,
        77,
        fixture_numerics,
    )
    .unwrap();
    resources.dense.release(&gpu).unwrap();
    assert!(resources.into_weights(&gpu).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn auxiliary_merge_failed_retirement_stays_owned_for_rollback() {
    let gpu = MockGpuBackend::new();
    let mut dense = Vec::new();
    for (slot, rows, value) in [
        (Exl3KdaDenseSlot::BProj, 1, 1u16),
        (Exl3KdaDenseSlot::FAProj, 2, 2),
        (Exl3KdaDenseSlot::GAProj, 2, 3),
    ] {
        let data = half::bf16::from_bits(value).to_le_bytes().repeat(rows * 4);
        let view = TensorView::new(Dtype::BF16, vec![rows, 4], &data).unwrap();
        let tensor = Exl3DenseDeviceTensor::upload(
            &gpu,
            &view,
            Exl3DenseDtype::Bf16,
            &single_range(0..rows),
        )
        .unwrap();
        dense.push((slot, tensor));
    }
    gpu.fail_next_free();
    assert!(super::aux::merge_auxiliary(&mut dense, &gpu).is_err());
    let slots: Vec<_> = dense.iter().map(|(slot, _)| *slot).collect();
    assert_eq!(
        slots,
        [Exl3KdaDenseSlot::AuxMerged, Exl3KdaDenseSlot::BProj],
        "merged base owned; failed B retained; FA/GA retired"
    );
    let merged = gpu.read_alloc(dense[0].1.ptr().unwrap()).unwrap();
    let expected: Vec<u8> = [(1u16, 4), (2, 8), (3, 8)]
        .into_iter()
        .flat_map(|(v, n)| v.to_le_bytes().repeat(n))
        .collect();
    assert_eq!(merged, expected);
    release_dense(&mut dense, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);

    let mut missing = Vec::new();
    assert!(super::aux::merge_auxiliary(&mut missing, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

fn fixture_numerics(
    _: usize,
    _: usize,
    _: usize,
    bits: u8,
    logical_n: usize,
    logical_offset: usize,
) -> Result<spark_runtime::weights::exl3::native::Exl3LinearNumerics> {
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

#[test]
fn pinned_loader_rejects_unsupported_geometry_and_kernel_failure_before_upload() {
    let fixture = Fixture::new(None);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    assert!(Glm5Exl3KdaResources::load(&router, &Fixture::config(), 0, &gpu, 3, 77).is_err());
    assert!(gpu.kernel_requests_snapshot().is_empty());
    assert_eq!(gpu.alloc_count(), 0);
    gpu.fail_next_kernel();
    assert!(
        Glm5Exl3KdaResources::load_with_numerics(
            &router,
            &Fixture::config(),
            0,
            &gpu,
            3,
            77,
            fixture_numerics
        )
        .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
}
