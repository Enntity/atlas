// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_core::config::LayerType;
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::atomic::{AtomicU64, Ordering};
struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn fixture(routed: bool) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "atlas-exl3-ffn-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let prefix = "model.language_model.layers.0.mlp";
    let groups = if routed {
        vec![
            format!("{prefix}.shared_experts"),
            format!("{prefix}.experts.0"),
            format!("{prefix}.experts.1"),
        ]
    } else {
        vec![prefix.into()]
    };
    let mut tensors = Vec::new();
    for group in groups {
        for (projection, k, n) in [
            ("gate_proj", 128, 256),
            ("up_proj", 128, 256),
            ("down_proj", 256, 128),
        ] {
            for (suffix, dtype, shape, data) in [
                (
                    "trellis",
                    Dtype::I16,
                    vec![k / 16, n / 16, 64],
                    vec![0; k * n / 2],
                ),
                (
                    "suh",
                    Dtype::F16,
                    vec![k],
                    0x3c00u16.to_le_bytes().repeat(k),
                ),
                (
                    "svh",
                    Dtype::F16,
                    vec![n],
                    0x3c00u16.to_le_bytes().repeat(n),
                ),
                (
                    "mul1",
                    Dtype::I32,
                    vec![],
                    0x83dc_d12du32.to_le_bytes().to_vec(),
                ),
            ] {
                tensors.push((format!("{group}.{projection}.{suffix}"), dtype, shape, data));
            }
        }
    }
    if routed {
        tensors.push((
            format!("{prefix}.gate.weight"),
            Dtype::F16,
            vec![2, 128],
            0x3c00u16.to_le_bytes().repeat(2 * 128),
        ));
        tensors.push((
            format!("{prefix}.gate.e_score_correction_bias"),
            Dtype::F32,
            vec![2],
            vec![0; 8],
        ));
    }
    safetensors::serialize_to_file(
        tensors.iter().map(|(name, dtype, shape, data)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        }),
        None,
        &root.join("weights.safetensors"),
    )
    .unwrap();
    let map: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, _, _, _)| (name.clone(), serde_json::json!("weights.safetensors")))
        .collect();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({"weight_map":map}).to_string(),
    )
    .unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}"#,
    )
    .unwrap();
    Fixture(root)
}
fn config(routed: bool) -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        weight_prefix: "model.language_model".into(),
        num_hidden_layers: 1,
        layer_types: vec![LayerType::LinearAttention],
        hidden_size: 128,
        intermediate_size: 256,
        moe_intermediate_size: 256,
        shared_expert_intermediate_size: 256,
        num_experts: 2,
        num_experts_per_tok: 1,
        scoring_func: "sigmoid".into(),
        moe_expert_groups: Some(1),
        moe_topk_groups: Some(1),
        routed_scaling_factor: 2.5,
        text_swiglu_limit: Some(10.0),
        mlp_only_layers: if routed { vec![] } else { vec![0] },
        tp_rank: 1,
        tp_world_size: 2,
        ep_rank: 0,
        ep_world_size: 1,
        expert_tp: false,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}
#[test]
fn checkpoint_loader_builds_dense_and_all_tp_experts_with_bf16_router() {
    for routed in [false, true] {
        let fixture = fixture(routed);
        let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
        let gpu = MockGpuBackend::new();
        let config = config(routed);
        let mut ffn = load::load_glm5_exl3_ffn_with_numerics(
            &router,
            &config,
            0,
            &gpu,
            3,
            77,
            fixture_numerics,
        )
        .unwrap();
        let crate::layers::FfnComponent::Exl3(native) = &ffn else {
            panic!("native FFN required")
        };
        assert_eq!(native.layer_index(), Some(0)); // independent of optional dumps
        match &**native {
            Glm5Exl3FfnLayer::Dense(d) => {
                assert!(!routed);
                assert_eq!(d.intermediate, 128);
                assert_eq!(d.role(), NativeDenseFfnRole::DensePacked);
                assert_eq!(
                    d.activation(),
                    NativeDenseActivation::PreSiLuClampedBf16 { limit: 10.0 }
                );
            }
            Glm5Exl3FfnLayer::Moe(m) => {
                assert!(routed);
                assert_eq!(m.shared.layer, Some(0));
                assert_eq!(m.experts, 2);
                assert_eq!(m.shared.role(), NativeDenseFfnRole::SharedExperts);
                assert!(m.shared.is_reconstructed());
                assert_eq!(
                    m.shared.activation(),
                    NativeDenseActivation::PreSiLuClampedBf16 { limit: 10.0 }
                );
                let mut bytes = vec![0; 2 * 128 * 2];
                gpu.copy_d2h(m.companions[0].ptr().unwrap(), &mut bytes)
                    .unwrap();
                assert!(bytes.chunks_exact(2).all(|b| b == 0x3f80u16.to_le_bytes()));
            }
        }
        ffn.release_exl3(&gpu).unwrap();
    }
}
#[test]
fn undeclared_group_policy_fails_before_gpu_upload() {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let mut config = config(true);
    config.moe_expert_groups = None;
    assert!(
        load::load_glm5_exl3_ffn_with_numerics(&router, &config, 0, &gpu, 3, 77, fixture_numerics)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn missing_down_projection_rolls_back_previously_uploaded_shards() {
    let fixture = fixture(false);
    let index = fixture.0.join("model.safetensors.index.json");
    let mut data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
    data["weight_map"]
        .as_object_mut()
        .unwrap()
        .retain(|key, _| !key.contains(".down_proj."));
    std::fs::write(index, data.to_string()).unwrap();
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    assert!(
        load::load_glm5_exl3_ffn_with_numerics(
            &router,
            &config(false),
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
fn routed_precision_uses_f32_slots_and_a_gpu_count_pass() {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let mut ffn = load::load_glm5_exl3_ffn_with_numerics(
        &router,
        &config(true),
        0,
        &gpu,
        3,
        77,
        fixture_numerics,
    )
    .unwrap();
    let crate::layers::FfnComponent::Exl3(native) = &ffn else {
        panic!()
    };
    let Glm5Exl3FfnLayer::Moe(m) = &**native else {
        panic!()
    };
    assert_eq!(m.precision, RoutedPrecisionPolicy::GLM53_TP2_REFERENCE);
    assert_eq!(m.buffers.len(), 11);
    let requested = gpu.kernel_requests_snapshot();
    let precision: Vec<_> = requested
        .iter()
        .filter(|(module, _)| module == "exl3_moe_precision" || module == "exl3_moe_mma")
        .map(|(_, name)| name.as_str())
        .collect();
    assert_eq!(
        precision,
        [
            "exl3_moe_input_hadamard_f16",
            "exl3_moe_mul1_matmul_precision_mma",
            "exl3_moe_output_hadamard_precision",
            "exl3_moe_combine_f32",
            "exl3_moe_activate_down_hadamard",
            "exl3_moe_route_counts",
            "exl3_moe_down_weighted_output",
        ]
    );
    for (index, values) in [(3, 3 * 128), (4, 3 * 128), (5, 3 * 128), (9, 2)] {
        gpu.copy_d2h(m.buffers[index], &mut vec![0; values * 4])
            .unwrap();
    }
    let input = gpu.alloc(128 * 2).unwrap();
    let output = gpu.alloc(128 * 2).unwrap();
    let before = gpu.launch_count();
    m.forward_local(&gpu, input, output, 1).unwrap();
    // Router, top-k, counts; 3 gate + 3 up; activation/down H;
    // down GEMM + weighted H; 1 shared activation launch plus 2 BF16 GEMMs; combine.
    assert_eq!(gpu.launch_count() - before, 14);
    let gemms = gpu.bf16_gemm_requests_snapshot();
    assert_eq!(gemms.len(), 3);
    assert_eq!(
        (
            gemms[0].rows,
            gemms[0].columns,
            gemms[0].inner,
            gemms[0].stream
        ),
        (1, 2, 128, 77)
    );
    let before = gpu.launch_count();
    m.forward_local(&gpu, input, output, 0).unwrap();
    assert_eq!(gpu.launch_count(), before);
    ffn.release_exl3(&gpu).unwrap();
    gpu.free(input).unwrap();
    gpu.free(output).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn unavailable_required_kernel_rolls_back_owned_experts_and_shared_weights() {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let cfg = config(true);
    let ffn =
        load::load_glm5_exl3_ffn_with_numerics(&router, &cfg, 0, &gpu, 3, 77, fixture_numerics)
            .unwrap();
    let crate::layers::FfnComponent::Exl3(native) = ffn else {
        panic!()
    };
    let Glm5Exl3FfnLayer::Moe(moe) = *native else {
        panic!()
    };
    let NativeMoeFfn {
        shared,
        tables,
        companions,
        buffers,
        ..
    } = *moe;
    for buffer in buffers {
        gpu.free(buffer).unwrap();
    }
    gpu.fail_next_kernel();
    assert!(
        NativeMoeFfn::new(
            &gpu,
            &cfg,
            shared,
            tables,
            companions,
            3,
            77,
            Exl3FfnActivation::ClampedSiLu { limit: 10.0 },
            RoutedPrecisionPolicy::GLM53_TP2_REFERENCE,
            None,
        )
        .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn moe_rejects_packed_shared_approximation_and_releases_owned_inputs() {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let cfg = config(true);
    let ffn =
        load::load_glm5_exl3_ffn_with_numerics(&router, &cfg, 0, &gpu, 3, 77, fixture_numerics)
            .unwrap();
    let crate::layers::FfnComponent::Exl3(native) = ffn else {
        panic!()
    };
    let Glm5Exl3FfnLayer::Moe(moe) = *native else {
        panic!()
    };
    let NativeMoeFfn {
        mut shared,
        tables,
        companions,
        buffers,
        ..
    } = *moe;
    shared.release(&gpu).unwrap();
    for buffer in buffers {
        gpu.free(buffer).unwrap();
    }
    let prefix = "model.language_model.layers.0.mlp.shared_experts";
    let shards: Vec<_> = ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|name| {
            crate::weight_loader::glm5::exl3::Exl3ProjectionSpec::from_rank_local_config(
                &cfg,
                &format!("{prefix}.{name}"),
            )
            .unwrap()
            .load(&router, &gpu)
            .unwrap()
        })
        .collect();
    let shared = NativeDenseFfn::new(
        &gpu,
        shards.try_into().ok().unwrap(),
        3,
        77,
        NativeDenseActivation::PreSiLuClampedBf16 { limit: 10.0 },
        NativeDenseFfnRole::DensePacked,
    )
    .unwrap();
    let result = NativeMoeFfn::new(
        &gpu,
        &cfg,
        shared,
        tables,
        companions,
        3,
        77,
        Exl3FfnActivation::ClampedSiLu { limit: 10.0 },
        RoutedPrecisionPolicy::GLM53_TP2_REFERENCE,
        None,
    );
    let rejected = result.is_err();
    if let Ok(mut incorrectly_accepted) = result {
        incorrectly_accepted.release(&gpu).unwrap();
    }
    assert!(
        rejected,
        "packed shared approximation must not enter native MoE"
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn zero_capacity_fails_before_shared_reconstruction_or_packed_upload() {
    for routed in [false, true] {
        let fixture = fixture(routed);
        let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
        let gpu = MockGpuBackend::new();
        assert!(
            load::load_glm5_exl3_ffn_with_numerics(
                &router,
                &config(routed),
                0,
                &gpu,
                0,
                77,
                fixture_numerics
            )
            .is_err()
        );
        assert!(gpu.kernel_requests_snapshot().is_empty());
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[path = "load_policy_tests.rs"]
mod policy_tests;
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

#[path = "scratch_load_tests.rs"]
mod scratch_load_tests;

fn load_perturbed(
    gpu: &MockGpuBackend,
    router: &Exl3CheckpointRouter,
    perturbation: crate::exl3_gate_perturbation::Exl3GatePerturbation,
) -> Result<crate::layers::FfnComponent> {
    load::load_glm5_exl3_ffn_with_scratch(
        router,
        &config(true),
        0,
        gpu,
        3,
        77,
        fixture_numerics,
        None,
        Some(perturbation),
    )
}

#[test]
fn drop_shared_control_skips_shared_expert_only_on_its_layer() {
    use crate::exl3_gate_perturbation::Exl3GatePerturbation as P;
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    // (perturbation on layer 0, other layer) -> launches/GEMMs per forward.
    for (perturbation, launches, gemms, dropped) in [
        (P::DropSharedLayer { layer: 0 }, 13, 1, true),
        (P::DropSharedLayer { layer: 1 }, 14, 3, false),
    ] {
        let gpu = MockGpuBackend::new();
        let mut ffn = load_perturbed(&gpu, &router, perturbation).unwrap();
        let crate::layers::FfnComponent::Exl3(native) = &ffn else {
            panic!()
        };
        let Glm5Exl3FfnLayer::Moe(m) = &**native else {
            panic!()
        };
        assert_eq!(m.drop_shared, dropped);
        assert_eq!(m.combine_flags(), [u32::from(!dropped), 0]);
        assert_eq!((m.topk, m.scale, m.round_routed_accum), (1, 2.5, false));
        let input = gpu.alloc(128 * 2).unwrap();
        let output = gpu.alloc(128 * 2).unwrap();
        let before = gpu.launch_count();
        m.forward_local(&gpu, input, output, 1).unwrap();
        assert_eq!(gpu.launch_count() - before, launches);
        assert_eq!(gpu.bf16_gemm_requests_snapshot().len(), gemms);
        ffn.release_exl3(&gpu).unwrap();
        gpu.free(input).unwrap();
        gpu.free(output).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn global_routed_controls_reach_the_native_moe() {
    use crate::exl3_gate_perturbation::Exl3GatePerturbation as P;
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    for (perturbation, scale, round) in [
        (P::RouterScale24, 2.4, false),
        (P::Bf16RoutedAccum, 2.5, true),
        (P::MhcPost19, 2.5, false),
    ] {
        let gpu = MockGpuBackend::new();
        let mut ffn = load_perturbed(&gpu, &router, perturbation).unwrap();
        let crate::layers::FfnComponent::Exl3(native) = &ffn else {
            panic!()
        };
        let Glm5Exl3FfnLayer::Moe(m) = &**native else {
            panic!()
        };
        assert_eq!(
            (m.scale, m.round_routed_accum, m.drop_shared),
            (scale, round, false)
        );
        assert_eq!(m.combine_flags(), [1, u32::from(round)]);
        ffn.release_exl3(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn topk_control_cannot_exceed_configured_scratch_and_rolls_back() {
    use crate::exl3_gate_perturbation::Exl3GatePerturbation as P;
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    // The fixture routes top-1; top-7 would overrun top-1 scratch.
    let error = load_perturbed(&gpu, &router, P::Topk7Layer { layer: 0 })
        .err()
        .expect("top-k increase must be rejected");
    assert!(format!("{error:#}").contains("only reduce routed top-k"));
    assert_eq!(gpu.alloc_count(), 0);
    // Another layer is unaffected.
    let mut ffn = load_perturbed(&gpu, &router, P::Topk7Layer { layer: 1 }).unwrap();
    ffn.release_exl3(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}
