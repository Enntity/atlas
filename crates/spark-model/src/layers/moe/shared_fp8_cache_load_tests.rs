// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::weights::WeightTensor;

#[test]
fn checkpoint_contract_checks_shape_dtype_pointer_and_checked_extent() {
    let tensor = |shape, dtype, ptr| {
        WeightStore::from_map(std::collections::HashMap::from([(
            "w".into(),
            WeightTensor { shape, dtype, ptr },
        )]))
    };
    let good = tensor(vec![2048, 2048], WeightDtype::UInt8, DevicePtr(16));
    assert!(checkpoint_weight(&good, "w", DevicePtr(16), 2048, WEIGHT_BYTES / 2, false).is_ok());
    for (shape, dtype, ptr) in [
        (vec![4096, 1024], WeightDtype::UInt8, DevicePtr(16)),
        (vec![2048, 2047], WeightDtype::UInt8, DevicePtr(16)),
        (vec![2048, 2048], WeightDtype::BF16, DevicePtr(16)),
        (vec![2048, 2048], WeightDtype::UInt8, DevicePtr(32)),
        (vec![usize::MAX, 2], WeightDtype::UInt8, DevicePtr(16)),
    ] {
        assert!(
            checkpoint_weight(
                &tensor(shape, dtype, ptr),
                "w",
                DevicePtr(16),
                2048,
                WEIGHT_BYTES / 2,
                false
            )
            .is_err()
        );
    }
    let scale = tensor(vec![2048, 256], WeightDtype::FP8E4M3, DevicePtr(16));
    assert!(checkpoint_weight(&scale, "w", DevicePtr(16), 2048, WEIGHT_BYTES / 16, true).is_ok());
    assert!(checkpoint_weight(&scale, "w", DevicePtr(16), 2048, WEIGHT_BYTES / 16, false).is_err());
}

fn fixture(gpu: &dyn GpuBackend) -> (MoeLayer, WeightStore, atlas_core::config::ModelConfig) {
    let config = atlas_core::config::ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 4096,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_hidden_layers: 45,
        mlp_only_layers: vec![0, 1, 2],
        num_experts: 288,
        num_experts_per_tok: 8,
        tp_world_size: 2,
        ep_world_size: 2,
        ..atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4()
    };
    let weights = |i: u64| QuantizedWeight {
        weight: DevicePtr(0x1_0000_0000 + i * 0x100_0000),
        weight_scale: DevicePtr(0x1_0080_0000 + i * 0x100_0000),
        weight_scale_2: 1.,
        ..QuantizedWeight::null()
    };
    let mut original = MoeWeights::empty(288);
    original.shared_expert = crate::weight_map::ExpertWeight {
        gate_proj: weights(0),
        up_proj: weights(1),
        down_proj: weights(2),
    };
    let mut layer = MoeLayer::new(original, 288, None, gpu, &config).unwrap();
    layer.shared_gate_t = Some(weights(3));
    layer.shared_up_t = Some(weights(4));
    layer.shared_down_t = Some(weights(5));
    let mut map = std::collections::HashMap::new();
    for (i, name) in ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .enumerate()
    {
        let (n, k) = if i == 2 { (4096, 2048) } else { (2048, 4096) };
        map.insert(
            format!("shared.{name}.weight"),
            WeightTensor {
                ptr: weights(i as u64).weight,
                shape: vec![n, k / 2],
                dtype: WeightDtype::UInt8,
            },
        );
        map.insert(
            format!("shared.{name}.weight_scale"),
            WeightTensor {
                ptr: weights(i as u64).weight_scale,
                shape: vec![n, k / 16],
                dtype: WeightDtype::FP8E4M3,
            },
        );
    }
    // Source addresses are typed metadata fixtures, not numerical weights;
    // MockGpu records conversion launches without reading these pointers.
    (layer, WeightStore::from_map(map), config)
}

#[test]
fn actual_target_install_is_atomic_once_and_preserves_original_views() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let (mut layer, store, config) = fixture(&gpu);
    let before = (gpu.alloc_count(), gpu.launch_count());
    let original = layer.weights.shared_expert.gate_proj.weight;
    let transposed = layer.shared_gate_t.unwrap().weight;
    layer
        .install_glm_target_shared_fp8(
            &store,
            "shared",
            &config,
            3,
            Nvfp4Variant::Standard,
            &gpu,
            0,
            (true, false),
        )
        .unwrap();
    assert_eq!(
        (gpu.alloc_count(), gpu.launch_count()),
        (before.0 + 3, before.1 + 3)
    );
    assert_eq!(layer.shared_fp8_cache.layer, Some(3));
    assert_eq!(layer.weights.shared_expert.gate_proj.weight, original);
    assert_eq!(layer.shared_gate_t.unwrap().weight, transposed);
    assert!(layer.gate_fp8.is_none());
    let ready = (gpu.alloc_count(), gpu.launch_count(), layer.shared_gate_fp8);
    assert!(
        layer
            .install_glm_target_shared_fp8(
                &store,
                "shared",
                &config,
                3,
                Nvfp4Variant::Standard,
                &gpu,
                0,
                (true, false)
            )
            .is_err()
    );
    assert_eq!(
        (gpu.alloc_count(), gpu.launch_count(), layer.shared_gate_fp8),
        ready
    );
    // Production success ownership is the backend ledger. Mock explicitly
    // releases the three outputs because its default sweep tracks nothing.
    for p in [
        layer.shared_gate_fp8,
        layer.shared_up_fp8,
        layer.shared_down_fp8,
    ] {
        gpu.free(p.unwrap()).unwrap();
    }
    assert_eq!(gpu.alloc_count(), before.0);
}

#[test]
fn actual_install_disabled_mtp_and_malformed_views_do_no_gpu_work() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let (mut layer, store, config) = fixture(&gpu);
    let before = (gpu.alloc_count(), gpu.launch_count());
    layer
        .install_glm_target_shared_fp8(
            &WeightStore::empty(),
            "missing",
            &config,
            usize::MAX,
            Nvfp4Variant::Bf16Raw,
            &gpu,
            0,
            (false, false),
        )
        .unwrap();
    layer
        .maybe_cache_glm_target_shared_fp8(
            &store,
            "shared",
            &config,
            45,
            false,
            Nvfp4Variant::Standard,
            &gpu,
            0,
        )
        .unwrap();
    let valid = layer.shared_down_t;
    for malformed in [
        None,
        Some(layer.weights.shared_expert.down_proj),
        Some(QuantizedWeight::null()),
    ] {
        layer.shared_down_t = malformed;
        assert!(
            layer
                .install_glm_target_shared_fp8(
                    &store,
                    "shared",
                    &config,
                    3,
                    Nvfp4Variant::Standard,
                    &gpu,
                    0,
                    (true, false)
                )
                .is_err()
        );
        assert!(
            layer.shared_gate_fp8.is_none()
                && layer.shared_up_fp8.is_none()
                && layer.shared_down_fp8.is_none()
        );
        assert_eq!((gpu.alloc_count(), gpu.launch_count()), before);
    }
    layer.shared_down_t = valid;
    layer.shared_up_fp8 = Some(DevicePtr(16));
    assert!(
        layer
            .install_glm_target_shared_fp8(
                &store,
                "shared",
                &config,
                3,
                Nvfp4Variant::Standard,
                &gpu,
                0,
                (true, false)
            )
            .is_err()
    );
    assert_eq!((gpu.alloc_count(), gpu.launch_count()), before);
}

#[test]
fn actual_install_byte_copy_failure_does_not_publish_or_leak_temporary() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let (mut layer, store, config) = fixture(&gpu);
    let before = (gpu.alloc_count(), gpu.launch_count());
    // The metadata-only source addresses have no Mock allocation. After
    // the actual converter launch, VERIFY's first D2H deterministically
    // fails. This exercises the real install error/publication boundary.
    assert!(
        layer
            .install_glm_target_shared_fp8(
                &store,
                "shared",
                &config,
                3,
                Nvfp4Variant::Standard,
                &gpu,
                0,
                (true, true)
            )
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), before.0);
    assert_eq!(gpu.launch_count(), before.1 + 1);
    assert!(
        layer.shared_gate_fp8.is_none()
            && layer.shared_up_fp8.is_none()
            && layer.shared_down_fp8.is_none()
    );
    assert!(layer.shared_fp8_cache.layer.is_none());
}
