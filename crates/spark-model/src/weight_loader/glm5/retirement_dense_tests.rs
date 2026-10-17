// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn actual_dense_conversion_and_tp_shard_preserve_checkpoint_origin() {
    for dtype in [WeightDtype::BF16, WeightDtype::FP32] {
        for rank in [0, 1] {
            let gpu = MockGpuBackend::new();
            let mut store = WeightStore::from_map(HashMap::from([(
                "q.weight".into(),
                WeightTensor {
                    ptr: gpu.alloc(16 * dtype.byte_size()).unwrap(),
                    shape: vec![4, 4],
                    dtype,
                },
            )]));
            let log = RetirementLog::new(&store, &gpu).unwrap();
            let source = LoadedDense::load(&store, "q.weight", &gpu, Some(&log), false).unwrap();
            let (local, _, _) = shard_dense_bf16(
                source.dense.weight,
                4,
                4,
                TpShardKind::ColumnParallel,
                rank,
                2,
                &gpu,
            )
            .unwrap();
            let local = source.replaced(local, 16, &gpu).unwrap();
            local.release(&gpu).unwrap();
            log.finish().rebuild(&mut store, &gpu).unwrap();
            assert_eq!(store.contains("q.weight"), dtype == WeightDtype::FP32);
            assert_eq!(gpu.alloc_count(), usize::from(dtype == WeightDtype::FP32));
        }
    }
}

#[test]
fn actual_fp8_packed_and_keep_f32_conversion_sources_keep_exact_provenance() {
    for (dtype, keep_f32) in [
        (WeightDtype::FP8E4M3, false),
        (WeightDtype::UInt8, false),
        (WeightDtype::BF16, true),
        (WeightDtype::FP32, true),
    ] {
        let gpu = MockGpuBackend::new();
        let mut map = HashMap::new();
        map.insert(
            "q.weight".into(),
            WeightTensor {
                ptr: gpu.alloc(16 * dtype.byte_size()).unwrap(),
                shape: vec![2, 8],
                dtype,
            },
        );
        let scale_dtype = if dtype == WeightDtype::UInt8 {
            WeightDtype::FP8E4M3
        } else {
            WeightDtype::FP32
        };
        map.insert(
            "q.weight_scale".into(),
            WeightTensor {
                ptr: gpu.alloc(scale_dtype.byte_size()).unwrap(),
                shape: vec![1],
                dtype: scale_dtype,
            },
        );
        map.insert(
            "q.weight_scale_2".into(),
            WeightTensor {
                ptr: gpu.alloc(4).unwrap(),
                shape: vec![1],
                dtype: WeightDtype::FP32,
            },
        );
        let mut store = WeightStore::from_map(map);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        LoadedDense::load(&store, "q.weight", &gpu, Some(&log), keep_f32)
            .unwrap()
            .release(&gpu)
            .unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        assert_eq!(
            store.contains("q.weight"),
            !(keep_f32 && dtype == WeightDtype::FP32)
        );
        assert!(store.contains("q.weight_scale"));
        assert!(store.contains("q.weight_scale_2"));
    }
}

/// Production loads without a log. A BF16 source is the store's own tensor:
/// replacing it by its TP shard must reclaim it, so teardown does not free it
/// a second time (FP4-Local's BF16 KDA conv1d, the NVFP4 release's BF16 KDA
/// projections). A converted source is a loader allocation, freed directly.
#[test]
fn logless_replacement_reclaims_a_store_alias_and_frees_a_conversion() {
    use atlas_core::scope::ModelResource;
    for dtype in [WeightDtype::BF16, WeightDtype::FP32] {
        let gpu = MockGpuBackend::new();
        let mut store = WeightStore::from_map(HashMap::from([(
            "q.weight".into(),
            WeightTensor {
                ptr: gpu.alloc(16 * dtype.byte_size()).unwrap(),
                shape: vec![4, 4],
                dtype,
            },
        )]));
        let source = LoadedDense::load(&store, "q.weight", &gpu, None, false).unwrap();
        let (local, _, _) = shard_dense_bf16(
            source.dense.weight,
            4,
            4,
            TpShardKind::ColumnParallel,
            0,
            2,
            &gpu,
        )
        .unwrap();
        source
            .replaced(local, 16, &gpu)
            .unwrap()
            .release(&gpu)
            .unwrap();
        assert_eq!(store.was_reclaimed("q.weight"), dtype == WeightDtype::BF16);
        assert_eq!(gpu.invalid_free_count(), 0);
        store.release(&gpu).unwrap();
        assert_eq!(
            gpu.invalid_free_count(),
            0,
            "{dtype:?}: no double free at teardown"
        );
        assert_eq!(gpu.alloc_count(), 0);
    }
}
