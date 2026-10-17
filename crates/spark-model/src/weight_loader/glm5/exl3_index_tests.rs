// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_core::scope::ModelResource;
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::exl3::{Exl3CheckpointRouter, dense::Exl3DenseDtype};
use std::sync::atomic::{AtomicUsize, Ordering};

const PREFIX: &str = "model.language_model.layers.3.self_attn.indexer";
struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture(bias_dim: usize, ape_dtype: Dtype) -> Fixture {
    fixture_values(bias_dim, ape_dtype, 0x3c01, 0x3f80_0001)
}
fn fixture_values(bias_dim: usize, ape_dtype: Dtype, weight_bits: u16, ape_bits: u32) -> Fixture {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "atlas-exl3-index-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}"#,
    )
    .unwrap();
    let ape_bytes = if ape_dtype == Dtype::F32 {
        ape_bits.to_le_bytes().repeat(4 * 128)
    } else {
        0x3c01u16.to_le_bytes().repeat(4 * 128)
    };
    let tensors = [
        (
            format!("{PREFIX}.weights_proj.weight"),
            Dtype::F16,
            vec![2, 3],
            0x3c00u16.to_le_bytes().repeat(6),
        ),
        (
            format!("{PREFIX}.k_norm.weight"),
            Dtype::F16,
            vec![128],
            weight_bits.to_le_bytes().repeat(128),
        ),
        (
            format!("{PREFIX}.k_norm.bias"),
            Dtype::F16,
            vec![bias_dim],
            0x3801u16.to_le_bytes().repeat(bias_dim),
        ),
        (
            format!("{PREFIX}.index_kpool_compress_ape"),
            ape_dtype,
            vec![4, 128],
            ape_bytes,
        ),
    ];
    safetensors::serialize_to_file(
        tensors
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), TensorView::new(*d, s.clone(), b).unwrap())),
        None,
        &root.join("a.safetensors"),
    )
    .unwrap();
    let map: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(n, _, _, _)| (n.clone(), serde_json::json!("a.safetensors")))
        .collect();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::json!({"weight_map":map}).to_string(),
    )
    .unwrap();
    Fixture(root)
}
fn store(f: &Fixture, gpu: &MockGpuBackend, include_ape: bool) -> WeightStore {
    let selected = if include_ape {
        vec![(
            format!("{PREFIX}.index_kpool_compress_ape"),
            Exl3DenseDtype::F32,
        )]
    } else {
        vec![]
    };
    WeightStore::from_exl3(Exl3CheckpointRouter::open(&f.0).unwrap(), &selected, gpu, 0).unwrap()
}

#[test]
fn loads_f32_norms_from_original_f16_and_borrows_exact_f32_ape() {
    let f = fixture(128, Dtype::F32);
    let gpu = MockGpuBackend::new();
    let mut store = store(&f, &gpu, true);
    let ape = store
        .get(&format!("{PREFIX}.index_kpool_compress_ape"))
        .unwrap()
        .ptr;
    let before = gpu.alloc_count();
    let mut params = Glm5Exl3IndexParams::load(&store, PREFIX, 128, 4, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), before + 2);
    let exact = params.f32_params().unwrap();
    assert_eq!(exact.ape, ape);
    let weight = gpu.read_alloc(exact.norm_weight).unwrap();
    let bias = gpu.read_alloc(exact.norm_bias).unwrap();
    assert_eq!(
        f32::from_le_bytes(weight[..4].try_into().unwrap()),
        1.000_976_6
    );
    assert_eq!(
        f32::from_le_bytes(bias[..4].try_into().unwrap()),
        0.500_488_3
    );
    assert_eq!(
        &gpu.read_alloc(ape).unwrap()[..4],
        &0x3f80_0001u32.to_le_bytes()
    );
    params.release(&gpu).unwrap();
    params.release(&gpu).unwrap();
    assert!(gpu.read_alloc(ape).is_some());
    assert!(params.f32_params().is_err());
    store.release(&gpu).unwrap();
    assert_eq!(gpu.sweep_unreleased(), 0);
}

#[test]
fn rejects_wrong_source_shape_dtype_or_missing_store_owner_before_upload() {
    for (dim, dtype, include) in [
        (127, Dtype::F32, true),
        (128, Dtype::F16, true),
        (128, Dtype::F32, false),
    ] {
        let f = fixture(dim, dtype);
        let gpu = MockGpuBackend::new();
        let mut store = store(&f, &gpu, include);
        let before = gpu.alloc_count();
        assert!(Glm5Exl3IndexParams::load(&store, PREFIX, 128, 4, &gpu).is_err());
        assert_eq!(gpu.alloc_count(), before);
        store.release(&gpu).unwrap();
        assert_eq!(gpu.sweep_unreleased(), 0);
    }
}

#[test]
fn release_failure_retains_owned_norm_for_retry_and_never_frees_borrowed_ape() {
    let f = fixture(128, Dtype::F32);
    let gpu = MockGpuBackend::new();
    let mut store = store(&f, &gpu, true);
    let mut params = Glm5Exl3IndexParams::load(&store, PREFIX, 128, 4, &gpu).unwrap();
    let ape = params.f32_params().unwrap().ape;
    gpu.fail_next_free();
    assert!(params.release(&gpu).is_err());
    assert_eq!(gpu.alloc_count(), 2);
    assert!(gpu.read_alloc(ape).is_some());
    params.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 1);
    store.release(&gpu).unwrap();
    assert_eq!(gpu.sweep_unreleased(), 0);
}

#[test]
fn rejects_nonfinite_companions_and_zero_pool_before_upload() {
    for (weight, ape) in [(0x7e00, 0x3f80_0001), (0x3c01, 0x7f80_0000)] {
        let f = fixture_values(128, Dtype::F32, weight, ape);
        let gpu = MockGpuBackend::new();
        let mut store = store(&f, &gpu, true);
        let before = gpu.alloc_count();
        assert!(Glm5Exl3IndexParams::load(&store, PREFIX, 128, 4, &gpu).is_err());
        assert_eq!(gpu.alloc_count(), before);
        store.release(&gpu).unwrap();
        assert_eq!(gpu.sweep_unreleased(), 0);
    }
    let f = fixture(128, Dtype::F32);
    let gpu = MockGpuBackend::new();
    let mut store = store(&f, &gpu, true);
    assert!(Glm5Exl3IndexParams::load(&store, PREFIX, 128, 0, &gpu).is_err());
    assert!(Glm5Exl3IndexParams::load(&store, PREFIX, 0, 4, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), 1);
    store.release(&gpu).unwrap();
    assert_eq!(gpu.sweep_unreleased(), 0);
}

#[test]
fn index_matrix_boundary_requires_checkpoint_and_live_bf16_owner_shape() {
    let f = fixture(128, Dtype::F32);
    let name = format!("{PREFIX}.weights_proj.weight");
    for dtype in [Exl3DenseDtype::Bf16, Exl3DenseDtype::F32] {
        let gpu = MockGpuBackend::new();
        let mut store = WeightStore::from_exl3(
            Exl3CheckpointRouter::open(&f.0).unwrap(),
            &[(name.clone(), dtype)],
            &gpu,
            0,
        )
        .unwrap();
        assert_eq!(
            validate_bf16_index_matrix(&store, &name, &[2, 3]).is_ok(),
            dtype == Exl3DenseDtype::Bf16
        );
        assert!(validate_bf16_index_matrix(&store, &name, &[3, 2]).is_err());
        assert_eq!(gpu.alloc_count(), 1);
        store.release(&gpu).unwrap();
        assert!(validate_bf16_index_matrix(&store, &name, &[2, 3]).is_err());
        assert_eq!(gpu.sweep_unreleased(), 0);
    }
}
