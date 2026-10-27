// SPDX-License-Identifier: AGPL-3.0-only
//! Exercise the production norm loader, including values BF16 cannot represent.
use super::f32_norm;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

fn store(gpu: &MockGpuBackend, bytes: &[u8], dtype: WeightDtype, shape: Vec<usize>) -> WeightStore {
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    WeightStore::from_map([("norm".into(), WeightTensor { ptr, shape, dtype })].into())
}

#[test]
fn fp32_low_bits_are_preserved_without_an_allocation() {
    let gpu = MockGpuBackend::new();
    let bits = [0x3f80_1234u32, 0xbeaa_abcdu32, 0x0000_0001u32];
    let bytes: Vec<u8> = bits.into_iter().flat_map(u32::to_le_bytes).collect();
    let store = store(&gpu, &bytes, WeightDtype::FP32, vec![3]);
    let allocations = gpu.alloc_count();
    let loaded = f32_norm(&store, "norm", &gpu, 3).unwrap();
    assert_eq!(loaded.weight, store.get("norm").unwrap().ptr);
    assert_eq!(gpu.alloc_count(), allocations);
    let mut actual = vec![0; bytes.len()];
    gpu.copy_d2h(loaded.weight, &mut actual).unwrap();
    assert_eq!(actual, bytes);
}

#[test]
fn bf16_is_widened_exactly() {
    let gpu = MockGpuBackend::new();
    let bits = [0x3f81u16, 0xbeabu16, 0x8000u16];
    let bytes: Vec<u8> = bits.into_iter().flat_map(u16::to_le_bytes).collect();
    let store = store(&gpu, &bytes, WeightDtype::BF16, vec![3]);
    let loaded = f32_norm(&store, "norm", &gpu, 3).unwrap();
    let mut actual = vec![0; 12];
    gpu.copy_d2h(loaded.weight, &mut actual).unwrap();
    let expected: Vec<u8> = bits
        .into_iter()
        .flat_map(|v| ((v as u32) << 16).to_le_bytes())
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn malformed_norms_fail_before_a_kernel_reads_them() {
    for (dtype, shape) in [
        (WeightDtype::UInt8, vec![3]),
        (WeightDtype::FP32, vec![2]),
        (WeightDtype::BF16, vec![1, 3]),
        (WeightDtype::FP32, vec![]),
    ] {
        let gpu = MockGpuBackend::new();
        let store = store(&gpu, &[0; 12], dtype, shape);
        assert!(f32_norm(&store, "norm", &gpu, 3).is_err());
    }
}
