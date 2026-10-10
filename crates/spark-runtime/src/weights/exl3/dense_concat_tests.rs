// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::gpu::mock::MockGpuBackend;
use safetensors::tensor::TensorView;

#[test]
fn concat_owned_bf16_rows_preserves_sources_and_retries_release() {
    let gpu = MockGpuBackend::new();
    let mut parts = Vec::new();
    for value in [1u16, 2, 3] {
        let data = value.to_le_bytes().repeat(4);
        let view = TensorView::new(Dtype::BF16, vec![1, 4], &data).unwrap();
        parts.push(
            Exl3DenseDeviceTensor::upload(&gpu, &view, Exl3DenseDtype::Bf16, &[0..1]).unwrap(),
        );
    }
    let mut merged =
        Exl3DenseDeviceTensor::concat_rows(&gpu, &parts.iter().collect::<Vec<_>>()).unwrap();
    assert_eq!(merged.shape(), &[3, 4]);
    let mut bytes = vec![0; 24];
    gpu.copy_d2h(merged.ptr().unwrap(), &mut bytes).unwrap();
    assert_eq!(
        bytes,
        [1u16, 2, 3]
            .into_iter()
            .flat_map(|v| v.to_le_bytes().repeat(4))
            .collect::<Vec<_>>()
    );
    for part in &mut parts {
        part.release(&gpu).unwrap();
    }
    gpu.fail_next_free();
    assert!(merged.release(&gpu).is_err());
    assert!(merged.ptr().is_ok());
    merged.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn concat_rejects_empty_and_released_parts_before_allocating() {
    let gpu = MockGpuBackend::new();
    assert!(Exl3DenseDeviceTensor::concat_rows(&gpu, &[]).is_err());
    let bytes = vec![0; 8];
    let view = TensorView::new(Dtype::BF16, vec![1, 4], &bytes).unwrap();
    let mut part =
        Exl3DenseDeviceTensor::upload(&gpu, &view, Exl3DenseDtype::Bf16, &[0..1]).unwrap();
    part.release(&gpu).unwrap();
    assert!(Exl3DenseDeviceTensor::concat_rows(&gpu, &[&part]).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}
