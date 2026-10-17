// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;
use half::{bf16, f16};

#[test]
fn f16_conversion_is_explicit_and_only_selected_rows_are_uploaded() {
    let gpu = MockGpuBackend::new();
    let values = [0.1f32, -0.75, 3.2, 0.0001, 7.0, -1.3, 0.0, 1.25];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::F16, vec![4, 2], &bytes).unwrap();
    let mut tensor =
        Exl3DenseDeviceTensor::upload(&gpu, &view, Exl3DenseDtype::Bf16, &[1..2, 3..4]).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), Exl3DenseDtype::Bf16);
    let expected: Vec<u8> = [2, 3, 6, 7]
        .into_iter()
        .flat_map(|i| bf16::from_f32(f16::from_f32(values[i]).to_f32()).to_le_bytes())
        .collect();
    assert_eq!(gpu.read_alloc(tensor.ptr().unwrap()).unwrap(), expected);
    let mut f32_tensor =
        Exl3DenseDeviceTensor::upload(&gpu, &view, Exl3DenseDtype::F32, &single(0..4)).unwrap();
    let expected_f32: Vec<u8> = values
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_f32().to_le_bytes())
        .collect();
    assert_eq!(
        gpu.read_alloc(f32_tensor.ptr().unwrap()).unwrap(),
        expected_f32
    );
    f32_tensor.release(&gpu).unwrap();
    tensor.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn bf16_and_f32_keep_checkpoint_bytes_and_rank_channel_order() {
    for (dtype, output, width) in [
        (Dtype::BF16, Exl3DenseDtype::Bf16, 2),
        (Dtype::F32, Exl3DenseDtype::F32, 4),
    ] {
        let gpu = MockGpuBackend::new();
        let bytes: Vec<u8> = (0..6 * 3 * width).map(|i| i as u8).collect();
        let view = TensorView::new(dtype, vec![6, 1, 3], &bytes).unwrap();
        let mut tensor =
            Exl3DenseDeviceTensor::upload(&gpu, &view, output, &[1..2, 3..4, 5..6]).unwrap();
        assert_eq!(tensor.shape(), &[3, 1, 3]);
        let expected: Vec<u8> = [1, 3, 5]
            .into_iter()
            .flat_map(|i| bytes[i * 3 * width..(i + 1) * 3 * width].to_vec())
            .collect();
        assert_eq!(gpu.read_alloc(tensor.ptr().unwrap()).unwrap(), expected);
        gpu.fail_next_free();
        assert!(tensor.release(&gpu).is_err());
        assert!(tensor.ptr().is_ok());
        tensor.release(&gpu).unwrap();
        assert!(tensor.ptr().is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn invalid_dense_shapes_dtypes_and_ranges_fail_before_allocating() {
    let gpu = MockGpuBackend::new();
    let bytes = vec![0u8; 16];
    let view = TensorView::new(Dtype::F32, vec![4], &bytes).unwrap();
    for ranges in [
        vec![],
        single(0..0),
        single(3..5),
        vec![2..3, 1..2],
        vec![0..2, 1..3],
    ] {
        assert!(Exl3DenseDeviceTensor::upload(&gpu, &view, Exl3DenseDtype::F32, &ranges).is_err());
    }
    assert!(
        Exl3DenseDeviceTensor::upload(&gpu, &view, Exl3DenseDtype::Bf16, &single(0..4)).is_err()
    );
    let scalar = TensorView::new(Dtype::F32, vec![], &bytes[..4]).unwrap();
    assert!(
        Exl3DenseDeviceTensor::upload(&gpu, &scalar, Exl3DenseDtype::F32, &single(0..1)).is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn checkpoint_dense_view_requires_exact_shape_and_dtype() {
    struct Dir(std::path::PathBuf);
    impl Dir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let dir = Dir(std::env::temp_dir().join(format!("atlas-exl3-dense-{}", std::process::id())));
    std::fs::create_dir(&dir.0).unwrap();
    let bytes = [0u8; 8];
    let view = TensorView::new(Dtype::F32, vec![2], &bytes).unwrap();
    safetensors::serialize_to_file([("param", view)], None, &dir.path().join("a.safetensors"))
        .unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"param":"a.safetensors"}}"#,
    )
    .unwrap();
    let router = Exl3CheckpointRouter::open(dir.path()).unwrap();
    assert_eq!(
        router.dense_view("param", Dtype::F32, &[2]).unwrap().data(),
        &bytes
    );
    assert!(router.dense_view("param", Dtype::F16, &[2]).is_err());
    assert!(router.dense_view("param", Dtype::F32, &[1, 2]).is_err());
    assert!(router.dense_view("missing", Dtype::F32, &[2]).is_err());
    assert!(router.dense_view("param", Dtype::I32, &[2]).is_err());
}

fn single(range: std::ops::Range<usize>) -> Vec<std::ops::Range<usize>> {
    vec![range]
}

#[test]
fn large_f16_row_selection_crosses_conversion_chunks_without_losing_elements() {
    let gpu = MockGpuBackend::new();
    let columns = 40000;
    let values: Vec<half::f16> = (0..4 * columns)
        .map(|i| half::f16::from_f32((i % 257) as f32 / 17.0 - 7.0))
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = TensorView::new(Dtype::F16, vec![4, columns], &bytes).unwrap();
    let selection = 1..3;
    let mut tensor = Exl3DenseDeviceTensor::upload(
        &gpu,
        &view,
        Exl3DenseDtype::Bf16,
        std::slice::from_ref(&selection),
    )
    .unwrap();
    let actual = gpu.read_alloc(tensor.ptr().unwrap()).unwrap();
    assert_eq!(actual.len(), 2 * columns * 2);
    for (index, word) in actual.chunks_exact(2).enumerate() {
        assert_eq!(
            bf16::from_le_bytes([word[0], word[1]]),
            bf16::from_f32(values[columns + index].to_f32())
        );
    }
    tensor.release(&gpu).unwrap();
}
