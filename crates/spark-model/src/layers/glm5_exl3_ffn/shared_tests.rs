// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn global_bf16_matrix_shards_follow_row_major_projection_axes() {
    let gpu = MockGpuBackend::new();
    let source = gpu.alloc(4 * 8 * 2).unwrap();
    let values: Vec<u8> = (0..32u16).flat_map(u16::to_le_bytes).collect();
    gpu.copy_h2d(&values, source).unwrap();
    for rank in [0, 1] {
        for axis in [ShardAxis::Output, ShardAxis::Input] {
            let local =
                shared::copy_global_bf16_shard(&gpu, source, 8, 4, axis, rank, 2, 77).unwrap();
            let expected: Vec<u8> = if axis == ShardAxis::Output {
                values[rank * 2 * 8 * 2..(rank + 1) * 2 * 8 * 2].to_vec()
            } else {
                (0..4)
                    .flat_map(|row| {
                        values[(row * 8 + rank * 4) * 2..(row * 8 + rank * 4 + 4) * 2]
                            .iter()
                            .copied()
                    })
                    .collect()
            };
            let mut actual = vec![0; expected.len()];
            gpu.copy_d2h(local, &mut actual).unwrap();
            assert_eq!(actual, expected);
            gpu.free(local).unwrap();
        }
    }
    assert_eq!(gpu.d2d_2d_count(), 2);
    gpu.free(source).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn reconstructed_shared_forward_uses_merged_gate_up_bf16_gemm_and_retries_release() {
    let gpu = MockGpuBackend::new();
    let pointers = [
        gpu.alloc(512 * 128 * 2).unwrap(),
        gpu.alloc(128 * 256 * 2).unwrap(),
    ];
    let weights = shared::SharedBf16Weights {
        pointers,
        hidden: 128,
        intermediate: 256,
        rank: 1,
        world: 2,
    };
    let mut ffn =
        NativeDenseFfn::from_shared(&gpu, weights, 3, 77, NativeDenseActivation::SiLuBf16).unwrap();
    let input = gpu.alloc(2 * 128 * 2).unwrap();
    let output = gpu.alloc(2 * 128 * 2).unwrap();
    ffn.forward_local(&gpu, input, output, 2).unwrap();
    let requests = gpu.bf16_gemm_requests_snapshot();
    assert!(
        requests
            .iter()
            .all(|r| { r.numerics == spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic })
    );
    assert_eq!(
        requests
            .iter()
            .map(|r| (r.rows, r.columns, r.inner, r.stream))
            .collect::<Vec<_>>(),
        [(2, 512, 128, 77), (2, 128, 256, 77)]
    );
    assert_eq!(
        requests.iter().map(|r| r.weight).collect::<Vec<_>>(),
        pointers
    );
    assert_eq!(gpu.launch_count(), 1);
    gpu.fail_next_free();
    assert!(ffn.release(&gpu).is_err());
    assert_eq!(gpu.alloc_count(), 3); // caller input/output and the failed owned free
    assert!(ffn.forward_local(&gpu, input, output, 2).is_err());
    ffn.release(&gpu).unwrap();
    gpu.free(input).unwrap();
    gpu.free(output).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn merged_local_weights_and_multirow_activation_split_preserve_bf16_bytes() {
    let gpu = MockGpuBackend::new();
    let values: Vec<u8> = (0..24u16).flat_map(u16::to_le_bytes).collect();
    let local = [
        gpu.alloc(12).unwrap(),
        gpu.alloc(12).unwrap(),
        gpu.alloc(12).unwrap(),
    ];
    gpu.copy_h2d(&values[..12], local[0]).unwrap();
    gpu.copy_h2d(&values[12..24], local[1]).unwrap();
    let merged = shared::merge_local_weights(&gpu, local, 3, 2, 77).unwrap();
    let mut actual = vec![0; 24];
    gpu.copy_d2h(merged[0], &mut actual).unwrap();
    assert_eq!(actual, values[..24]);
    assert_eq!(gpu.alloc_count(), 2);
    gpu.free(merged[0]).unwrap();
    gpu.free(merged[1]).unwrap();
    let source = gpu.alloc(48).unwrap();
    let gate = gpu.alloc(24).unwrap();
    let up = gpu.alloc(24).unwrap();
    gpu.copy_h2d(&values, source).unwrap();
    shared::split_merged_bf16(&gpu, source, gate, up, 3, 4, 77).unwrap();
    let mut actual_gate = vec![0; 24];
    let mut actual_up = vec![0; 24];
    gpu.copy_d2h(gate, &mut actual_gate).unwrap();
    gpu.copy_d2h(up, &mut actual_up).unwrap();
    assert_eq!(
        actual_gate,
        (0..3)
            .flat_map(|row| values[row * 16..row * 16 + 8].iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual_up,
        (0..3)
            .flat_map(|row| values[row * 16 + 8..row * 16 + 16].iter().copied())
            .collect::<Vec<_>>()
    );
    for p in [source, gate, up] {
        gpu.free(p).unwrap();
    }
    let local = [
        gpu.alloc(12).unwrap(),
        gpu.alloc(12).unwrap(),
        gpu.alloc(12).unwrap(),
    ];
    gpu.fail_next_free();
    assert!(shared::merge_local_weights(&gpu, local, 3, 2, 77).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}
