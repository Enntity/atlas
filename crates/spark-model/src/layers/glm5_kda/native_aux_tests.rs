// SPDX-License-Identifier: AGPL-3.0-only
use super::native_aux::*;
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::{Bf16GemmNumerics, DevicePtr, GpuBackend, mock::MockGpuBackend};

#[test]
fn merged_kda_aux_pads_single_row_and_scatters_only_physical_rows() {
    let gpu = MockGpuBackend::new();
    let weight = gpu.alloc(288 * 4096 * 2).unwrap();
    let mut aux = NativeKdaAux::new(&gpu, weight, 4096, 32, 128, 128, 77).unwrap();
    let input = gpu.alloc(4096 * 5 * 2).unwrap();
    gpu.copy_h2d(&vec![0x3f; 4096 * 5 * 2], input).unwrap();
    let outputs = [
        gpu.alloc(5 * 32 * 2).unwrap(),
        gpu.alloc(5 * 128 * 2).unwrap(),
        gpu.alloc(5 * 128 * 2).unwrap(),
    ];
    let pattern = (0..288 * 5)
        .flat_map(|v| (v as u16).to_le_bytes())
        .collect::<Vec<_>>();
    gpu.copy_h2d(&pattern, aux.output()).unwrap();
    for rows in [5, 1] {
        aux.forward(&gpu, input, outputs, rows, 77).unwrap();
    }
    let requests = gpu.bf16_gemm_requests_snapshot();
    assert_eq!(
        requests.iter().map(|r| r.rows).collect::<Vec<_>>(),
        vec![5, 3]
    );
    // Three or more rows read the caller's input directly; fewer are staged.
    assert_eq!(requests[0].input, input);
    assert_ne!(requests[1].input, input);
    assert!(
        requests
            .iter()
            .all(|r| r.numerics == Bf16GemmNumerics::Glm53AuxiliaryClassic
                && r.columns == 288
                && r.inner == 4096)
    );
    let mut padded = vec![0; 3 * 4096 * 2];
    gpu.copy_d2h(requests[1].input, &mut padded).unwrap();
    assert!(padded[..4096 * 2].iter().all(|v| *v == 0x3f));
    assert!(padded[4096 * 2..].iter().all(|v| *v == 0));
    for (i, (offset, width)) in [(0, 32), (32, 128), (160, 128)].into_iter().enumerate() {
        let mut first = vec![0; width * 2];
        gpu.copy_d2h(outputs[i], &mut first).unwrap();
        assert_eq!(first, pattern[offset * 2..(offset + width) * 2]);
    }
    let before = gpu.bf16_gemm_requests_snapshot().len();
    assert!(aux.forward(&gpu, input, outputs, 129, 77).is_err());
    assert!(aux.forward(&gpu, input, outputs, 1, 78).is_err());
    assert_eq!(gpu.bf16_gemm_requests_snapshot().len(), before);
    gpu.fail_next_free();
    assert!(aux.release(&gpu).is_err());
    aux.release(&gpu).unwrap();
    gpu.free(weight).unwrap();
    gpu.free(input).unwrap();
    for output in outputs {
        gpu.free(output).unwrap();
    }
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn merged_kda_aux_rejects_bad_geometry_before_allocation() {
    let gpu = MockGpuBackend::new();
    assert!(NativeKdaAux::new(&gpu, DevicePtr(1), 4096, 32, 128, 0, 77).is_err());
    assert!(NativeKdaAux::new(&gpu, DevicePtr::NULL, 4096, 32, 128, 128, 77).is_err());
    assert!(NativeKdaAux::new(&gpu, DevicePtr(1), 4096, 0, 128, 128, 77).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}
