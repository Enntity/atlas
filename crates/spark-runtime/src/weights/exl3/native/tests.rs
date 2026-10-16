// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::scope::ModelResource;
use safetensors::{Dtype, tensor::TensorView};

use super::Exl3LinearWorkspace;
use crate::gpu::{DevicePtr, mock::MockGpuBackend};
use crate::weights::exl3::{
    Exl3LinearView,
    tp::{Exl3DeviceShard, ShardAxis, TpShardPlan},
};

fn weight(gpu: &MockGpuBackend, k: usize, n: usize) -> Exl3DeviceShard {
    let trellis = vec![0; k * n / 2];
    let input = 0x3c00u16.to_le_bytes().repeat(k);
    let output = 0x3c00u16.to_le_bytes().repeat(n);
    let marker = 0x83dc_d12du32.to_le_bytes();
    let view = Exl3LinearView::new(
        TensorView::new(Dtype::I16, vec![k / 16, n / 16, 64], &trellis).unwrap(),
        TensorView::new(Dtype::F16, vec![k], &input).unwrap(),
        TensorView::new(Dtype::F16, vec![n], &output).unwrap(),
        TensorView::new(Dtype::I32, vec![], &marker).unwrap(),
        k,
        n,
    )
    .unwrap();
    let plan = TpShardPlan::new(&view, ShardAxis::Replicated, 0, 1).unwrap();
    Exl3DeviceShard::upload(gpu, &view, &plan).unwrap()
}

#[test]
fn rejects_invalid_capacity_and_shapes_before_allocating() {
    let gpu = MockGpuBackend::new();
    for (k, n, rows) in [
        (0, 128, 1),
        (128, 0, 1),
        (127, 128, 1),
        (128, 127, 1),
        (128, 128, 0),
        (128, 128, 65536),
        (usize::MAX, 128, 1),
    ] {
        assert!(Exl3LinearWorkspace::new(&gpu, k, n, rows, 0).is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn invalid_launches_do_not_submit_partial_work() {
    let gpu = MockGpuBackend::new();
    let shard = weight(&gpu, 128, 256);
    let mismatched = weight(&gpu, 256, 128);
    let workspace = Exl3LinearWorkspace::new(&gpu, 128, 256, 3, 0).unwrap();
    for (weight, input, output, rows) in [
        (&shard, DevicePtr(123), DevicePtr(456), 4),
        (&shard, DevicePtr::NULL, DevicePtr(456), 1),
        (&shard, DevicePtr(123), DevicePtr::NULL, 1),
        (&mismatched, DevicePtr(123), DevicePtr(456), 1),
    ] {
        assert!(
            workspace
                .launch_bf16(&gpu, weight, input, output, rows)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[test]
fn batched_launch_has_three_stages_and_empty_batch_submits_nothing() {
    let gpu = MockGpuBackend::new();
    let mut shard = weight(&gpu, 128, 256);
    let mut workspace = Exl3LinearWorkspace::new(&gpu, 128, 256, 3, 0).unwrap();
    assert_eq!(gpu.alloc_count(), 5); // weights/scales + two F32 activation buffers
    workspace
        .launch_bf16(&gpu, &shard, DevicePtr(123), DevicePtr(456), 0)
        .unwrap();
    assert_eq!(gpu.launch_count(), 0);
    workspace
        .launch_bf16(&gpu, &shard, DevicePtr(123), DevicePtr(456), 3)
        .unwrap();
    assert_eq!(gpu.launch_count(), 3);
    let launches = gpu.launches_snapshot();
    assert_eq!(launches[0].grid, [1, 3, 1]);
    assert_eq!(launches[1].grid, [2, 3, 1]);
    assert_eq!(launches[2].grid, [2, 3, 1]);
    assert!(launches.iter().all(|launch| launch.block == [128, 1, 1]));
    gpu.fail_next_free();
    assert!(workspace.release(&gpu).is_err());
    assert!(
        workspace
            .launch_bf16(&gpu, &shard, DevicePtr(123), DevicePtr(456), 1)
            .is_err()
    );
    workspace.release(&gpu).unwrap();
    workspace.release(&gpu).unwrap();
    shard.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn cooperative_invalid_geometry_bits_and_flat_overflow_fail_before_gpu_work() {
    use super::{Exl3CooperativeProfile, Exl3LinearNumerics};
    let geometry = Exl3CooperativeProfile {
        tile_k: 32,
        tile_n: 128,
        slices: 48,
        half_partial: false,
    };
    let valid = Exl3LinearNumerics {
        logical_n: 384,
        logical_offset: 128,
        bits: 4,
        reference_min_rows: 3,
        reference_pad: true,
        row_profiles: [Some(geometry); 4],
        verify_rows_5_8: None,
    };
    for invalid in [
        Exl3LinearNumerics { bits: 0, ..valid },
        Exl3LinearNumerics {
            reference_min_rows: 0,
            ..valid
        },
        Exl3LinearNumerics {
            logical_offset: 1,
            ..valid
        },
        Exl3LinearNumerics {
            logical_offset: usize::MAX,
            ..valid
        },
        Exl3LinearNumerics {
            row_profiles: [Some(Exl3CooperativeProfile {
                tile_k: 64,
                ..geometry
            }); 4],
            ..valid
        },
        Exl3LinearNumerics {
            row_profiles: [Some(Exl3CooperativeProfile {
                tile_n: 256,
                ..geometry
            }); 4],
            ..valid
        },
        Exl3LinearNumerics {
            row_profiles: [Some(Exl3CooperativeProfile {
                slices: 49,
                ..geometry
            }); 4],
            ..valid
        },
        Exl3LinearNumerics {
            row_profiles: [Some(Exl3CooperativeProfile {
                half_partial: true,
                ..geometry
            }); 4],
            ..valid
        },
    ] {
        let gpu = MockGpuBackend::new();
        assert!(Exl3LinearWorkspace::new_with_numerics(&gpu, 128, 128, 3, 77, invalid).is_err());
        assert!(gpu.kernel_requests_snapshot().is_empty());
        assert_eq!(gpu.alloc_count(), 0);
    }
    let gpu = MockGpuBackend::new();
    assert!(
        Exl3LinearWorkspace::new_with_numerics(
            &gpu,
            0xffff_ff80,
            128,
            3,
            77,
            Exl3LinearNumerics {
                logical_n: 0xffff_ff80,
                ..valid
            }
        )
        .is_err()
    );
    assert!(gpu.kernel_requests_snapshot().is_empty());
}

#[test]
fn explicit_input_precision_resolves_required_transform_before_allocating() {
    use super::Exl3InputPrecision;
    for (precision, name) in [
        (Exl3InputPrecision::F32, "exl3_input_hadamard"),
        (Exl3InputPrecision::F16, "exl3_input_hadamard_f16"),
    ] {
        let gpu = MockGpuBackend::new();
        let mut workspace =
            Exl3LinearWorkspace::new_with_input_precision(&gpu, 128, 256, 3, 77, precision)
                .unwrap();
        assert_eq!(
            gpu.kernel_requests_snapshot()[0],
            ("exl3".into(), name.into())
        );
        assert_eq!(gpu.alloc_count(), 2);
        workspace.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);

        let gpu = MockGpuBackend::new();
        gpu.fail_next_kernel();
        assert!(
            Exl3LinearWorkspace::new_with_input_precision(&gpu, 128, 256, 3, 77, precision,)
                .is_err()
        );
        assert_eq!(
            gpu.kernel_requests_snapshot(),
            [("exl3".into(), name.into())]
        );
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[test]
fn cooperative_profile_selects_explicit_padded_buckets_and_resolves_before_scratch() {
    use super::{Exl3CooperativeProfile, Exl3LinearNumerics};
    let profiles = [2, 4, 8, 16].map(|slices| {
        Some(Exl3CooperativeProfile {
            tile_k: 32,
            tile_n: 128,
            slices,
            half_partial: false,
        })
    });
    let numerical = Exl3LinearNumerics {
        logical_n: 384,
        logical_offset: 128,
        bits: 4,
        reference_min_rows: 3,
        reference_pad: true,
        row_profiles: profiles,
        verify_rows_5_8: None,
    };
    assert_eq!(numerical.profile_for_rows(1).unwrap().slices, 4);
    assert_eq!(numerical.profile_for_rows(5).unwrap().slices, 8);
    assert_eq!(numerical.profile_for_rows(128).unwrap().slices, 16);
    assert_eq!(numerical.kernel_args(1).unwrap(), [384, 128, 32, 128, 4, 0]);
    assert_eq!(
        Exl3LinearNumerics {
            reference_pad: false,
            ..numerical
        }
        .profile_for_rows(1)
        .unwrap()
        .slices,
        2
    );
    let gpu = MockGpuBackend::new();
    let mut workspace =
        Exl3LinearWorkspace::new_with_numerics(&gpu, 128, 128, 3, 77, numerical).unwrap();
    assert_eq!(
        gpu.kernel_requests_snapshot()[1],
        ("exl3_linear_mma".into(), "exl3_mul1_matmul_mma".into())
    );
    workspace.release(&gpu).unwrap();
    let gpu = MockGpuBackend::new();
    gpu.fail_next_kernel();
    assert!(Exl3LinearWorkspace::new_with_numerics(&gpu, 128, 128, 3, 77, numerical).is_err());
    assert_eq!(gpu.alloc_count(), 0);
    let gpu = MockGpuBackend::new();
    assert!(
        Exl3LinearWorkspace::new_with_numerics(
            &gpu,
            128,
            128,
            3,
            77,
            Exl3LinearNumerics {
                logical_offset: 384,
                ..numerical
            }
        )
        .is_err()
    );
    assert!(gpu.kernel_requests_snapshot().is_empty());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn cooperative_forward_checks_checkpoint_contract_and_preserves_empty_batches() {
    use super::{Exl3CooperativeProfile, Exl3LinearNumerics};
    let policy = Exl3LinearNumerics {
        logical_n: 384,
        logical_offset: 128,
        bits: 4,
        reference_min_rows: 3,
        reference_pad: true,
        row_profiles: [Some(Exl3CooperativeProfile {
            tile_k: 32,
            tile_n: 128,
            slices: 48,
            half_partial: false,
        }); 4],
        verify_rows_5_8: None,
    };
    let gpu = MockGpuBackend::new();
    let mut shard = weight(&gpu, 128, 128);
    let mut workspace =
        Exl3LinearWorkspace::new_with_numerics(&gpu, 128, 128, 3, 77, policy).unwrap();
    let mut wrong = Exl3LinearWorkspace::new_with_numerics(
        &gpu,
        128,
        128,
        3,
        77,
        Exl3LinearNumerics { bits: 6, ..policy },
    )
    .unwrap();
    assert!(
        wrong
            .launch_bf16(&gpu, &shard, DevicePtr(123), DevicePtr(456), 1)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    workspace
        .launch_bf16(&gpu, &shard, DevicePtr::NULL, DevicePtr::NULL, 0)
        .unwrap();
    assert_eq!(gpu.launch_count(), 0);
    workspace
        .launch_bf16(&gpu, &shard, DevicePtr(123), DevicePtr(456), 3)
        .unwrap();
    assert_eq!(gpu.launch_count(), 3);
    assert!(
        gpu.launches_snapshot()
            .iter()
            .all(|launch| launch.grid == [1, 3, 1] && launch.block == [128, 1, 1])
    );
    gpu.fail_next_free();
    assert!(workspace.release(&gpu).is_err());
    workspace.release(&gpu).unwrap();
    wrong.release(&gpu).unwrap();
    shard.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn absent_unreachable_bucket_is_explicit_but_capacity_requires_every_reachable_bucket() {
    use super::{Exl3CooperativeProfile, Exl3LinearNumerics};
    let geometry = Exl3CooperativeProfile {
        tile_k: 32,
        tile_n: 128,
        slices: 48,
        half_partial: false,
    };
    let policy = Exl3LinearNumerics {
        logical_n: 128,
        logical_offset: 0,
        bits: 4,
        reference_min_rows: 3,
        reference_pad: true,
        row_profiles: [None, Some(geometry), Some(geometry), Some(geometry)],
        verify_rows_5_8: None,
    };
    let gpu = MockGpuBackend::new();
    let mut workspace =
        Exl3LinearWorkspace::new_with_numerics(&gpu, 128, 128, 128, 77, policy).unwrap();
    workspace.release(&gpu).unwrap();
    for invalid in [
        Exl3LinearNumerics {
            reference_pad: false,
            ..policy
        },
        Exl3LinearNumerics {
            row_profiles: [None, Some(geometry), None, Some(geometry)],
            ..policy
        },
    ] {
        let gpu = MockGpuBackend::new();
        assert!(Exl3LinearWorkspace::new_with_numerics(&gpu, 128, 128, 128, 77, invalid).is_err());
        assert!(gpu.kernel_requests_snapshot().is_empty());
        assert_eq!(gpu.alloc_count(), 0);
    }
    assert!(
        Exl3LinearNumerics {
            reference_pad: false,
            ..policy
        }
        .profile_for_rows(1)
        .is_err()
    );
}

#[test]
fn verify_profile_applies_only_inside_a_verify_scope_and_only_to_5_8_rows() {
    use super::{Exl3CooperativeProfile, Exl3LinearNumerics, verify_scope};
    let p = |tile_k, tile_n| Exl3CooperativeProfile {
        tile_k,
        tile_n,
        slices: 48,
        half_partial: false,
    };
    let numerics = Exl3LinearNumerics {
        logical_n: 4096,
        logical_offset: 0,
        bits: 6,
        reference_min_rows: 3,
        reference_pad: true,
        row_profiles: [None, Some(p(16, 512)), Some(p(16, 512)), Some(p(32, 256))],
        verify_rows_5_8: Some(p(32, 256)),
    };
    assert_eq!(numerics.profile_for_rows(6).unwrap(), p(16, 512));
    {
        let _scope = verify_scope();
        assert_eq!(numerics.profile_for_rows(6).unwrap(), p(32, 256));
        assert_eq!(numerics.profile_slot(8).unwrap(), 4);
        assert_eq!(numerics.profile_for_rows(1).unwrap(), p(16, 512));
        assert_eq!(numerics.profile_slot(4).unwrap(), 1);
        assert_eq!(numerics.profile_slot(9).unwrap(), 3);
    }
    assert_eq!(numerics.profile_slot(6).unwrap(), 2);
}
