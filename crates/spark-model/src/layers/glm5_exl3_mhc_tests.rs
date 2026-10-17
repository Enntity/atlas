// SPDX-License-Identifier: AGPL-3.0-only
//! Native mHC numerical-policy, precision-boundary and launch-alias tests.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn mhc_post_perturbation_replaces_consumed_multiplier_only() {
    let reference = MhcNumericalPolicy::GLM53_TP2_REFERENCE;
    let perturbed = reference.with_perturbation(Some(Exl3GatePerturbation::MhcPost19));
    assert_eq!(perturbed.post_multiplier(None).unwrap(), 1.9);
    // A checkpoint declaring the reference 2.0 must not reject the control.
    assert_eq!(perturbed.post_multiplier(Some(2.0)).unwrap(), 1.9);
    assert!(perturbed.post_multiplier(Some(1.9)).is_err());
    for other in [None, Some(Exl3GatePerturbation::RouterScale24)] {
        let policy = reference.with_perturbation(other);
        assert_eq!(policy.post_multiplier(Some(2.0)).unwrap(), 2.0);
    }
}

#[test]
fn perturbed_mhc_load_stores_the_perturbed_multiplier() {
    let gpu = MockGpuBackend::new();
    let policy = MhcNumericalPolicy::GLM53_TP2_REFERENCE
        .with_perturbation(Some(Exl3GatePerturbation::MhcPost19));
    let mhc = NativeExl3Mhc::load(&gpu, 4096, 4, 20, 1e-6, 1e-6, policy, Some(2.0)).unwrap();
    assert_eq!(mhc.post_multiplier, 1.9);
}

#[test]
fn named_reference_policy_validates_declared_checkpoint_values() {
    let policy = MhcNumericalPolicy::GLM53_TP2_REFERENCE;
    assert_eq!(policy.post_multiplier(None).unwrap(), 2.0);
    assert_eq!(policy.post_multiplier(Some(2.0)).unwrap(), 2.0);
    for value in [0.0, 2.5, f32::NAN] {
        assert!(policy.post_multiplier(Some(value)).is_err());
    }
}

#[test]
fn rounded_projection_uses_explicit_balanced_reference_split_policy() {
    let policy = MhcNumericalPolicy::GLM53_TP2_REFERENCE;
    for (rows, splits) in [
        (1, 48),
        (10, 48),
        (17, 48),
        (64, 48),
        (65, 24),
        (128, 24),
        (129, 16),
        (4096, 1),
    ] {
        assert_eq!(policy.rounded_splits(4096, rows).unwrap(), splits);
    }
    assert_eq!(policy.rounded_splits(128, 1).unwrap(), 2);
    for (hidden, rows) in [(0, 1), (4096, 0), (127, 1), (usize::MAX, 1)] {
        assert!(policy.rounded_splits(hidden, rows).is_err());
    }
}

#[test]
fn closure_is_required_and_precision_boundary_is_explicit() {
    let gpu = MockGpuBackend::new();
    let mhc = NativeExl3Mhc::load(
        &gpu,
        4096,
        4,
        20,
        1e-5,
        1e-6,
        MhcNumericalPolicy::GLM53_TP2_REFERENCE,
        None,
    )
    .unwrap();
    assert_eq!(
        gpu.kernel_requests_snapshot(),
        vec![
            ("exl3_mhc".into(), "exl3_mhc_pre_norm".into()),
            ("exl3_mhc".into(), "exl3_mhc_post_unrounded".into()),
            ("exl3_mhc".into(), "exl3_mhc_contract".into())
        ]
    );
    assert_eq!(
        MhcResidualPrecision::for_fused_rows(16),
        MhcResidualPrecision::UnroundedPost
    );
    assert_eq!(
        MhcResidualPrecision::for_fused_rows(17),
        MhcResidualPrecision::Rounded
    );
    gpu.fail_next_kernel();
    assert!(
        NativeExl3Mhc::load(
            &gpu,
            4096,
            4,
            20,
            1e-5,
            1e-6,
            MhcNumericalPolicy::GLM53_TP2_REFERENCE,
            None
        )
        .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    mhc.contract(&gpu, DevicePtr::NULL, DevicePtr::NULL, 0, 7)
        .unwrap();
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn post_launch_permits_only_the_reference_highway_alias() {
    let gpu = MockGpuBackend::new();
    let mhc = NativeExl3Mhc::load(
        &gpu,
        4096,
        4,
        20,
        1e-5,
        1e-6,
        MhcNumericalPolicy::GLM53_TP2_REFERENCE,
        None,
    )
    .unwrap();
    let p: Vec<_> = (0..4).map(|_| gpu.alloc(16).unwrap()).collect();
    mhc.post(&gpu, p[0], p[1], p[2], p[3], p[1], 5, 9).unwrap();
    assert_eq!(gpu.launches_snapshot()[0].grid, [5, 1, 1]);
    assert!(mhc.post(&gpu, p[0], p[1], p[2], p[3], p[0], 5, 9).is_err());
    assert!(mhc.post(&gpu, p[0], p[0], p[2], p[3], p[1], 5, 9).is_err());
    assert_eq!(gpu.launch_count(), 1);
    for pointer in p {
        gpu.free(pointer).unwrap();
    }
}

#[test]
fn preflight_rejects_alias_and_invalid_policy_before_launch() {
    let gpu = MockGpuBackend::new();
    for (hidden, streams, iters, eps) in [
        (0, 4, 20, 1e-5),
        (4096, 3, 20, 1e-5),
        (4096, 4, 0, 1e-5),
        (4096, 4, 20, f32::NAN),
    ] {
        assert!(
            NativeExl3Mhc::load(
                &gpu,
                hidden,
                streams,
                iters,
                eps,
                1e-6,
                MhcNumericalPolicy::GLM53_TP2_REFERENCE,
                None
            )
            .is_err()
        );
    }
    assert!(gpu.kernel_requests_snapshot().is_empty());
    let mhc = NativeExl3Mhc::load(
        &gpu,
        4096,
        4,
        20,
        1e-5,
        1e-6,
        MhcNumericalPolicy::GLM53_TP2_REFERENCE,
        None,
    )
    .unwrap();
    let pointers: Vec<_> = (0..8).map(|_| gpu.alloc(16).unwrap()).collect();
    let site = HcSiteWeights {
        hc_fn: pointers[4],
        hc_scale: pointers[5],
        hc_base: pointers[6],
        hc_fn_bf16: DevicePtr::NULL,
        lowrank: None,
    };
    let mut buffers = NativeMhcBuffers {
        highway: pointers[0],
        normed: pointers[1],
        post: pointers[2],
        comb: pointers[3],
    };
    mhc.pre_norm(
        &gpu,
        &site,
        pointers[7],
        buffers,
        2,
        MhcResidualPrecision::Rounded,
        9,
    )
    .unwrap();
    assert_eq!(gpu.launches_snapshot()[0].grid, [2, 1, 1]);
    assert_eq!(gpu.launches_snapshot()[0].block, [256, 1, 1]);
    buffers.normed = buffers.highway;
    assert!(
        mhc.pre_norm(
            &gpu,
            &site,
            pointers[7],
            buffers,
            2,
            MhcResidualPrecision::Rounded,
            9
        )
        .is_err()
    );
    assert_eq!(gpu.launch_count(), 1);
    for pointer in pointers {
        gpu.free(pointer).unwrap();
    }
}
