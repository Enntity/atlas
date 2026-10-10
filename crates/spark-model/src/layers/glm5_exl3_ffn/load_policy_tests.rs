// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit routed accumulation policy rejection and ownership.
use super::*;

#[test]
fn unsupported_thin_mma_profile_rejects_before_kernel_lookup_and_releases_inputs() {
    reject_mma_profile(RoutedPrecisionPolicy {
        half_partial: false,
        ..RoutedPrecisionPolicy::GLM53_TP2_REFERENCE
    });
}

#[test]
fn unsupported_fat_mma_profile_rejects_before_kernel_lookup_and_releases_inputs() {
    reject_mma_profile(RoutedPrecisionPolicy {
        fat_tile_k: 32,
        ..RoutedPrecisionPolicy::GLM53_TP2_REFERENCE
    });
}

fn reject_mma_profile(precision: RoutedPrecisionPolicy) {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let cfg = config(true);
    let ffn =
        load::load_glm5_exl3_ffn_with_numerics(&router, &cfg, 0, &gpu, 3, 77, fixture_numerics)
            .unwrap();
    let crate::layers::FfnComponent::Exl3(native) = ffn else {
        panic!()
    };
    let Glm5Exl3FfnLayer::Moe(moe) = *native else {
        panic!()
    };
    let NativeMoeFfn {
        shared,
        tables,
        companions,
        buffers,
        ..
    } = *moe;
    for buffer in buffers {
        gpu.free(buffer).unwrap();
    }
    let requested = gpu.kernel_requests_snapshot();
    let launched = gpu.launch_count();
    assert!(
        NativeMoeFfn::new(
            &gpu,
            &cfg,
            shared,
            tables,
            companions,
            3,
            77,
            Exl3FfnActivation::ClampedSiLu { limit: 10.0 },
            precision,
            None,
        )
        .is_err()
    );
    assert_eq!(gpu.kernel_requests_snapshot(), requested);
    assert_eq!(gpu.launch_count(), launched);
    assert_eq!(gpu.alloc_count(), 0);
}
