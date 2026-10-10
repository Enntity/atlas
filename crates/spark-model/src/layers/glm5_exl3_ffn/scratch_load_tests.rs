// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn two_native_layers_borrow_one_scratch_and_release_only_their_weights() {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    let cfg = config(true);
    let mut owner =
        MoeScratch::allocate(&gpu, MoeScratchGeometry::new(&cfg, 3, 77).unwrap()).unwrap();
    let baseline = gpu.alloc_count();
    let mut first = load::load_glm5_exl3_ffn_with_scratch(
        &router,
        &cfg,
        0,
        &gpu,
        3,
        77,
        fixture_numerics,
        Some(owner.borrow()),
    )
    .unwrap();
    let one = gpu.alloc_count() - baseline;
    let mut second = load::load_glm5_exl3_ffn_with_scratch(
        &router,
        &cfg,
        0,
        &gpu,
        3,
        77,
        fixture_numerics,
        Some(owner.borrow()),
    )
    .unwrap();
    assert_eq!(gpu.alloc_count() - baseline, one * 2);
    let pointers = owner.borrow().pointers().unwrap();
    for ffn in [&first, &second] {
        let crate::layers::FfnComponent::Exl3(native) = ffn else {
            panic!()
        };
        let Glm5Exl3FfnLayer::Moe(m) = native.as_ref() else {
            panic!()
        };
        assert_eq!(m.buffers, pointers);
        assert!(m.owned_scratch.is_none());
    }
    let input = gpu.alloc(128 * 2).unwrap();
    let output = gpu.alloc(128 * 2).unwrap();
    for ffn in [&first, &second] {
        let crate::layers::FfnComponent::Exl3(native) = ffn else {
            panic!()
        };
        let Glm5Exl3FfnLayer::Moe(m) = native.as_ref() else {
            panic!()
        };
        m.forward_local(&gpu, input, output, 1).unwrap();
    }
    first.release_exl3(&gpu).unwrap();
    second.release_exl3(&gpu).unwrap();
    for p in pointers {
        assert!(gpu.read_alloc(p).is_some());
    }
    owner.release(&gpu).unwrap();
    gpu.free(input).unwrap();
    gpu.free(output).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn shared_required_exports_fail_before_any_global_gpu_upload() {
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    gpu.fail_next_kernel();
    assert!(
        load::load_glm5_exl3_ffn_with_numerics(
            &router,
            &config(true),
            0,
            &gpu,
            3,
            77,
            fixture_numerics
        )
        .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
}
