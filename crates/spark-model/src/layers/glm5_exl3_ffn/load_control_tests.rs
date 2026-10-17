// SPDX-License-Identifier: AGPL-3.0-only
//! Gate-perturbation controls applied at load (drop-shared, routed, top-k).
use super::*;

fn load_perturbed(
    gpu: &MockGpuBackend,
    router: &Exl3CheckpointRouter,
    perturbation: crate::exl3_gate_perturbation::Exl3GatePerturbation,
) -> Result<crate::layers::FfnComponent> {
    load::load_glm5_exl3_ffn_with_scratch(
        router,
        &config(true),
        0,
        gpu,
        3,
        77,
        fixture_numerics,
        None,
        Some(perturbation),
    )
}

#[test]
fn drop_shared_control_skips_shared_expert_only_on_its_layer() {
    use crate::exl3_gate_perturbation::Exl3GatePerturbation as P;
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    // (perturbation on layer 0, other layer) -> launches/GEMMs per forward.
    for (perturbation, launches, gemms, dropped) in [
        (P::DropSharedLayer { layer: 0 }, 13, 1, true),
        (P::DropSharedLayer { layer: 1 }, 14, 3, false),
    ] {
        let gpu = MockGpuBackend::new();
        let mut ffn = load_perturbed(&gpu, &router, perturbation).unwrap();
        let crate::layers::FfnComponent::Exl3(native) = &ffn else {
            panic!()
        };
        let Glm5Exl3FfnLayer::Moe(m) = &**native else {
            panic!()
        };
        assert_eq!(m.drop_shared, dropped);
        assert_eq!(m.combine_flags(), [u32::from(!dropped), 0]);
        assert_eq!((m.topk, m.scale, m.round_routed_accum), (1, 2.5, false));
        let input = gpu.alloc(128 * 2).unwrap();
        let output = gpu.alloc(128 * 2).unwrap();
        let before = gpu.launch_count();
        m.forward_local(&gpu, input, output, 1).unwrap();
        assert_eq!(gpu.launch_count() - before, launches);
        assert_eq!(gpu.bf16_gemm_requests_snapshot().len(), gemms);
        ffn.release_exl3(&gpu).unwrap();
        gpu.free(input).unwrap();
        gpu.free(output).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn global_routed_controls_reach_the_native_moe() {
    use crate::exl3_gate_perturbation::Exl3GatePerturbation as P;
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    for (perturbation, scale, round) in [
        (P::RouterScale24, 2.4, false),
        (P::Bf16RoutedAccum, 2.5, true),
        (P::MhcPost19, 2.5, false),
    ] {
        let gpu = MockGpuBackend::new();
        let mut ffn = load_perturbed(&gpu, &router, perturbation).unwrap();
        let crate::layers::FfnComponent::Exl3(native) = &ffn else {
            panic!()
        };
        let Glm5Exl3FfnLayer::Moe(m) = &**native else {
            panic!()
        };
        assert_eq!(
            (m.scale, m.round_routed_accum, m.drop_shared),
            (scale, round, false)
        );
        assert_eq!(m.combine_flags(), [1, u32::from(round)]);
        ffn.release_exl3(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn topk_control_cannot_exceed_configured_scratch_and_rolls_back() {
    use crate::exl3_gate_perturbation::Exl3GatePerturbation as P;
    let fixture = fixture(true);
    let router = Exl3CheckpointRouter::open(&fixture.0).unwrap();
    let gpu = MockGpuBackend::new();
    // The fixture routes top-1; top-7 would overrun top-1 scratch.
    let error = load_perturbed(&gpu, &router, P::Topk7Layer { layer: 0 })
        .err()
        .expect("top-k increase must be rejected");
    assert!(format!("{error:#}").contains("only reduce routed top-k"));
    assert_eq!(gpu.alloc_count(), 0);
    // Another layer is unaffected.
    let mut ffn = load_perturbed(&gpu, &router, P::Topk7Layer { layer: 1 }).unwrap();
    ffn.release_exl3(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}
