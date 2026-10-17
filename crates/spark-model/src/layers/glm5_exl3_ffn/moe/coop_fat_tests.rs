// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn oversized_gate_up_scratch_is_rejected_before_gpu_setup() {
    // Includes equal widths, a just-too-wide intermediate, and a width whose
    // doubled element count overflows usize. All widths are tile-aligned.
    for (hidden, intermediate) in [(4096, 4096), (4096, 2304), (256, usize::MAX - 255)] {
        let gpu = MockGpuBackend::new();
        let resolved = CoopFat::resolve(
            &gpu,
            &[],
            hidden,
            intermediate,
            33,
            8,
            RoutedPrecisionPolicy::GLM53_TP2_REFERENCE,
        );
        assert!(resolved.is_none());
        assert!(gpu.kernel_requests_snapshot().is_empty());
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn supported_gate_up_scratch_geometry_reaches_kernel_resolution() {
    // The actual TP2 geometry and the exact capacity boundary both fit the
    // existing [slots, max(hidden, intermediate)] F32 matmul scratch.
    for intermediate in [1024, 2048] {
        let gpu = MockGpuBackend::new();
        let _ = CoopFat::resolve(
            &gpu,
            &[],
            4096,
            intermediate,
            33,
            8,
            RoutedPrecisionPolicy::GLM53_TP2_REFERENCE,
        );
        assert!(
            gpu.kernel_requests_snapshot()
                .iter()
                .any(|(_, name)| { name == "exl3_coop_moe_fat_multirow_b4_k16_n128" })
        );
    }
}
