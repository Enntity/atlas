// SPDX-License-Identifier: AGPL-3.0-only
//! Upstream-capture tests: gated-norm and hc-boundary bindings, disabled-trace
//! no-I/O, and the supported layer boundary.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
#[test]
fn gated_norm_binding_captures_resident_weight_and_effective_epsilon() {
    let gpu = MockGpuBackend::new();
    let root = std::env::temp_dir().join(format!("atlas-norm-binding-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let trace = UpstreamTrace(Exl3FfnRouter::new(root.clone(), 0, 1, 2).unwrap());
    let ptr = gpu.alloc(4).unwrap();
    let weight = [0, 0x3f, 0, 0x40];
    gpu.copy_h2d(&weight, ptr).unwrap();
    let eps = 0.00003125f32;
    // Disabled capture returns before validation, GPU access or filesystem writes.
    trace
        .capture_gated_norm_binding(
            None,
            &gpu,
            DevicePtr::NULL,
            KernelHandle(0),
            0,
            0,
            0,
            f32::NAN,
            8,
        )
        .unwrap();
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    assert_eq!(gpu.d2h_blocking_count() + gpu.d2h_async_count(), 0);
    std::fs::write(root.join("capture-enabled"), b"").unwrap();
    let invocation = trace.begin().unwrap();
    trace
        .capture_gated_norm_binding(invocation, &gpu, ptr, KernelHandle(123), 3, 4, 2, eps, 8)
        .unwrap();
    assert_eq!(
        std::fs::read(root.join("ffn-L0-rank1-000000-gated_norm_binding-weight.bin")).unwrap(),
        weight
    );
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("ffn-L0-rank1-000000-gated_norm_binding.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["binding"]["eps_f32_bits"], eps.to_bits());
    assert_eq!(metadata["binding"]["kernel_handle"], 123);
    assert_eq!(metadata["binding"]["grid"], serde_json::json!([12, 1, 1]));
    assert_eq!(metadata["binding"]["block"], serde_json::json!([128, 1, 1]));
    // Exercise the real launch wrapper and require its dispatched geometry to
    // match the captured binding, rather than trusting copied constants.
    ops::kda_sigmoid_gated_norm(&gpu, KernelHandle(123), ptr, ptr, ptr, ptr, 3, 4, 2, eps, 8)
        .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    assert_eq!(metadata["binding"]["kernel_handle"], launches[0].func);
    assert_eq!(
        metadata["binding"]["grid"],
        serde_json::json!(launches[0].grid)
    );
    assert_eq!(
        metadata["binding"]["block"],
        serde_json::json!(launches[0].block)
    );
    assert_eq!(metadata["binding"]["activation"], "sigmoid");
    assert_eq!(metadata["tensors"][0]["shape"], serde_json::json!([2]));
    assert_eq!(metadata["tensors"][0]["dtype"], "bf16-le");
    for invalid in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(
            trace
                .capture_gated_norm_binding(
                    Some(1),
                    &gpu,
                    ptr,
                    KernelHandle(123),
                    3,
                    4,
                    2,
                    invalid,
                    8
                )
                .is_err()
        );
    }
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 3);
    gpu.free(ptr).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn disabled_trace_performs_no_io_and_preserves_marker_identity() {
    let gpu = MockGpuBackend::new();
    let root = std::env::temp_dir().join(format!("atlas-upstream-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let trace = UpstreamTrace(Exl3FfnRouter::new(root.clone(), 0, 1, 2).unwrap());
    let ptr = gpu.alloc(4).unwrap();
    gpu.copy_h2d(&[0, 0x3f, 0, 0x40], ptr).unwrap();
    assert_eq!(trace.begin().unwrap(), None);
    trace
        .capture(
            None,
            &gpu,
            "embedding",
            &[("hidden", DiagnosticDtype::Bf16, ptr, &[1, 2])],
            8,
        )
        .unwrap();
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    std::fs::write(root.join("capture-enabled"), b"").unwrap();
    let invocation = trace.begin().unwrap();
    trace
        .capture(
            invocation,
            &gpu,
            "embedding",
            &[("hidden", DiagnosticDtype::Bf16, ptr, &[1, 2])],
            8,
        )
        .unwrap();
    assert_eq!(
        std::fs::read(root.join("ffn-L0-rank1-000000-embedding-hidden.bin")).unwrap(),
        [0, 0x3f, 0, 0x40]
    );
    assert_eq!(trace.begin().unwrap(), Some(1));
    gpu.free(ptr).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn upstream_capture_supports_only_the_l0_l1_boundary() {
    for layer in 0..=1 {
        ensure_supported_layer(layer).unwrap();
    }
    for layer in 2..64 {
        assert!(ensure_supported_layer(layer).is_err());
    }
}
#[test]
fn hc_boundary_binding_binds_rows_precision_and_request() {
    use crate::layers::glm5_exl3_mhc::MhcResidualPrecision;
    let make = |rows: usize,
                hc_mult: usize,
                sinkhorn_iters: u32,
                hc_eps: f32,
                rms_eps: f32,
                request_id: Option<String>| {
        HcBoundaryFacts {
            rows,
            hc_mult,
            sinkhorn_iters,
            hc_eps,
            rms_eps,
            residual_precision: MhcResidualPrecision::for_fused_rows(rows),
            request_id,
        }
    };
    let bound = hc_boundary_binding(&make(16, 4, 20, 1e-6, 1e-5, None)).unwrap();
    assert_eq!(bound["rows"], 16);
    assert!(bound.get("local_rows").is_none());
    assert_eq!(bound["hc_mult"], 4);
    assert_eq!(bound["sinkhorn_iters"], 20);
    assert_eq!(bound["hc_eps_bits"], 1e-6f32.to_bits());
    assert_eq!(bound["rms_eps_bits"], 1e-5f32.to_bits());
    assert_eq!(bound["residual_precision"], "UnroundedPost");
    assert_eq!(bound["sequence_parallel"], false);
    assert!(bound.get("request_id").is_none());
    let bound =
        hc_boundary_binding(&make(39, 4, 20, 1e-6, 1e-5, Some("req-1".to_string()))).unwrap();
    assert_eq!(bound["residual_precision"], "Rounded");
    assert_eq!(bound["request_id"], "req-1");
    for (rows, hc_mult, iters, hc_eps, rms_eps) in [
        (0usize, 4usize, 20u32, 1e-6f32, 1e-5f32),
        (1, 0, 20, 1e-6, 1e-5),
        (1, 4, 0, 1e-6, 1e-5),
        (1, 4, 20, f32::NAN, 1e-5),
        (1, 4, 20, 1e-6, 0.0),
    ] {
        assert!(hc_boundary_binding(&make(rows, hc_mult, iters, hc_eps, rms_eps, None)).is_err());
    }
}
#[test]
fn layer1_boundary_capture_records_mhc_binding() {
    use crate::layers::glm5_exl3_mhc::MhcResidualPrecision;
    let gpu = MockGpuBackend::new();
    let root = std::env::temp_dir().join(format!("atlas-upstream-l1-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let trace = UpstreamTrace(Exl3FfnRouter::new(root.clone(), 1, 0, 2).unwrap());
    let ptr = gpu.alloc(4096).unwrap();
    gpu.copy_h2d(&[0u8; 4096], ptr).unwrap();
    std::fs::write(root.join("capture-enabled"), b"").unwrap();
    let facts = HcBoundaryFacts {
        rows: 39,
        hc_mult: 4,
        sinkhorn_iters: 20,
        hc_eps: 1e-6,
        rms_eps: 1e-5,
        residual_precision: MhcResidualPrecision::for_fused_rows(39),
        request_id: Some("req-1".to_string()),
    };
    let invocation = trace.begin().unwrap();
    trace
        .capture_boundary(
            invocation,
            &gpu,
            "attn_mhc_norm",
            &[
                ("normed", DiagnosticDtype::Bf16, ptr, &[39, 2]),
                ("highway", DiagnosticDtype::Bf16, ptr, &[39, 4, 2]),
                ("post", DiagnosticDtype::F32, ptr, &[39, 4]),
                ("comb", DiagnosticDtype::F32, ptr, &[39, 4, 4]),
            ],
            8,
            &facts,
        )
        .unwrap();
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("ffn-L1-rank0-000000-attn_mhc_norm.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        metadata["binding"]["schema"],
        "atlas-mhc-boundary-binding-v1"
    );
    assert_eq!(metadata["binding"]["rows"], 39);
    assert_eq!(metadata["binding"]["request_id"], "req-1");
    assert_eq!(metadata["tensors"][0]["name"], "normed");
    assert_eq!(metadata["tensors"][0]["shape"], serde_json::json!([39, 2]));
    gpu.free(ptr).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn disabled_boundary_capture_returns_before_binding_validation() {
    use crate::layers::glm5_exl3_mhc::MhcResidualPrecision;
    let gpu = MockGpuBackend::new();
    let root = std::env::temp_dir().join(format!("atlas-upstream-disabled-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let trace = UpstreamTrace(Exl3FfnRouter::new(root.clone(), 1, 1, 2).unwrap());
    // Invalid facts on purpose: disabled capture must return before any
    // validation, GPU access or filesystem write (module invariant).
    let facts = HcBoundaryFacts {
        rows: 0,
        hc_mult: 0,
        sinkhorn_iters: 0,
        hc_eps: f32::NAN,
        rms_eps: 0.0,
        residual_precision: MhcResidualPrecision::Rounded,
        request_id: None,
    };
    assert_eq!(trace.begin().unwrap(), None);
    trace
        .capture_boundary(None, &gpu, "attn_mhc_norm", &[], 8, &facts)
        .unwrap();
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    // Enabled: the same facts fail fast.
    std::fs::write(root.join("capture-enabled"), b"").unwrap();
    let invocation = trace.begin().unwrap();
    assert!(
        trace
            .capture_boundary(invocation, &gpu, "attn_mhc_norm", &[], 8, &facts)
            .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}
