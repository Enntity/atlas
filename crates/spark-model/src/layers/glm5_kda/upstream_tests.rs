// SPDX-License-Identifier: AGPL-3.0-only
//! CPU tests for the upstream trace, including the C1 o_proj binding record.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
#[test]
fn gated_norm_binding_captures_resident_weight_and_effective_epsilon() {
    let gpu = MockGpuBackend::new();
    let root = std::env::temp_dir().join(format!("atlas-norm-binding-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let trace = UpstreamTrace {
        router: Exl3FfnRouter::new(root.clone(), 0, 1, 2).unwrap(),
        rank: 1,
        o_proj: None,
    };
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
    let trace = UpstreamTrace {
        router: Exl3FfnRouter::new(root.clone(), 0, 1, 2).unwrap(),
        rank: 1,
        o_proj: None,
    };
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

fn valid_load_record(rank: usize) -> serde_json::Value {
    serde_json::json!({
        "schema": o_proj_binding::LOAD_SCHEMA,
        "layer": 0,
        "rank": rank,
        "tp_world": 2,
        "module": "model.layers.0.self_attn.o_proj",
        "projection": {
            "module": "model.layers.0.self_attn.o_proj",
            "tp_pre_reduce": true,
            "weight_layout": "in-out",
            "input_dtype": "bf16",
            "weight_dtype": "bf16",
            "accumulator_dtype": "f32",
            "output_dtype": "bf16",
        },
        "files": {"weight": {"file": "o-proj-weight-L0-rank1.npy",
                              "shape": [4096, 4096], "dtype": "bf16-le",
                              "sha256": "ab".repeat(32)}},
        "resident": {"form": "exl3-trellis-mul1", "file": "o-proj-resident-L0-rank1.bin",
                     "dtype": "exl3-shard-pack-v1", "sha256": "cd".repeat(32),
                     "recipe": {"name": "exl3-mul1-reference-reconstruct-rne-bf16",
                                "inputs": {"resident_sha256": "cd".repeat(32)}}},
    })
}

#[test]
fn o_proj_binding_record_round_trips_v1_fields_and_fails_closed() {
    let base = std::env::temp_dir().join(format!(
        "atlas-o-proj-binding-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let router_dir = base.join("router");
    let load_dir = base.join("load");
    std::fs::create_dir_all(&router_dir).unwrap();
    std::fs::create_dir_all(&load_dir).unwrap();
    std::fs::write(
        load_dir.join(o_proj_binding::load_file(1)),
        valid_load_record(1).to_string(),
    )
    .unwrap();
    let make_trace = |load_dir: PathBuf| UpstreamTrace {
        router: Exl3FfnRouter::new(router_dir.clone(), 0, 1, 2).unwrap(),
        rank: 1,
        o_proj: Some(OProjBinding { dir: load_dir }),
    };
    let trace = make_trace(load_dir.clone());
    let receipt = serde_json::json!({"provider": "exl3-native", "dispatch": {"tokens": 10}});
    // No marker and no invocation: nothing is written.
    assert_eq!(trace.begin().unwrap(), None);
    trace.capture_o_proj_binding(None, 10, &receipt).unwrap();
    assert_eq!(std::fs::read_dir(&router_dir).unwrap().count(), 0);
    std::fs::write(router_dir.join("capture-enabled"), b"").unwrap();
    let invocation = trace.begin().unwrap().unwrap();
    trace
        .capture_o_proj_binding(Some(invocation), 10, &receipt)
        .unwrap();
    let path = router_dir.join("ffn-L0-rank1-000000-o_proj_binding.json");
    let envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(envelope["schema"], "atlas-exl3-ffn-v1");
    assert_eq!(envelope["stage"], "o_proj_binding");
    let binding = &envelope["binding"];
    assert_eq!(binding["schema"], o_proj_binding::BINDING_SCHEMA);
    assert_eq!(binding["engine"], "candidate");
    assert_eq!(binding["rank"], 1);
    assert_eq!(binding["layer"], 0);
    assert_eq!(binding["invocation"], 0);
    assert_eq!(binding["tokens"], 10);
    assert_eq!(binding["projection"]["tp_pre_reduce"], true);
    assert_eq!(binding["projection"]["weight_layout"], "in-out");
    assert_eq!(binding["projection"]["accumulator_dtype"], "f32");
    assert_eq!(
        binding["files"]["weight"]["file"],
        "o-proj-weight-L0-rank1.npy"
    );
    assert_eq!(
        binding["files"]["weight"]["shape"],
        serde_json::json!([4096, 4096])
    );
    assert_eq!(binding["resident"]["form"], "exl3-trellis-mul1");
    assert_eq!(
        binding["resident"]["recipe"]["name"],
        "exl3-mul1-reference-reconstruct-rne-bf16"
    );
    assert_eq!(envelope["receipt"]["provider"], "exl3-native");
    assert_eq!(
        envelope["receipt"]["weight_basis"],
        "exl3-mul1-reference-reconstruct-rne-bf16"
    );
    // Immutable evidence: the same (stage, invocation) cannot be rewritten.
    assert!(
        trace
            .capture_o_proj_binding(Some(invocation), 10, &receipt)
            .is_err()
    );
    // A load record missing a required field fails closed on first use.
    let mut broken = valid_load_record(1);
    broken.as_object_mut().unwrap().remove("resident");
    std::fs::write(
        load_dir.join(o_proj_binding::load_file(1)),
        broken.to_string(),
    )
    .unwrap();
    let fresh = make_trace(load_dir.clone());
    assert!(fresh.capture_o_proj_binding(Some(1), 10, &receipt).is_err());
    // A load record bound to another rank fails closed.
    std::fs::write(
        load_dir.join(o_proj_binding::load_file(1)),
        valid_load_record(0).to_string(),
    )
    .unwrap();
    let fresh = make_trace(load_dir.clone());
    assert!(fresh.capture_o_proj_binding(Some(1), 10, &receipt).is_err());
    std::fs::remove_dir_all(&base).unwrap();
}
