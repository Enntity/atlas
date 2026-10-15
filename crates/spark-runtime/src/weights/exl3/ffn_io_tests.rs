// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::gpu::mock::MockGpuBackend;

#[test]
fn selected_layers_require_explicit_unique_in_range_ids() {
    assert_eq!(parse_layers("0,3", 45).unwrap(), BTreeSet::from([0, 3]));
    for invalid in ["", "0,0", "45", "-1", "0,", "all"] {
        assert!(parse_layers(invalid, 45).is_err());
    }
}

#[test]
fn capture_preserves_typed_bytes_marker_gate_and_exclusive_identity() {
    let root = std::env::temp_dir().join(format!(
        "atlas-ffn-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let router = Exl3FfnRouter::new(root.clone(), 3, 1, 2).unwrap();
    assert_eq!(router.begin().unwrap(), None);
    std::fs::write(root.join("capture-enabled"), b"").unwrap();
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    let values = [1.25f32, -2.5];
    let bytes: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    gpu.copy_h2d(&bytes, ptr).unwrap();
    let invocation = router.begin().unwrap().unwrap();
    let tensor = DiagnosticTensor {
        name: "weights",
        dtype: DiagnosticDtype::F32,
        shape: vec![1, 2],
        ptr,
    };
    router
        .capture(&gpu, invocation, "router", &[tensor], 17)
        .unwrap();
    assert_eq!(
        std::fs::read(root.join("ffn-L3-rank1-000000-router-weights.bin")).unwrap(),
        bytes
    );
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("ffn-L3-rank1-000000-router.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["tensors"][0]["dtype"], "f32-le");
    assert_eq!(metadata["tensors"][0]["shape"], serde_json::json!([1, 2]));
    assert_eq!(metadata["tp_rank"], 1);
    assert!(
        router
            .capture(
                &gpu,
                invocation,
                "router",
                &[DiagnosticTensor {
                    name: "weights",
                    dtype: DiagnosticDtype::F32,
                    shape: vec![1, 2],
                    ptr
                }],
                17
            )
            .is_err()
    );
    assert!(
        router
            .capture(&gpu, invocation, "../unsafe", &[], 17)
            .is_err()
    );
    assert!(
        router
            .capture_with_metadata(&gpu, 1, "binding", &[], 17, &serde_json::json!(["invalid"]))
            .is_err()
    );
    router
        .capture_with_metadata(
            &gpu,
            1,
            "binding",
            &[DiagnosticTensor {
                name: "weights",
                dtype: DiagnosticDtype::F32,
                shape: vec![1, 2],
                ptr,
            }],
            17,
            &serde_json::json!({"eps_f32_bits": 0x3727c5acu32}),
        )
        .unwrap();
    let bound: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("ffn-L3-rank1-000001-binding.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(bound["binding"]["eps_f32_bits"], 0x3727c5acu32);
    assert_eq!(bound["invocation"], 1);
    assert!(metadata.get("binding").is_none());
    // Records commit by rename: no partial file survives, and an existing
    // record is never overwritten.
    assert!(std::fs::read_dir(&root).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".partial")
    }));
    std::fs::write(root.join("ffn-L3-rank1-000002-stale.json"), b"{}").unwrap();
    assert!(
        router
            .capture(
                &gpu,
                2,
                "stale",
                &[DiagnosticTensor {
                    name: "weights",
                    dtype: DiagnosticDtype::F32,
                    shape: vec![1, 2],
                    ptr
                }],
                17
            )
            .is_err()
    );
    assert_eq!(
        std::fs::read(root.join("ffn-L3-rank1-000002-stale.json")).unwrap(),
        b"{}"
    );
    gpu.free(ptr).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
