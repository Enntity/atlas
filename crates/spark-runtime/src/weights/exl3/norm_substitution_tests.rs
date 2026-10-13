// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::gpu::mock::MockGpuBackend;

pub(super) fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "atlas-norm-sub-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    root
}

pub(super) fn entry_json(
    request_id: &str,
    tp_rank: usize,
    invocation: usize,
    layer: usize,
    shape: [usize; 2],
    file: &str,
    sha256: &str,
) -> serde_json::Value {
    serde_json::json!({
        "request_id": request_id, "layer": layer, "tp_rank": tp_rank, "tp_world": 2,
        "invocation": invocation, "stage": "gated_norm", "tensor": "gated",
        "dtype": "bf16-le", "shape": shape, "file": file, "sha256": sha256
    })
}

pub(super) fn manifest_json(entries: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "schema": MANIFEST_SCHEMA,
        "variant": "reference-exact",
        "source": {"capture_root": "/ev/reference", "engine": "reference",
                   "schedule_sha256": "0".repeat(64)},
        "entries": entries
    })
}

/// Writes `manifest` and `payloads` under a fresh fixture dir plus a fresh
/// record dir; returns (fixture dir, record dir).
pub(super) fn write_fixture(
    tag: &str,
    manifest: &serde_json::Value,
    payloads: &[(&str, &[u8])],
) -> (PathBuf, PathBuf) {
    let root = temp_root(tag);
    let fixture = root.join("fixtures");
    let records = root.join("records");
    std::fs::create_dir(&fixture).unwrap();
    std::fs::create_dir(&records).unwrap();
    std::fs::write(
        fixture.join(MANIFEST_NAME),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
    for (name, bytes) in payloads {
        std::fs::write(fixture.join(name), bytes).unwrap();
    }
    (fixture, records)
}

/// Eight deterministic BF16 bytes for a `[2, 2]` `gated` fixture.
pub(super) const PAYLOAD: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
pub(super) const PRE_IMAGE: [u8; 8] = [9, 9, 9, 9, 9, 9, 9, 9];

pub(super) fn payload_entry(request_id: &str, invocation: usize) -> serde_json::Value {
    entry_json(
        request_id,
        0,
        invocation,
        0,
        [2, 2],
        "L0-rank0-gated.bin",
        &hex_sha256(&PAYLOAD),
    )
}

#[test]
fn knob_off_is_a_noop_with_zero_file_io_and_zero_device_copies() {
    // The knob env is process-wide and `env_tests` mutates it under the same
    // lock (repo ENV_LOCK pattern); take the lock around the `from_env` read
    // so this test cannot observe a transient engaged state.
    let _env_guard = super::env_tests::lock();
    assert!(NormSubstitution::from_env(0, 2).unwrap().is_none());
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    gpu.copy_h2d(&PRE_IMAGE, ptr).unwrap();
    let d2h_before = gpu.d2h_blocking_count() + gpu.d2h_async_count();
    let sync_before = gpu.sync_count();
    let pending = engage(
        &gpu,
        None,
        0,
        ptr,
        DiagnosticDtype::Bf16,
        &[2, 2],
        Some("req-a"),
        7,
    )
    .unwrap();
    assert!(pending.is_none());
    assert_eq!(gpu.d2h_blocking_count() + gpu.d2h_async_count(), d2h_before);
    assert_eq!(gpu.sync_count(), sync_before);
    assert_eq!(gpu.read_alloc(ptr).unwrap(), PRE_IMAGE);
    gpu.free(ptr).unwrap();
}

#[test]
fn load_rejects_missing_manifest_bad_schema_and_bad_variant() {
    let root = temp_root("no-manifest");
    let records = root.join("records");
    std::fs::create_dir(&records).unwrap();
    assert!(NormSubstitution::load(&root, &records, 0, 2).is_err());

    for (tag, mutate) in [
        (
            "bad-schema",
            Box::new(|m: &mut serde_json::Value| m["schema"] = "other-v0".into())
                as Box<dyn Fn(&mut serde_json::Value)>,
        ),
        (
            "bad-variant",
            Box::new(|m: &mut serde_json::Value| m["variant"] = "v25".into()),
        ),
        (
            "bad-source",
            Box::new(|m: &mut serde_json::Value| m["source"]["schedule_sha256"] = "zz".into()),
        ),
    ] {
        let mut manifest = manifest_json(vec![]);
        mutate(&mut manifest);
        let (fixture, records) = write_fixture(tag, &manifest, &[]);
        assert!(
            NormSubstitution::load(&fixture, &records, 0, 2).is_err(),
            "{tag}"
        );
    }
}

#[test]
fn load_rejects_bad_sha_and_size_mismatch() {
    // Right size, wrong content: the declared sha256 does not match.
    let bad_sha = manifest_json(vec![entry_json(
        "req-a",
        0,
        0,
        0,
        [2, 2],
        "bad-sha.bin",
        &hex_sha256(b"different"),
    )]);
    let (fixture, records) = write_fixture("bad-sha", &bad_sha, &[("bad-sha.bin", &PAYLOAD)]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());

    // Honest sha256 over 6 bytes, declared shape [2, 2] (8 bytes): size mismatch.
    let short: [u8; 6] = [1, 2, 3, 4, 5, 6];
    let bad_size = manifest_json(vec![entry_json(
        "req-a",
        0,
        0,
        0,
        [2, 2],
        "short.bin",
        &hex_sha256(&short),
    )]);
    let (fixture, records) = write_fixture("bad-size", &bad_size, &[("short.bin", &short)]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());
}

#[test]
fn load_rejects_duplicate_entries_nonzero_layer_and_bad_topology() {
    let dupe = manifest_json(vec![payload_entry("req-a", 0), payload_entry("req-b", 0)]);
    let (fixture, records) = write_fixture("dupe", &dupe, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());

    let layer1 = manifest_json(vec![entry_json(
        "req-a",
        0,
        0,
        1,
        [2, 2],
        "L0-rank0-gated.bin",
        &hex_sha256(&PAYLOAD),
    )]);
    let (fixture, records) = write_fixture("layer1", &layer1, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());

    let bad_rank = manifest_json(vec![entry_json(
        "req-a",
        2,
        0,
        0,
        [2, 2],
        "L0-rank0-gated.bin",
        &hex_sha256(&PAYLOAD),
    )]);
    let (fixture, records) = write_fixture("rank", &bad_rank, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());
    assert!(NormSubstitution::load(&fixture, &records, 0, 1).is_err());
}

#[test]
fn load_rejects_unsafe_or_missing_payloads_and_bad_record_dir() {
    let escape = manifest_json(vec![entry_json(
        "req-a",
        0,
        0,
        0,
        [2, 2],
        "../escape.bin",
        &hex_sha256(&PAYLOAD),
    )]);
    let (fixture, records) = write_fixture("escape", &escape, &[("escape.bin", &PAYLOAD)]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());

    let missing = manifest_json(vec![payload_entry("req-a", 0)]);
    let (fixture, records) = write_fixture("missing-payload", &missing, &[]);
    assert!(NormSubstitution::load(&fixture, &records, 0, 2).is_err());

    let ok = manifest_json(vec![payload_entry("req-a", 0)]);
    let (fixture, records) =
        write_fixture("missing-records", &ok, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    assert!(
        NormSubstitution::load(&fixture, &fixture.join("absent"), 0, 2).is_err(),
        "record directory must already exist"
    );
    assert!(
        NormSubstitution::load(&fixture, &fixture, 0, 2).is_err(),
        "records must not land in the fixture directory"
    );
    drop(records);
}

#[test]
fn engaged_invocation_without_entry_errors_the_run() {
    let manifest = manifest_json(vec![payload_entry("req-a", 0)]);
    let (fixture, records) = write_fixture(
        "missing-entry",
        &manifest,
        &[("L0-rank0-gated.bin", &PAYLOAD)],
    );
    let sub = NormSubstitution::load(&fixture, &records, 0, 2).unwrap();
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    gpu.copy_h2d(&PRE_IMAGE, ptr).unwrap();
    assert!(
        sub.substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[2, 2],
            Some("req-a"),
            7
        )
        .unwrap()
        .is_some()
    );
    // Invocation 1 has no fixture entry: the engaged run must error.
    let err = sub
        .substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[2, 2],
            Some("req-a"),
            7,
        )
        .unwrap_err();
    assert!(err.to_string().contains("no fixture entry"), "{err:#}");
    gpu.free(ptr).unwrap();
}

#[test]
fn request_binding_must_be_present_and_match_the_fixture_entry() {
    let manifest = manifest_json(vec![
        payload_entry("req-a", 0),
        payload_entry("req-a", 1),
        payload_entry("req-a", 2),
    ]);
    let (fixture, records) =
        write_fixture("binding", &manifest, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    let sub = NormSubstitution::load(&fixture, &records, 0, 2).unwrap();
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    gpu.copy_h2d(&PRE_IMAGE, ptr).unwrap();
    let missing = sub
        .substitute(&gpu, 0, ptr, DiagnosticDtype::Bf16, &[2, 2], None, 7)
        .unwrap_err();
    assert!(
        missing.to_string().contains("request-id binding"),
        "{missing:#}"
    );
    let mismatch = sub
        .substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[2, 2],
            Some("req-b"),
            7,
        )
        .unwrap_err();
    assert!(
        mismatch.to_string().contains("request-id mismatch"),
        "{mismatch:#}"
    );
    assert!(
        sub.substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[2, 2],
            Some("req-a"),
            7
        )
        .unwrap()
        .is_some()
    );
    gpu.free(ptr).unwrap();
}

#[test]
fn shape_and_dtype_mismatch_error_the_run() {
    let manifest = manifest_json(vec![payload_entry("req-a", 0), payload_entry("req-a", 1)]);
    let (fixture, records) = write_fixture("shape", &manifest, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    let sub = NormSubstitution::load(&fixture, &records, 0, 2).unwrap();
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(16).unwrap();
    let shape = sub
        .substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[4, 1],
            Some("req-a"),
            7,
        )
        .unwrap_err();
    assert!(shape.to_string().contains("shape mismatch"), "{shape:#}");
    let dtype = sub
        .substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::F32,
            &[2, 2],
            Some("req-a"),
            7,
        )
        .unwrap_err();
    assert!(dtype.to_string().contains("dtype mismatch"), "{dtype:#}");
    gpu.free(ptr).unwrap();
}

#[test]
fn happy_path_copies_fixture_bytes_records_pre_hash_and_post_payload() {
    let manifest = manifest_json(vec![payload_entry("req-a", 0)]);
    let (fixture, records) = write_fixture("happy", &manifest, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    let sub = NormSubstitution::load(&fixture, &records, 0, 2).unwrap();
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    gpu.copy_h2d(&PRE_IMAGE, ptr).unwrap();
    let pending = sub
        .substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[2, 2],
            Some("req-a"),
            7,
        )
        .unwrap()
        .unwrap();
    // The staged fixture bytes replaced `gated`; the pre-image hash is kept.
    assert_eq!(gpu.read_alloc(ptr).unwrap(), PAYLOAD);
    assert_eq!(pending.pre_sha256, hex_sha256(&PRE_IMAGE));
    assert_eq!(pending.fixture_sha256, hex_sha256(&PAYLOAD));
    pending.record(&gpu, ptr, 7).unwrap();
    assert_eq!(
        std::fs::read(records.join("ffn-L0-rank0-000000-norm_substitution-gated.bin")).unwrap(),
        PAYLOAD
    );
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(records.join("ffn-L0-rank0-000000-norm_substitution.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(record["binding"]["schema"], RECORD_SCHEMA);
    assert_eq!(record["binding"]["payload_stage"], "post-substitution");
    assert_eq!(record["binding"]["variant"], "reference-exact");
    assert_eq!(record["binding"]["request_id"], "req-a");
    assert_eq!(record["binding"]["invocation"], 0);
    assert_eq!(record["binding"]["shape"], serde_json::json!([2, 2]));
    assert_eq!(record["binding"]["fixture_file"], "L0-rank0-gated.bin");
    assert_eq!(record["binding"]["fixture_sha256"], hex_sha256(&PAYLOAD));
    assert_eq!(
        record["binding"]["pre_substitution_sha256"],
        hex_sha256(&PRE_IMAGE)
    );
    assert_eq!(record["tensors"][0]["dtype"], "bf16-le");
    // Atomic commit: no partial records survive, the fixture dir is untouched.
    assert!(std::fs::read_dir(&records).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".partial")
    }));
    assert_eq!(std::fs::read_dir(&fixture).unwrap().count(), 2);
    gpu.free(ptr).unwrap();
}

#[test]
fn later_layers_do_not_substitute_and_multi_seq_fails_fast() {
    let manifest = manifest_json(vec![payload_entry("req-a", 0)]);
    let (fixture, records) =
        write_fixture("layer-gate", &manifest, &[("L0-rank0-gated.bin", &PAYLOAD)]);
    let sub = NormSubstitution::load(&fixture, &records, 0, 2).unwrap();
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    gpu.copy_h2d(&PRE_IMAGE, ptr).unwrap();
    // Layer 1..: no substitution and no device copies (assumption A12), and
    // the invocation counter does not advance — layer 0 still sees entry 0.
    for layer in 1..4 {
        assert!(
            sub.substitute(
                &gpu,
                layer,
                ptr,
                DiagnosticDtype::Bf16,
                &[2, 2],
                Some("req-a"),
                7
            )
            .unwrap()
            .is_none()
        );
    }
    assert_eq!(gpu.d2h_blocking_count() + gpu.d2h_async_count(), 0);
    assert_eq!(gpu.read_alloc(ptr).unwrap(), PRE_IMAGE);
    assert!(
        sub.substitute(
            &gpu,
            0,
            ptr,
            DiagnosticDtype::Bf16,
            &[2, 2],
            Some("req-a"),
            7
        )
        .unwrap()
        .is_some()
    );
    // Assumption A9: the multi-sequence path errors, never substitutes.
    let err = sub.reject_multi_seq().unwrap_err();
    assert!(err.to_string().contains("multi-sequence"), "{err:#}");
    gpu.free(ptr).unwrap();
}

#[test]
fn dtype_labels_stay_pinned_to_the_diagnostic_tensor_table() {
    for dtype in [
        DiagnosticDtype::Bf16,
        DiagnosticDtype::F32,
        DiagnosticDtype::I32,
    ] {
        assert_eq!(
            dtype_width(dtype.label()),
            Some(dtype.width()),
            "{}",
            dtype.label()
        );
    }
    assert_eq!(dtype_width("f16-le"), None);
}
