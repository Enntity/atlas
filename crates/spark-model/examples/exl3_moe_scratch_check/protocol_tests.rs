// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[test]
fn explicit_plan_rejects_same_layer_wrong_rank_stream_and_unbounded_rows() {
    let mut plan = Plan {
        checkpoint: "checkpoint".into(),
        config_sha256: "a".repeat(64),
        index_sha256: "b".repeat(64),
        rank: 0,
        world: 2,
        capacity: 129,
        device: 0,
        stream: "default".into(),
        layers: [3, 4],
        cases: vec![InputCase {
            id: "actual_m5".into(),
            rows: 5,
            input_a: FileBinding {
                file: "a.bin".into(),
                sha256: "c".repeat(64),
            },
            input_b: FileBinding {
                file: "b.bin".into(),
                sha256: "d".repeat(64),
            },
        }],
        parent_evidence: vec![FileBinding {
            file: "checkpoint-full-sha.json".into(),
            sha256: "e".repeat(64),
        }],
    };
    plan.validate().unwrap();
    plan.layers = [3, 3];
    assert!(plan.validate().is_err());
    plan.layers = [3, 4];
    plan.rank = 2;
    assert!(plan.validate().is_err());
    plan.rank = 0;
    plan.stream = "guessed".into();
    assert!(plan.validate().is_err());
    plan.stream = "default".into();
    plan.cases[0].rows = 130;
    assert!(plan.validate().is_err());
}
#[test]
fn bf16_input_requires_exact_shape_and_finite_values() {
    validate_bf16(&0x3f80u16.to_le_bytes().repeat(128), 1, 128).unwrap();
    assert!(validate_bf16(&[0; 254], 1, 128).is_err());
    assert!(validate_bf16(&0x7fc0u16.to_le_bytes().repeat(128), 1, 128).is_err());
}
#[test]
fn actual_file_boundary_hashes_parameters_and_never_overwrites_outputs() {
    let root = std::env::temp_dir().join(format!("atlas-scratch-plan-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let config = root.join("config.json");
    let index = root.join("model.safetensors.index.json");
    std::fs::write(&config, b"{}").unwrap();
    std::fs::write(&index, b"{}").unwrap();
    let input = root.join("input.bin");
    std::fs::write(&input, 0x3f80u16.to_le_bytes().repeat(128)).unwrap();
    let bound = serde_json::json!({"file":input,"sha256":digest(&std::fs::read(&input).unwrap())});
    let plan = root.join("plan.json");
    std::fs::write(&plan,serde_json::to_vec(&serde_json::json!({"checkpoint":root,"config_sha256":digest(b"{}"),"index_sha256":digest(b"{}"),"rank":0,"world":2,"device":0,"capacity":129,"stream":"default","layers":[3,4],"cases":[{"id":"actual_m1","rows":1,"input_a":bound,"input_b":bound}],"parent_evidence":[bound]})).unwrap()).unwrap();
    let io = ReplayIORouter::open(&plan, &root.join("output")).unwrap();
    assert_eq!(io.config().unwrap(), "{}");
    assert_eq!(io.plan.device, 0);
    assert_eq!(io.plan_sha256, digest(&std::fs::read(&plan).unwrap()));
    assert_eq!(ReplayIORouter::executable_hash().unwrap().len(), 64);
    // CPU test has no BLAS libraries; explicit missing-provider error is expected.
    assert!(ReplayIORouter::loaded_blas().is_err());
    io.write("result.bin", b"bytes").unwrap();
    assert!(io.write("result.bin", b"replacement").is_err());
    std::fs::remove_dir_all(root).unwrap();
}
