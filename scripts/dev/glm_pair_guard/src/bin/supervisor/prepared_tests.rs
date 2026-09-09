// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{symlink, MetadataExt, OpenOptionsExt, PermissionsExt};

struct Fixture {
    root: PathBuf,
    input: PathBuf,
    launch: Launch,
    original: [wire::Recipe; 2],
}
fn write(path: &Path, bytes: &[u8]) {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}
fn recipe(rank: u8) -> wire::Recipe {
    wire::Recipe {
        argv: vec!["/opt/atlas/spark".into(), "serve".into()],
        environment: vec![("ATLAS_GLM_PAIR_FD".into(), "3".into())],
        mounts: vec![wire::Mount {
            source: "/run/atlas-glm-pairs/template".into(),
            destination: "/run/atlas-pair".into(),
            read_only: false,
            propagation: "rprivate".into(),
        }],
        image_digest: [1; 32],
        guard_elf_digest: [2; 32],
        server_elf_digest: [3; 32],
        rank,
        world: 2,
        profile: wire::Profile {
            tp: 2,
            ep: 2,
            ep_protocol: 2,
            max_sequences: 2,
            context: 2044,
            prefill: 1024,
            drafts: 4,
            eager: true,
            kv_format: 1,
            cold_min: 2,
            cold_max: 1024,
        },
        resources: wire::Resources {
            memory: 114 << 30,
            swap: 114 << 30,
            cpuset: "0-19".into(),
            shm: 1 << 30,
            device_requests: vec![],
            ulimits: vec![],
            cap_add: vec![],
            cap_drop: vec![],
            uid: 0,
            gid: 0,
            network_mode: "host".into(),
            pid_mode: "private".into(),
            restart_policy: "no".into(),
            init: false,
            no_new_privileges: true,
            ipc_mode: "private".into(),
            devices: vec![],
            security_options: vec![],
        },
    }
}
impl Fixture {
    fn new() -> Self {
        let mut template = b"/tmp/atlas-prepared-XXXXXX\0".to_vec();
        assert!(!unsafe { libc::mkdtemp(template.as_mut_ptr().cast()) }.is_null());
        let root = PathBuf::from(std::str::from_utf8(&template[..template.len() - 1]).unwrap());
        let original = [recipe(0), recipe(1)];
        let recipes = [root.join("template0.bin"), root.join("template1.bin")];
        let bytes = original.each_ref().map(|r| r.encode().unwrap());
        for r in 0..2 {
            write(&recipes[r], &bytes[r]);
        }
        let payload = root.join("source.input");
        write(&payload, b"print('literal workload')\n");
        let hash = |b: &[u8]| super::super::docker::hex(&raw_hash(b));
        let node = |r: usize| {
            json!({"rank":r,"destination":format!("abc@node{r}"),"supervisor_path":"/opt/atlas/supervisor",
            "supervisor_sha256":"01".repeat(32),"relay_path":"/opt/atlas/relay","relay_sha256":"02".repeat(32),
            "guard_container_path":"/opt/atlas/guard","recipe_file":recipes[r],"recipe_sha256":hash(&bytes[r])})
        };
        let value = json!({"version":1,"policy":{"startup":1000,"lease":1000,"challenge":100,"frame":50,"campaign":10000,
            "poll":10,"reap":100,"child_handshake":500,"quiescent_wait":500,"exit":200},
            "controller":{"readiness_ms":2000,"workload_ms":2000,"drain_ms":1000,"status_max_age_ms":400,"command_ms":40,"status_interval_ms":100,"cleanup_ms":500},
            "ssh":{"executable":"/usr/bin/ssh","key":"/home/abc/key","known_hosts":"/home/abc/known_hosts","environment":[]},
            "nodes":[node(0),node(1)],"readiness":{"curl":"/usr/bin/curl","url":"http://node0:8000/health","expected_model":"glm","per_attempt_ms":30},
            "workload":{"program":"/usr/bin/python3","program_sha256":"03".repeat(32),"argv":["-"],"environment":[],
                "input_file":payload,"input_sha256":hash(b"print('literal workload')\n"),
                "limits":{"timeout_ms":1000,"stdout_bytes":4096,"stderr_bytes":4096,"stdin_bytes":4096,"bytes_per_turn":64}}});
        let launch: Launch = serde_json::from_value(value).unwrap();
        let input = root.join("source.json");
        write(&input, &serde_json::to_vec(&launch).unwrap());
        Self {
            root,
            input,
            launch,
            original,
        }
    }
    fn prepare(&self) -> Prepared {
        Prepared::prepare(&self.input, &self.root.join("prepared")).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn actual_prepare_load_bundles_exact_bytes_without_original_files() {
    let f = Fixture::new();
    let p = f.prepare();
    assert_ne!(p.session, [0; 32]);
    assert_eq!(fs::metadata(&p.directory).unwrap().mode() & 0o7777, 0o700);
    for name in NAMES {
        assert_eq!(
            fs::metadata(p.directory.join(name)).unwrap().mode() & 0o7777,
            0o600
        );
    }
    for r in 0..2 {
        let mut expected = f.original[r].clone();
        expected.mounts[0].source = format!(
            "/run/atlas-glm-pairs/{}/rank{r}",
            super::super::docker::hex(&p.session)
        );
        assert_eq!(
            p.recipes[r], expected,
            "only the selected session source changes"
        );
        assert_eq!(p.launch.nodes[r].recipe_file, PathBuf::from(NAMES[r + 2]));
        fs::remove_file(&f.launch.nodes[r].recipe_file).unwrap();
    }
    fs::remove_file(&f.launch.workload.input_file).unwrap();
    fs::remove_file(&f.input).unwrap();
    let loaded = Prepared::load(&p.directory, p.digest)
        .expect("actual prepared files reload without templates");
    assert_eq!(loaded.session, p.session);
    assert_eq!(loaded.recipes, p.recipes);
    assert_eq!(loaded.workload_input, b"print('literal workload')\n");
    assert_eq!(
        serde_json::to_vec(&loaded.launch).unwrap(),
        serde_json::to_vec(&p.launch).unwrap()
    );
    loaded.consume().unwrap();
    assert!(loaded.consume().is_err());
}
#[test]
fn actual_prepare_is_exclusive_fresh_and_consumption_is_one_shot() {
    let f = Fixture::new();
    let p = f.prepare();
    assert!(Prepared::prepare(&f.input, &p.directory).is_err());
    let second = Prepared::prepare(&f.input, &f.root.join("second")).unwrap();
    assert_ne!(p.session, second.session);
    assert_ne!(p.digest, second.digest);
    p.consume().unwrap();
    assert!(p.consume().is_err());
    assert_eq!(
        fs::read(p.directory.join("consumed")).unwrap(),
        [p.session.as_slice(), p.digest.as_slice()].concat()
    );
}
#[test]
fn strict_actual_json_rejects_duplicates_unknowns_missing_fields_and_unsorted_env() {
    for failure in 0..5 {
        let f = Fixture::new();
        let mut text = String::from_utf8(fs::read(&f.input).unwrap()).unwrap();
        match failure {
            0 => text = text.replacen("\"version\":1", "\"version\":1,\"version\":1", 1),
            1 => text = text.replacen("\"version\":1", "\"version\":1,\"unknown\":0", 1),
            2 => text = text.replacen("\"version\":1,", "", 1),
            _ => {
                let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
                value["ssh"]["environment"] = if failure == 3 {
                    json!([["Z", "1"], ["A", "2"]])
                } else {
                    json!([["A", "1"], ["A", "2"]])
                };
                text = serde_json::to_string(&value).unwrap();
            }
        }
        fs::write(&f.input, text).unwrap();
        let out = f.root.join("refused");
        assert!(Prepared::prepare(&f.input, &out).is_err());
        assert!(!out.exists());
    }
}
#[test]
fn actual_load_rejects_modified_bytes_digest_symlinks_and_directory_replacement() {
    for failure in 0..5 {
        let f = Fixture::new();
        let p = f.prepare();
        Prepared::load(&p.directory, p.digest).expect("positive reload control before tampering");
        match failure {
            0 => fs::write(p.directory.join("workload.input"), b"changed").unwrap(),
            1 => {
                fs::remove_file(p.directory.join("rank0.recipe.bin")).unwrap();
                symlink(
                    &f.launch.nodes[0].recipe_file,
                    p.directory.join("rank0.recipe.bin"),
                )
                .unwrap();
            }
            2 => fs::set_permissions(
                p.directory.join("launch.json"),
                fs::Permissions::from_mode(0o644),
            )
            .unwrap(),
            3 => {
                let mut wrong = p.digest;
                wrong[0] ^= 1;
                assert!(Prepared::load(&p.directory, wrong).is_err());
                continue;
            }
            _ => {
                let moved = f.root.join("moved");
                fs::rename(&p.directory, &moved).unwrap();
                symlink(&moved, &p.directory).unwrap();
                assert!(p.consume().is_err());
            }
        }
        assert!(Prepared::load(&p.directory, p.digest).is_err());
    }
}
#[test]
fn actual_input_hash_size_and_symlink_fail_before_creating_bundle() {
    for failure in 0..3 {
        let f = Fixture::new();
        match failure {
            0 => fs::write(&f.launch.workload.input_file, b"wrong literal input").unwrap(),
            1 => {
                let mut file = OpenOptions::new().write(true).open(&f.input).unwrap();
                file.write_all(&vec![b' '; MAX_JSON + 1]).unwrap();
            }
            _ => {
                let original = f.root.join("original.json");
                fs::rename(&f.input, &original).unwrap();
                symlink(original, &f.input).unwrap();
            }
        }
        let out = f.root.join("refused");
        assert!(Prepared::prepare(&f.input, &out).is_err());
        assert!(!out.exists());
    }
}

#[test]
fn actual_evidence_is_exclusive_bounded_and_cannot_overwrite_bundle_or_marker() {
    let f = Fixture::new();
    let p = f.prepare();
    p.consume().unwrap();
    p.record("evidence-node0-0001.json", b"literal observation")
        .unwrap();
    assert_eq!(
        fs::read(p.directory.join("evidence-node0-0001.json")).unwrap(),
        b"literal observation"
    );
    assert!(p.record("evidence-node0-0001.json", b"overwrite").is_err());
    for name in [
        "launch.json",
        "consumed",
        "evidence-",
        "evidence-../outside",
        "evidence-bad\nname",
    ] {
        assert!(p.record(name, b"forbidden").is_err());
    }
    assert!(p
        .record("evidence-too-large", &vec![0; MAX_INPUT + 1])
        .is_err());
    symlink(&f.input, p.directory.join("evidence-symlink")).unwrap();
    assert!(p.record("evidence-symlink", b"forbidden").is_err());
}
