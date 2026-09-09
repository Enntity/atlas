// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sample() -> Recipe {
    Recipe {
        argv: vec!["/spark".into(), "serve".into()],
        environment: vec![
            ("ATLAS_GLM_PAIR_FD".into(), "3".into()),
            ("PATH".into(), "/usr/bin:/bin".into()),
        ],
        mounts: vec![Mount {
            source: "/session".into(),
            destination: "/run/atlas-pair".into(),
            read_only: false,
            propagation: "rprivate".into(),
        }],
        image_digest: [1; 32],
        guard_elf_digest: [2; 32],
        server_elf_digest: [3; 32],
        rank: 1,
        world: 2,
        profile: Profile {
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
        resources: Resources {
            memory: 114 << 30,
            swap: 114 << 30,
            cpuset: "0-3".into(),
            shm: 1 << 30,
            device_requests: vec![DeviceRequest {
                driver: "nvidia".into(),
                count: -1,
                device_ids: vec![],
                capabilities: vec![vec!["gpu".into()]],
                options: vec![],
            }],
            ulimits: vec![Ulimit {
                name: "memlock".into(),
                soft: -1,
                hard: -1,
            }],
            cap_add: vec![],
            cap_drop: vec!["ALL".into()],
            uid: 0,
            gid: 0,
            network_mode: "host".into(),
            pid_mode: "private".into(),
            restart_policy: "no".into(),
            init: false,
            no_new_privileges: true,
            ipc_mode: "private".into(),
        },
    }
}

// Independently assembled schema v1 bytes, not the production encoder.
fn expected() -> Vec<u8> {
    fn s(b: &mut Vec<u8>, value: &str) {
        b.extend_from_slice(&(value.len() as u32).to_be_bytes());
        b.extend_from_slice(value.as_bytes());
    }
    fn n(b: &mut Vec<u8>, value: u16) {
        b.extend_from_slice(&value.to_be_bytes());
    }
    let mut b = vec![];
    n(&mut b, 1);
    n(&mut b, 2);
    s(&mut b, "/spark");
    s(&mut b, "serve");
    n(&mut b, 2);
    s(&mut b, "ATLAS_GLM_PAIR_FD");
    s(&mut b, "3");
    s(&mut b, "PATH");
    s(&mut b, "/usr/bin:/bin");
    n(&mut b, 1);
    s(&mut b, "/session");
    s(&mut b, "/run/atlas-pair");
    b.push(0);
    s(&mut b, "rprivate");
    b.extend([1; 32]);
    b.extend([2; 32]);
    b.extend([3; 32]);
    b.extend([1, 2]);
    b.extend([2, 2, 2]);
    n(&mut b, 2);
    for value in [2044u32, 1024] {
        b.extend(value.to_be_bytes());
    }
    b.extend([4, 1, 1]);
    for value in [2u32, 1024] {
        b.extend(value.to_be_bytes());
    }
    for value in [114u64 << 30, 114 << 30] {
        b.extend(value.to_be_bytes());
    }
    s(&mut b, "0-3");
    b.extend((1u64 << 30).to_be_bytes());
    n(&mut b, 1);
    s(&mut b, "nvidia");
    b.extend((-1i64).to_be_bytes());
    n(&mut b, 0);
    n(&mut b, 1);
    n(&mut b, 1);
    s(&mut b, "gpu");
    n(&mut b, 0);
    n(&mut b, 1);
    s(&mut b, "memlock");
    b.extend((-1i64).to_be_bytes());
    b.extend((-1i64).to_be_bytes());
    n(&mut b, 0);
    n(&mut b, 1);
    s(&mut b, "ALL");
    b.extend([0; 8]);
    s(&mut b, "host");
    s(&mut b, "private");
    s(&mut b, "no");
    b.extend([0, 1]);
    s(&mut b, "private");
    b
}

#[test]
fn actual_recipe_decode_matches_independent_full_schema() {
    let bytes = expected();
    let actual = Recipe::decode(&bytes).expect("valid complete explicit recipe");
    assert_eq!(actual, sample());
    assert_eq!(actual.encode().unwrap(), bytes);
    assert_eq!(actual.digest().unwrap(), recipe_digest(&bytes).unwrap());
}

#[test]
fn truncated_noncanonical_and_oversized_bytes_refuse() {
    let bytes = expected();
    for cut in 0..bytes.len() {
        assert!(Recipe::decode(&bytes[..cut]).is_err(), "cut{cut}");
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Recipe::decode(&bad).is_err());
    bad = bytes.clone();
    bad[1] = 2;
    assert!(Recipe::decode(&bad).is_err());
    bad = bytes.clone();
    bad[2..4].copy_from_slice(&65u16.to_be_bytes());
    assert!(Recipe::decode(&bad).is_err());
    bad = bytes.clone();
    bad[4..8].copy_from_slice(&4097u32.to_be_bytes());
    assert!(Recipe::decode(&bad).is_err());
    bad = bytes.clone();
    bad[8] = 0;
    assert!(Recipe::decode(&bad).is_err());
    bad = bytes.clone();
    bad[8] = 0xff;
    assert!(Recipe::decode(&bad).is_err());
    let marker = b"/run/atlas-pair";
    let mount_bool = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .unwrap()
        + marker.len();
    bad = bytes.clone();
    assert_eq!(bad[mount_bool], 0);
    bad[mount_bool] = 2;
    assert!(Recipe::decode(&bad).is_err());
    assert!(Recipe::decode(&vec![0; 65537]).is_err());
}

#[test]
fn every_profile_field_is_fixed_but_no_environment_values_are_invented() {
    let original = sample();
    for which in 0..13 {
        let mut v = original.clone();
        match which {
            0 => v.rank = 2,
            1 => v.world = 1,
            2 => v.profile.tp = 1,
            3 => v.profile.ep = 1,
            4 => v.profile.ep_protocol = 1,
            5 => v.profile.max_sequences = 1,
            6 => v.profile.context = 2048,
            7 => v.profile.prefill = 512,
            8 => v.profile.drafts = 3,
            9 => v.profile.eager = false,
            10 => v.profile.kv_format = 0,
            11 => v.profile.cold_min = 1,
            _ => v.profile.cold_max = 1025,
        }
        assert!(v.encode().is_err(), "field{which}");
    }
    let mut v = original.clone();
    // Data codec does not manufacture NCCL or reject an explicitly supplied
    // different environment value; later literal recipe admission owns that.
    v.environment[0].1 = "controller-must-reject-this-at-admission".into();
    assert_eq!(Recipe::decode(&v.encode().unwrap()).unwrap(), v);
    assert_ne!(v.digest().unwrap(), original.digest().unwrap());
    v = original;
    v.argv.push("serve".into()); // argv duplicates retain their meaning.
    assert_eq!(Recipe::decode(&v.encode().unwrap()).unwrap(), v);
}

#[test]
fn duplicates_order_and_all_explicit_collection_caps_refuse() {
    for which in 0..14 {
        let mut v = sample();
        match which {
            0 => v.environment.swap(0, 1),
            1 => v.environment.push(v.environment[1].clone()),
            2 => v.mounts.push(v.mounts[0].clone()),
            3 => v
                .resources
                .device_requests
                .push(v.resources.device_requests[0].clone()),
            4 => v.resources.ulimits.push(v.resources.ulimits[0].clone()),
            5 => v.resources.cap_drop.push("ALL".into()),
            6 => {
                v.resources.device_requests[0].options =
                    vec![("z".into(), "1".into()), ("a".into(), "2".into())]
            }
            7 => v.resources.device_requests[0].device_ids = vec!["1".into(), "0".into()],
            8 => {
                v.resources.device_requests[0].capabilities = vec![vec!["gpu".into(), "gpu".into()]]
            }
            9 => {
                v.resources.device_requests[0].capabilities =
                    vec![vec!["gpu".into()], vec!["gpu".into()]]
            }
            10 => v.resources.device_requests[0].count = -2,
            11 => v.resources.ulimits[0].soft = -2,
            12 => v.environment[0].0 = "lowercase".into(),
            _ => v.resources.ulimits[0].hard = -2,
        }
        assert!(v.encode().is_err(), "invalid{which}");
    }
    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("X{i:03}")).collect()
    }
    for which in 0..11 {
        let mut v = sample();
        match which {
            0 => v.argv = names(65),
            1 => v.environment = names(129).into_iter().map(|s| (s, "x".into())).collect(),
            2 => {
                v.mounts = names(33)
                    .into_iter()
                    .map(|s| Mount {
                        destination: s,
                        ..v.mounts[0].clone()
                    })
                    .collect()
            }
            3 => {
                v.resources.device_requests = names(9)
                    .into_iter()
                    .map(|s| DeviceRequest {
                        driver: s,
                        ..v.resources.device_requests[0].clone()
                    })
                    .collect()
            }
            4 => {
                v.resources.ulimits = names(33)
                    .into_iter()
                    .map(|s| Ulimit {
                        name: s,
                        soft: 0,
                        hard: 0,
                    })
                    .collect()
            }
            5 => v.resources.cap_add = names(65),
            6 => v.resources.cap_drop = names(65),
            7 => v.resources.device_requests[0].device_ids = names(9),
            8 => {
                v.resources.device_requests[0].capabilities =
                    names(9).into_iter().map(|s| vec![s]).collect()
            }
            9 => v.resources.device_requests[0].capabilities = vec![names(17)],
            _ => {
                v.resources.device_requests[0].options =
                    names(33).into_iter().map(|s| (s, "x".into())).collect()
            }
        }
        assert!(v.encode().is_err(), "cap{which}");
    }
    let mut v = sample();
    v.argv = vec!["x".repeat(4096); 16];
    assert!(v.validate().is_err()); // Strings fit individually, complete record does not.
    v = sample();
    v.argv[0] = "x".repeat(4097);
    assert!(v.encode().is_err());
    v = sample();
    v.environment[0].1 = "x\0y".into();
    assert!(v.encode().is_err());
    v = sample();
    v.argv.clear();
    assert!(v.encode().is_err());
    v = sample();
    v.environment.clear();
    assert!(v.encode().is_err());
}

#[test]
fn populated_resources_roundtrip_every_nested_field_without_defaults() {
    let mut v = sample();
    v.resources.device_requests[0].count = 0;
    v.resources.device_requests[0].device_ids = vec!["GPU-0".into(), "GPU-1".into()];
    v.resources.device_requests[0].capabilities =
        vec![vec!["compute".into(), "gpu".into()], vec!["utility".into()]];
    v.resources.device_requests[0].options =
        vec![("one".into(), "".into()), ("two".into(), "value".into())];
    v.resources.ulimits.push(Ulimit {
        name: "nofile".into(),
        soft: 1024,
        hard: 2048,
    });
    v.resources.cap_add = vec!["IPC_LOCK".into()];
    v.resources.uid = 123;
    v.resources.gid = 456;
    v.resources.init = true;
    v.resources.no_new_privileges = false;
    assert_eq!(Recipe::decode(&v.encode().unwrap()).unwrap(), v);
    assert_ne!(v.digest().unwrap(), sample().digest().unwrap());
}
