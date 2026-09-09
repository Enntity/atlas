// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use sha2::{Digest as _, Sha256};

fn populated() -> Recipe {
    let mut value = sample();
    value.resources.devices = vec![DeviceMapping {
        path_on_host: "/dev/infiniband".into(),
        path_in_container: "/dev/infiniband".into(),
        cgroup_permissions: "rwm".into(),
    }];
    value.resources.security_options = vec!["seccomp=unconfined".into()];
    value
}

#[test]
fn actual_v2_recipe_retains_rdma_security_and_versioned_digest() {
    let value = populated();
    let bytes = value.encode().unwrap();
    assert_eq!(&bytes[..2], &2u16.to_be_bytes());
    assert_eq!(Recipe::decode(&bytes).unwrap(), value);
    let mut expected = Sha256::new();
    expected.update(b"atlas.glm.pair.recipe.v2\0");
    expected.update((bytes.len() as u32).to_be_bytes());
    expected.update(&bytes);
    let expected: crate::Digest = expected.finalize().into();
    assert_eq!(value.digest().unwrap(), expected);
    let mut old = bytes.clone();
    old[..2].copy_from_slice(&1u16.to_be_bytes());
    assert!(Recipe::decode(&old).is_err());
    for end in 0..bytes.len() {
        assert!(Recipe::decode(&bytes[..end]).is_err());
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert!(Recipe::decode(&trailing).is_err());
}

#[test]
fn actual_v2_fields_reject_duplicate_noncanonical_or_second_nnp_authority() {
    for permissions in ["", "wr", "rr", "rx", "mrw"] {
        let mut value = populated();
        value.resources.devices[0].cgroup_permissions = permissions.into();
        assert!(value.encode().is_err(), "permissions {permissions}");
    }
    for option in [
        "no-new-privileges",
        "no-new-privileges=true",
        "no-new-privileges=false",
        "no-new-privileges:1",
    ] {
        let mut value = populated();
        value.resources.security_options = vec![option.into()];
        assert!(value.encode().is_err(), "duplicate authority {option}");
    }
    for case in 0..8 {
        let mut value = populated();
        match case {
            0 => value
                .resources
                .devices
                .push(value.resources.devices[0].clone()),
            1 => value.resources.devices[0].path_on_host = "".into(),
            2 => value.resources.devices[0].path_in_container = "bad\0path".into(),
            3 => value
                .resources
                .security_options
                .push("seccomp=unconfined".into()),
            4 => value
                .resources
                .security_options
                .push("apparmor=unconfined".into()),
            5 => value.resources.security_options = vec!["x".repeat(4097)],
            6 => {
                value.resources.devices = (0..33)
                    .map(|i| DeviceMapping {
                        path_in_container: format!("/dev/{i:02}"),
                        ..value.resources.devices[0].clone()
                    })
                    .collect()
            }
            _ => {
                value.resources.security_options =
                    (0..33).map(|i| format!("key{i:02}=value")).collect()
            }
        }
        assert!(value.encode().is_err(), "case {case}");
    }
    let base = populated();
    for permissions in ["r", "w", "m", "rw", "rm", "wm", "rwm"] {
        let mut value = base.clone();
        value.resources.devices[0].cgroup_permissions = permissions.into();
        assert_eq!(Recipe::decode(&value.encode().unwrap()).unwrap(), value);
    }
}
