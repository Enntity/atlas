// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use serde_json::json;

fn recipe() -> wire::Recipe {
    wire::Recipe {
        argv: vec!["/opt/atlas/spark".into(), "serve".into()],
        environment: vec![("ATLAS_GLM_PAIR_FD".into(), "3".into())],
        mounts: vec![wire::Mount {
            source: "/run/atlas-glm-pairs/session/rank0".into(),
            destination: "/run/atlas-pair".into(),
            read_only: false,
            propagation: "rprivate".into(),
        }],
        image_digest: [1; 32],
        guard_elf_digest: [2; 32],
        server_elf_digest: [3; 32],
        rank: 0,
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
            devices: vec![wire::DeviceMapping {
                path_on_host: "/dev/infiniband".into(),
                path_in_container: "/dev/infiniband".into(),
                cgroup_permissions: "rwm".into(),
            }],
            security_options: vec!["label=disable".into(), "seccomp=unconfined".into()],
            memory: 114 << 30,
            swap: 114 << 30,
            cpuset: "0-19".into(),
            shm: 1 << 30,
            device_requests: vec![wire::DeviceRequest {
                driver: "nvidia".into(),
                count: -1,
                device_ids: vec![],
                capabilities: vec![vec!["gpu".into()]],
                options: vec![],
            }],
            ulimits: vec![wire::Ulimit {
                name: "memlock".into(),
                soft: -1,
                hard: -1,
            }],
            cap_add: vec!["IPC_LOCK".into(), "SYS_NICE".into()],
            cap_drop: vec![],
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

#[test]
fn create_request_uses_explicit_image_guard_and_recipe_resources() {
    let recipe = recipe();
    let actual = create_request(&recipe, "/opt/atlas/glm-pair-guard", "runc").unwrap();
    assert_eq!(actual["Image"], format!("sha256:{}", "01".repeat(32)));
    assert_eq!(actual["Entrypoint"], json!(["/opt/atlas/glm-pair-guard"]));
    assert_eq!(actual["Cmd"], json!(["--live"]));
    assert_eq!(actual["Env"], json!(["ATLAS_GLM_PAIR_FD=3"]));
    assert_eq!(actual["User"], "0:0");
    let host = &actual["HostConfig"];
    assert_eq!(host["Memory"], 114_u64 << 30);
    assert_eq!(host["MemorySwap"], host["Memory"]);
    assert_eq!(host["PidMode"], "");
    assert_eq!(host["IpcMode"], "private");
    assert_eq!(host["ShmSize"], 1_u64 << 30);
    assert_eq!(host["Privileged"], false);
    assert_eq!(host["AutoRemove"], false);
    assert_eq!(host["Init"], false);
    assert_eq!(host["Runtime"], "runc");
    assert_eq!(
        host["Devices"],
        json!([{"PathOnHost":"/dev/infiniband","PathInContainer":"/dev/infiniband","CgroupPermissions":"rwm"}])
    );
    assert_eq!(
        host["SecurityOpt"],
        json!([
            "label=disable",
            "no-new-privileges=true",
            "seccomp=unconfined"
        ])
    );
    assert_eq!(
        host["RestartPolicy"],
        json!({"Name":"no", "MaximumRetryCount":0})
    );
    assert_eq!(
        host["Ulimits"],
        json!([{"Name":"memlock","Soft":-1,"Hard":-1}])
    );
    assert_eq!(host["Mounts"][0]["BindOptions"]["Propagation"], "rprivate");
}

#[test]
fn launch_rejects_unsafe_or_ambiguous_recipe_resource_modes() {
    for which in 0..11 {
        let mut r = recipe();
        match which {
            0 => r.resources.swap += 1,
            1 => r.resources.memory = 0,
            2 => r.resources.pid_mode = "host".into(),
            3 => r.resources.init = true,
            4 => r.resources.restart_policy = "always".into(),
            5 => r.resources.ipc_mode = "host".into(),
            6 => r.resources.uid = 1,
            7 => r.resources.gid = 1,
            8 => r.resources.no_new_privileges = false,
            9 => r.mounts[0].source = "relative".into(),
            _ => r.mounts[0].destination = "/run/../run/atlas-pair".into(),
        }
        assert!(
            create_request(&r, "/guard", "runc").is_err(),
            "case {which}"
        );
    }
    assert!(create_request(&recipe(), "relative", "runc").is_err());
    assert!(create_request(&recipe(), "/guard", "nvidia").is_err());
    assert!(create_request(&recipe(), "/guard", "").is_err());
}

// Docker observation-shaped data, not a container/runtime qualification. The
// expected fields above are independently asserted; the adapter must compare
// every new observation and cannot trust this helper or a previous inspection.
fn observed(r: &wire::Recipe) -> Value {
    let mut request = create_request(r, "/guard", "runc").unwrap();
    let host = request
        .as_object_mut()
        .unwrap()
        .remove("HostConfig")
        .unwrap();
    json!({
        "Id":"04".repeat(32),"Image":format!("sha256:{}","01".repeat(32)),
        "Path":"/guard","Args":["--live"],"Config":request,"HostConfig":host,
        "RestartCount":0,
        "State":{"Status":"created","Running":false,"Paused":false,"Restarting":false,
            "OOMKilled":false,"Dead":false,"Pid":0,"ExitCode":0,"Error":""},
        "Mounts":[{"Type":"bind","Source":r.mounts[0].source,
            "Destination":"/run/atlas-pair","RW":true,"Propagation":"rprivate"}]
    })
}

#[test]
fn engine_omits_empty_tmpfs_but_cannot_hide_additional_mounts() {
    let r = recipe();
    let mut value = observed(&r);
    value["HostConfig"].as_object_mut().unwrap().remove("Tmpfs");
    assert_eq!(
        inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &value).unwrap(),
        0
    );
    value["HostConfig"]["Tmpfs"] = json!({"/extra":""});
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &value).is_err());
}

#[test]
fn engine_null_oom_disable_is_not_permission_to_disable_killer() {
    let r = recipe();
    let mut value = observed(&r);
    value["HostConfig"]["OomKillDisable"] = Value::Null;
    assert_eq!(
        inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &value).unwrap(),
        0
    );
    value["HostConfig"]["OomKillDisable"] = json!(true);
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &value).is_err());
    value["HostConfig"]
        .as_object_mut()
        .unwrap()
        .remove("OomKillDisable");
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &value).is_err());
}

#[test]
fn exact_inspection_requires_original_full_identity_and_matching_phase() {
    let r = recipe();
    let mut v = observed(&r);
    assert_eq!(
        inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &v).unwrap(),
        0
    );
    assert!(inspect(&r, "/guard", "runc", &[5; 32], Stage::Created, &v).is_err());
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Exited, &v).is_err());
    v["State"]["Status"] = json!("running");
    v["State"]["Running"] = json!(true);
    v["State"]["Pid"] = json!(1234);
    assert_eq!(
        inspect(&r, "/guard", "runc", &[4; 32], Stage::Running, &v).unwrap(),
        1234
    );
    v["State"]["Status"] = json!("exited");
    v["State"]["Running"] = json!(false);
    v["State"]["Pid"] = json!(0);
    assert_eq!(
        inspect(&r, "/guard", "runc", &[4; 32], Stage::Exited, &v).unwrap(),
        0
    );
    v["State"]["OOMKilled"] = json!(true);
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Exited, &v).is_err());
}

#[test]
fn inspect_refuses_configuration_drift_missing_fields_and_extra_authorities() {
    let r = recipe();
    let original = observed(&r);
    let cases = [
        ("/Image", json!(format!("sha256:{}", "09".repeat(32)))),
        (
            "/Config/Env",
            json!(["ATLAS_GLM_PAIR_FD=3", "LD_PRELOAD=/bad"]),
        ),
        ("/Config/Cmd", json!(["--live", "extra"])),
        ("/Config/User", json!("")),
        ("/HostConfig/Privileged", json!(true)),
        ("/HostConfig/Init", json!(true)),
        ("/HostConfig/AutoRemove", json!(true)),
        ("/HostConfig/Runtime", json!("other")),
        ("/HostConfig/DeviceCgroupRules", json!(["a *:* rwm"])),
        ("/HostConfig/GroupAdd", json!(["123"])),
        ("/HostConfig/PidMode", json!("host")),
        ("/HostConfig/IpcMode", json!("host")),
        ("/HostConfig/MemorySwap", json!(-1)),
        (
            "/HostConfig/Devices",
            json!([{"PathOnHost":"/dev/mem","PathInContainer":"/dev/mem","CgroupPermissions":"rwm"}]),
        ),
        (
            "/HostConfig/SecurityOpt",
            json!(["no-new-privileges=false"]),
        ),
        ("/HostConfig/Binds", json!(["/:/host"])),
        ("/HostConfig/Tmpfs", json!({"/extra":""})),
        ("/HostConfig/Ulimits", json!([])),
        ("/HostConfig/CapAdd", json!(["CAP_SYS_ADMIN"])),
        ("/State/Paused", json!(true)),
        ("/State/Restarting", json!(true)),
        ("/State/OOMKilled", json!(true)),
        ("/State/Dead", json!(true)),
        ("/State/ExitCode", json!(74)),
        ("/RestartCount", json!(1)),
        ("/Mounts/0/RW", json!(false)),
        ("/Mounts/0/Type", json!("volume")),
    ];
    for (path, value) in cases {
        let mut v = original.clone();
        *v.pointer_mut(path).unwrap() = value;
        assert!(
            inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &v).is_err(),
            "{path}"
        );
    }
    let mut v = original;
    v["HostConfig"].as_object_mut().unwrap().remove("Memory");
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &v).is_err());
}

#[test]
fn docker_omits_false_bind_readonly_but_never_true() {
    let r = recipe();
    let mut v = observed(&r);
    v["HostConfig"]["Mounts"][0]
        .as_object_mut()
        .unwrap()
        .remove("ReadOnly");
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &v).is_ok());
    v["HostConfig"]["Mounts"][0]["ReadOnly"] = json!(true);
    assert!(inspect(&r, "/guard", "runc", &[4; 32], Stage::Created, &v).is_err());
}

fn running_and_exited(r: &wire::Recipe) -> (Value, Value) {
    let mut running = observed(r);
    running["State"]["Status"] = json!("running");
    running["State"]["Running"] = json!(true);
    running["State"]["Pid"] = json!(1234);
    let mut exited = observed(r);
    exited["State"]["Status"] = json!("exited");
    (running, exited)
}

#[test]
fn observation_window_accepts_only_clean_forward_exit() {
    let r = recipe();
    let (running, exited) = running_and_exited(&r);
    for (before, after, expected) in [
        (&running, &running, (Stage::Running, 1234)),
        (&exited, &exited, (Stage::Exited, 0)),
        (&running, &exited, (Stage::Exited, 0)),
    ] {
        assert_eq!(
            observation_pair(&r, "/guard", &[4; 32], false, before, after).unwrap(),
            expected
        );
    }
}

#[test]
fn observation_window_never_hides_identity_health_or_reverse_transition() {
    let r = recipe();
    let (running, exited) = running_and_exited(&r);
    assert!(observation_pair(&r, "/guard", &[4; 32], false, &exited, &running).is_err());
    assert!(observation_pair(&r, "/guard", &[4; 32], true, &running, &exited).is_err());
    let mut replaced = running.clone();
    replaced["State"]["Pid"] = json!(5678);
    assert!(observation_pair(&r, "/guard", &[4; 32], false, &running, &replaced).is_err());
    for (path, value) in [
        ("/Id", json!("05".repeat(32))),
        ("/Image", json!(format!("sha256:{}", "06".repeat(32)))),
        ("/State/OOMKilled", json!(true)),
        ("/State/ExitCode", json!(74)),
        ("/State/Dead", json!(true)),
        ("/State/Restarting", json!(true)),
        ("/State/Paused", json!(true)),
        ("/RestartCount", json!(1)),
        ("/Config/Env", json!(["UNEXPECTED=1"])),
    ] {
        let mut bad_after = exited.clone();
        *bad_after.pointer_mut(path).unwrap() = value.clone();
        assert!(
            observation_pair(&r, "/guard", &[4; 32], false, &running, &bad_after).is_err(),
            "after {path}"
        );
        let mut bad_before = running.clone();
        *bad_before.pointer_mut(path).unwrap() = value;
        assert!(
            observation_pair(&r, "/guard", &[4; 32], false, &bad_before, &exited).is_err(),
            "before {path}"
        );
    }
}
