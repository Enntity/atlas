// SPDX-License-Identifier: AGPL-3.0-only
//! Docker API v1.48 data boundary; pure conversion/validation, never host I/O.
use super::{error, io, wire};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    Created,
    Running,
    Exited,
}

pub(super) fn hex(value: &wire::Digest) -> String {
    value.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) fn parse_id(s: &str) -> io::Result<wire::Digest> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(error("full lowercase Docker identity required"));
    }
    let mut out = [0; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)
            .map_err(|_| error("invalid Docker identity"))?;
    }
    if out == [0; 32] {
        return Err(error("zero Docker identity"));
    }
    Ok(out)
}

fn absolute(path: &str) -> bool {
    path.starts_with('/')
        && path != "/"
        && !path.contains('\0')
        && path[1..].split('/').all(|s| !matches!(s, "" | "." | ".."))
}

fn admit(recipe: &wire::Recipe, guard: &str, runtime: &str) -> io::Result<()> {
    recipe.validate().map_err(io::Error::other)?;
    let r = &recipe.resources;
    if !absolute(guard)
        || !absolute(&recipe.argv[0])
        || runtime != "runc"
        || r.memory == 0
        || r.memory > 114 << 30
        || r.swap != r.memory
        || r.pid_mode != "private"
        || r.ipc_mode != "private"
        || r.network_mode != "host"
        || r.init
        || r.restart_policy != "no"
        || r.uid != 0
        || r.gid != 0
        || !r.no_new_privileges
        || r.shm == 0
        || r.shm > 1 << 30
    {
        return Err(error("unsupported or unsafe paired launch resources"));
    }
    for m in &recipe.mounts {
        if !absolute(&m.source) || !absolute(&m.destination) || m.propagation != "rprivate" {
            return Err(error("noncanonical paired bind mount"));
        }
    }
    if !recipe
        .mounts
        .iter()
        .any(|m| m.destination == "/run/atlas-pair" && !m.read_only)
    {
        return Err(error("missing writable private pair session bind"));
    }
    for d in &r.devices {
        if !absolute(&d.path_on_host)
            || !absolute(&d.path_in_container)
            || !d.path_on_host.starts_with("/dev/")
            || !d.path_in_container.starts_with("/dev/")
        {
            return Err(error("device mappings must name absolute device paths"));
        }
    }
    Ok(())
}

fn cap(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|s| format!("CAP_{}", s.strip_prefix("CAP_").unwrap_or(s)))
        .collect()
}

pub(super) fn create_request(
    recipe: &wire::Recipe,
    guard: &str,
    runtime: &str,
) -> io::Result<Value> {
    admit(recipe, guard, runtime)?;
    let r = &recipe.resources;
    let mut security = r.security_options.clone();
    security.push("no-new-privileges=true".into());
    security.sort();
    let requests: Vec<_> = r.device_requests.iter().map(|d| json!({
        "Driver":d.driver,"Count":d.count,"DeviceIDs":d.device_ids,
        "Capabilities":d.capabilities,"Options":d.options.iter().map(|(k,v)|(k.clone(),Value::String(v.clone()))).collect::<serde_json::Map<String,Value>>()
    })).collect();
    let devices: Vec<_> = r
        .devices
        .iter()
        .map(|d| {
            json!({
                "PathOnHost":d.path_on_host,"PathInContainer":d.path_in_container,
                "CgroupPermissions":d.cgroup_permissions
            })
        })
        .collect();
    let mounts: Vec<_> = recipe
        .mounts
        .iter()
        .map(|m| {
            json!({
                "Type":"bind","Source":m.source,"Target":m.destination,"ReadOnly":m.read_only,
                "BindOptions":{"Propagation":m.propagation}
            })
        })
        .collect();
    let ulimits: Vec<_> = r
        .ulimits
        .iter()
        .map(|u| json!({"Name":u.name,"Soft":u.soft,"Hard":u.hard}))
        .collect();
    Ok(json!({
        "Image":format!("sha256:{}",hex(&recipe.image_digest)),
        "Entrypoint":[guard],"Cmd":["--live"],"User":format!("{}:{}",r.uid,r.gid),
        "Env":recipe.environment.iter().map(|(k,v)|format!("{k}={v}")).collect::<Vec<_>>(),
        "WorkingDir":"/","Tty":false,"OpenStdin":false,"StdinOnce":false,
        "AttachStdin":false,"AttachStdout":false,"AttachStderr":false,
        "Healthcheck":{"Test":["NONE"]},"StopSignal":"SIGTERM",
        "HostConfig":{
            "Memory":r.memory,"MemorySwap":r.swap,"CpusetCpus":r.cpuset,"ShmSize":r.shm,
            "DeviceRequests":requests,"Devices":devices,"SecurityOpt":security,
            "Ulimits":ulimits,"CapAdd":cap(&r.cap_add),"CapDrop":cap(&r.cap_drop),
            "NetworkMode":r.network_mode,"PidMode":"","IpcMode":r.ipc_mode,
            "RestartPolicy":{"Name":"no","MaximumRetryCount":0},"Init":false,
            "Runtime":runtime,"Privileged":false,"AutoRemove":false,"ReadonlyRootfs":false,
            "PublishAllPorts":false,"Mounts":mounts,"Binds":[],"VolumesFrom":[],"Tmpfs":{},
            "DeviceCgroupRules":[],"GroupAdd":[],"PortBindings":{},"Links":[],
            "UsernsMode":"","UTSMode":"","CgroupParent":"","CgroupnsMode":"private",
            "MemoryReservation":0,"OomKillDisable":false,"CpuShares":0,"NanoCpus":0,
            "CpuPeriod":0,"CpuQuota":0,"CpuRealtimePeriod":0,"CpuRealtimeRuntime":0,
            "CpusetMems":"","PidsLimit":0
        }
    }))
}

pub(super) fn inspect(
    recipe: &wire::Recipe,
    guard: &str,
    runtime: &str,
    id: &wire::Digest,
    stage: Stage,
    observed: &Value,
) -> io::Result<u32> {
    let expected = create_request(recipe, guard, runtime)?;
    if *id == [0; 32]
        || observed["Id"] != hex(id)
        || observed["Image"] != expected["Image"]
        || observed["Path"] != guard
        || observed["Args"] != json!(["--live"])
        || observed["RestartCount"] != 0
    {
        return Err(error("Docker full identity or executable mismatch"));
    }
    let config = observed
        .get("Config")
        .and_then(Value::as_object)
        .ok_or_else(|| error("missing Docker Config"))?;
    for (key, want) in expected
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.as_str() != "HostConfig")
    {
        let actual = config
            .get(key)
            .ok_or_else(|| error("missing Docker configuration field"))?;
        if key == "Env" {
            if sorted(actual)? != sorted(want)? {
                return Err(error("Docker literal environment mismatch"));
            }
        } else if actual != want {
            return Err(io::Error::other(format!("Docker Config mismatch: {key}")));
        }
    }
    for key in ["Volumes", "OnBuild"] {
        if config.get(key).is_some_and(|v| !empty(v)) {
            return Err(error("unexpected inherited Docker configuration"));
        }
    }
    let host = observed
        .get("HostConfig")
        .and_then(Value::as_object)
        .ok_or_else(|| error("missing Docker HostConfig"))?;
    for (key, want) in expected["HostConfig"].as_object().unwrap() {
        // Moby declares Tmpfs with omitempty; an absent map means no tmpfs
        // entries. Keep every other expected field mandatory.
        if key == "Tmpfs" && !host.contains_key(key) && want == &json!({}) {
            continue;
        }
        let actual = host.get(key).ok_or_else(|| {
            io::Error::other(format!("missing Docker host resource field: {key}"))
        })?;
        if normalize(key, actual)? != normalize(key, want)? {
            return Err(io::Error::other(format!(
                "Docker HostConfig mismatch: {key}"
            )));
        }
    }
    // Unsupported non-default resource/security authorities cannot hide outside
    // the generated request. Empty/null forms below are Docker representations,
    // not invented values for required recipe fields.
    for key in [
        "Dns",
        "DnsOptions",
        "DnsSearch",
        "ExtraHosts",
        "StorageOpt",
        "Sysctls",
        "BlkioWeightDevice",
        "BlkioDeviceReadBps",
        "BlkioDeviceWriteBps",
        "BlkioDeviceReadIOps",
        "BlkioDeviceWriteIOps",
        "Annotations",
    ] {
        if host.get(key).is_some_and(|v| !empty(v)) {
            return Err(error("extra Docker resource authority"));
        }
    }
    for key in [
        "OomScoreAdj",
        "BlkioWeight",
        "CpuCount",
        "CpuPercent",
        "IOMaximumIOps",
        "IOMaximumBandwidth",
    ] {
        if host.get(key).is_some_and(|v| v != &json!(0)) {
            return Err(error("unsupported Docker resource tuning"));
        }
    }
    let mounts = observed
        .get("Mounts")
        .and_then(Value::as_array)
        .ok_or_else(|| error("missing Docker mount observations"))?;
    if mounts.len() != recipe.mounts.len() {
        return Err(error("extra or missing Docker mounts"));
    }
    let mut mounts: Vec<_> = mounts.iter().collect();
    mounts.sort_by_key(|v| v["Destination"].as_str().unwrap_or(""));
    for (actual, want) in mounts.into_iter().zip(&recipe.mounts) {
        if actual["Type"] != "bind"
            || actual["Source"] != want.source
            || actual["Destination"] != want.destination
            || actual["RW"] != !want.read_only
            || actual["Propagation"] != want.propagation
        {
            return Err(error("Docker observed bind mismatch"));
        }
    }
    let s = &observed["State"];
    for key in ["Paused", "Restarting", "OOMKilled", "Dead"] {
        if s[key] != false {
            return Err(error("unhealthy or incomplete Docker state"));
        }
    }
    let status = match stage {
        Stage::Created => "created",
        Stage::Running => "running",
        Stage::Exited => "exited",
    };
    let pid = s["Pid"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| error("invalid Docker host init PID"))?;
    if s["Status"] != status
        || s["Running"] != (stage == Stage::Running)
        || (pid > 0) != (stage == Stage::Running)
        || s["ExitCode"] != 0
        || s["Error"] != ""
    {
        return Err(error("Docker phase or final exit mismatch"));
    }
    Ok(pid)
}

fn empty(v: &Value) -> bool {
    v.is_null()
        || v.as_array().is_some_and(Vec::is_empty)
        || v.as_object().is_some_and(serde_json::Map::is_empty)
}
fn sorted(v: &Value) -> io::Result<Value> {
    let mut a: Vec<String> = v
        .as_array()
        .ok_or_else(|| error("expected Docker string array"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| error("invalid Docker string"))
        })
        .collect::<io::Result<_>>()?;
    a.sort();
    Ok(json!(a))
}
fn normalize(key: &str, value: &Value) -> io::Result<Value> {
    let mut v = value.clone();
    if matches!(
        key,
        "Binds"
            | "VolumesFrom"
            | "DeviceCgroupRules"
            | "GroupAdd"
            | "Links"
            | "CapAdd"
            | "CapDrop"
            | "Devices"
            | "DeviceRequests"
            | "Ulimits"
    ) && v.is_null()
    {
        v = json!([]);
    }
    if matches!(key, "Tmpfs" | "PortBindings") && v.is_null() {
        v = json!({});
    }
    if key == "PidsLimit" && v.is_null() {
        v = json!(0);
    }
    // Moby clears this pointer when disabling the OOM killer is unsupported
    // (notably cgroup v2). Null grants no disable authority; true still refuses.
    if key == "OomKillDisable" && v.is_null() {
        v = json!(false);
    }
    if matches!(key, "CapAdd" | "CapDrop" | "SecurityOpt") {
        let a = v
            .as_array()
            .ok_or_else(|| error("invalid Docker options"))?;
        let mut strings = Vec::new();
        for s in a {
            let s = s
                .as_str()
                .ok_or_else(|| error("invalid Docker option type"))?;
            strings.push(if key == "SecurityOpt" {
                // Engine stores the true spelling bare on some API versions.
                if s == "no-new-privileges" {
                    "no-new-privileges=true".into()
                } else {
                    s.to_owned()
                }
            } else {
                format!("CAP_{}", s.strip_prefix("CAP_").unwrap_or(s))
            });
        }
        strings.sort();
        v = json!(strings);
    }
    if key == "DeviceRequests" {
        for d in v
            .as_array_mut()
            .ok_or_else(|| error("invalid Docker GPU requests"))?
        {
            if d.get("DeviceIDs").is_some_and(Value::is_null) {
                d["DeviceIDs"] = json!([]);
            }
            if d.get("Options").is_some_and(Value::is_null) {
                d["Options"] = json!({});
            }
        }
    }
    if key == "Mounts" {
        for m in v
            .as_array_mut()
            .ok_or_else(|| error("invalid Docker bind requests"))?
        {
            if m["Type"] == "bind" && m.get("ReadOnly").is_none() {
                m["ReadOnly"] = json!(false);
            }
        }
    }
    Ok(v)
}

#[cfg(test)]
#[path = "docker_tests.rs"]
mod tests;
