// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit bounded-owner native recipe materialization; never creates containers.
use atlas_glm_pair_wire as wire;
use serde_json::{Map, Value};
use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Component, Path};

fn error(s: &str) -> io::Error {
    io::Error::other(s.to_owned())
}
fn object<'a>(v: &'a Value, keys: &[&str]) -> io::Result<&'a Map<String, Value>> {
    let o = v.as_object().ok_or_else(|| error("JSON object required"))?;
    if o.len() != keys.len() || keys.iter().any(|k| !o.contains_key(*k)) {
        return Err(error("missing or unknown explicit JSON fields"));
    }
    Ok(o)
}
fn text(v: &Value) -> io::Result<&str> {
    v.as_str()
        .filter(|s| !s.is_empty() && !s.contains('\0'))
        .ok_or_else(|| error("nonempty literal string required"))
}
fn absolute(s: &str) -> io::Result<&Path> {
    let path = Path::new(s);
    if !path.is_absolute()
        || s == "/"
        || s.len() > 4096
        || path
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
        || s[1..]
            .split('/')
            .any(|c| c.is_empty() || c == "." || c == "..")
    {
        return Err(error("canonical absolute path required"));
    }
    Ok(path)
}
fn digest(v: &Value) -> io::Result<wire::Digest> {
    let s = text(v)?;
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(error("full lowercase SHA256 required, without prefix"));
    }
    let mut out = [0; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(io::Error::other)?;
    }
    if out == [0; 32] {
        return Err(error("zero digest refused"));
    }
    Ok(out)
}
fn array(v: &Value) -> io::Result<&Vec<Value>> {
    v.as_array().ok_or_else(|| error("explicit array required"))
}
fn capacity_flag(argv: &[String], flag: &str, owners: u16) -> io::Result<()> {
    let prefix = format!("{flag}=");
    let expected = format!("{flag}={owners}");
    let supplied: Vec<_> = argv
        .iter()
        .filter(|arg| arg.as_str() == flag || arg.starts_with(&prefix))
        .collect();
    if supplied.len() != 1 || supplied[0] != &expected {
        return Err(error(
            "exact single CLI capacity flag must equal explicit owner_capacity",
        ));
    }
    Ok(())
}
fn recipe(v: &Value, rank: u8, root: &Map<String, Value>) -> io::Result<wire::Recipe> {
    let n = object(v, &["rank", "image_sha256", "argv", "environment"])?;
    if n["rank"].as_u64() != Some(u64::from(rank)) {
        return Err(error("ordered ranks0,1 required"));
    }
    let argv = array(&n["argv"])?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| error("argv strings required"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let owners = root["owner_capacity"]
        .as_u64()
        .filter(|n| (2..=8).contains(n))
        .ok_or_else(|| error("explicit owner_capacity must be between 2 and 8"))?
        as u16;
    capacity_flag(&argv, "--max-batch-size", owners)?;
    capacity_flag(&argv, "--max-num-seqs", owners)?;
    let mut environment = Vec::new();
    for entry in array(&n["environment"])? {
        let pair = array(entry)?;
        if pair.len() != 2 {
            return Err(error("environment entries require key/value pairs"));
        }
        let key = text(&pair[0])?;
        if key.contains('=') {
            return Err(error("invalid environment key"));
        }
        let value = pair[1]
            .as_str()
            .ok_or_else(|| error("environment value string required"))?;
        environment.push((key.to_owned(), value.to_owned()));
    }
    environment.sort_by(|a, b| a.0.cmp(&b.0));
    // The canonical encoder rejects duplicate keys; no environment merging or
    // inferred model flags occurs here. Runtime recipe validation remains real.
    Ok(wire::Recipe {
        argv,
        environment,
        image_digest: digest(&n["image_sha256"])?,
        guard_elf_digest: digest(&root["guard_elf_sha256"])?,
        server_elf_digest: digest(&root["server_elf_sha256"])?,
        rank,
        world: 2,
        mounts: vec![
            wire::Mount {
                source: format!("/run/atlas-glm-pairs/prepare/rank{rank}"),
                destination: "/run/atlas-pair".into(),
                read_only: false,
                propagation: "rprivate".into(),
            },
            wire::Mount {
                source: text(&root["weights_host_path"])?.into(),
                destination: "/var/tmp/models/glm53-flash-nvfp4".into(),
                read_only: true,
                propagation: "rprivate".into(),
            },
        ],
        profile: wire::Profile {
            tp: 2,
            ep: 2,
            ep_protocol: 2,
            max_sequences: owners,
            context: 2044,
            prefill: 1024,
            drafts: 4,
            eager: true,
            kv_format: 1,
            cold_min: 2,
            cold_max: 1024,
        },
        resources: wire::Resources {
            memory: 114_u64 << 30,
            swap: 114_u64 << 30,
            cpuset: "0-19".into(),
            shm: 1 << 30,
            device_requests: vec![wire::DeviceRequest {
                driver: String::new(),
                count: -1,
                device_ids: vec![],
                capabilities: vec![vec!["gpu".into()]],
                options: vec![],
            }],
            devices: vec![wire::DeviceMapping {
                path_on_host: "/dev/infiniband".into(),
                path_in_container: "/dev/infiniband".into(),
                cgroup_permissions: "rwm".into(),
            }],
            security_options: vec!["label=disable".into(), "seccomp=unconfined".into()],
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
    })
}
fn run() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 1 {
        return Err(error("usage: glm-release-recipe-generator INPUT.json"));
    }
    let mut bytes = Vec::new();
    File::open(&args[0])?
        .take(200 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 200 * 1024 {
        return Err(error("JSON input exceeds200KiB"));
    }
    let value: Value = serde_json::from_slice(&bytes)?;
    let input = object(
        &value,
        &[
            "output_directory",
            "weights_host_path",
            "server_elf_sha256",
            "guard_elf_sha256",
            "owner_capacity",
            "ranks",
        ],
    )?;
    let out = absolute(text(&input["output_directory"])?)?;
    absolute(text(&input["weights_host_path"])?)?;
    let ranks = array(&input["ranks"])?;
    if ranks.len() != 2 {
        return Err(error("exactly two rank records required"));
    }
    let recipes = [recipe(&ranks[0], 0, input)?, recipe(&ranks[1], 1, input)?];
    let encoded = recipes
        .each_ref()
        .map(|r| r.encode().map_err(io::Error::other));
    let [a, b] = encoded;
    let encoded = [a?, b?];
    // Existing directories and files are never overwritten or removed.
    DirBuilder::new().mode(0o700).create(out)?;
    File::open(out.parent().ok_or_else(|| error("missing output parent"))?)?.sync_all()?;
    for (rank, bytes) in encoded.iter().enumerate() {
        let path = out.join(format!("rank{rank}.recipe.bin"));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        let digest = wire::recipe_digest(bytes).map_err(io::Error::other)?;
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        println!(
            "rank={rank} recipe_domain_digest={hex} bytes={} path={}",
            bytes.len(),
            path.display()
        );
    }
    File::open(out)?.sync_all()?;
    // runc is the controller's fixed deployment runtime, not a Recipe field.
    println!("runtime=runc; container IDs are not generated by this tool");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("recipe generator: {e}");
        std::process::exit(1);
    }
}
