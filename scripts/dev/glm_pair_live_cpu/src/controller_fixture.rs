// SPDX-License-Identifier: AGPL-3.0-only
//! Literal recipes for real root-operated CPU Docker qualification only.
//! No container IDs, sessions, tickets or Model capabilities are manufactured.
use crate::{identity, wire};
use anyhow::{ensure, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

pub(crate) fn environment(mode: &str, delay: &str) -> Vec<(String, String)> {
    let mut values = vec![
        ("ATLAS_GLM_PAIR_FD", "3"),
        ("ATLAS_PAIR_CPU_DELAY_MS", delay),
        ("ATLAS_PAIR_CPU_MODE", mode),
        ("PATH", "/usr/bin:/bin"),
        ("ATLAS_EP_PROTOCOL", "v2"),
        ("ATLAS_GLM_MTP_HIDDEN_TRACE", "0"),
        ("ATLAS_GLM_MTP_REPAIR", "0"),
        ("ATLAS_GLM_MTP_BATCHED_PREFILL", "1"),
        ("ATLAS_GLM_MTP_DISTRIBUTED", "1"),
        ("ATLAS_GLM_MTP_ALL_GATHER", "1"),
        ("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0"),
        ("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1"),
    ];
    if mode == "registered-drain-controller" {
        values.extend([
            ("ATLAS_PAIR_CPU_HTTP_ADDR", "127.0.0.1:18761"),
            ("ATLAS_PAIR_CPU_HTTP_MODEL", "glm-pair-cpu"),
            ("ATLAS_PAIR_CPU_WAIT_MS", "60000"),
        ]);
    }
    let mut values: Vec<_> = values
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    values.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    values
}
fn absolute(text: &str) -> Result<PathBuf> {
    let path = PathBuf::from(text);
    ensure!(
        path.is_absolute() && text.len() <= 4096 && !text.contains('\0'),
        "absolute bounded fixture path required"
    );
    ensure!(
        text[1..].split('/').all(|s| !matches!(s, "" | "." | "..")),
        "canonical fixture path required"
    );
    Ok(path)
}
fn directory(path: &Path, private: bool) -> Result<()> {
    let m = std::fs::symlink_metadata(path)?;
    ensure!(
        m.is_dir() && m.uid() == 0 && m.gid() == 0 && m.mode() & 0o022 == 0,
        "root-owned real fixture directory required"
    );
    ensure!(
        !private || m.mode() & 0o7777 == 0o700,
        "private fixture directory requires0700"
    );
    ensure!(
        std::fs::canonicalize(path)? == path,
        "fixture directory path aliases are unsupported"
    );
    Ok(())
}
fn elf(path: &Path) -> Result<wire::Digest> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)?;
    let m = file.metadata()?;
    ensure!(
        m.uid() == 0 && m.gid() == 0 && m.mode() & 0o022 == 0,
        "root-owned immutable fixture ELF required"
    );
    let end = identity::boot_time_ms()?
        .checked_add(30000)
        .context("ELF deadline overflow")?;
    Ok(identity::PinnedExecutable::from_file(file, 512 * 1024 * 1024, end)?.digest())
}
pub(crate) fn recipes(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == 6 && unsafe { libc::geteuid() } == 0 && unsafe { libc::getegid() } == 0,
        "controller-recipes IMAGEHEX ABSGUARD ABSSERVER ABSASSETS ABSCOORD ABSOUT requires root"
    );
    ensure!(
        args[0].len() == 64
            && args[0]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "canonical full image digest required"
    );
    let image: wire::Digest = crate::unhex(&args[0])?
        .try_into()
        .map_err(|_| anyhow::anyhow!("image digest length"))?;
    ensure!(image != [0; 32], "zero image digest");
    let guard = absolute(&args[1])?;
    let server = absolute(&args[2])?;
    let assets = absolute(&args[3])?;
    let coord = absolute(&args[4])?;
    let output = absolute(&args[5])?;
    directory(&assets, false)?;
    directory(&coord, true)?;
    ensure!(
        std::fs::read_dir(&coord)?.next().is_none(),
        "fresh empty coordination directory required"
    );
    ensure!(
        guard.starts_with(&assets) && server.starts_with(&assets),
        "both actual ELFs must be in pinned assets bind"
    );
    let guard_digest = elf(&guard)?;
    let server_digest = elf(&server)?;
    let environment = environment("registered-drain-controller", "0");
    std::fs::DirBuilder::new().mode(0o700).create(&output)?;
    directory(&output, true)?;
    for rank in 0..2 {
        let mut mounts = vec![
            wire::Mount {
                source: assets.to_str().unwrap().into(),
                destination: assets.to_str().unwrap().into(),
                read_only: true,
                propagation: "rprivate".into(),
            },
            wire::Mount {
                source: coord.to_str().unwrap().into(),
                destination: "/run/atlas-cpu".into(),
                read_only: false,
                propagation: "rprivate".into(),
            },
            // Prepared::prepare replaces only this writable bind with the
            // genuinely fresh session/rank directory before any Docker create.
            wire::Mount {
                source: format!("/run/atlas-cpu-unprepared/rank{rank}"),
                destination: "/run/atlas-pair".into(),
                read_only: false,
                propagation: "rprivate".into(),
            },
        ];
        mounts.sort_unstable_by(|a, b| a.destination.cmp(&b.destination));
        let recipe = wire::Recipe {
            argv: vec![
                server.to_str().unwrap().into(),
                "--consumer-registered".into(),
            ],
            environment: environment.clone(),
            mounts,
            image_digest: image,
            guard_elf_digest: guard_digest,
            server_elf_digest: server_digest,
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
                memory: 2 << 30,
                swap: 2 << 30,
                cpuset: "0,1".into(),
                shm: 64 << 20,
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
        };
        let bytes = recipe.encode()?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(output.join(format!("rank{rank}.recipe.bin")))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    File::open(output)?.sync_all()?;
    Ok(())
}
