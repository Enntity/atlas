// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed local Engine API transport. Never pulls, builds, or removes containers.
use super::{error, Metadata};
use crate::process::{Limits, Process, Spec};
use atlas_glm_pair_io::identity::{boot_time_ms, PinnedExecutable};
use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

pub(super) const MAX_ELF: u64 = 1_073_741_824;
pub(super) fn executable(
    path: &Path,
    expected: [u8; 32],
    deadline: u64,
) -> io::Result<(File, PinnedExecutable)> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    let m = file.metadata()?;
    if m.uid() != 0 || m.gid() != 0 || m.mode() & 0o022 != 0 || m.mode() & 0o111 == 0 {
        return Err(error(
            "node executable must be root-owned and non-writable by others",
        ));
    }
    let pin = PinnedExecutable::from_file(file.try_clone()?, MAX_ELF, deadline)?;
    if pin.digest() != expected {
        return Err(error("node executable digest mismatch"));
    }
    Ok((file, pin))
}
pub(super) fn self_check(expected: [u8; 32]) -> io::Result<()> {
    // /proc/self/exe is the kernel's executable link, not caller pathname input.
    let file = File::open("/proc/self/exe")?;
    let m = file.metadata()?;
    if m.uid() != 0 || m.gid() != 0 || m.mode() & 0o022 != 0 {
        return Err(error("node supervisor ownership/mode"));
    }
    let pin = PinnedExecutable::from_file(
        file,
        MAX_ELF,
        boot_time_ms()?
            .checked_add(30_000)
            .ok_or_else(|| error("deadline overflow"))?,
    )?;
    if pin.digest() != expected {
        return Err(error("running supervisor SHA256 mismatch"));
    }
    Ok(())
}
fn socket_identity() -> io::Result<(u64, u64)> {
    // Native Docker's documented /var/run socket alias is admitted explicitly.
    if std::fs::canonicalize("/var/run")? != Path::new("/run") {
        return Err(error("unexpected Docker runtime-directory alias"));
    }
    let m = std::fs::symlink_metadata("/var/run/docker.sock")?;
    if m.uid() != 0 || m.mode() & libc::S_IFMT != libc::S_IFSOCK || m.mode() & 0o002 != 0 {
        return Err(error("Docker socket type, owner or world-write policy"));
    }
    let curl = std::fs::symlink_metadata("/usr/bin/curl")?;
    if !curl.is_file() || curl.uid() != 0 || curl.gid() != 0 || curl.mode() & 0o022 != 0 {
        return Err(error("fixed curl executable ownership"));
    }
    Ok((m.dev(), m.ino()))
}
pub(super) fn request(
    metadata: &Metadata,
    method: &str,
    endpoint: &str,
    body: Vec<u8>,
    accepted: u16,
) -> io::Result<Vec<u8>> {
    let socket = socket_identity()?;
    let policy = metadata.policy.to_wire()?;
    let mut args: Vec<std::ffi::OsString> = [
        "--disable",
        "--silent",
        "--show-error",
        "--noproxy",
        "*",
        "--unix-socket",
        "/var/run/docker.sock",
        "--max-time",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    args.push(
        format!(
            "{}.{:03}",
            metadata.command_ms / 1000,
            metadata.command_ms % 1000
        )
        .into(),
    );
    args.extend([
        "--request".into(),
        method.into(),
        "--header".into(),
        "Content-Type: application/json".into(),
        "--write-out".into(),
        "\n%{http_code}".into(),
    ]);
    if !body.is_empty() {
        args.extend(["--data-binary".into(), "@-".into()]);
    }
    args.push(format!("http://localhost/v1.48{endpoint}").into());
    let mut child = Process::spawn(
        Spec {
            program: "/usr/bin/curl".into(),
            args,
            env: vec![],
            stdin: body,
        },
        Limits {
            timeout_ms: metadata.command_ms,
            stdout_bytes: 1_048_576,
            stderr_bytes: 65_536,
            stdin_bytes: 1_048_576,
            bytes_per_turn: 65_536,
        },
        boot_time_ms()?,
    )?;
    let result = (|| {
        let mut output = Vec::new();
        loop {
            let p = child.poll(boot_time_ms()?)?;
            output.extend(p.stdout);
            if p.done {
                if !p.exit.is_some_and(|s| s.success()) {
                    return Err(error("fixed Docker curl command failed"));
                }
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(policy.poll.min(5)));
        }
        if socket_identity()? != socket {
            return Err(error("Docker socket replaced during command"));
        }
        let split = output
            .iter()
            .rposition(|&b| b == b'\n')
            .ok_or_else(|| error("missing Docker HTTP status"))?;
        let status = std::str::from_utf8(&output[split + 1..])
            .ok()
            .and_then(|s| s.parse::<u16>().ok());
        if status != Some(accepted) {
            return Err(error("Docker HTTP status differs"));
        }
        output.truncate(split);
        Ok(output)
    })();
    if result.is_err() {
        let until = boot_time_ms()?
            .checked_add(policy.reap)
            .ok_or_else(|| error("reap deadline overflow"))?;
        child.abort()?;
        while child.reap()?.is_none() {
            if boot_time_ms()? >= until {
                return Err(error("Docker curl direct child not reaped by deadline"));
            }
            std::thread::sleep(std::time::Duration::from_millis(policy.poll.min(5)));
        }
    }
    result
}
pub(super) fn relay(metadata: &Metadata, session: &str, rank: u8) -> io::Result<()> {
    let policy = metadata.policy.to_wire()?;
    let until = boot_time_ms()?
        .checked_add(metadata.command_ms)
        .ok_or_else(|| error("relay exec deadline"))?;
    let (file, pin) = executable(
        &metadata.relay_path,
        crate::docker::parse_id(&metadata.relay_sha256)?,
        until,
    )?;
    let argv = vec![
        metadata
            .relay_path
            .to_str()
            .ok_or_else(|| error("non-UTF8 relay path"))?
            .to_owned(),
        "--session".into(),
        session.into(),
        "--rank".into(),
        rank.to_string(),
        "--connect-ms".into(),
        metadata.command_ms.to_string(),
        "--frame-ms".into(),
        policy.frame.to_string(),
        "--campaign-ms".into(),
        policy.campaign.to_string(),
        "--poll-ms".into(),
        policy.poll.to_string(),
    ];
    let strings = argv
        .iter()
        .map(|s| CString::new(s.as_str()).map_err(io::Error::from))
        .collect::<io::Result<Vec<_>>>()?;
    let mut pointers: Vec<_> = strings.iter().map(|s| s.as_ptr()).collect();
    pointers.push(std::ptr::null());
    pin.revalidate()?;
    atlas_glm_pair_io::identity::check_deadline(until)?;
    unsafe {
        libc::fexecve(
            file.as_raw_fd(),
            pointers.as_ptr(),
            [std::ptr::null()].as_ptr(),
        );
    }
    Err(io::Error::last_os_error())
}
