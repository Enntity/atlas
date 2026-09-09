// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed root-only node verbs; no pull/build, shell command, or name-based kill.
use super::{docker, prepared, wire};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::path::PathBuf;
#[path = "node_commands.rs"]
mod commands;
#[path = "node_engine.rs"]
mod engine;
#[path = "node_files.rs"]
mod files;
#[path = "node_proc.rs"]
mod proc;
const MAX_JSON: usize = 200 * 1024;
fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}
fn now() -> io::Result<u64> {
    atlas_glm_pair_io::identity::boot_time_ms()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub command_ms: u64,
    pub policy: prepared::PolicyConfig,
    pub relay_path: PathBuf,
    pub relay_sha256: String,
    pub guard_container_path: String,
}
impl Metadata {
    fn validate(&self) -> io::Result<()> {
        let policy = self.policy.to_wire()?;
        if self.command_ms == 0
            || self.command_ms > policy.frame
            || self.command_ms < policy.poll
            || !self.relay_path.is_absolute()
            || !std::path::Path::new(&self.guard_container_path).is_absolute()
            || self.relay_path.to_str().is_none()
            || self.relay_path.as_os_str().len() > 4096
            || self.guard_container_path.len() > 4096
        {
            return Err(error("invalid explicit node metadata"));
        }
        docker::parse_id(&self.relay_sha256)?;
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareInput {
    pub recipe_hex: String,
    pub metadata: Metadata,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeProcess {
    pub boot_id: [u8; 16],
    pub pid_namespace_device: u64,
    pub pid_namespace_inode: u64,
    pub guard_pid: u32,
    pub child_pid: u32,
    pub guard_start_ticks: u64,
    pub child_start_ticks: u64,
    pub host_guard_pid: u32,
    pub host_child_pid: u32,
    pub uid: u32,
    pub gid: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeObservation {
    pub inspect: serde_json::Value,
    pub process: Option<NodeProcess>,
    pub socket_ready: bool,
    pub mem_available_kib: u64,
    pub swap_used_kib: u64,
}
struct Command {
    verb: String,
    session: String,
    rank: u8,
    id: Option<wire::Digest>,
    self_hash: wire::Digest,
}
impl Command {
    fn parse(args: &[String]) -> io::Result<Self> {
        if args.len() != 5 && args.len() != 6 {
            return Err(error(
                "fixed node verb/session/rank/[ID]/self-hash required",
            ));
        }
        let with_id = match args[0].as_str() {
            "node-prepare" | "node-create" => false,
            "node-seal" | "node-start" | "node-observe" | "node-socket" | "node-relay"
            | "node-kill" => true,
            _ => return Err(error("unsupported node verb")),
        };
        if args.len() != 5 + usize::from(with_id)
            || args[args.len() - 2] != "--self-sha256"
            || !matches!(args[2].as_str(), "0" | "1")
        {
            return Err(error("noncanonical node arguments"));
        }
        docker::parse_id(&args[1])?;
        Ok(Self {
            verb: args[0].clone(),
            session: args[1].clone(),
            rank: args[2].as_bytes()[0] - b'0',
            id: if with_id {
                Some(docker::parse_id(&args[3])?)
            } else {
                None
            },
            self_hash: docker::parse_id(&args[args.len() - 1])?,
        })
    }
}
fn decode_hex(text: &str, max: usize) -> io::Result<Vec<u8>> {
    if text.len() > 2 * max
        || !text.len().is_multiple_of(2)
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(error("bounded lowercase recipe hex required"));
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|_| error("invalid hex")))
        .collect()
}
fn input(max: usize) -> io::Result<Vec<u8>> {
    let flags = unsafe { libc::fcntl(0, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(0, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let end = now()?
        .checked_add(30_000)
        .ok_or_else(|| error("stdin deadline"))?;
    let mut bytes = Vec::new();
    loop {
        if now()? >= end {
            return Err(error("node stdin deadline"));
        }
        let mut part = [0; 8192];
        match io::stdin().read(&mut part) {
            Ok(0) => return Ok(bytes),
            Ok(n) => {
                if n > max - bytes.len() {
                    return Err(error("node stdin bound"));
                }
                bytes.extend_from_slice(&part[..n]);
            }
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(2))
            }
            Err(e) => return Err(e),
        }
    }
}
pub fn run(args: &[String]) -> io::Result<()> {
    let command = Command::parse(args)?;
    if unsafe { libc::getuid() } != 0
        || unsafe { libc::geteuid() } != 0
        || unsafe { libc::getgid() } != 0
        || unsafe { libc::getegid() } != 0
    {
        return Err(error("node verbs require root identity"));
    }
    engine::self_check(command.self_hash)?;
    let value = if command.verb == "node-prepare" {
        commands::prepare(&command, &input(MAX_JSON)?)?
    } else {
        let loaded = commands::Loaded::open(&command)?;
        match command.verb.as_str() {
            "node-create" => loaded.create()?,
            "node-seal" => loaded.seal(&command, &input(288)?)?,
            "node-start" => loaded.start(&command)?,
            "node-observe" | "node-socket" => {
                serde_json::to_value(loaded.observation(&command, command.verb == "node-socket")?)?
            }
            "node-relay" => return loaded.relay(&command),
            "node-kill" => loaded.kill(&command)?,
            _ => return Err(error("unreachable node verb")),
        }
    };
    output(&value)
}

fn output(value: &serde_json::Value) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    if bytes.len() > 1_048_576 {
        return Err(error("node stdout bound"));
    }
    let flags = unsafe { libc::fcntl(1, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(1, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let end = now()?
        .checked_add(30_000)
        .ok_or_else(|| error("stdout deadline"))?;
    let mut written = 0;
    while written < bytes.len() {
        if now()? >= end {
            return Err(error("node stdout deadline"));
        }
        let until = (written + 8192).min(bytes.len());
        match io::stdout().write(&bytes[written..until]) {
            Ok(0) => return Err(error("zero node stdout write")),
            Ok(n) => written += n,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(2))
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "node_tests.rs"]
mod tests;
