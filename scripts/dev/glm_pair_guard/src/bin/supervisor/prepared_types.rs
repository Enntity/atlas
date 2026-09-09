// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    pub version: u16,
    pub policy: PolicyConfig,
    pub controller: ControllerConfig,
    pub ssh: Ssh,
    pub nodes: [Node; 2],
    pub readiness: Readiness,
    pub workload: Workload,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    pub startup: u64,
    pub lease: u64,
    pub challenge: u64,
    pub frame: u64,
    pub campaign: u64,
    pub poll: u64,
    pub reap: u64,
    pub child_handshake: u64,
    pub quiescent_wait: u64,
    pub exit: u64,
}
impl PolicyConfig {
    pub fn to_wire(&self) -> io::Result<wire::Policy> {
        let p = wire::Policy {
            startup: self.startup,
            lease: self.lease,
            challenge: self.challenge,
            frame: self.frame,
            campaign: self.campaign,
            poll: self.poll,
            reap: self.reap,
            child_handshake: self.child_handshake,
            quiescent_wait: self.quiescent_wait,
            exit: self.exit,
        };
        p.validate().map_err(io::Error::other)?;
        Ok(p)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControllerConfig {
    pub readiness_ms: u64,
    pub workload_ms: u64,
    pub drain_ms: u64,
    pub status_max_age_ms: u64,
    pub command_ms: u64,
    pub status_interval_ms: u64,
    pub cleanup_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ssh {
    pub executable: PathBuf,
    pub key: PathBuf,
    pub known_hosts: PathBuf,
    pub environment: Vec<(String, String)>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub rank: u8,
    pub destination: String,
    pub supervisor_path: PathBuf,
    pub supervisor_sha256: String,
    pub relay_path: PathBuf,
    pub relay_sha256: String,
    pub guard_container_path: String,
    pub recipe_file: PathBuf,
    pub recipe_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Readiness {
    pub curl: PathBuf,
    pub url: String,
    pub expected_model: String,
    pub per_attempt_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub program: PathBuf,
    pub program_sha256: String,
    pub argv: Vec<String>,
    pub environment: Vec<(String, String)>,
    pub input_file: PathBuf,
    pub input_sha256: String,
    pub limits: ProcessLimits,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessLimits {
    pub timeout_ms: u64,
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
    pub stdin_bytes: usize,
    pub bytes_per_turn: usize,
}
impl ProcessLimits {
    pub fn to_process(&self) -> super::super::process::Limits {
        super::super::process::Limits {
            timeout_ms: self.timeout_ms,
            stdout_bytes: self.stdout_bytes,
            stderr_bytes: self.stderr_bytes,
            stdin_bytes: self.stdin_bytes,
            bytes_per_turn: self.bytes_per_turn,
        }
    }
}
fn text(s: &str, empty: bool) -> io::Result<()> {
    if (!empty && s.is_empty())
        || s.len() > 4096
        || s.bytes().any(|b| b == 0 || b == b'\n' || b == b'\r')
    {
        return Err(error("invalid bounded literal string"));
    }
    Ok(())
}
fn absolute(p: &Path) -> io::Result<()> {
    files::path_parts(p).map(|_| ())
}
fn remote_path(p: &Path) -> io::Result<()> {
    absolute(p)?;
    if !p
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || b"/_-.".contains(b))
    {
        return Err(error(
            "remote executable path is not a fixed shell-safe token",
        ));
    }
    Ok(())
}
fn environment(env: &[(String, String)]) -> io::Result<()> {
    if env.len() > 256 {
        return Err(error("environment bound"));
    }
    let mut previous: Option<&str> = None;
    for (k, v) in env {
        text(k, false)?;
        text(v, true)?;
        if k.contains('=') || previous.is_some_and(|p| p >= k.as_str()) {
            return Err(error("environment must be sorted with unique valid keys"));
        }
        previous = Some(k);
    }
    Ok(())
}
impl Launch {
    pub fn validate(&self, bundled: bool) -> io::Result<()> {
        let p = self.policy.to_wire()?;
        let c = &self.controller;
        if self.version != 1
            || [
                c.readiness_ms,
                c.workload_ms,
                c.drain_ms,
                c.status_max_age_ms,
                c.command_ms,
                c.status_interval_ms,
                c.cleanup_ms,
            ]
            .iter()
            .any(|&v| v == 0 || v > p.campaign)
            || c.command_ms > p.frame
            || c.status_interval_ms >= c.status_max_age_ms
        {
            return Err(error("invalid launch version/controller bounds"));
        }
        for path in [
            &self.ssh.executable,
            &self.ssh.key,
            &self.ssh.known_hosts,
            &self.readiness.curl,
            &self.workload.program,
        ] {
            absolute(path)?;
        }
        environment(&self.ssh.environment)?;
        environment(&self.workload.environment)?;
        for (rank, node) in self.nodes.iter().enumerate() {
            if usize::from(node.rank) != rank {
                return Err(error("nodes must be ordered rank0/rank1"));
            }
            let Some((user, host)) = node.destination.split_once('@') else {
                return Err(error("explicit user@host required"));
            };
            if user.is_empty()
                || host.is_empty()
                || node.destination.len() > 255
                || !user
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                || !host
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-".contains(&b))
            {
                return Err(error("unsupported SSH destination token"));
            }
            remote_path(&node.supervisor_path)?;
            remote_path(&node.relay_path)?;
            absolute(Path::new(&node.guard_container_path))?;
            for digest in [
                &node.supervisor_sha256,
                &node.relay_sha256,
                &node.recipe_sha256,
            ] {
                super::super::docker::parse_id(digest)?;
            }
            if bundled {
                if node.recipe_file != Path::new(&format!("rank{rank}.recipe.bin")) {
                    return Err(error("noncanonical bundled recipe path"));
                }
            } else {
                absolute(&node.recipe_file)?;
            }
        }
        if self.nodes[0].destination == self.nodes[1].destination {
            return Err(error("two distinct node destinations required"));
        }
        text(&self.readiness.url, false)?;
        text(&self.readiness.expected_model, false)?;
        if !(self.readiness.url.starts_with("http://")
            || self.readiness.url.starts_with("https://"))
            || self.readiness.url.bytes().any(|b| b.is_ascii_whitespace())
            || !self.readiness.url.ends_with("/health")
            || self.readiness.per_attempt_ms == 0
            || self.readiness.per_attempt_ms > c.command_ms
        {
            return Err(error("explicit bounded HTTP readiness required"));
        }
        super::super::docker::parse_id(&self.workload.program_sha256)?;
        super::super::docker::parse_id(&self.workload.input_sha256)?;
        if bundled {
            if self.workload.input_file != Path::new("workload.input") {
                return Err(error("noncanonical bundled workload path"));
            }
        } else {
            absolute(&self.workload.input_file)?;
        }
        if self.workload.argv.len() > 256 {
            return Err(error("argv bound"));
        }
        for arg in &self.workload.argv {
            text(arg, true)?;
        }
        let l = &self.workload.limits;
        if l.timeout_ms == 0
            || l.timeout_ms > c.workload_ms
            || l.bytes_per_turn == 0
            || l.bytes_per_turn > 65536
            || [l.stdout_bytes, l.stderr_bytes, l.stdin_bytes]
                .iter()
                .any(|&n| n > MAX_INPUT)
        {
            return Err(error("invalid workload process bounds"));
        }
        Ok(())
    }
}
