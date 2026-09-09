// SPDX-License-Identifier: AGPL-3.0-only
//! Finite command owners and evidence I/O used by the connected driver.
use crate::{docker, error, prepared, process, remote, wire};
use atlas_glm_pair_io::identity::{boot_time_ms, PinnedExecutable};
use std::{
    fs::File,
    io,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
};

pub(super) struct Job {
    pub process: process::Process,
    pub started: u64,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}
pub(super) struct Completion {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit: std::process::ExitStatus,
}
impl Job {
    pub fn spawn(spec: process::Spec, limits: process::Limits) -> io::Result<Self> {
        let started = boot_time_ms()?;
        Ok(Self {
            process: process::Process::spawn(spec, limits, started)?,
            started,
            stdout: vec![],
            stderr: vec![],
        })
    }
    pub fn poll(&mut self) -> io::Result<Option<Vec<u8>>> {
        let Some(progress) = self.poll_any()? else {
            return Ok(None);
        };
        if !progress.exit.success() {
            return Err(io::Error::other(format!(
                "finite command failed: {:?}: {}",
                progress.exit,
                String::from_utf8_lossy(&progress.stderr)
            )));
        }
        Ok(Some(progress.stdout))
    }
    pub fn poll_any(&mut self) -> io::Result<Option<Completion>> {
        let progress = self.process.poll(boot_time_ms()?)?;
        self.stdout.extend(progress.stdout);
        self.stderr.extend(progress.stderr);
        if !progress.done {
            return Ok(None);
        }
        Ok(Some(Completion {
            stdout: std::mem::take(&mut self.stdout),
            stderr: std::mem::take(&mut self.stderr),
            exit: progress
                .exit
                .ok_or_else(|| error("missing completed command status"))?,
        }))
    }
}

pub(super) fn limits(p: &prepared::Prepared) -> process::Limits {
    process::Limits {
        timeout_ms: p.launch.controller.command_ms,
        stdout_bytes: 1_048_576,
        stderr_bytes: 65_536,
        stdin_bytes: 1_048_576,
        bytes_per_turn: 16_384,
    }
}
pub(super) fn remote_spec(
    p: &prepared::Prepared,
    r: usize,
    verb: remote::Verb,
    id: Option<wire::Digest>,
    input: Vec<u8>,
) -> io::Result<process::Spec> {
    let n = &p.launch.nodes[r];
    let ssh = &p.launch.ssh;
    remote::spec(
        remote::Endpoint {
            executable: &ssh.executable,
            key: &ssh.key,
            known_hosts: &ssh.known_hosts,
            environment: &ssh.environment,
            destination: &n.destination,
            supervisor: &n.supervisor_path,
            self_sha256: &n.supervisor_sha256,
        },
        verb,
        p.session,
        r as u8,
        id,
        input,
        p.launch.controller.command_ms,
    )
}
pub(super) fn remote_job(
    p: &prepared::Prepared,
    r: usize,
    verb: remote::Verb,
    id: Option<wire::Digest>,
    input: Vec<u8>,
) -> io::Result<Job> {
    Job::spawn(remote_spec(p, r, verb, id, input)?, limits(p))
}
pub(super) fn pause(ms: u64) -> io::Result<()> {
    if unsafe {
        libc::poll(
            std::ptr::null_mut(),
            0,
            i32::try_from(ms).map_err(|_| error("poll overflow"))?,
        )
    } < 0
        && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
pub(super) struct Journal<'a> {
    prepared: &'a prepared::Prepared,
    sequence: u32,
    bytes: usize,
}
impl<'a> Journal<'a> {
    pub fn new(p: &'a prepared::Prepared) -> Self {
        Self {
            prepared: p,
            sequence: 0,
            bytes: 0,
        }
    }
    pub fn record(&mut self, kind: &str, value: &serde_json::Value) -> io::Result<()> {
        let bytes = serde_json::to_vec(
            &serde_json::json!({"boot_ms":boot_time_ms()?,"kind":kind,"value":value}),
        )?;
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .ok_or_else(|| error("journal overflow"))?;
        if self.bytes > 64 * 1024 * 1024 || self.sequence >= 100_000 {
            return Err(error("campaign evidence exceeds fixed bound"));
        }
        let name = format!("evidence-{:06}.json", self.sequence);
        self.sequence += 1;
        self.prepared.record(&name, &bytes)
    }
}

pub(super) struct WorkloadExecutable {
    file: File,
    pinned: PinnedExecutable,
}
impl WorkloadExecutable {
    pub fn revalidate(&self) -> io::Result<()> {
        self.pinned.revalidate()
    }
    pub fn program(&self) -> std::path::PathBuf {
        format!("/proc/self/fd/{}", self.file.as_raw_fd()).into()
    }
}
pub(super) fn pin_workload(p: &prepared::Prepared) -> io::Result<WorkloadExecutable> {
    let file = File::open(&p.launch.workload.program)?;
    let m = file.metadata()?;
    if m.uid() != 0 || m.mode() & 0o022 != 0 {
        return Err(error(
            "workload interpreter must be immutable to unprivileged users",
        ));
    }
    let end = boot_time_ms()?
        .checked_add(p.launch.controller.command_ms)
        .ok_or_else(|| error("pin deadline overflow"))?;
    let pinned = PinnedExecutable::from_file(file.try_clone()?, 512 * 1024 * 1024, end)?;
    if pinned.digest() != docker::parse_id(&p.launch.workload.program_sha256)? {
        return Err(error("workload executable digest mismatch"));
    }
    Ok(WorkloadExecutable { file, pinned })
}
