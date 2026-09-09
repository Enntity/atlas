// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded local subprocess adapter. The caller supplies freshly observed
//! CLOCK_BOOTTIME milliseconds and owns the bounded event/reap loop. No SSH or
//! Docker policy lives here; killing the direct child is not descendant cleanup.
use std::collections::HashSet;
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::{ffi::OsStrExt, process::CommandExt};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

const MAX_BYTES: usize = 1024 * 1024;
const MAX_TURN: usize = 64 * 1024;

pub struct Spec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    pub stdin: Vec<u8>,
}
#[derive(Clone, Copy)]
pub struct Limits {
    pub timeout_ms: u64,
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
    pub stdin_bytes: usize,
    pub bytes_per_turn: usize,
}
pub struct Progress {
    /// Deltas, not complete retained output; the adapter enforces total caps.
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit: Option<ExitStatus>,
    pub done: bool,
}
pub struct Process {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    input: Vec<u8>,
    written: usize,
    counts: [usize; 2],
    limits: Limits,
    deadline: u64,
    last_now: u64,
    exit: Option<ExitStatus>,
    failed: bool,
    kill_sent: bool,
}
fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}
fn nonblocking(fd: i32) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
pub(super) fn validate(spec: &Spec, limits: Limits) -> io::Result<()> {
    if !spec.program.is_absolute()
        || limits.timeout_ms == 0
        || limits.timeout_ms > 86_400_000
        || limits.bytes_per_turn == 0
        || limits.bytes_per_turn > MAX_TURN
        || [limits.stdout_bytes, limits.stderr_bytes, limits.stdin_bytes]
            .iter()
            .any(|&n| n > MAX_BYTES)
        || spec.stdin.len() > limits.stdin_bytes
        || spec.args.len() > 256
        || spec.env.len() > 256
    {
        return Err(error("invalid explicit process bounds or executable"));
    }
    let mut bytes = spec.program.as_os_str().as_bytes().len();
    let mut keys = HashSet::new();
    for (key, value) in &spec.env {
        let key_bytes = key.as_bytes();
        if key_bytes.is_empty()
            || key_bytes.contains(&b'=')
            || key_bytes.contains(&0)
            || value.as_bytes().contains(&0)
            || !keys.insert(key)
        {
            return Err(error("invalid or duplicate explicit environment"));
        }
        bytes = bytes
            .checked_add(key_bytes.len())
            .and_then(|n| n.checked_add(value.as_bytes().len()))
            .ok_or_else(|| error("argument size overflow"))?;
    }
    for arg in &spec.args {
        if arg.as_bytes().contains(&0) {
            return Err(error("NUL in argument"));
        }
        bytes = bytes
            .checked_add(arg.as_bytes().len())
            .ok_or_else(|| error("argument size overflow"))?;
    }
    if bytes > MAX_BYTES || spec.program.as_os_str().as_bytes().contains(&0) {
        return Err(error("executable/argument byte bound"));
    }
    Ok(())
}

impl Process {
    pub fn spawn(spec: Spec, limits: Limits, now_ms: u64) -> io::Result<Self> {
        validate(&spec, limits)?;
        let deadline = now_ms
            .checked_add(limits.timeout_ms)
            .ok_or_else(|| error("deadline overflow"))?;
        let child = Command::new(&spec.program)
            .args(&spec.args)
            .env_clear()
            .envs(spec.env)
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // Establish the retained owner before any post-spawn fallible setup.
        let mut process = Self {
            child,
            stdin: None,
            stdout: None,
            stderr: None,
            input: spec.stdin,
            written: 0,
            counts: [0; 2],
            limits,
            deadline,
            last_now: now_ms,
            exit: None,
            failed: false,
            kill_sent: false,
        };
        process.stdin = process.child.stdin.take();
        process.stdout = process.child.stdout.take();
        process.stderr = process.child.stderr.take();
        nonblocking(
            process
                .stdin
                .as_ref()
                .ok_or_else(|| error("missing stdin"))?
                .as_raw_fd(),
        )?;
        nonblocking(
            process
                .stdout
                .as_ref()
                .ok_or_else(|| error("missing stdout"))?
                .as_raw_fd(),
        )?;
        nonblocking(
            process
                .stderr
                .as_ref()
                .ok_or_else(|| error("missing stderr"))?
                .as_raw_fd(),
        )?;
        if process.input.is_empty() {
            process.stdin.take();
        }
        Ok(process)
    }
    /// At most one bounded write and one bounded read per output pipe. No
    /// blocking wait, write_all, or internal retry loop hides controller work.
    pub fn poll(&mut self, now_ms: u64) -> io::Result<Progress> {
        let result = self.poll_inner(now_ms);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn poll_inner(&mut self, now_ms: u64) -> io::Result<Progress> {
        if self.failed || now_ms < self.last_now || now_ms >= self.deadline {
            return Err(error(
                "terminal subprocess adapter, deadline, or clock regression",
            ));
        }
        self.last_now = now_ms;
        self.reap()?;
        if let Some(stdin) = self.stdin.as_mut() {
            let end = self
                .input
                .len()
                .min(self.written + self.limits.bytes_per_turn);
            match stdin.write(&self.input[self.written..end]) {
                Ok(0) => return Err(error("zero-byte child stdin write")),
                Ok(n) => self.written += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(e),
            }
            if self.written == self.input.len() {
                self.stdin.take();
            }
        }
        let stdout = read_one(
            &mut self.stdout,
            &mut self.counts[0],
            self.limits.stdout_bytes,
            self.limits.bytes_per_turn,
        )?;
        let stderr = read_one(
            &mut self.stderr,
            &mut self.counts[1],
            self.limits.stderr_bytes,
            self.limits.bytes_per_turn,
        )?;
        self.reap()?;
        Ok(Progress {
            stdout,
            stderr,
            exit: self.exit,
            done: self.exit.is_some()
                && self.stdout.is_none()
                && self.stderr.is_none()
                && self.stdin.is_none(),
        })
    }
    /// Nonblocking, valid after an error/abort. The caller bounds cleanup time.
    pub fn reap(&mut self) -> io::Result<Option<ExitStatus>> {
        if self.exit.is_none() {
            self.exit = self.child.try_wait()?;
        }
        Ok(self.exit)
    }
    /// No numeric-PID lookup, process-group kill or waiting. try_wait may reap;
    /// if it does, never signal again. Otherwise the retained child cannot have
    /// its PID recycled before this owner's kill (no other reaper is allowed).
    pub fn abort(&mut self) -> io::Result<Option<ExitStatus>> {
        self.failed = true;
        self.stdin.take();
        if let Some(status) = self.reap()? {
            return Ok(Some(status));
        }
        if !self.kill_sent {
            self.child.kill()?;
            self.kill_sent = true;
        }
        self.reap()
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        // Best effort only. Explicit abort + caller-bounded reap supplies proof.
        let _ = self.abort();
    }
}
fn read_one<T: Read>(
    pipe: &mut Option<T>,
    count: &mut usize,
    cap: usize,
    turn: usize,
) -> io::Result<Vec<u8>> {
    let Some(reader) = pipe.as_mut() else {
        return Ok(Vec::new());
    };
    // Probe at most one byte past the cumulative limit; overflow is a failure,
    // not silently truncated successful output. Each read is still turn-bounded.
    let mut bytes = vec![0; turn.min(cap - *count + 1)];
    match reader.read(&mut bytes) {
        Ok(0) => {
            pipe.take();
            Ok(Vec::new())
        }
        Ok(n) => {
            if n > cap - *count {
                return Err(error("subprocess output exceeded explicit cap"));
            }
            *count += n;
            bytes.truncate(n);
            Ok(bytes)
        }
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(Vec::new())
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
#[path = "process_tests.rs"]
mod tests;
