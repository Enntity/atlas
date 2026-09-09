// SPDX-License-Identifier: AGPL-3.0-only
//! Persistent local SSH transport, not remote completion or lease authority.
use crate::{frame, process};
use atlas_glm_pair_wire as wire;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

#[allow(dead_code)] // Shared bounded codec/queues also contain guard-only helpers.
#[path = "../../live_io.rs"]
mod io_state;
#[allow(dead_code)] // Use the existing pipe adapter, not a second I/O implementation.
#[path = "../relay/io.rs"]
mod platform;
pub(crate) use io_state::Incoming;
use io_state::{Output, Outputs, Reader};
use platform::Io;
#[path = "relay_poll.rs"]
mod poll;

fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}
fn wire_error(e: wire::Error) -> io::Error {
    io::Error::other(e)
}
fn end(started: u64, limit: u64) -> io::Result<u64> {
    started
        .checked_add(limit)
        .ok_or_else(|| error("relay deadline overflow"))
}

#[derive(Clone, Copy)]
pub struct Limits {
    pub frame_ms: u64,
    pub poll_ms: u64,
    pub campaign_ms: u64,
    pub stderr_bytes: usize,
    pub stderr_per_turn: usize,
    pub reap_ms: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Written {
    /// Exact GLP2 kind, not acknowledgement by the remote guard/server.
    pub kind: u8,
    pub started: u64,
    pub completed: u64,
}
pub struct Progress {
    pub incoming: Option<(Incoming, u64)>,
    pub sent_live: Option<Written>,
    /// One-shot clean stream-boundary EOF; caller State must authorize it.
    pub eof_idle: bool,
    /// Actual local child status, published only with/after clean stdout EOF.
    pub exit: Option<ExitStatus>,
    pub stderr: Vec<u8>,
    /// Local pipes drained and child reaped, never remote/pair success.
    pub done: bool,
}
pub struct Relay {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    reader: Reader,
    output: Outputs,
    live_kind: Option<(u8, u64)>,
    rank: u8,
    limits: Limits,
    started: u64,
    last_now: u64,
    stderr_count: usize,
    exit: Option<ExitStatus>,
    exit_reported: bool,
    failed: bool,
    abort_started: Option<u64>,
    kill_sent: bool,
}
impl Relay {
    pub fn spawn(spec: process::Spec, rank: u8, limits: Limits, started: u64) -> io::Result<Self> {
        process::validate(
            &spec,
            process::Limits {
                timeout_ms: limits.campaign_ms,
                stdout_bytes: wire::MAX_ENCODED,
                stderr_bytes: limits.stderr_bytes,
                stdin_bytes: 0,
                bytes_per_turn: limits.stderr_per_turn,
            },
        )?;
        if rank > 1
            || !spec.stdin.is_empty()
            || limits.frame_ms == 0
            || limits.frame_ms > limits.campaign_ms
            || limits.poll_ms == 0
            || limits.poll_ms > 250
            || limits.poll_ms > limits.frame_ms
            || limits.reap_ms == 0
            || limits.reap_ms > 10000
            || limits.reap_ms > limits.campaign_ms
        {
            return Err(error("invalid explicit persistent relay limits"));
        }
        end(started, limits.campaign_ms)?;
        frame::check_deadline(started, Io::now()?, limits.campaign_ms).map_err(error)?;
        let child = Command::new(spec.program)
            .args(spec.args)
            .env_clear()
            .envs(spec.env)
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        // Retain the actual child before any fallible post-spawn setup.
        let mut relay = Self {
            child,
            stdin: None,
            stdout: None,
            stderr: None,
            reader: Reader::directional(wire::Direction::GuardToController),
            output: Outputs::new(),
            live_kind: None,
            rank,
            limits,
            started,
            last_now: started,
            stderr_count: 0,
            exit: None,
            exit_reported: false,
            failed: false,
            abort_started: None,
            kill_sent: false,
        };
        relay.stdin = relay.child.stdin.take();
        relay.stdout = relay.child.stdout.take();
        relay.stderr = relay.child.stderr.take();
        Io::nonblocking(
            relay
                .stdin
                .as_ref()
                .ok_or_else(|| error("relay missing stdin"))?
                .as_raw_fd(),
        )?;
        Io::nonblocking(
            relay
                .stdout
                .as_ref()
                .ok_or_else(|| error("relay missing stdout"))?
                .as_raw_fd(),
        )?;
        Io::nonblocking(
            relay
                .stderr
                .as_ref()
                .ok_or_else(|| error("relay missing stderr"))?
                .as_raw_fd(),
        )?;
        relay.tick()?;
        Ok(relay)
    }
    pub fn poll_ms(&self) -> u64 {
        self.limits.poll_ms
    }
    fn tick(&mut self) -> io::Result<u64> {
        let now = Io::now()?;
        if self.failed || now < self.last_now {
            return Err(error("terminal relay or regressed clock"));
        }
        frame::check_deadline(self.started, now, self.limits.campaign_ms).map_err(error)?;
        self.reader.check(now, self.limits.frame_ms)?;
        self.output.check(now, self.limits.frame_ms)?;
        self.last_now = now;
        Ok(now)
    }
    fn writable(&mut self, started: u64) -> io::Result<()> {
        let now = self.tick()?;
        frame::check_deadline(started, now, self.limits.frame_ms).map_err(error)?;
        self.observe_exit()?;
        if self.exit.is_some() || self.stdout.is_none() || self.stdin.is_none() {
            return Err(error("cannot queue to exited or closed relay"));
        }
        Ok(())
    }
    pub fn queue_lease(&mut self, value: &frame::Frame, started: u64) -> io::Result<()> {
        let result = (|| {
            self.writable(started)?;
            if !matches!(value.kind, frame::RENEW | frame::REVOKE) {
                return Err(error("relay lease direction/direct START"));
            }
            let output = Output::new(&value.encode(), started)?;
            output.check(self.tick()?, self.limits.frame_ms)?;
            self.output.lease(output)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn queue_live(&mut self, value: &wire::Frame, started: u64) -> io::Result<()> {
        let result = (|| {
            self.writable(started)?;
            if value.rank != self.rank {
                return Err(error("relay output rank mismatch"));
            }
            let encoded = value.encode().map_err(wire_error)?;
            wire::Frame::decode(encoded.as_slice(), wire::Direction::ControllerToGuard)
                .map_err(wire_error)?;
            let output = Output::new(encoded.as_slice(), started)?;
            output.check(self.tick()?, self.limits.frame_ms)?;
            self.output.live(output)?;
            self.live_kind = Some((encoded.as_slice()[10], started));
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn poll(&mut self) -> io::Result<Progress> {
        let result = self.poll_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn observe_exit(&mut self) -> io::Result<()> {
        if self.exit.is_none() {
            self.exit = self.child.try_wait()?;
        }
        Ok(())
    }
    /// Nonblocking cleanup of this retained local process only; no SSH command
    /// or remote PID/PGID is inferred. Caller supplies the bounded reap loop.
    pub fn abort(&mut self) -> io::Result<Option<ExitStatus>> {
        self.failed = true;
        self.stdin.take();
        let now = Io::now()?;
        end(now, self.limits.reap_ms)?;
        self.abort_started.get_or_insert(now);
        self.observe_exit()?;
        if self.exit.is_none() && !self.kill_sent {
            self.child.kill()?;
            self.kill_sent = true;
        }
        self.reap()
    }
    pub fn reap(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(started) = self.abort_started {
            frame::check_deadline(started, Io::now()?, self.limits.reap_ms).map_err(error)?;
        }
        self.observe_exit()?;
        Ok(self.exit)
    }
}
impl Drop for Relay {
    fn drop(&mut self) {
        let _ = self.abort();
    }
}

#[cfg(test)]
#[path = "relay_tests.rs"]
mod tests;
