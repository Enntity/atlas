// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-GPU inherited-channel handshake only; no Model or serving registration.
//! Deliberately usable by the GPU-free CPU harness through an exact source include.

use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, ensure, Context, Result};
use atlas_glm_pair_io::identity::{boot_time_ms, fresh_nonce, LocalIdentity, PinnedExecutable};
use atlas_glm_pair_io::Channel;
use atlas_glm_pair_wire::{policy_digest, Body, ChildHello, ChildTicket, Direction, Frame, Policy};

static INHERITED_SPENT: AtomicBool = AtomicBool::new(false);

/// Trusted, resolved launch assertions; recipe digest includes the fixed profile.
/// The remote rank is a trusted controller assertion, not a local proc observation.
#[derive(Clone, Copy)]
pub(crate) struct ExpectedSession {
    pub rank: u8,
    pub pair_session: [u8; 32],
    pub policy: Policy,
    pub container_id: [u8; 32],
    pub image_digest: [u8; 32],
    pub recipe_digest: [u8; 32],
    pub server_elf_digest: [u8; 32],
    pub guard_elf_digest: [u8; 32],
    pub max_executable_bytes: u64,
}

impl ExpectedSession {
    fn validate(&self, handshake_ms: u64) -> Result<()> {
        self.policy.validate()?;
        ensure!(self.rank < 2, "invalid local paired rank");
        ensure!(
            handshake_ms > 0 && handshake_ms <= self.policy.child_handshake,
            "handshake exceeds explicit policy bound"
        );
        ensure!(
            self.max_executable_bytes >= 4,
            "invalid explicit ELF size bound"
        );
        for digest in [
            self.pair_session,
            self.container_id,
            self.image_digest,
            self.recipe_digest,
            self.server_elf_digest,
            self.guard_elf_digest,
        ] {
            ensure!(digest != [0; 32], "zero expected launch identity");
        }
        Ok(())
    }
}

pub(crate) struct InheritedSession {
    channel: Channel,
    ticket: ChildTicket,
    local: LocalIdentity,
    // Retain the opened identities; a pathname/digest alone does not own the ELF.
    _server: PinnedExecutable,
    _guard: PinnedExecutable,
}

impl InheritedSession {
    pub(crate) fn ticket(&self) -> &ChildTicket {
        &self.ticket
    }

    /// Consume only inherited FD3, irreversibly, before selected GPU startup.
    ///
    /// # Safety
    /// The caller exclusively owns FD3; it must have no other owner/user. This
    /// is an early single-session ingress, never a recoverable ordinary fallback.
    pub(crate) unsafe fn receive(expected: ExpectedSession, handshake_ms: u64) -> Result<Self> {
        expected.validate(handshake_ms)?;
        let mut deadline = Deadline::new(handshake_ms)?;
        ensure!(
            std::env::var_os("ATLAS_GLM_PAIR_FD").as_deref() == Some(std::ffi::OsStr::new("3")),
            "selected channel requires exact ATLAS_GLM_PAIR_FD=3"
        );
        ensure!(
            !INHERITED_SPENT.swap(true, Ordering::AcqRel),
            "inherited channel already consumed"
        );
        let channel = unsafe { Channel::consume_inherited(3) }.context("consume paired FD3")?;
        let local = LocalIdentity::observe()?;
        local.require_guard_parent()?;
        deadline.check()?;
        let server = PinnedExecutable::open_process(
            local.child.pid as u32,
            expected.max_executable_bytes,
            deadline.at,
        )?;
        let guard = PinnedExecutable::open_process(
            local.parent.pid as u32,
            expected.max_executable_bytes,
            deadline.at,
        )?;
        ensure!(
            server.digest() == expected.server_elf_digest,
            "local server ELF mismatch"
        );
        ensure!(
            guard.digest() == expected.guard_elf_digest,
            "local guard ELF mismatch"
        );
        let hello = hello(&local, fresh_nonce()?);
        let encoded = Frame {
            rank: expected.rank,
            body: Body::ChildHello(hello),
        }
        .encode()?;
        let mut frame_deadline = deadline.stage(expected.policy.frame)?;
        loop {
            deadline.check()?;
            frame_deadline.check()?;
            let sent = channel.send(encoded.as_slice())?;
            frame_deadline.check()?;
            deadline.check()?;
            if sent {
                break;
            }
            frame_deadline.wait(&channel, libc::POLLOUT)?;
        }
        let mut frame_deadline = deadline.stage(expected.policy.frame)?;
        let packet = loop {
            deadline.check()?;
            frame_deadline.check()?;
            let received = channel.receive(local.parent)?;
            frame_deadline.check()?;
            deadline.check()?;
            if let Some(bytes) = received {
                break bytes;
            }
            frame_deadline.wait(&channel, libc::POLLIN)?;
        };
        let frame = Frame::decode(&packet, Direction::GuardToChild)?;
        let ticket = check_ticket(&expected, &hello, frame)?;
        ensure!(
            LocalIdentity::observe()? == local,
            "local identity changed during handshake"
        );
        local.require_guard_parent()?;
        server.revalidate_process(local.child.pid as u32)?;
        guard.revalidate_process(local.parent.pid as u32)?;
        deadline.check()?;
        Ok(Self {
            channel,
            ticket,
            local,
            _server: server,
            _guard: guard,
        })
    }
}

fn hello(local: &LocalIdentity, challenge: [u8; 32]) -> ChildHello {
    ChildHello {
        boot_id: local.boot_id,
        pid_namespace_device: local.pid_namespace_device,
        pid_namespace_inode: local.pid_namespace_inode,
        guard_pid: local.parent.pid as u32,
        child_pid: local.child.pid as u32,
        guard_start_ticks: local.parent_start_ticks,
        child_start_ticks: local.child_start_ticks,
        server_challenge: challenge,
    }
}

fn check_ticket(
    expected: &ExpectedSession,
    hello: &ChildHello,
    frame: Frame,
) -> Result<ChildTicket> {
    ensure!(
        frame.rank == expected.rank,
        "ticket recipient rank mismatch"
    );
    let Body::ChildTicket(ticket) = frame.body else {
        bail!("expected one child ticket")
    };
    ticket.validate_hello(expected.rank, hello)?;
    let manifest = &ticket.manifest;
    ensure!(
        manifest.pair_session == expected.pair_session
            && manifest.policy_digest == policy_digest(&expected.policy)?,
        "ticket session/policy mismatch"
    );
    let local = &manifest.ranks[usize::from(expected.rank)];
    ensure!(
        local.container_id == expected.container_id
            && local.image_digest == expected.image_digest
            && local.recipe_digest == expected.recipe_digest
            && local.server_elf_digest == expected.server_elf_digest
            && local.guard_elf_digest == expected.guard_elf_digest,
        "ticket local launch mismatch"
    );
    Ok(ticket)
}

struct Deadline {
    at: u64,
    last: u64,
}
impl Deadline {
    fn new(duration_ms: u64) -> Result<Self> {
        let last = boot_time_ms()?;
        let at = last
            .checked_add(duration_ms)
            .context("handshake deadline overflow")?;
        Ok(Self { at, last })
    }
    fn stage(&mut self, duration_ms: u64) -> Result<Self> {
        self.check()?;
        ensure!(duration_ms > 0, "zero frame deadline");
        let at = self
            .last
            .checked_add(duration_ms)
            .context("frame deadline overflow")?
            .min(self.at);
        Ok(Self {
            at,
            last: self.last,
        })
    }
    fn check(&mut self) -> Result<u64> {
        let now = boot_time_ms()?;
        ensure!(
            now >= self.last && now < self.at,
            "handshake deadline/clock failure"
        );
        self.last = now;
        Ok(self.at - now)
    }
    fn wait(&mut self, channel: &Channel, events: i16) -> Result<()> {
        let remaining = self.check()?.min(i32::MAX as u64) as i32;
        let mut fd = libc::pollfd {
            fd: channel.as_raw_fd(),
            events,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut fd, 1, remaining) };
        self.check()?;
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
        ensure!(
            fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) == 0,
            "paired channel closed during handshake"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "inherited_tests.rs"]
mod tests;
