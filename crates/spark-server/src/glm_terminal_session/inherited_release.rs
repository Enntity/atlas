// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot release exchange for an already quiescent caller. No owner teardown.

use anyhow::{Result, bail, ensure};
use atlas_glm_pair_io::identity::{LocalIdentity, fresh_nonce};
use atlas_glm_pair_wire::{
    Body, DRAIN_EPOCH, Direction, Frame, Quiescent, SHUTDOWN_COMMAND, manifest_digest,
};

use super::{Deadline, InheritedSession};

pub(super) fn exchange(session: &InheritedSession) -> Result<()> {
    // Conservative origin before receipt send; never extend by retry or by a
    // later local observation. Guard independently enforces lease/campaign.
    let mut deadline = Deadline::new(session.policy.quiescent_wait)?;
    validate_local(session)?;
    deadline.check()?;
    let manifest = &session.ticket.manifest;
    let local = manifest
        .ranks
        .get(usize::from(session.rank))
        .ok_or_else(|| anyhow::anyhow!("invalid retained paired rank"))?;
    let pending = Quiescent {
        pair_digest: manifest_digest(manifest)?,
        child_instance: local.process.child_instance,
        epoch: DRAIN_EPOCH,
        last_command: SHUTDOWN_COMMAND,
        receipt_nonce: fresh_nonce()?,
    };
    let encoded = Frame {
        rank: session.rank,
        body: Body::Quiescent(pending),
    }
    .encode()?;
    let mut frame_deadline = deadline.stage(session.policy.frame)?;
    loop {
        deadline.check()?;
        frame_deadline.check()?;
        let sent = session.channel.send(encoded.as_slice())?;
        frame_deadline.check()?;
        deadline.check()?;
        if sent {
            break;
        }
        frame_deadline.wait(&session.channel, libc::POLLOUT)?;
    }
    let (packet, mut frame_deadline) = loop {
        deadline.check()?;
        // No partial frame exists on SOCK_SEQPACKET. An absent packet waits
        // under QWAIT; only an actual receive/validation uses the frame bound.
        // Capture that bound before recvmsg, retaining it only for Some(packet).
        let mut processing = deadline.stage(session.policy.frame)?;
        let packet = session.channel.receive(session.local.parent)?;
        deadline.check()?;
        if let Some(packet) = packet {
            processing.check()?;
            break (packet, processing);
        }
        deadline.wait(&session.channel, libc::POLLIN)?;
    };
    let frame = Frame::decode(&packet, Direction::GuardToChild)?;
    validate_release(session, &pending, frame)?;
    validate_local(session)?;
    // Reject time crossed while decoding or observing identity, not just time
    // observed around recvmsg. Neither phase clock may be extended on success.
    frame_deadline.check()?;
    deadline.check()?;
    Ok(())
}

fn validate_local(session: &InheritedSession) -> Result<()> {
    ensure!(
        LocalIdentity::observe()? == session.local,
        "local identity changed before paired release"
    );
    session.local.require_guard_parent()?;
    session
        ._server
        .revalidate_process(session.local.child.pid as u32)?;
    session
        ._guard
        .revalidate_process(session.local.parent.pid as u32)?;
    Ok(())
}

fn validate_release(session: &InheritedSession, pending: &Quiescent, frame: Frame) -> Result<()> {
    ensure!(
        frame.rank == session.rank,
        "release recipient rank mismatch"
    );
    let Body::PairRelease(release) = frame.body else {
        bail!("expected one paired release")
    };
    // SSOT validates both ordered receipt digests, their manifest/child bindings,
    // epoch and shutdown word, plus byte-exact equality to our pending receipt.
    // Authenticated transport and retained process identity remain separate gates.
    release.validate(&session.ticket.manifest, session.rank, pending)?;
    Ok(())
}
