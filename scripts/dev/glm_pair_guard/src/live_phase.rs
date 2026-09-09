// SPDX-License-Identifier: AGPL-3.0-only
//! Transitions over the actual retained Child, never supplied PID authority.
use super::*;

fn quiescent_end(previous: Option<u64>, now: u64, wait: u64) -> io::Result<u64> {
    let received = end(now, wait)?;
    Ok(previous.map_or(received, |drain| drain.min(received)))
}

impl Runner<'_> {
    pub(super) fn initialize(&mut self) -> io::Result<()> {
        self.startup.validate().map_err(wire_error)?;
        self.record.validate().map_err(wire_error)?;
        if self.rank > 1
            || self.hello.kind != frame::HELLO
            || self.hello.ordinal != 0
            || self.hello.session != self.record.local_control_session
            || self.hello.instance != self.record.guard_instance
            || self.hello.challenge != self.record.original_startup_challenge
            || self.record.container_id != self.startup.container_id
            || self.record.image_digest != self.startup.image_digest
            || self.record.recipe_digest != self.startup.recipe_digest
            || self.record.server_elf_digest != self.startup.server_elf_digest
            || self.record.guard_elf_digest != self.startup.guard_elf_digest
        {
            return Err(error("LIVE original local startup binding"));
        }
        for fd in [
            self.control.as_raw_fd(),
            self.signals.as_raw_fd(),
            self.channel.as_raw_fd(),
        ] {
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if flags < 0 || flags & libc::O_NONBLOCK == 0 {
                return Err(error("LIVE requires nonblocking descriptors"));
            }
        }
        self.tick()?;
        self.observe_child()?;
        let queued = self.tick()?;
        self.output
            .lease(Output::new(&self.hello.encode(), queued)?)?;
        let report = wire::Frame {
            rank: self.rank,
            body: wire::Body::GatedReport(wire::GatedReport {
                pair_session: self.startup.pair_session,
                record: self.record,
            }),
        }
        .encode()
        .map_err(wire_error)?;
        self.output.live(Output::new(report.as_slice(), queued)?)?;
        self.tick()?;
        Ok(())
    }

    fn observe_child(&self) -> io::Result<()> {
        if self.child.exit_status()?.is_some() {
            return Err(error("LIVE child already exited"));
        }
        let actual = LocalIdentity::observe_child(self.child.pid() as u32)?;
        if actual.child != self.child.credentials() {
            return Err(error("LIVE actual child credentials changed"));
        }
        wire::ChildHello {
            boot_id: actual.boot_id,
            pid_namespace_device: actual.pid_namespace_device,
            pid_namespace_inode: actual.pid_namespace_inode,
            guard_pid: actual.parent.pid as u32,
            child_pid: actual.child.pid as u32,
            guard_start_ticks: actual.parent_start_ticks,
            child_start_ticks: actual.child_start_ticks,
            server_challenge: [1; 32],
        }
        .validate_process(&self.record.process)
        .map_err(wire_error)?;
        if self.child.exit_status()?.is_some() {
            return Err(error("LIVE child exited during observation"));
        }
        Ok(())
    }

    fn acceptance(&mut self, began: u64) -> io::Result<u64> {
        let now = self.tick()?;
        frame::check_deadline(began, now, self.startup.policy.frame).map_err(error)?;
        Ok(now)
    }

    pub(super) fn accept_control(&mut self, incoming: Incoming, began: u64) -> io::Result<()> {
        match incoming {
            Incoming::Legacy(frame) => {
                if frame.kind == frame::START {
                    return Err(error("LIVE refuses direct legacy START"));
                }
                let accepted = self.acceptance(began)?;
                if self.state.accept(accepted, &frame).map_err(error)? {
                    return Err(error("LIVE unexpected legacy gate release"));
                }
            }
            Incoming::Live(frame) => {
                if frame.rank != self.rank {
                    return Err(error("LIVE control recipient rank"));
                }
                match frame.body {
                    wire::Body::DrainRequest(request) => {
                        if self.drain_requested {
                            return Err(error("LIVE repeated drain request"));
                        }
                        if self.rank != 0
                            || self.phase != Phase::Running
                            || self.pending.is_some()
                            || self.child_output.is_some()
                            || self.output.live_pending()
                        {
                            return Err(error(
                                "LIVE drain requires running rank0 without pending release",
                            ));
                        }
                        request
                            .validate(
                                self.manifest
                                    .as_ref()
                                    .ok_or_else(|| error("missing manifest"))?,
                            )
                            .map_err(wire_error)?;
                        self.observe_child()?;
                        let accepted = self.acceptance(began)?;
                        // Consume before signaling: even an uncertain signal
                        // result cannot admit a retry or earn another window.
                        self.drain_requested = true;
                        self.phase_end = Some(end(accepted, self.startup.policy.quiescent_wait)?);
                        self.child.request_drain()?;
                        self.tick()?;
                    }
                    wire::Body::PairedStart(manifest) => {
                        if self.phase != Phase::Gated || self.output.pending() {
                            return Err(error("LIVE unexpected or premature PAIRED_START"));
                        }
                        manifest
                            .validate_local(self.rank, self.startup, &self.record)
                            .map_err(wire_error)?;
                        self.observe_child()?;
                        let accepted = self.acceptance(began)?;
                        let mut start = self.hello.clone();
                        start.kind = frame::START;
                        if !self.state.accept(accepted, &start).map_err(error)? {
                            return Err(error("LIVE original START did not release"));
                        }
                        self.phase_end = Some(end(accepted, self.startup.policy.child_handshake)?);
                        self.manifest = Some(manifest);
                        self.phase = Phase::Hello;
                        self.tick()?;
                        self.child.release()?;
                        self.tick()?;
                    }
                    wire::Body::PairRelease(release) => {
                        if self.phase != Phase::Quiescent
                            || self.output.live_pending()
                            || self.child_output.is_some()
                        {
                            return Err(error("LIVE release without forwarded pending receipt"));
                        }
                        release
                            .validate(
                                self.manifest
                                    .as_ref()
                                    .ok_or_else(|| error("missing manifest"))?,
                                self.rank,
                                self.pending
                                    .as_ref()
                                    .ok_or_else(|| error("missing local receipt"))?,
                            )
                            .map_err(wire_error)?;
                        let encoded = frame.encode().map_err(wire_error)?;
                        // Check the OLD quiescent/frame bounds before replacing
                        // them with the acceptance-anchored exit window.
                        let now = self.acceptance(began)?;
                        // Original acceptance clock; delivery cannot extend it.
                        self.phase_end = Some(end(now, self.startup.policy.exit)?);
                        self.child_output = Some(Output::new(encoded.as_slice(), now)?);
                        self.phase = Phase::DeliverRelease;
                    }
                    _ => return Err(error("LIVE phase-inappropriate control frame")),
                }
            }
        }
        Ok(())
    }

    pub(super) fn accept_child(&mut self, frame: wire::Frame, began: u64) -> io::Result<()> {
        if frame.rank != self.rank {
            return Err(error("LIVE child recipient rank"));
        }
        match frame.body {
            wire::Body::ChildHello(hello) => {
                if self.phase != Phase::Hello || self.child_output.is_some() {
                    return Err(error("LIVE duplicate or premature CHILD_HELLO"));
                }
                hello
                    .validate_process(&self.record.process)
                    .map_err(wire_error)?;
                self.observe_child()?;
                self.tick()?;
                let ticket = wire::Frame {
                    rank: self.rank,
                    body: wire::Body::ChildTicket(wire::ChildTicket {
                        manifest: self.manifest.ok_or_else(|| error("missing manifest"))?,
                        echoed_server_challenge: hello.server_challenge,
                        guard_ticket_challenge: fresh_nonce()?,
                    }),
                }
                .encode()
                .map_err(wire_error)?;
                // Keep the pre-recv packet window through actual proc checks,
                // decoding and nonce creation, before publishing a ticket.
                let now = self.acceptance(began)?;
                self.child_output = Some(Output::new(ticket.as_slice(), now)?);
                self.phase = Phase::Ticket;
            }
            wire::Body::Quiescent(receipt) => {
                if self.phase != Phase::Running || self.pending.is_some() {
                    return Err(error("LIVE duplicate or premature QUIESCENT"));
                }
                receipt
                    .validate(
                        self.manifest
                            .as_ref()
                            .ok_or_else(|| error("missing manifest"))?,
                        self.rank,
                    )
                    .map_err(wire_error)?;
                let encoded = frame.encode().map_err(wire_error)?;
                let now = self.acceptance(began)?;
                self.output.live(Output::new(encoded.as_slice(), now)?)?;
                self.pending = Some(receipt);
                // Spontaneous Q starts its usual wait. Requested drain retains
                // its earlier bound: receiving Q cannot restart that clock.
                self.phase_end = Some(quiescent_end(
                    self.phase_end,
                    now,
                    self.startup.policy.quiescent_wait,
                )?);
                self.phase = Phase::Quiescent;
            }
            _ => return Err(error("LIVE phase-inappropriate child frame")),
        }
        Ok(())
    }

    pub(super) fn send_child(&mut self) -> io::Result<()> {
        self.tick()?;
        let output = self
            .child_output
            .as_ref()
            .ok_or_else(|| error("missing child output"))?;
        let sent = self.channel.send(output.packet())?;
        self.tick()?;
        if sent {
            self.child_output = None;
            match self.phase {
                Phase::Ticket => {
                    self.phase = Phase::Running;
                    self.phase_end = None;
                }
                Phase::DeliverRelease => self.phase = Phase::Exit,
                _ => return Err(error("LIVE unexpected child output completion")),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod drain_deadline_tests {
    use super::*;

    #[test]
    fn actual_quiescent_deadline_helper_never_extends_requested_drain() {
        assert_eq!(quiescent_end(None, 100, 100).unwrap(), 200);
        assert_eq!(quiescent_end(Some(200), 150, 100).unwrap(), 200);
        assert_eq!(quiescent_end(Some(200), 199, 100).unwrap(), 200);
        // The runner's phase check rejects equality; preserve that exact
        // absolute boundary even when Q arrives near the old deadline.
        let until = quiescent_end(Some(200), 199, 100).unwrap();
        assert!(199 < until);
        assert!(200 >= until);
        assert_eq!(quiescent_end(Some(500), 100, 100).unwrap(), 200);
        assert!(quiescent_end(None, u64::MAX, 1).is_err());
        assert!(quiescent_end(Some(1), u64::MAX, 1).is_err());
    }
}
