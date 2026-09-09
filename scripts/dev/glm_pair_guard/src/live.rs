// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot LIVE loop. Process setup and recipe authority belong to the caller.
use crate::{
    child::Child,
    core::State,
    frame,
    linux::{error, Io},
};
use atlas_glm_pair_io::{
    identity::{fresh_nonce, LocalIdentity},
    Channel,
};
use atlas_glm_pair_wire as wire;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};

#[path = "live_io.rs"]
mod io_state;
#[path = "live_phase.rs"]
mod phase;
use io_state::{Incoming, Output, Outputs, Reader};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Gated,
    Hello,
    Ticket,
    Running,
    Quiescent,
    DeliverRelease,
    Exit,
}

struct Runner<'a> {
    control: &'a OwnedFd,
    signals: &'a OwnedFd,
    child: &'a mut Child,
    channel: &'a Channel,
    state: &'a mut State,
    hello: frame::Frame,
    startup: &'a wire::StartupRecord,
    record: wire::RankRecord,
    rank: u8,
    started: u64,
    phase: Phase,
    phase_end: Option<u64>,
    manifest: Option<wire::Manifest>,
    pending: Option<wire::Quiescent>,
    input: Reader,
    output: Outputs,
    child_output: Option<Output>,
    child_eof: bool,
}

/// Caller supplies the original State/HELLO and clock constructed from this
/// startup policy, retaining fail-stop ownership on Err. Ok requires delivered
/// paired release, exact normal child exit0, and reap.
#[allow(clippy::too_many_arguments)] // Reviewed explicit integration ownership boundary.
pub fn drive(
    control: &OwnedFd,
    signals: &OwnedFd,
    child: &mut Child,
    channel: &Channel,
    state: &mut State,
    legacy_hello: frame::Frame,
    startup: &wire::StartupRecord,
    record: wire::RankRecord,
    rank: u8,
    started: u64,
) -> io::Result<()> {
    let mut runner = Runner {
        control,
        signals,
        child,
        channel,
        state,
        hello: legacy_hello,
        startup,
        record,
        rank,
        started,
        phase: Phase::Gated,
        phase_end: None,
        manifest: None,
        pending: None,
        input: Reader::new(),
        output: Outputs::new(),
        child_output: None,
        child_eof: false,
    };
    let result = runner.run();
    if result.is_err() {
        runner.state.stop();
    }
    result
}

fn wire_error(error: wire::Error) -> io::Error {
    io::Error::other(error)
}
fn end(now: u64, duration: u64) -> io::Result<u64> {
    now.checked_add(duration)
        .ok_or_else(|| error("LIVE deadline overflow"))
}
fn pollfd(fd: i32, events: i16) -> libc::pollfd {
    libc::pollfd {
        fd,
        events,
        revents: 0,
    }
}

fn decode_child_packet(packet: &[u8], began: u64, limit: u64) -> io::Result<wire::Frame> {
    frame::check_deadline(began, Io::now()?, limit).map_err(error)?;
    let decoded = wire::Frame::decode(packet, wire::Direction::ChildToGuard).map_err(wire_error)?;
    frame::check_deadline(began, Io::now()?, limit).map_err(error)?;
    Ok(decoded)
}

impl Runner<'_> {
    fn tick(&mut self) -> io::Result<u64> {
        let now = Io::now()?;
        self.state.check(now).map_err(error)?;
        frame::check_deadline(self.started, now, self.startup.policy.campaign).map_err(error)?;
        if self.phase == Phase::Gated {
            frame::check_deadline(self.started, now, self.startup.policy.startup).map_err(error)?;
        }
        if self.phase_end.is_some_and(|until| now >= until) {
            return Err(error("LIVE phase deadline"));
        }
        self.input.check(now, self.startup.policy.frame)?;
        self.output.check(now, self.startup.policy.frame)?;
        if let Some(output) = &self.child_output {
            output.check(now, self.startup.policy.frame)?;
        }
        Ok(now)
    }

    fn run(&mut self) -> io::Result<()> {
        self.initialize()?;
        loop {
            let now = self.tick()?;
            if self.state.needs_challenge(now) {
                let challenge = self.state.issue(now, fresh_nonce()?).map_err(error)?;
                self.output.lease(Output::new(&challenge.encode(), now)?)?;
            }
            let mut polls = [
                pollfd(
                    self.control.as_raw_fd(),
                    libc::POLLIN
                        | if self.output.pending() {
                            libc::POLLOUT
                        } else {
                            0
                        },
                ),
                pollfd(self.signals.as_raw_fd(), libc::POLLIN),
                pollfd(self.child.fd(), libc::POLLIN),
                pollfd(
                    if self.child_eof {
                        -1
                    } else {
                        self.channel.as_raw_fd()
                    },
                    libc::POLLIN
                        | if self.child_output.is_some() {
                            libc::POLLOUT
                        } else {
                            0
                        },
                ),
            ];
            Io::poll(&mut polls, self.startup.policy.poll)?;
            self.tick()?;
            if polls[1].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                return Err(error("LIVE signal descriptor failed"));
            }
            if polls[1].revents & libc::POLLIN != 0 {
                io_state::signals(self.signals.as_raw_fd())?;
                self.tick()?;
            }
            if polls[0].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                return Err(error("LIVE control closed or failed"));
            }
            if polls[2].revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
                return Err(error("LIVE held pidfd failed"));
            }
            // Already-exited children cannot become authorized by buffered
            // RELEASE. SIGCHLD alone is only a notification.
            let exited = self.child.exit_status()?;
            self.tick()?;
            if let Some(status) = exited {
                if self.phase != Phase::Exit
                    || status.code != libc::CLD_EXITED
                    || status.status != 0
                {
                    return Err(error("LIVE child exit without delivered healthy release"));
                }
            }
            // Control FIRST: at most one stream frame and one child packet/turn.
            if polls[0].revents & libc::POLLIN != 0 {
                let now = self.tick()?;
                let received = self.input.read(self.control.as_raw_fd(), now)?;
                let accepted = self.tick()?;
                if let Some((frame, began)) = received {
                    frame::check_deadline(began, accepted, self.startup.policy.frame)
                        .map_err(error)?;
                    self.accept_control(frame, began)?;
                    self.tick()?;
                }
            }
            if polls[0].revents & libc::POLLOUT != 0 {
                self.tick()?;
                self.output
                    .send(self.control.as_raw_fd(), self.startup.policy.frame)?;
                self.tick()?;
            }
            if polls[3].revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
                return Err(error("LIVE child channel failed"));
            }
            if polls[3].revents & (libc::POLLIN | libc::POLLHUP) != 0 {
                let began = self.tick()?;
                let packet = self.channel.receive(self.child.credentials());
                self.tick()?;
                match packet {
                    Ok(Some(packet)) => {
                        let frame = decode_child_packet(&packet, began, self.startup.policy.frame)?;
                        self.tick()?;
                        self.accept_child(frame, began)?;
                    }
                    Ok(None) => {}
                    Err(e)
                        if e.kind() == io::ErrorKind::UnexpectedEof
                            && self.phase == Phase::Exit =>
                    {
                        self.child_eof = true
                    }
                    Err(e) => return Err(e),
                }
                self.tick()?;
            }
            if polls[3].revents & libc::POLLOUT != 0 && self.child_output.is_some() {
                self.send_child()?;
            }
            self.tick()?;
            // A turn that consumed control bytes cannot prove the stream is
            // quiet: another buffered frame may follow. Observe another turn,
            // still one frame/channel and with the original partial deadline.
            if exited.is_some() && self.input.idle() && polls[0].revents & libc::POLLIN == 0 {
                // Recheck after all same-turn control/channel failures.
                let status = self
                    .child
                    .exit_status()?
                    .ok_or_else(|| error("lost held child status"))?;
                if self.phase != Phase::Exit
                    || status.code != libc::CLD_EXITED
                    || status.status != 0
                {
                    return Err(error("LIVE exit status changed"));
                }
                self.tick()?;
                if !self.child.reap()? {
                    return Err(error("LIVE child reap not confirmed"));
                }
                self.tick()?;
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod packet_timing_tests {
    use super::*;

    #[test]
    fn actual_received_packet_cannot_outlive_its_processing_window() {
        let (sender, receiver) = Channel::pair().unwrap();
        let credentials = atlas_glm_pair_io::Credentials {
            pid: unsafe { libc::getpid() },
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
        };
        // Untrusted wire record only; this helper does not establish PID1 or
        // Model authority. The connected namespace fixture owns that proof.
        let encoded = wire::Frame {
            rank: 0,
            body: wire::Body::Quiescent(wire::Quiescent {
                pair_digest: [1; 32],
                child_instance: [2; 32],
                epoch: wire::DRAIN_EPOCH,
                last_command: wire::SHUTDOWN_COMMAND,
                receipt_nonce: [3; 32],
            }),
        }
        .encode()
        .unwrap();
        assert!(sender.send(encoded.as_slice()).unwrap());
        let began = Io::now().unwrap();
        let packet = receiver.receive(credentials).unwrap().unwrap();
        assert!(decode_child_packet(&packet, began, 10_000).is_ok());
        std::thread::sleep(std::time::Duration::from_millis(3));
        assert!(decode_child_packet(&packet, began, 1).is_err());
    }
}
