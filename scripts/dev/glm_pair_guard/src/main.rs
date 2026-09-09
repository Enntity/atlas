// SPDX-License-Identifier: AGPL-3.0-only

//! Local CPU guard core only. No serving activation or pair-disarm authority.
#![cfg(target_os = "linux")]
mod child;
mod core;
mod frame;
mod linux;

use child::{Child, Spec};
use core::{Policy, State};
use frame::{check_deadline, Frame, Reader, LEN};
use linux::{error, Io};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};

struct Output {
    bytes: [u8; LEN],
    sent: usize,
    started: u64,
}
impl Output {
    fn new(frame: Frame, now: u64) -> Self {
        Self {
            bytes: frame.encode(),
            sent: 0,
            started: now,
        }
    }
}

fn drive(
    control: &OwnedFd,
    signal: &OwnedFd,
    child: &mut Child,
    state: &mut State,
    hello: Frame,
    policy: Policy,
    started: u64,
) -> io::Result<()> {
    let mut input = Reader::new();
    let mut output = Some(Output::new(hello, started));
    loop {
        let now = Io::now()?;
        state.check(now).map_err(error)?;
        if input
            .started
            .is_some_and(|t| now.saturating_sub(t) >= policy.frame)
            || output
                .as_ref()
                .is_some_and(|o| now.saturating_sub(o.started) >= policy.frame)
        {
            return Err(error("frame or output deadline"));
        }
        if state.needs_challenge(now) {
            if output.is_some() {
                return Err(error("outbound frame still pending"));
            }
            output = Some(Output::new(
                state.issue(now, Io::random()?).map_err(error)?,
                now,
            ));
        }
        let mut polls = [
            libc::pollfd {
                fd: control.as_raw_fd(),
                events: libc::POLLIN | if output.is_some() { libc::POLLOUT } else { 0 },
                revents: 0,
            },
            libc::pollfd {
                fd: signal.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: child.fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        Io::poll(&mut polls, policy.poll)?;
        let now = Io::now()?;
        state.check(now).map_err(error)?;
        if polls[1].revents != 0 || polls[2].revents != 0 {
            return Err(error("signal or child exit before disarm"));
        }
        if polls[0].revents & (libc::POLLERR | libc::POLLNVAL | libc::POLLHUP) != 0 {
            return Err(error("control failed or closed"));
        }
        if polls[0].revents & libc::POLLOUT != 0 {
            let o = output
                .as_mut()
                .ok_or_else(|| error("unexpected output readiness"))?;
            if now.saturating_sub(o.started) >= policy.frame {
                return Err(error("blocked output"));
            }
            let progress = Io::write(control.as_raw_fd(), &o.bytes[o.sent..])?;
            let sent_at = Io::now()?;
            check_deadline(o.started, sent_at, policy.frame).map_err(error)?;
            state.check(sent_at).map_err(error)?;
            if let Some(n) = progress {
                if n == 0 {
                    return Err(error("empty control write"));
                }
                o.sent += n;
                if o.sent == LEN {
                    output = None;
                }
            }
        }
        // At most one bounded frame each turn; no drain loop that starves clocks.
        if polls[0].revents & libc::POLLIN != 0 {
            let now = Io::now()?;
            state.check(now).map_err(error)?;
            if input
                .started
                .is_some_and(|t| now.saturating_sub(t) >= policy.frame)
            {
                return Err(error("partial frame deadline"));
            }
            // Reader clears started on final decode: retain the original value
            // through the fresh clock immediately before accepting the frame.
            let frame_started = input.started.unwrap_or(now);
            match Io::read(control.as_raw_fd(), input.remaining())? {
                Some(0) => return Err(error("control EOF")),
                Some(n) => {
                    if let Some(frame) = input.received(n, now).map_err(error)? {
                        let accepted_at = Io::now()?;
                        check_deadline(frame_started, accepted_at, policy.frame).map_err(error)?;
                        if state.accept(accepted_at, &frame).map_err(error)? {
                            child.release()?;
                        }
                    }
                }
                None => {}
            }
        }
    }
}

fn run(control: OwnedFd, spec: Spec, policy: Policy) -> io::Result<()> {
    let (signals, old_mask) = Io::signals()?;
    let now = Io::now()?;
    let (mut state, hello) =
        State::new(now, policy, Io::random()?, Io::random()?, Io::random()?).map_err(error)?;
    let deadline = now
        .checked_add(policy.startup)
        .ok_or_else(|| error("startup overflow"))?;
    let mut child = Child::prepare(spec, &old_mask, deadline)?;
    let failure = drive(
        &control, &signals, &mut child, &mut state, hello, policy, now,
    );
    state.stop();
    child.terminate()?;
    let reap_until = Io::now()?
        .checked_add(policy.reap)
        .ok_or_else(|| error("reap overflow"))?;
    loop {
        if child.reap()? {
            break;
        }
        let now = Io::now()?;
        if now >= reap_until {
            return Err(error("unconfirmed child exit"));
        }
        Io::poll(
            &mut [libc::pollfd {
                fd: child.fd(),
                events: libc::POLLIN,
                revents: 0,
            }],
            policy.poll,
        )?;
    }
    failure
}

fn entry() -> io::Result<()> {
    // Explicit CLI for local harness. No default durations or authority flags.
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 9 {
        return Err(error("usage: FD STARTUP_MS LEASE_MS CHALLENGE_MS FRAME_MS CAMPAIGN_MS POLL_MS REAP_MS ELF [ARG ...]"));
    }
    let number = |i: usize| {
        args[i]
            .parse::<u64>()
            .map_err(|_| error("invalid numeric argument"))
    };
    let fd = i32::try_from(number(0)?).map_err(|_| error("invalid descriptor"))?;
    let policy = Policy {
        startup: number(1)?,
        lease: number(2)?,
        challenge: number(3)?,
        frame: number(4)?,
        campaign: number(5)?,
        poll: number(6)?,
        reap: number(7)?,
    }
    .validate()
    .map_err(error)?;
    let spec = Spec::new(&args[8..])?;
    run(Io::control(fd)?, spec, policy)
}

fn main() {
    std::panic::set_hook(Box::new(|_| unsafe { libc::_exit(74) }));
    // No successful completion is defined until the later pair-drain slice.
    let _ = entry();
    unsafe { libc::_exit(74) };
}
