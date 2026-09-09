// SPDX-License-Identifier: AGPL-3.0-only
//! One-shot root-local SSH byte relay; EOF is failure, never pair certification.
#![cfg(target_os = "linux")]
use atlas_glm_pair_wire as wire;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};

#[allow(dead_code)] // Reuse the exact guard codecs; legacy-only helpers are not relay entry points.
#[path = "../frame.rs"]
mod frame;
#[allow(dead_code)] // Fixed shared queues/codec, including guard-only signal handling.
#[path = "../live_io.rs"]
mod io_state;
#[path = "relay/io.rs"]
mod platform;
#[path = "relay/path.rs"]
mod secure_path;
use io_state::{Incoming, Output, Outputs, Reader};
use platform::Io;

fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}
fn wire_error(message: wire::Error) -> io::Error {
    io::Error::other(message.to_string())
}
fn end(started: u64, duration: u64) -> io::Result<u64> {
    started
        .checked_add(duration)
        .ok_or_else(|| error("relay deadline overflow"))
}

struct Config {
    session: String,
    rank: u8,
    connect: u64,
    frame: u64,
    campaign: u64,
    poll: u64,
}
impl Config {
    fn parse(args: &[String]) -> io::Result<Self> {
        if args.len() != 12
            || args[0] != "--session"
            || args[2] != "--rank"
            || args[4] != "--connect-ms"
            || args[6] != "--frame-ms"
            || args[8] != "--campaign-ms"
            || args[10] != "--poll-ms"
        {
            return Err(error(
                "explicit session/rank/connect/frame/campaign/poll arguments required",
            ));
        }
        let session = &args[1];
        if session.len() != 64
            || !session
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || session.bytes().all(|b| b == b'0')
            || !matches!(args[3].as_str(), "0" | "1")
        {
            return Err(error("invalid exact session or rank"));
        }
        let number = |at: usize| -> io::Result<u64> {
            if args[at].is_empty() || !args[at].bytes().all(|b| b.is_ascii_digit()) {
                return Err(error("invalid duration"));
            }
            let n = args[at]
                .parse::<u64>()
                .map_err(|_| error("duration overflow"))?;
            if n == 0 || n > 86_400_000 {
                return Err(error("duration bound"));
            }
            Ok(n)
        };
        let value = Self {
            session: session.clone(),
            rank: args[3].as_bytes()[0] - b'0',
            connect: number(5)?,
            frame: number(7)?,
            campaign: number(9)?,
            poll: number(11)?,
        };
        if value.connect > value.campaign
            || value.frame > value.campaign
            || value.poll > value.frame
            || value.poll > value.connect
            || value.poll > 250
        {
            return Err(error("inconsistent relay deadlines"));
        }
        Ok(value)
    }
    fn check(&self, started: u64, now: u64) -> io::Result<()> {
        end(started, self.campaign)?;
        frame::check_deadline(started, now, self.campaign).map_err(error)
    }
}

fn drive(
    input: &OwnedFd,
    output: &OwnedFd,
    socket: &OwnedFd,
    config: &Config,
    started: u64,
) -> io::Result<()> {
    let mut control = Reader::new();
    let mut guard = Reader::directional(wire::Direction::GuardToController);
    let mut to_guard = Outputs::new();
    let mut to_controller = Outputs::new();
    loop {
        let now = Io::now()?;
        config.check(started, now)?;
        for reader in [&control, &guard] {
            reader.check(now, config.frame)?;
        }
        for output in [&to_guard, &to_controller] {
            output.check(now, config.frame)?;
        }
        // One bounded read per channel, controller first. Extra frames never
        // grow a queue: a same-class occupied output slot is terminal.
        forward(
            &mut control,
            input.as_raw_fd(),
            &mut to_guard,
            config,
            started,
            true,
        )?;
        forward(
            &mut guard,
            socket.as_raw_fd(),
            &mut to_controller,
            config,
            started,
            false,
        )?;
        for (queue, fd) in [
            (&mut to_guard, socket.as_raw_fd()),
            (&mut to_controller, output.as_raw_fd()),
        ] {
            config.check(started, Io::now()?)?;
            queue.send(fd, config.frame)?;
            config.check(started, Io::now()?)?;
        }
        let mut polls = [
            libc::pollfd {
                fd: input.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: socket.as_raw_fd(),
                events: libc::POLLIN | if to_guard.pending() { libc::POLLOUT } else { 0 },
                revents: 0,
            },
            libc::pollfd {
                fd: output.as_raw_fd(),
                events: if to_controller.pending() {
                    libc::POLLOUT
                } else {
                    0
                },
                revents: 0,
            },
        ];
        Io::poll(&mut polls, config.poll)?;
        config.check(started, Io::now()?)?;
        if polls
            .iter()
            .any(|p| p.revents & (libc::POLLERR | libc::POLLNVAL) != 0)
            || polls[2].revents & libc::POLLHUP != 0
        {
            return Err(error("relay transport failure"));
        }
    }
}

fn forward(
    reader: &mut Reader,
    fd: i32,
    output: &mut Outputs,
    config: &Config,
    started: u64,
    inbound: bool,
) -> io::Result<()> {
    let began = Io::now()?;
    config.check(started, began)?;
    reader.check(began, config.frame)?;
    let packet = reader.read(fd, began)?;
    let observed = Io::now()?;
    config.check(started, observed)?;
    reader.check(observed, config.frame)?;
    if let Some((packet, original)) = packet {
        // Decode has already completed. Preserve the first-byte origin through
        // validation and queue admission instead of starting a new frame clock.
        frame::check_deadline(original, observed, config.frame).map_err(error)?;
        let (encoded, legacy) = match packet {
            Incoming::Legacy(value) => {
                let valid = if inbound {
                    matches!(value.kind, frame::RENEW | frame::REVOKE)
                } else {
                    matches!(value.kind, frame::HELLO | frame::CHALLENGE)
                };
                if !valid {
                    return Err(error("legacy frame direction or direct START"));
                }
                (Output::new(&value.encode(), original)?, true)
            }
            Incoming::Live(value) => {
                if value.rank != config.rank {
                    return Err(error("relay frame rank mismatch"));
                }
                (
                    Output::new(value.encode().map_err(wire_error)?.as_slice(), original)?,
                    false,
                )
            }
        };
        let accepted = Io::now()?;
        config.check(started, accepted)?;
        encoded.check(accepted, config.frame)?;
        if legacy {
            output.lease(encoded)?;
        } else {
            output.live(encoded)?;
        }
    }
    Ok(())
}

fn run() -> io::Result<()> {
    let args = std::env::args_os()
        .skip(1)
        .map(|s| s.into_string().map_err(|_| error("non-Unicode argument")))
        .collect::<io::Result<Vec<_>>>()?;
    let config = Config::parse(&args)?;
    let started = Io::now()?;
    config.check(started, started)?;
    if unsafe { libc::geteuid() } != 0 || unsafe { libc::getegid() } != 0 {
        return Err(error("relay requires authorized root invocation"));
    }
    if unsafe { libc::isatty(0) } != 0 || unsafe { libc::isatty(1) } != 0 {
        return Err(error("relay requires non-PTY transport"));
    }
    Io::ignore_sigpipe()?;
    let input = Io::duplicate(0)?;
    let output = Io::duplicate(1)?;
    Io::nonblocking(input.as_raw_fd())?;
    Io::nonblocking(output.as_raw_fd())?;
    let socket = secure_path::connect(&config, started)?;
    drive(&input, &output, &socket, &config, started)
}

fn main() {
    // Do not block the relay on a full diagnostic pipe at a terminal failure.
    let _ = Io::nonblocking(2);
    if let Err(e) = run() {
        let message = format!("glm-pair-relay: {e}\n");
        let _ = Io::write(2, message.as_bytes());
    }
    std::process::exit(74);
}

#[cfg(test)]
#[path = "relay/tests.rs"]
mod tests;
