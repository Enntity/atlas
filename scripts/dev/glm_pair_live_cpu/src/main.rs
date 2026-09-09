// SPDX-License-Identifier: AGPL-3.0-only

//! Controller-only connection/release fixture. No Docker, GPU or Model proof.
//! Executes exact production Child and server-consumer sources in two actual
//! private PID1/proc namespaces; fixture launch IDs are not Docker assertions.
#![allow(dead_code)] // Included production modules expose later integration APIs.

#[path = "../../glm_pair_guard/src/child.rs"]
mod child;
#[path = "../../glm_pair_guard/src/core.rs"]
mod core;
#[path = "../../glm_pair_guard/src/frame.rs"]
mod frame;
// The owning workspace formats these edition-2024 sources. Do not rewrite them
// with this edition-2021 fixture's formatting rules when traversing path modules.
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/inherited.rs"]
mod inherited;
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/inherited_startup.rs"]
mod inherited_startup;
#[path = "../../glm_pair_guard/src/linux.rs"]
mod linux;
#[path = "../../glm_pair_guard/src/live.rs"]
mod live;
mod namespace;

use anyhow::{bail, ensure, Context, Result};
use atlas_glm_pair_io::{identity, Channel, Credentials};
use atlas_glm_pair_wire as wire;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;
use std::time::Duration;

fn read_frame(socket: &mut UnixStream, direction: wire::Direction) -> Result<wire::Frame> {
    let mut prefix = [0; 4];
    socket.read_exact(&mut prefix)?;
    let size = u32::from_be_bytes(prefix) as usize;
    ensure!(
        (12..=wire::MAX_ENCODED - 4).contains(&size),
        "bounded frame"
    );
    let mut bytes = vec![0; size + 4];
    bytes[..4].copy_from_slice(&prefix);
    socket.read_exact(&mut bytes[4..])?;
    Ok(wire::Frame::decode(&bytes, direction)?)
}

fn write_frame(socket: &mut UnixStream, frame: wire::Frame) -> Result<()> {
    socket.write_all(frame.encode()?.as_slice())?;
    Ok(())
}

fn packet(channel: &Channel, expected: Credentials, deadline: u64) -> Result<Vec<u8>> {
    loop {
        identity::check_deadline(deadline)?;
        if let Some(bytes) = channel.receive(expected)? {
            return Ok(bytes);
        }
        linux::Io::poll(
            &mut [libc::pollfd {
                fd: channel.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            }],
            5,
        )?;
    }
}

fn send_packet(channel: &Channel, bytes: &[u8], deadline: u64) -> Result<()> {
    loop {
        identity::check_deadline(deadline)?;
        if channel.send(bytes)? {
            return Ok(());
        }
        linux::Io::poll(
            &mut [libc::pollfd {
                fd: channel.as_raw_fd(),
                events: libc::POLLOUT,
                revents: 0,
            }],
            5,
        )?;
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(text: &str) -> Result<Vec<u8>> {
    ensure!(
        text.len() % 2 == 0 && text.is_ascii() && text.len() <= 4096,
        "bounded hex"
    );
    (0..text.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&text[i..i + 2], 16)?))
        .collect()
}

fn policy() -> wire::Policy {
    wire::Policy {
        startup: 30000,
        lease: 30000,
        challenge: 5000,
        frame: 3000,
        campaign: 60000,
        poll: 250,
        reap: 10000,
        child_handshake: 30000,
        quiescent_wait: 10000,
        exit: 5000,
    }
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--guard") if args.len() == 4 => namespace::guard(args[2].parse()?, &args[3]),
        Some("--consumer") if args.len() == 3 => namespace::consumer(&unhex(&args[2])?),
        Some("--consumer-files") if args.len() == 2 => {
            if std::env::var("ATLAS_PAIR_CPU_MODE").as_deref() == Ok("bad-environment") {
                // CPU-only pre-ingress mismatch, never a production test flag.
                std::env::set_var("ATLAS_UNRECORDED_CPU_KEY", "1");
            }
            let session = unsafe { inherited_startup::receive(20000, 512 * 1024 * 1024) }?;
            let delay: u64 = std::env::var("ATLAS_PAIR_CPU_DELAY_MS")?.parse()?;
            ensure!(delay <= 6000, "bounded CPU-only delay");
            std::thread::sleep(Duration::from_millis(delay));
            if std::env::var("ATLAS_PAIR_CPU_MODE")? == "unreleased-zero" {
                unsafe { libc::_exit(0) }
            }
            session.exit_after_quiescence();
        }
        Some("--mount-guard") if args.len() == 4 => namespace::mount_guard(&args[2], &args[3]),
        Some("--run-main") if args.len() == 3 => namespace::main_controller(&args[2], "valid"),
        Some("--run-main") if args.len() == 4 => namespace::main_controller(&args[2], &args[3]),
        Some("--run") if args.len() == 2 => namespace::controller("valid"),
        Some("--run") if args.len() == 3 => namespace::controller(&args[2]),
        _ => bail!("usage: glm-pair-live-cpu --run (controller root; no GPUs)"),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CPU boundary FAIL: {error:#}");
        // Namespace PID1 must not return into ordinary teardown with a live child.
        unsafe { libc::_exit(74) }
    }
}
