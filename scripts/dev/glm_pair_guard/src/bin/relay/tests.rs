// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use std::io::Write;
use std::os::unix::net::UnixStream;

pub(super) fn config() -> Config {
    Config::parse(
        &[
            "--session",
            &"1".repeat(64),
            "--rank",
            "0",
            "--connect-ms",
            "1000",
            "--frame-ms",
            "500",
            "--campaign-ms",
            "2000",
            "--poll-ms",
            "10",
        ]
        .map(String::from),
    )
    .unwrap()
}
fn pipe() -> (OwnedFd, OwnedFd) {
    let mut fds = [-1; 2];
    assert_eq!(
        unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) },
        0
    );
    (Io::owned(fds[0]).unwrap(), Io::owned(fds[1]).unwrap())
}
fn receive(fd: i32, length: usize) -> Vec<u8> {
    let mut bytes = vec![0; length];
    let mut at = 0;
    let started = Io::now().unwrap();
    while at != length {
        frame::check_deadline(started, Io::now().unwrap(), 1000).unwrap();
        match Io::read(fd, &mut bytes[at..]).unwrap() {
            Some(0) => panic!("relay closed before forwarding actual frame"),
            Some(n) => at += n,
            None => Io::poll(
                &mut [libc::pollfd {
                    fd,
                    events: libc::POLLIN,
                    revents: 0,
                }],
                10,
            )
            .unwrap(),
        }
    }
    bytes
}
fn old(kind: u8) -> frame::Frame {
    frame::Frame {
        kind,
        session: [1; 32],
        instance: [2; 32],
        ordinal: 1,
        challenge: [3; 32],
    }
}
#[test]
fn actual_pipe_and_socket_relay_preserves_both_directions_then_fails_eof() {
    let (input, writer) = pipe();
    let (reader, output) = pipe();
    let (socket, mut peer) = UnixStream::pair().unwrap();
    socket.set_nonblocking(true).unwrap();
    peer.set_nonblocking(true).unwrap();
    let renew = old(frame::RENEW).encode();
    let receipt = wire::Frame {
        rank: 0,
        body: wire::Body::Quiescent(wire::Quiescent {
            pair_digest: [1; 32],
            child_instance: [2; 32],
            epoch: 1,
            last_command: u32::MAX,
            receipt_nonce: [3; 32],
        }),
    }
    .encode()
    .unwrap();
    assert_eq!(
        Io::write(writer.as_raw_fd(), &renew).unwrap(),
        Some(renew.len())
    );
    peer.write_all(receipt.as_slice()).unwrap();
    let task = std::thread::spawn(move || {
        drive(
            &input,
            &output,
            &socket.into(),
            &config(),
            Io::now().unwrap(),
        )
    });
    assert_eq!(
        receive(reader.as_raw_fd(), receipt.as_slice().len()),
        receipt.as_slice()
    );
    assert_eq!(receive(peer.as_raw_fd(), renew.len()), renew);
    drop(writer);
    assert!(task.join().unwrap().is_err());
}

#[test]
fn cli_rejects_path_rank_and_duration_substitution() {
    let valid = [
        "--session",
        &"1".repeat(64),
        "--rank",
        "0",
        "--connect-ms",
        "1000",
        "--frame-ms",
        "500",
        "--campaign-ms",
        "2000",
        "--poll-ms",
        "10",
    ]
    .map(String::from);
    for (index, value) in [
        (1, "../rank0"),
        (1, &"A".repeat(64)),
        (1, &"0".repeat(64)),
        (3, "2"),
        (5, "0"),
        (7, "+1"),
        (9, "86400001"),
        (11, "501"),
    ] {
        let mut args = valid.clone();
        args[index] = value.into();
        assert!(Config::parse(&args).is_err());
    }
    assert!(Config::parse(&valid[..10]).is_err());
    assert!(config().check(u64::MAX, u64::MAX).is_err());
}

#[test]
fn actual_pump_rejects_wrong_direction_rank_and_guard_eof() {
    let wrong_rank = wire::Frame {
        rank: 1,
        body: wire::Body::Quiescent(wire::Quiescent {
            pair_digest: [1; 32],
            child_instance: [2; 32],
            epoch: 1,
            last_command: u32::MAX,
            receipt_nonce: [3; 32],
        }),
    }
    .encode()
    .unwrap();
    for case in 0..5 {
        let (input, writer) = pipe();
        let (reader, output) = pipe();
        let (socket, mut peer) = UnixStream::pair().unwrap();
        socket.set_nonblocking(true).unwrap();
        peer.set_nonblocking(true).unwrap();
        match case {
            0 => {
                Io::write(writer.as_raw_fd(), &old(frame::START).encode()).unwrap();
            }
            1 => {
                Io::write(writer.as_raw_fd(), &old(frame::HELLO).encode()).unwrap();
            }
            2 => {
                peer.write_all(wrong_rank.as_slice()).unwrap();
            }
            3 => {
                peer.write_all(&old(frame::RENEW).encode()).unwrap();
            }
            4 => {
                peer.shutdown(std::net::Shutdown::Both).unwrap();
            }
            _ => unreachable!(),
        }
        let socket: OwnedFd = socket.into();
        let failure = drive(&input, &output, &socket, &config(), Io::now().unwrap()).unwrap_err();
        assert!(
            failure.to_string().contains(match case {
                0 | 1 | 3 => "legacy frame direction",
                2 => "rank mismatch",
                4 => "EOF",
                _ => unreachable!(),
            }),
            "case {case}: {failure}"
        );
        if case <= 1 {
            // Assert before any invalid command reaches the guard; a later
            // campaign timeout is not evidence of pre-forward refusal.
            assert_eq!(Io::read(peer.as_raw_fd(), &mut [0; 1]).unwrap(), None);
        }
        assert_eq!(Io::read(reader.as_raw_fd(), &mut [0; 1]).unwrap(), None);
    }
}

#[test]
fn actual_partial_input_and_blocked_output_keep_finite_deadlines() {
    for blocked_output in [false, true] {
        let (input, writer) = pipe();
        let (_reader, output) = pipe();
        let (socket, mut peer) = UnixStream::pair().unwrap();
        socket.set_nonblocking(true).unwrap();
        if blocked_output {
            while Io::write(output.as_raw_fd(), &[7; 4096]).unwrap().is_some() {}
            peer.write_all(&old(frame::HELLO).encode()).unwrap();
        } else {
            Io::write(writer.as_raw_fd(), &[0]).unwrap();
        }
        let mut config = config();
        config.frame = 30;
        config.poll = 5;
        config.campaign = 1000;
        let started = Io::now().unwrap();
        let failure = drive(&input, &output, &socket.into(), &config, started).unwrap_err();
        assert!(failure.to_string().contains("frame deadline"), "{failure}");
        assert!(Io::now().unwrap() - started >= config.frame);
    }
}
