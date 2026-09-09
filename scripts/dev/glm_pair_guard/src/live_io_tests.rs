// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Unix stream I/O and bounded buffer checks, not PID1/release authority.
use super::*;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

fn pair() -> (UnixStream, UnixStream) {
    let (sender, receiver) = UnixStream::pair().unwrap();
    sender.set_nonblocking(true).unwrap();
    receiver.set_nonblocking(true).unwrap();
    (sender, receiver)
}
fn old(kind: u8, ordinal: u64) -> frame::Frame {
    frame::Frame {
        kind,
        session: [1; 32],
        instance: [2; 32],
        ordinal,
        challenge: [3; 32],
    }
}

#[test]
fn actual_stream_one_frame_per_read_keeps_original_clock() {
    let (mut sender, receiver) = pair();
    let first = old(frame::RENEW, 1);
    let second = old(frame::REVOKE, 2);
    sender.write_all(&first.encode()[..2]).unwrap();
    let mut input = Reader::new();
    assert!(input.read(receiver.as_raw_fd(), 100).unwrap().is_none());
    assert!(input.check(149, 50).is_ok());
    assert!(input.check(150, 50).is_err());
    assert!(input.check(99, 50).is_err());
    sender.write_all(&first.encode()[2..]).unwrap();
    sender.write_all(&second.encode()).unwrap();
    assert!(input.read(receiver.as_raw_fd(), 110).unwrap().is_none());
    let (Incoming::Legacy(decoded), began) =
        input.read(receiver.as_raw_fd(), 120).unwrap().unwrap()
    else {
        panic!("legacy frame")
    };
    assert_eq!(decoded, first);
    assert_eq!(began, 100);
    assert!(input.idle());
    assert!(input.read(receiver.as_raw_fd(), 130).unwrap().is_none());
    let (Incoming::Legacy(decoded), began) =
        input.read(receiver.as_raw_fd(), 140).unwrap().unwrap()
    else {
        panic!("second frame")
    };
    assert_eq!(decoded, second);
    assert_eq!(began, 130);
}

#[test]
fn actual_stream_rejects_unknown_prefix_and_truncation() {
    for prefix in [u32::MAX, 0, 4092, 100] {
        let (mut sender, receiver) = pair();
        sender.write_all(&prefix.to_be_bytes()).unwrap();
        assert!(Reader::new().read(receiver.as_raw_fd(), 1).is_err());
    }
    let (mut sender, receiver) = pair();
    sender
        .write_all(&old(frame::RENEW, 1).encode()[..20])
        .unwrap();
    let mut input = Reader::new();
    assert!(input.read(receiver.as_raw_fd(), 1).unwrap().is_none());
    assert!(input.read(receiver.as_raw_fd(), 2).unwrap().is_none());
    drop(sender);
    assert!(input.read(receiver.as_raw_fd(), 3).is_err());
}

#[test]
fn actual_outputs_prioritize_lease_without_interleaving_partial_live() {
    for partial in [false, true] {
        let (sender, mut receiver) = pair();
        let now = Io::now().unwrap();
        let mut queue = Outputs::new();
        let mut live = Output::new(b"LIVE-RECORD", now).unwrap();
        if partial {
            // Real completed prefix write establishes this continuation offset;
            // this does not claim the kernel was forced into a short send.
            assert_eq!(Io::write(sender.as_raw_fd(), b"LIVE-").unwrap(), Some(5));
            live.sent = 5;
        }
        queue.live(live).unwrap();
        queue.lease(Output::new(b"LEASE", now).unwrap()).unwrap();
        assert!(queue.lease(Output::new(b"second", now).unwrap()).is_err());
        assert!(queue.live(Output::new(b"second", now).unwrap()).is_err());
        queue.send(sender.as_raw_fd(), 1000).unwrap();
        queue.send(sender.as_raw_fd(), 1000).unwrap();
        assert!(!queue.pending());
        let mut bytes = [0u8; 16];
        receiver.read_exact(&mut bytes).unwrap();
        assert_eq!(
            &bytes,
            if partial {
                b"LIVE-RECORDLEASE"
            } else {
                b"LEASELIVE-RECORD"
            }
        );
    }
}

#[test]
fn every_queued_output_and_overflow_bound_is_checked() {
    let mut queue = Outputs::new();
    queue.lease(Output::new(b"lease", 100).unwrap()).unwrap();
    queue.live(Output::new(b"live", 120).unwrap()).unwrap();
    assert!(queue.check(149, 50).is_ok());
    assert!(queue.check(150, 50).is_err());
    assert!(queue.check(99, 50).is_err());
    assert!(Output::new(&[], 0).is_err());
    assert!(Output::new(&[0; wire::MAX_ENCODED + 1], 0).is_err());
    assert!(end(u64::MAX, 1).is_err());
}

#[test]
fn actual_stream_eof_is_typed_without_erasing_partial_frame_state() {
    for partial in [false, true] {
        let (mut sender, receiver) = pair();
        let mut reader = Reader::new();
        if partial {
            sender.write_all(&[0, 0]).unwrap();
            assert!(reader.read(receiver.as_raw_fd(), 1).unwrap().is_none());
        }
        drop(sender);
        let failure = match reader.read(receiver.as_raw_fd(), 2) {
            Err(e) => e,
            Ok(_) => panic!("actual stream EOF must be an error"),
        };
        assert_eq!(failure.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(reader.idle(), !partial);
    }
}
