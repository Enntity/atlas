// SPDX-License-Identifier: AGPL-3.0-only

//! Exact real codec/state boundaries, not clock sleeps or a copied policy.
#[path = "../src/core.rs"]
mod core;
#[path = "../src/frame.rs"]
mod frame;
use frame::{Frame, Reader, HELLO, LEN};

#[test]
fn fixed_frame_validates_every_header_and_never_resets_partial_start() {
    let frame = Frame {
        kind: HELLO,
        session: [1; 32],
        instance: [2; 32],
        ordinal: 7,
        challenge: [3; 32],
    };
    let encoded = frame.encode();
    assert_eq!(Frame::decode(&encoded).unwrap(), frame);
    for offset in [0, 1, 2, 3, 4, 5, 6, 7] {
        let mut bad = encoded;
        bad[offset] ^= 0x80;
        assert!(Frame::decode(&bad).is_err());
    }
    let mut reader = Reader::new();
    for (i, byte) in encoded.iter().enumerate() {
        reader.remaining()[0] = *byte;
        let output = reader.received(1, 100 + i as u64).unwrap();
        if i == LEN - 1 {
            assert_eq!(output, Some(frame.clone()));
            assert_eq!(reader.started, None);
        } else {
            assert!(output.is_none());
            assert_eq!(reader.started, Some(100));
        }
    }
    for length in [0u32, 1, 107, 109, 4097, u32::MAX] {
        let mut reader = Reader::new();
        reader.remaining()[..4].copy_from_slice(&length.to_be_bytes());
        assert!(reader.received(4, 0).is_err());
    }
}

#[test]
fn completion_checks_original_frame_start_at_fresh_observed_clock() {
    use frame::check_deadline;
    let frame = Frame {
        kind: HELLO,
        session: [1; 32],
        instance: [2; 32],
        ordinal: 7,
        challenge: [3; 32],
    };
    let bytes = frame.encode();
    let mut reader = Reader::new();
    reader.remaining()[..4].copy_from_slice(&bytes[..4]);
    assert!(reader.received(4, 100).unwrap().is_none());
    let original_start = reader.started.unwrap();
    assert!(check_deadline(original_start, 399, 300).is_ok());
    reader.remaining().copy_from_slice(&bytes[4..]);
    assert_eq!(reader.received(LEN - 4, 399).unwrap(), Some(frame));
    assert_eq!(reader.started, None);
    // Actual codec cleared its partial state; the runner retains this start
    // across decode and uses this exact predicate before State::accept.
    assert!(check_deadline(original_start, 400, 300).is_err());
    assert!(check_deadline(original_start, 401, 300).is_err());
    // The post-send runner invokes the same predicate before clearing output.
    assert!(check_deadline(100, 399, 300).is_ok());
    assert!(check_deadline(100, 400, 300).is_err());
    assert!(check_deadline(100, 99, 300).is_err());
    assert!(check_deadline(u64::MAX, u64::MAX, 0).is_err());
}
