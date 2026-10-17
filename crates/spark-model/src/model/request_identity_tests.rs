// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::model::impl_a2_ep_worker::idle_command_fixture as fixture;
use crate::traits::Model;
use fixture::{Event, Fixture};

#[test]
fn request_id_codec_preserves_utf8_and_refuses_invalid_payloads() {
    for value in [
        None,
        Some(""),
        Some("req-c2-arithmetic-r0"),
        Some("req-é-🦀"),
    ] {
        let (length, words) = encode(value).unwrap();
        assert_eq!(decode(length, &words).unwrap().as_deref(), value);
    }
    assert_eq!(encode(Some("")).unwrap(), (0, vec![]));
    assert!(encode(Some(&"x".repeat(MAX_ID_BYTES + 1))).is_err());
    assert!(decode(MAX_ID_BYTES as u32 + 1, &[]).is_err());
    assert!(decode(1, &[]).is_err());
    assert!(decode(1, &[0xff]).is_err());
    assert!(decode(1, &[0x10061]).is_err());
}

#[test]
fn actual_worker_binding_targets_v1_v2_slots_and_clears_reused_identity() {
    for v2 in [false, true] {
        let slot = if v2 { 1 } else { 0 };
        let (length, payload) = encode(Some("req-c2-longer-r0")).unwrap();
        let mut wire = vec![];
        if v2 {
            wire.push(slot);
        }
        wire.extend([EP_CMD_REQUEST_ID, length]);
        wire.extend(payload);
        if v2 {
            wire.push(slot);
        }
        wire.extend([EP_CMD_REQUEST_ID, NO_REQUEST_ID]);
        let f = Fixture::new(1, v2, &wire);
        let mut slots = vec![
            Some(f.model.alloc_sequence().unwrap()),
            Some(f.model.alloc_sequence().unwrap()),
        ];
        let other = (1 - slot) as usize;
        slots[other].as_mut().unwrap().request_id = Some("other-request".into());
        assert!(f.model.ep_worker_step(&mut slots).unwrap());
        assert_eq!(
            slots[slot as usize].as_ref().unwrap().request_id.as_deref(),
            Some("req-c2-longer-r0")
        );
        assert_eq!(
            slots[other].as_ref().unwrap().request_id.as_deref(),
            Some("other-request")
        );
        assert!(f.model.ep_worker_step(&mut slots).unwrap());
        assert_eq!(slots[slot as usize].as_ref().unwrap().request_id, None);
        assert!(f.record.words.lock().is_empty());
    }
}

#[test]
fn actual_sender_writes_length_and_packed_bytes_before_following_commands() {
    for v2 in [false, true] {
        let f = Fixture::new(0, v2, &[]);
        f.model
            .ep_broadcast_request_id_for_seq(1, Some("req-c2"))
            .unwrap();
        let writes: Vec<Vec<u8>> = f
            .events()
            .into_iter()
            .filter_map(|event| {
                if let Event::H2d(_, bytes) = event {
                    Some(bytes)
                } else {
                    None
                }
            })
            .collect();
        let mut expected = vec![];
        if v2 {
            expected.push(1u32.to_le_bytes().to_vec());
        }
        expected.extend([
            EP_CMD_REQUEST_ID.to_le_bytes().to_vec(),
            6u32.to_le_bytes().to_vec(),
            b"req-c2\0\0".to_vec(),
        ]);
        assert_eq!(writes, expected);
    }
}

#[test]
fn malformed_worker_identity_never_overwrites_existing_binding() {
    for payload in [
        vec![1, 0xff],
        vec![1, 0x10061],
        vec![MAX_ID_BYTES as u32 + 1],
    ] {
        let mut wire = vec![EP_CMD_REQUEST_ID];
        wire.extend(payload);
        let f = Fixture::new(1, false, &wire);
        let mut seq = f.model.alloc_sequence().unwrap();
        seq.request_id = Some("earlier-request".into());
        let mut slots = vec![Some(seq)];
        assert!(f.model.ep_worker_step(&mut slots).is_err());
        assert_eq!(
            slots[0].as_ref().unwrap().request_id.as_deref(),
            Some("earlier-request")
        );
    }
}
