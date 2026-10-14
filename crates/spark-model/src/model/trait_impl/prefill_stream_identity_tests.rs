// SPDX-License-Identifier: AGPL-3.0-only
//! Request identity reaches the worker's layer forward. Split out of
//! `prefill_stream_tests.rs` (500-LoC cap).

use super::*;

#[test]
fn actual_worker_prefill_entries_preserve_request_identity_in_forward_context() {
    if isolated("actual_worker_prefill_entries_preserve_request_identity_in_forward_context") {
        return;
    }
    for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
        for identity in [None, Some("req-c2-worker".into())] {
            let mut f = Fixture::new(2, 2, 1);
            f.seq.request_id = identity.clone();
            invoke(&mut f, entry).unwrap();
            assert_eq!(f.request_ids(), vec![identity], "{entry:?}");
        }
    }
}

/// The head's identity command, then its prefill chunk: the worker binds the
/// identity on the slot and its layer forward sees it.
#[test]
fn actual_worker_identity_then_f0_keeps_identity_through_layer_forward() {
    if isolated("actual_worker_identity_then_f0_keeps_identity_through_layer_forward") {
        return;
    }
    for v2 in [false, true] {
        let mut f = Fixture::new(2, 2, 1);
        f.model.ep_protocol_v2 = v2;
        let id = b"req-c2-worker";
        let mut wire = vec![];
        if v2 {
            wire.push(0);
        }
        wire.extend([
            crate::model::request_identity::EP_CMD_REQUEST_ID,
            id.len() as u32,
        ]);
        for chunk in id.chunks(4) {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            wire.push(u32::from_le_bytes(word));
        }
        if v2 {
            wire.push(0);
        }
        wire.extend([0xFFFF_FFF0, 4, 0, 4, 1, 2, 3, 4]);
        f.script(&wire);
        let seq = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        let mut slots = vec![Some(seq)];
        assert!(f.model.ep_worker_step(&mut slots).unwrap());
        assert!(f.model.ep_worker_step(&mut slots).unwrap());
        assert_eq!(f.unread(), 0, "command words left unread");
        assert_eq!(
            f.request_ids()
                .iter()
                .map(|id| id.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("req-c2-worker")]
        );
        f.seq = slots.remove(0).unwrap();
        assert_eq!(f.seq.request_id.as_deref(), Some("req-c2-worker"));
        assert_eq!(f.seq.tokens, vec![1, 2, 3, 4]);
    }
}
