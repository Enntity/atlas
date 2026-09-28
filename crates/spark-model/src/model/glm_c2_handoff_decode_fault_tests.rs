// SPDX-License-Identifier: AGPL-3.0-only
//! Actual outer decode failures; host sentinels prove ownership, not numerics.
use super::{fixture::*, isolated};
use crate::layers::glm5_mtp::Glm5MtpProposerState;
use crate::speculative::DraftProposer;
use crate::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Target,
    Bonus,
    Completion,
}

fn private(seq: &SequenceState) -> &Glm5MtpProposerState {
    seq.proposer_state
        .as_ref()
        .unwrap()
        .as_any()
        .downcast_ref::<Glm5MtpProposerState>()
        .unwrap()
}

fn slot(f: &Fixture, owner: usize) -> DevicePtr {
    f.gpu.slab().offset(owner * 6 * ROW_BYTES)
}

fn bonus_copy(f: &Fixture, owner: usize) -> Event {
    Event::Copy(
        f.model.buffers.norm_output(),
        slot(f, owner).offset(5 * ROW_BYTES),
        ROW_BYTES,
        DEFAULT,
    )
}

fn prepared(rank: usize, victim: usize) -> Fixture {
    let mut f = Fixture::new(rank);
    f.gpu.write_span(f.gpu.slab(), &vec![0xa5; SLAB_BYTES]);
    for (owner, prompt) in [[1, 2, 3, 4], [4, 3, 2, 1]].iter().enumerate() {
        f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
        assert_eq!(private(&f.seqs[owner]).seq_len, 3);
    }
    assert_ne!(
        f.gpu.read_span(slot(&f, 0), ROW_BYTES),
        f.gpu.read_span(slot(&f, 1), ROW_BYTES)
    );
    let peer = 1 - victim;
    f.model
        .decode(5 + peer as u32, &mut f.seqs[peer], CALLER)
        .unwrap();
    assert_eq!(f.seqs[peer].seq_len, 5);
    assert_eq!(
        f.gpu
            .read_span(slot(&f, peer).offset(5 * ROW_BYTES), ROW_BYTES),
        vec![5 + peer as u8 + 0x24; ROW_BYTES]
    );
    assert_eq!(
        f.gpu
            .read_span(slot(&f, victim).offset(5 * ROW_BYTES), ROW_BYTES),
        vec![0xa5; ROW_BYTES]
    );
    f.gpu.clear();
    f
}

fn unique(events: &[Event], expected: &Event) -> usize {
    let found: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| *event == expected)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "boundary {expected:?} must execute once; events={events:#?}"
    );
    found[0]
}

fn failure_ordinal(rank: usize, victim: usize, boundary: Boundary) -> usize {
    let mut f = prepared(rank, victim);
    f.model
        .decode(5 + victim as u32, &mut f.seqs[victim], CALLER)
        .unwrap();
    assert_eq!(f.seqs[victim].seq_len, 5);
    assert_eq!(
        private(&f.seqs[victim]).seq_len,
        3,
        "target decode must not consume the bootstrap pair"
    );
    let events = f.gpu.trace();
    let target = unique(&events, &Event::Target(1, 4, DEFAULT));
    let bonus = unique(&events, &bonus_copy(&f, victim));
    assert!(target < bonus);
    assert_eq!(events.get(bonus + 1), Some(&Event::Sync(DEFAULT)));
    assert_eq!(
        f.gpu
            .read_span(slot(&f, victim).offset(5 * ROW_BYTES), ROW_BYTES),
        vec![5 + victim as u8 + 0x24; ROW_BYTES]
    );
    assert_ne!(
        f.gpu
            .read_span(slot(&f, 0).offset(5 * ROW_BYTES), ROW_BYTES),
        f.gpu
            .read_span(slot(&f, 1).offset(5 * ROW_BYTES), ROW_BYTES)
    );
    match boundary {
        Boundary::Target => target + 1,
        Boundary::Bonus => bonus + 1,
        Boundary::Completion => bonus + 2,
    }
}

struct Peer {
    bytes: Vec<u8>,
    blocks: BTreeSet<u32>,
    tokens: Vec<u32>,
    rows: Vec<(DevicePtr, DevicePtr, Vec<u8>, Vec<u8>)>,
}

impl Peer {
    fn snapshot(f: &Fixture, owner: usize) -> Self {
        let seq = &f.seqs[owner];
        let rows = f
            .head
            .paired_test_kv_rows(
                seq.proposer_state.as_ref().unwrap().as_ref(),
                f.model.gpu.as_ref(),
                3,
            )
            .unwrap();
        Self {
            bytes: f.gpu.read_span(slot(f, owner), 6 * ROW_BYTES),
            blocks: private(seq).block_table.iter().copied().collect(),
            tokens: seq.tokens.clone(),
            rows: rows
                .into_iter()
                .map(|(k, v)| (k, v, f.gpu.read_span(k, 1024), f.gpu.read_span(v, 1024)))
                .collect(),
        }
    }

    fn unchanged(&self, f: &Fixture, owner: usize) {
        assert_eq!(f.gpu.read_span(slot(f, owner), 6 * ROW_BYTES), self.bytes);
        assert_eq!(private(&f.seqs[owner]).seq_len, 3);
        assert_eq!(
            private(&f.seqs[owner])
                .block_table
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            self.blocks
        );
        assert_eq!(f.seqs[owner].tokens, self.tokens);
        assert_eq!(f.seqs[owner].seq_len, 5);
        self.prefix_unchanged(f);
    }

    fn prefix_unchanged(&self, f: &Fixture) {
        for (k, v, kb, vb) in &self.rows {
            assert_eq!(f.gpu.read_span(*k, 1024), *kb);
            assert_eq!(f.gpu.read_span(*v, 1024), *vb);
        }
    }
}

fn fails_closed(boundary: Boundary) {
    for rank in 0..2 {
        for victim in 0..2 {
            let ordinal = failure_ordinal(rank, victim, boundary);
            let mut f = prepared(rank, victim);
            let peer = 1 - victim;
            let saved = Peer::snapshot(&f, peer);
            let failed_blocks: BTreeSet<_> = private(&f.seqs[victim])
                .block_table
                .iter()
                .copied()
                .collect();
            let tail = f.gpu.read_span(slot(&f, victim), ROW_BYTES);
            assert_eq!((failed_blocks.len(), saved.blocks.len()), (128, 128));
            assert!(failed_blocks.is_disjoint(&saved.blocks));
            f.gpu.fail.store(ordinal, Ordering::Relaxed);
            let error = f
                .model
                .decode(5 + victim as u32, &mut f.seqs[victim], CALLER)
                .unwrap_err();
            assert!(
                format!("{error:#}").contains("injected fixture operation failure"),
                "actual {boundary:?}/rank{rank}/victim{victim} injection must execute: {error:#}"
            );
            let events = f.gpu.trace();
            let expected = match boundary {
                Boundary::Target => Event::Target(1, 4, DEFAULT),
                Boundary::Bonus => bonus_copy(&f, victim),
                Boundary::Completion => Event::Sync(DEFAULT),
            };
            assert_eq!(
                events.get(ordinal - 1),
                Some(&expected),
                "fresh fixture must fail the intended boundary"
            );
            if matches!(boundary, Boundary::Completion) {
                assert_eq!(events.get(ordinal - 2), Some(&bonus_copy(&f, victim)));
            }
            saved.unchanged(&f, peer);
            assert_eq!(f.gpu.read_span(slot(&f, victim), ROW_BYTES), tail);
            // The existing test-only accessor validates the real lease before
            // returning read-only addresses: no receipt or Ready is forged.
            assert!(
                f.head
                    .paired_test_kv_rows(
                        f.seqs[victim].proposer_state.as_ref().unwrap().as_ref(),
                        f.model.gpu.as_ref(),
                        1
                    )
                    .is_err(),
                "failed producer lease must already be unreadable"
            );
            f.gpu.clear();
            f.model.gpu.synchronize(DEFAULT).unwrap();
            f.gpu.clear();
            assert!(
                f.model
                    .decode(5 + victim as u32, &mut f.seqs[victim], CALLER)
                    .is_err()
            );
            assert!(
                f.gpu.trace().is_empty(),
                "later successful sync cannot allow any retry work"
            );
            assert!(
                f.model
                    .run_mtp_propose_inner(7, 5, 4, &mut f.seqs[victim], None)
                    .is_err()
            );
            assert!(
                f.gpu.trace().is_empty(),
                "failed bonus bytes are not a publishable view"
            );
            assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
            saved.unchanged(&f, peer);
            // Healthy peer progresses through the actual owned-tail repair and
            // first four-draft proposer; no verdict/admission path is enabled.
            let drafts = f
                .model
                .run_mtp_propose_inner(7, 5, 4, &mut f.seqs[peer], None)
                .unwrap();
            assert_eq!(drafts.len(), 4);
            assert_eq!(private(&f.seqs[peer]).seq_len, 8);
            saved.prefix_unchanged(&f);
            assert_eq!(f.gpu.read_span(slot(&f, victim), ROW_BYTES), tail);
            f.model.free_sequence(&mut f.seqs[peer]).unwrap();
            f.model.free_sequence(&mut f.seqs[peer]).unwrap();
            let mut replacement = f.head.alloc_state(f.model.gpu.as_ref()).unwrap();
            let blocks: BTreeSet<_> = replacement
                .as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .unwrap()
                .block_table
                .iter()
                .copied()
                .collect();
            assert_eq!(
                blocks, saved.blocks,
                "only healthy peer reserve may be reused"
            );
            assert!(blocks.is_disjoint(&failed_blocks));
            assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
            f.head
                .free_state(f.model.gpu.as_ref(), None, replacement.as_mut())
                .unwrap();
            // Producer failure is owner-local; the later selected cleanup
            // error is session-terminal, after the healthy peer control.
            let free = f.head.paired_test_free_blocks();
            f.gpu.clear();
            assert!(f.model.free_sequence(&mut f.seqs[victim]).is_err());
            assert!(f.model.alloc_sequence().is_err());
            assert!(f.gpu.trace().is_empty());
            assert_eq!(f.head.paired_test_free_blocks(), free);
            assert!(!f.model.ssm_pool.slot_is_free(victim));
        }
    }
}

#[test]
fn actual_outer_decode_target_failure_quarantines_only_its_owner() {
    if isolated("decode_fault_tests::actual_outer_decode_target_failure_quarantines_only_its_owner")
    {
        return;
    }
    fails_closed(Boundary::Target);
}

#[test]
fn actual_outer_decode_bonus_copy_failure_quarantines_only_its_owner() {
    if isolated(
        "decode_fault_tests::actual_outer_decode_bonus_copy_failure_quarantines_only_its_owner",
    ) {
        return;
    }
    fails_closed(Boundary::Bonus);
}

#[test]
fn actual_outer_decode_completion_failure_quarantines_only_its_owner() {
    if isolated(
        "decode_fault_tests::actual_outer_decode_completion_failure_quarantines_only_its_owner",
    ) {
        return;
    }
    fails_closed(Boundary::Completion);
}
