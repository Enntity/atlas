// SPDX-License-Identifier: AGPL-3.0-only
//! Actual selected scheduler/Model/local worker replay; no native arithmetic claim.
use super::{issued, step_selected_serial, tests::context};
use crate::scheduler::{
    ActiveSeq,
    sched_ctx::SchedCtx,
    test_support::{RespRx, test_owned_seq},
};
use spark_model::{
    model::{
        TransformerModel,
        glm_c2_test_support::{Event, Fixture, Observer, Snapshot, Wire},
    },
    traits::{Model, SequenceState},
};

#[path = "glm_c2_fixture_test_process.rs"]
mod process;

const SLOT_BYTES: usize = 6 * 8192;
const E6: u32 = 0xffff_ffe6;
const F5: u32 = 0xffff_fff5;
const E1: u32 = 0xffff_ffe1;

fn prepared(rank: usize) -> (TransformerModel, [SequenceState; 4], Observer, Wire) {
    let mut fixture = Fixture::paired_compute_with_owner_capacity(rank, 4);
    fixture.deterministic_logits(true);
    let wire = fixture.install_wire();
    let (model, [s0, s1], observer) = fixture.into_parts();
    let s2 = model.alloc_sequence().unwrap();
    let s3 = model.alloc_sequence().unwrap();
    assert_eq!((s2.slot_idx, s3.slot_idx), (2, 3));
    let mut seqs = [s0, s1, s2, s3];
    let prompts = [
        vec![1, 2, 3, 4],
        vec![6, 5, 4, 3, 2, 1],
        vec![2, 4, 6, 1, 3],
        vec![7, 6, 5, 4, 3, 2, 1],
    ];
    for (seq, prompt) in seqs.iter_mut().zip(prompts) {
        seq.prompt_len = prompt.len();
        model.prefill(&prompt, seq, 37).unwrap();
    }
    assert!(
        wire.packets().is_empty(),
        "local prefill setup is not F0 transport proof"
    );
    assert_eq!(
        model
            .glm_paired_execution()
            .unwrap()
            .owner_capacity()
            .unwrap(),
        4
    );
    (model, seqs, observer, wire)
}

#[derive(Clone, Copy)]
struct Before {
    position: usize,
    outputs: usize,
    remaining: usize,
    tokens: [u32; 5],
}

struct Run {
    model: TransformerModel,
    worker: TransformerModel,
    active: Vec<ActiveSeq>,
    slots: [Option<SequenceState>; 4],
    observer: Observer,
    peer_observer: Observer,
    tx: Wire,
    rx: Wire,
    responses: Vec<RespRx>,
    sched: SchedCtx,
}

impl Run {
    fn new(reverse: bool) -> Self {
        let (model, seqs, observer, tx) = prepared(0);
        let (worker, seqs_peer, peer_observer, rx) = prepared(1);
        let (mut active, responses): (Vec<_>, Vec<_>) = seqs
            .into_iter()
            .map(|seq| {
                let first = (5 + seq.slot_idx as u32) % 8;
                let (mut a, response) = test_owned_seq(seq, vec![first], 128, None);
                a.finished = false;
                a.min_tokens = 0;
                a.lz_penalty = 0.0;
                (a, response)
            })
            .unzip();
        if reverse {
            active.reverse();
        }
        let mut run = Self {
            model,
            worker,
            active,
            slots: seqs_peer.map(Some),
            observer,
            peer_observer,
            tx,
            rx,
            responses,
            sched: SchedCtx::for_test(),
        };
        run.step(); // Intentional runtime RED at the existing 1..2 occupancy gate.
        let packets = run.tx.packets();
        assert_eq!(packets.len(), 20);
        for owner in 0..4 {
            let start = owner * 5;
            assert_eq!(packets[start], [owner as u32]);
            assert_eq!(packets[start + 1], [(5 + owner as u32) % 8]);
            assert_eq!(packets[start + 2], [owner as u32]);
            assert_eq!(packets[start + 3], [E1]);
            let a = run.owner(owner);
            assert_eq!(a.seq.seq_len, a.seq.prompt_len + 1);
            assert_eq!(a.output_tokens.len(), 2);
            assert_eq!(a.pending_drafts.len(), 4);
            assert_eq!(packets[start + 4].last(), Some(&a.last_token));
        }
        run.rx.queue(&packets);
        for _ in 0..8 {
            assert!(run.worker.ep_worker_step(&mut run.slots).unwrap());
        }
        run.rx.assert_drained();
        run.compare();
        for a in &mut run.active {
            a.presence_penalty = -0.25;
        }
        run.clear();
        run
    }
    fn owner(&self, slot: usize) -> &ActiveSeq {
        self.active.iter().find(|a| a.seq.slot_idx == slot).unwrap()
    }
    fn clear(&self) {
        self.tx.clear();
        self.observer.clear();
    }
    fn step(&mut self) {
        step_selected_serial(
            &self.model,
            &mut self.active,
            &self.sched,
            &context(&self.sched),
        )
        .unwrap();
    }
    fn compare(&self) {
        for a in &self.active {
            let peer = self.slots[a.seq.slot_idx].as_ref().unwrap();
            assert_eq!(a.seq.tokens, peer.tokens);
            assert_eq!(a.seq.seq_len, peer.seq_len);
            let rows = self.observer.private_cursor(&self.model, &a.seq).unwrap();
            assert_eq!(
                rows,
                self.peer_observer
                    .private_cursor(&self.worker, peer)
                    .unwrap()
            );
            assert_eq!(
                self.observer
                    .snapshot(&self.model, &a.seq, rows)
                    .unwrap()
                    .initial(),
                self.peer_observer
                    .snapshot(&self.worker, peer, rows)
                    .unwrap()
                    .initial()
            );
            if !a.finished {
                assert_eq!(rows, a.seq.seq_len + 3);
            }
        }
    }
    fn committed(&self, owner: usize, accepted: usize, before: Before) {
        assert!(accepted <= 4);
        let a = self.owner(owner);
        assert_eq!(a.seq.seq_len, before.position + accepted + 1);
        assert_eq!(
            &a.seq.tokens[before.position..],
            &before.tokens[..accepted + 1]
        );
        let emitted = (accepted + 1).min(before.remaining);
        assert_eq!(a.output_tokens.len(), before.outputs + emitted);
        let copied = accepted.min(emitted);
        assert_eq!(
            &a.output_tokens[before.outputs..before.outputs + copied],
            &before.tokens[1..1 + copied]
        );
        assert_eq!(a.output_tokens.last(), Some(&a.last_token));
        assert_eq!(a.pending_drafts.len(), if a.finished { 0 } else { 4 });
        if a.finished {
            self.model
                .glm_paired_execution()
                .unwrap()
                .validate_propose(&a.seq, a.last_token, a.seq.seq_len, 4, None)
                .unwrap();
        }
    }
    fn round(&mut self, expected_groups: &[usize], expected_singles: &[usize]) {
        let before: [Option<Before>; 4] = std::array::from_fn(|owner| {
            self.active
                .iter()
                .find(|a| a.seq.slot_idx == owner)
                .map(|a| Before {
                    position: a.seq.seq_len,
                    outputs: a.output_tokens.len(),
                    remaining: a.remaining,
                    tokens: issued(a, self.model.vocab_size()).unwrap(),
                })
        });
        self.clear();
        self.step();
        let packets = self.tx.packets();
        let (mut offset, mut groups, mut singles, mut commands) =
            (0, Vec::new(), Vec::new(), Vec::new());
        while offset < packets.len() {
            assert_eq!(packets[offset].len(), 1);
            assert_eq!(packets[offset + 1].len(), 1);
            let slot = packets[offset][0] as usize;
            let command = packets[offset + 1][0];
            commands.push((slot, command));
            match command {
                E6 => {
                    groups.push(slot);
                    assert!(matches!(slot, 0 | 2));
                    let payload = &packets[offset + 2];
                    assert_eq!(payload.len(), 26);
                    assert_eq!(&payload[..4], &[1, 2, 10, 1]);
                    assert_eq!(&packets[offset + 3][..2], &[1, 2]);
                    for ordinal in 0..2 {
                        let owner = slot + ordinal;
                        let previous = before[owner].unwrap();
                        let record = 4 + ordinal * 11;
                        assert_eq!(payload[record], owner as u32);
                        assert_eq!(payload[record + 5], previous.position as u32);
                        assert_eq!(&payload[record + 6..record + 11], &previous.tokens);
                        self.committed(owner, packets[offset + 3][2 + ordinal] as usize, previous);
                    }
                    offset += 4;
                }
                F5 => {
                    singles.push(slot);
                    assert_eq!(packets[offset + 2], [5]);
                    assert_eq!(packets[offset + 3], before[slot].unwrap().tokens);
                    self.committed(slot, packets[offset + 4][0] as usize, before[slot].unwrap());
                    offset += 5;
                }
                E1 => {
                    assert!(!self.owner(slot).finished);
                    assert_eq!(packets[offset + 2].len(), 8);
                    assert_eq!(
                        &packets[offset + 2][5..],
                        &[
                            self.owner(slot).seq.seq_len as u32,
                            4,
                            self.owner(slot).last_token
                        ]
                    );
                    offset += 3;
                }
                _ => panic!("unexpected steady selected command {command:x}"),
            }
        }
        assert_eq!(groups, expected_groups);
        assert_eq!(singles, expected_singles);
        assert_eq!(
            self.observer
                .events()
                .iter()
                .filter(|event| **event == Event::Read(80, 7))
                .count(),
            expected_groups.len() * 2 + expected_singles.len(),
            "every owner uses actual checked full-row selection"
        );
        self.rx.queue(&packets);
        for (slot, command) in commands {
            let untouched: Vec<_> = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(owner, seq)| {
                    if owner == slot || (command == E6 && owner == slot + 1) {
                        return None;
                    }
                    seq.as_ref().map(|seq| {
                        let rows = self
                            .peer_observer
                            .private_cursor(&self.worker, seq)
                            .unwrap();
                        (
                            owner,
                            seq.tokens.clone(),
                            seq.seq_len,
                            self.peer_observer
                                .snapshot(&self.worker, seq, rows)
                                .unwrap(),
                        )
                    })
                })
                .collect();
            assert!(self.worker.ep_worker_step(&mut self.slots).unwrap());
            for (owner, tokens, position, saved) in untouched {
                let seq = self.slots[owner].as_ref().unwrap();
                assert_eq!(seq.tokens, tokens);
                assert_eq!(seq.seq_len, position);
                unchanged_owner(&self.peer_observer, &saved, owner);
            }
        }
        self.rx.assert_drained();
        self.compare();
        assert!(
            self.responses
                .iter_mut()
                .all(|response| response.try_recv().is_err()),
            "step alone must not send retirement responses"
        );
    }
    fn retire(&mut self, owner: usize) {
        let index = self
            .active
            .iter()
            .position(|a| a.seq.slot_idx == owner)
            .unwrap();
        let mut a = self.active.remove(index);
        self.model.free_sequence(&mut a.seq).unwrap();
        let mut seq = self.slots[owner].take().unwrap();
        self.worker.free_sequence(&mut seq).unwrap();
        // CPU fixture retirement is explicit Model cleanup, not serving F1 proof.
    }
    fn close(mut self) {
        while let Some(a) = self.active.first() {
            self.retire(a.seq.slot_idx);
        }
        drop(self.active);
        drop(self.slots);
        drop(self.responses);
        self.model.teardown().unwrap();
        self.worker.teardown().unwrap();
    }
}

fn unchanged_owner(observer: &Observer, saved: &Snapshot, owner: usize) {
    let after = observer.read_snapshot(saved).unwrap();
    let slab = after.len() - 1;
    assert_eq!(after[slab].len(), 4 * SLOT_BYTES);
    assert_eq!(&after[..slab], &saved.initial()[..slab]);
    let start = owner * SLOT_BYTES;
    assert_eq!(
        &after[slab][start..start + SLOT_BYTES],
        &saved.initial()[slab][start..start + SLOT_BYTES]
    );
}

#[test]
fn actual_c4_reversed_groups_then_c3_singleton2_replay() {
    if process::isolated(
        "scheduler::glm_c2_serial::group_tests::actual_c4_reversed_groups_then_c3_singleton2_replay",
    ) {
        return;
    }
    let mut run = Run::new(true);
    assert_eq!(
        run.active
            .iter()
            .map(|a| a.seq.slot_idx)
            .collect::<Vec<_>>(),
        [3, 2, 1, 0]
    );
    for _ in 0..2 {
        run.round(&[0, 2], &[]);
    }
    run.retire(3);
    assert_eq!(run.active.len(), 3);
    run.round(&[0], &[2]);
    run.close();
}

#[test]
fn actual_c4_drain_retains_physical_survivor3_then_finishes_without_e1() {
    if process::isolated(
        "scheduler::glm_c2_serial::group_tests::actual_c4_drain_retains_physical_survivor3_then_finishes_without_e1",
    ) {
        return;
    }
    let mut run = Run::new(false);
    for a in &mut run.active {
        if a.seq.slot_idx != 3 {
            a.remaining = 1;
        }
    }
    run.round(&[0, 2], &[]);
    for owner in 0..3 {
        assert!(run.owner(owner).finished);
        run.retire(owner);
    }
    assert_eq!(run.active.len(), 1);
    assert_eq!(run.active[0].seq.slot_idx, 3);
    run.round(&[], &[3]);
    run.active[0].remaining = 1;
    run.round(&[], &[3]);
    assert!(run.active[0].finished && run.active[0].pending_drafts.is_empty());
    assert!(!run.tx.packets().iter().any(|packet| packet == &[E1]));
    run.clear();
    run.step();
    assert!(run.tx.packets().is_empty() && run.observer.events().is_empty());
    run.close();
}
