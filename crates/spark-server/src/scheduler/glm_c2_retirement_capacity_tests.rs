// SPDX-License-Identifier: AGPL-3.0-only
//! Actual bounded-owner allocation/retirement and addressed F1; no native proof.
use super::*;
use spark_model::model::glm_c2_test_support::Snapshot;

struct Owners<const N: usize> {
    model: TransformerModel,
    worker: TransformerModel,
    active: Vec<ActiveSeq>,
    slots: Vec<Option<SequenceState>>,
    observer: Observer,
    peer_observer: Observer,
    tx: Wire,
    rx: Wire,
    responses: Vec<RespRx>,
}

type Four = Owners<4>;
type Eight = Owners<8>;

impl<const N: usize> Owners<N> {
    fn new(reverse: bool) -> Self {
        let prepare = |rank| {
            let mut f = if N <= 4 {
                Fixture::paired_compute_with_owner_capacity(rank, N)
            } else {
                Fixture::owner_compute_with_owner_capacity(rank, N)
            };
            f.deterministic_logits(true);
            let wire = f.install_wire();
            let (model, seqs, observer) = f.into_parts();
            let mut seqs = Vec::from(seqs);
            for slot in 2..N {
                let seq = model.alloc_sequence().unwrap();
                assert_eq!(seq.slot_idx, slot);
                seqs.push(seq);
            }
            assert_eq!(
                model
                    .glm_paired_execution()
                    .unwrap()
                    .owner_capacity()
                    .unwrap(),
                N
            );
            for seq in &mut seqs {
                let prompt = [1, 2, 3, (4 + seq.slot_idx as u32) % 8];
                seq.prompt_len = prompt.len();
                model.prefill(&prompt, seq, 37).unwrap();
            }
            wire.clear();
            observer.clear();
            (model, seqs, observer, wire)
        };
        let (model, seqs, observer, tx) = prepare(0);
        let (worker, worker_seqs, peer_observer, rx) = prepare(1);
        let (mut active, responses): (Vec<_>, Vec<_>) = seqs
            .into_iter()
            .map(|seq| {
                let (mut a, response) = test_owned_seq(seq, vec![5], 128, None);
                a.finished = false;
                (a, response)
            })
            .unzip();
        if reverse {
            active.reverse();
        }
        Self {
            model,
            worker,
            active,
            slots: worker_seqs.into_iter().map(Some).collect(),
            observer,
            peer_observer,
            tx,
            rx,
            responses,
        }
    }

    fn seq(&self, slot: usize) -> &SequenceState {
        &self
            .active
            .iter()
            .find(|a| a.seq.slot_idx == slot)
            .unwrap()
            .seq
    }
    fn finish(&mut self, slot: usize) {
        let a = self
            .active
            .iter_mut()
            .find(|a| a.seq.slot_idx == slot)
            .unwrap();
        a.finished = true;
        a.remaining = 0;
    }
    fn retire(&mut self) -> anyhow::Result<()> {
        retire_selected_finished_sequences(&self.model, &mut self.active, 2044)
    }
    fn saved(&self, slot: usize) -> (Snapshot, usize, Vec<u32>, Vec<u32>) {
        let seq = self.seq(slot);
        let cursor = self.observer.private_cursor(&self.model, seq).unwrap();
        (
            self.observer.snapshot(&self.model, seq, cursor).unwrap(),
            cursor,
            seq.tokens.clone(),
            seq.block_table.clone(),
        )
    }
    fn unchanged(&self, slot: usize, saved: &(Snapshot, usize, Vec<u32>, Vec<u32>), live: bool) {
        let seq = self.seq(slot);
        assert_eq!(seq.ssm_slot_idx(), Some(slot));
        assert_eq!(seq.tokens, saved.2);
        assert_eq!(seq.block_table, saved.3);
        let bytes = self.observer.read_snapshot(&saved.0).unwrap();
        let original = saved.0.initial();
        let last = bytes.len() - 1;
        assert_eq!(bytes[..last], original[..last], "survivor private K/V");
        let slab = slot * 6 * 8192..(slot + 1) * 6 * 8192;
        assert_eq!(
            bytes[last][slab.clone()],
            original[last][slab],
            "survivor slab rows"
        );
        if live {
            assert_eq!(
                self.observer.private_cursor(&self.model, seq).unwrap(),
                saved.1
            );
        }
    }
    fn replay(&mut self, count: usize) {
        self.rx.queue(&self.tx.packets());
        for _ in 0..count {
            assert!(self.worker.ep_worker_step(&mut self.slots).unwrap());
        }
        self.rx.assert_drained();
    }
    fn close(mut self) {
        // Only byte-fixture disposal after observations; never serving recovery.
        // A failed local retirement deliberately cannot resume ordinary cleanup.
        drop(self.active);
        drop(self.slots);
        drop(self.responses);
        self.model.teardown().unwrap();
        self.worker.teardown().unwrap();
    }
}

#[test]
fn actual_four_owner_retirement_reuses_slots2_and3_without_compaction() {
    if isolated("capacity::actual_four_owner_retirement_reuses_slots2_and3_without_compaction") {
        return;
    }
    reuse_last_pair::<4>();
}

#[test]
fn actual_eight_owner_retirement_reuses_slots6_and7_without_compaction() {
    if isolated("capacity::actual_eight_owner_retirement_reuses_slots6_and7_without_compaction") {
        return;
    }
    reuse_last_pair::<8>();
}

fn reuse_last_pair<const N: usize>() {
    for reverse in [false, true] {
        let mut r = Owners::<N>::new(reverse);
        let saved: Vec<_> = (0..N - 2).map(|slot| r.saved(slot)).collect();
        r.finish(N - 1);
        r.finish(N - 2);
        r.retire()
            .expect("actual retirement must accept the last physical pair");
        assert_eq!(
            r.tx.packets(),
            [
                vec![(N - 2) as u32],
                vec![0xfffffff1],
                vec![(N - 1) as u32],
                vec![0xfffffff1]
            ]
        );
        assert_eq!(r.active.len(), N - 2);
        for (slot, before) in saved.iter().enumerate() {
            r.unchanged(slot, before, true);
        }
        for slot in 0..N {
            if slot < N - 2 {
                assert!(r.responses[slot].try_recv().is_err());
            } else {
                assert_eq!(
                    r.responses[slot].try_recv().unwrap().unwrap().finish_reason,
                    "length"
                );
            }
        }
        r.replay(2);
        for slot in N - 2..N {
            assert_eq!(r.slots[slot].as_ref().unwrap().slot_idx, slot);
            assert_eq!(
                r.peer_observer
                    .private_cursor(&r.worker, r.slots[slot].as_ref().unwrap())
                    .unwrap(),
                0
            );
            let mut replacement = r.model.alloc_sequence().unwrap();
            assert_eq!(replacement.slot_idx, slot);
            let prompt = [4, 3, 2, 1];
            replacement.prompt_len = prompt.len();
            r.model.prefill(&prompt, &mut replacement, 37).unwrap();
            let peer = r.slots[slot].as_mut().unwrap();
            peer.prompt_len = prompt.len();
            r.worker.prefill(&prompt, peer, 37).unwrap();
            r.tx.clear();
            r.model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut replacement, 5)
                .unwrap();
            r.replay(1);
            assert_eq!(replacement.tokens, r.slots[slot].as_ref().unwrap().tokens);
            assert_eq!(replacement.seq_len, r.slots[slot].as_ref().unwrap().seq_len);
            let (a, response) = test_owned_seq(replacement, vec![5], 128, None);
            r.active.push(a);
            r.responses.push(response);
            for (slot, before) in saved.iter().enumerate() {
                r.unchanged(slot, before, true);
            }
        }
        r.close();
    }
}

#[test]
fn failed_last_slot_retirement_keeps_owner_and_response_at_capacity_eight() {
    if isolated("capacity::failed_last_slot_retirement_keeps_owner_and_response_at_capacity_eight")
    {
        return;
    }
    let mut control = Eight::new(false);
    control.finish(7);
    control.retire().unwrap();
    let sync = control
        .observer
        .events()
        .iter()
        .position(|event| matches!(event, Event::Sync(7)))
        .unwrap()
        + 1;
    control.close();
    for fault in 0..3 {
        let mut r = Eight::new(true);
        let saved: Vec<_> = (0..7).map(|slot| r.saved(slot)).collect();
        r.finish(7);
        if fault == 0 {
            r.observer.fail_at(sync);
        } else {
            r.tx.fail_at(fault);
        }
        let error = r.retire().unwrap_err();
        assert!(format!("{error:#}").contains("injected"), "{error:#}");
        assert_eq!(r.active.len(), 8);
        assert!(r.active.iter().any(|a| a.seq.slot_idx == 7 && a.finished));
        for response in &mut r.responses {
            assert!(response.try_recv().is_err());
        }
        if fault == 0 {
            assert!(r.tx.packets().is_empty());
            assert_eq!(r.observer.events().len(), sync);
        } else {
            assert_eq!(r.tx.packets().len(), fault);
            assert_eq!(r.tx.packets()[0], vec![7]);
            if fault == 2 {
                assert_eq!(r.tx.packets()[1], vec![0xfffffff1]);
            }
        }
        for (slot, before) in saved.iter().enumerate() {
            // Inspect saved actual backing after failure, not revoked leases.
            r.unchanged(slot, before, false);
        }
        r.close();
    }
}

#[test]
fn failed_second_group_retirement_keeps_owner_and_stops_following_f1() {
    if isolated("capacity::failed_second_group_retirement_keeps_owner_and_stops_following_f1") {
        return;
    }
    let mut control = Four::new(false);
    control.finish(0);
    control.retire().unwrap();
    let first_events = control.observer.events().len();
    control.finish(2);
    control.retire().unwrap();
    let second_sync = control.observer.events()[first_events..]
        .iter()
        .position(|event| matches!(event, Event::Sync(7)))
        .unwrap()
        + first_events
        + 1;
    control.close();
    for fault in 0..3 {
        let mut r = Four::new(true);
        let saved1 = r.saved(1);
        let saved3 = r.saved(3);
        for slot in [0, 2, 3] {
            r.finish(slot);
        }
        if fault == 0 {
            r.observer.fail_at(second_sync);
        } else {
            r.tx.fail_at(2 + fault);
        }
        let error = r.retire().unwrap_err();
        assert!(format!("{error:#}").contains("injected"), "{error:#}");
        assert_eq!(r.active.len(), 3);
        assert!(!r.active.iter().any(|a| a.seq.slot_idx == 0));
        assert!(r.active.iter().any(|a| a.seq.slot_idx == 2));
        assert_eq!(
            r.responses[0].try_recv().unwrap().unwrap().finish_reason,
            "length"
        );
        for slot in 1..4 {
            assert!(r.responses[slot].try_recv().is_err());
        }
        assert_eq!(&r.tx.packets()[..2], &[vec![0], vec![0xfffffff1]]);
        assert_eq!(r.tx.packets().len(), if fault == 0 { 2 } else { 2 + fault });
        if fault != 0 {
            assert_eq!(r.tx.packets()[2], vec![2]);
        } else {
            assert_eq!(r.observer.events().len(), second_sync);
        }
        r.unchanged(1, &saved1, false);
        r.unchanged(3, &saved3, false);
        r.close();
    }
}
