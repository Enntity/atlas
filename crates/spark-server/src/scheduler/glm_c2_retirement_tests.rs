// SPDX-License-Identifier: AGPL-3.0-only
//! Actual local retirement + worker F1 replay; not distributed completion proof.
use super::retire_selected_finished_sequences;
use crate::scheduler::{
    ActiveSeq,
    glm_c2_serial::step_selected_serial,
    logit_processors::LogitsContext,
    sched_ctx::SchedCtx,
    test_support::{RespRx, test_owned_seq},
};
use spark_model::{
    model::{
        TransformerModel,
        glm_c2_test_support::{Event, Fixture, Observer, Wire},
    },
    traits::{Model, SequenceState},
};

#[path = "glm_c2_fixture_test_process.rs"]
mod process;
fn isolated(name: &str) -> bool {
    process::isolated(&format!(
        "scheduler::mod_helpers::selected_retirement_tests::{name}"
    ))
}

struct Run {
    model: TransformerModel,
    worker: TransformerModel,
    active: Vec<ActiveSeq>,
    slots: [Option<SequenceState>; 2],
    observer: Observer,
    peer_observer: Observer,
    tx: Wire,
    rx: Wire,
    responses: Vec<RespRx>,
    sched: SchedCtx,
}
impl Run {
    fn new(reverse: bool) -> Self {
        let prepare = |rank| {
            let mut f = Fixture::paired(rank);
            f.deterministic_logits(true);
            let (model, seqs) = f.parts_mut();
            for (slot, tokens) in [[1, 2, 3, 4], [4, 3, 2, 1]].iter().enumerate() {
                seqs[slot].prompt_len = tokens.len();
                model.prefill(tokens, &mut seqs[slot], 37).unwrap();
            }
            f
        };
        let mut head = prepare(0);
        let mut peer = prepare(1);
        let tx = head.install_wire();
        let rx = peer.install_wire();
        let (model, seqs, observer) = head.into_parts();
        let (worker, worker_seqs, peer_observer) = peer.into_parts();
        let (mut active, responses): (Vec<_>, Vec<_>) = seqs
            .into_iter()
            .map(|seq| {
                let seed = 5 + seq.slot_idx as u32;
                let (mut a, response) = test_owned_seq(seq, vec![seed], 128, None);
                a.finished = false;
                a.min_tokens = 0;
                a.lz_penalty = 0.0;
                (a, response)
            })
            .unzip();
        if reverse {
            active.reverse();
        }
        let mut r = Self {
            model,
            worker,
            active,
            slots: worker_seqs.map(Some),
            observer,
            peer_observer,
            tx,
            rx,
            responses,
            sched: SchedCtx::for_test(),
        };
        r.step();
        r.replay(4);
        r.clear();
        r
    }
    fn step(&mut self) {
        let s = &self.sched;
        let ctx = LogitsContext {
            scratch: &s.scratch,
            dumps: &s.dumps,
            stats: s.stats.clone(),
            watchdog: s.watchdog,
            boundary_mask: None,
            mid_word_mask: None,
            sampling: s.levers.sampling(),
            timing: s.timing.clone(),
            think_end_token: None,
            think_start_token: None,
            tool_call_start_token: None,
            tool_call_end_token: None,
        };
        step_selected_serial(&self.model, &mut self.active, s, &ctx).unwrap();
    }
    fn replay(&mut self, count: usize) {
        self.rx.queue(&self.tx.packets());
        for _ in 0..count {
            assert!(self.worker.ep_worker_step(&mut self.slots).unwrap());
        }
        self.rx.assert_drained();
    }
    fn clear(&self) {
        self.tx.clear();
        self.observer.clear();
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
        retire_selected_finished_sequences(
            &self.model,
            &mut self.active,
            self.sched.limits.max_seq_len,
        )
    }
    fn close(mut self, failed: bool) {
        // Only after observations: local byte-fixture disposal, never serving recovery.
        self.observer.clear();
        if !failed {
            for a in &mut self.active {
                self.model.free_sequence(&mut a.seq).unwrap();
            }
        }
        for s in &mut self.slots {
            self.worker.free_sequence(s.as_mut().unwrap()).unwrap();
        }
        drop(self.active);
        drop(self.slots);
        drop(self.responses);
        self.model.teardown().unwrap();
        self.worker.teardown().unwrap();
    }
}

#[test]
fn actual_retirement_keeps_slot_survivor_and_replays_worker_f1() {
    if isolated("actual_retirement_keeps_slot_survivor_and_replays_worker_f1") {
        return;
    }
    for reverse in [false, true] {
        for retired in 0..2 {
            let mut r = Run::new(reverse);
            let survivor = 1 - retired;
            let a = r
                .active
                .iter()
                .find(|a| a.seq.slot_idx == survivor)
                .unwrap();
            let cursor = r.observer.private_cursor(&r.model, &a.seq).unwrap();
            let before = r.observer.snapshot(&r.model, &a.seq, cursor).unwrap();
            r.finish(retired);
            r.retire().unwrap();
            assert_eq!(r.tx.packets(), vec![vec![retired as u32], vec![0xfffffff1]]);
            assert_eq!(r.active.len(), 1);
            assert_eq!(r.active[0].seq.slot_idx, survivor);
            assert_eq!(r.observer.read_snapshot(&before).unwrap(), before.initial());
            assert_eq!(
                r.observer
                    .private_cursor(&r.model, &r.active[0].seq)
                    .unwrap(),
                cursor
            );
            assert_eq!(
                r.responses[retired]
                    .try_recv()
                    .unwrap()
                    .unwrap()
                    .finish_reason,
                "length"
            );
            assert!(r.responses[survivor].try_recv().is_err());
            r.replay(1);
            assert_eq!(r.slots[retired].as_ref().unwrap().slot_idx, retired);
            assert_eq!(
                r.peer_observer
                    .private_cursor(&r.worker, r.slots[retired].as_ref().unwrap())
                    .unwrap(),
                0
            );
            for _ in 0..2 {
                r.clear();
                r.step();
                r.replay(2);
                let head = &r.active[0].seq;
                let peer = r.slots[survivor].as_ref().unwrap();
                assert_eq!(head.seq_len, peer.seq_len);
                assert_eq!(head.tokens, peer.tokens);
            }
            r.close(false);
        }
    }
}

#[test]
fn actual_free_and_f1_faults_return_before_completion_removal_or_peer() {
    if isolated("actual_free_and_f1_faults_return_before_completion_removal_or_peer") {
        return;
    }
    let mut control = Run::new(false);
    control.finish(0);
    control.retire().unwrap();
    let events = control.observer.events();
    let sync = events
        .iter()
        .position(|e| matches!(e, Event::Sync(7)))
        .unwrap()
        + 1;
    assert!(events.iter().any(|e| matches!(e, Event::Upload(4, _))));
    control.close(false);
    for retired in 0..2 {
        for fault in 0..3 {
            let mut r = Run::new(true);
            let survivor = 1 - retired;
            let a = r
                .active
                .iter()
                .find(|a| a.seq.slot_idx == survivor)
                .unwrap();
            let cursor = r.observer.private_cursor(&r.model, &a.seq).unwrap();
            let snapshot = r.observer.snapshot(&r.model, &a.seq, cursor).unwrap();
            let tokens = a.seq.tokens.clone();
            r.finish(retired);
            // Slot0 failure must also stop another finished owner queued behind it.
            if retired == 0 {
                r.finish(survivor);
            }
            if fault == 0 {
                r.observer.fail_at(sync);
            } else {
                r.tx.fail_at(fault);
            }
            let error = r.retire().unwrap_err();
            assert!(format!("{error:#}").contains("injected"));
            assert_eq!(r.active.len(), 2);
            assert!(r.active.iter().any(|a| a.seq.slot_idx == retired));
            for response in &mut r.responses {
                assert!(response.try_recv().is_err());
            }
            if fault == 0 {
                assert!(r.tx.packets().is_empty());
                assert_eq!(r.observer.events().len(), sync);
            } else {
                assert_eq!(r.tx.packets().len(), fault);
                assert_eq!(r.tx.packets()[0], vec![retired as u32]);
            }
            assert_eq!(
                r.observer.read_snapshot(&snapshot).unwrap(),
                snapshot.initial()
            );
            let a = r
                .active
                .iter()
                .find(|a| a.seq.slot_idx == survivor)
                .unwrap();
            assert_eq!(a.seq.tokens, tokens);
            // A failed local free may revoke the whole Model session; raw peer bytes
            // remain observable but are not authority to resume after this error.
            r.close(true);
        }
    }
}

#[test]
fn whole_slice_identity_preflight_and_two_finished_slot_order() {
    if isolated("whole_slice_identity_preflight_and_two_finished_slot_order") {
        return;
    }
    let mut r = Run::new(true);
    r.finish(0);
    r.finish(1);
    let original = r.active[0].seq.slot_idx;
    for invalid in [0, 2] {
        r.active[0].seq.slot_idx = invalid;
        assert!(r.retire().is_err());
        assert_eq!(r.active.len(), 2);
        assert!(r.observer.events().is_empty());
        assert!(r.tx.packets().is_empty());
        for response in &mut r.responses {
            assert!(response.try_recv().is_err());
        }
    }
    r.active[0].seq.slot_idx = original;
    r.retire().unwrap();
    assert!(r.active.is_empty());
    assert_eq!(
        r.tx.packets(),
        vec![vec![0], vec![0xfffffff1], vec![1], vec![0xfffffff1]]
    );
    for response in &mut r.responses {
        assert_eq!(
            response.try_recv().unwrap().unwrap().finish_reason,
            "length"
        );
    }
    r.replay(2);
    r.close(false);
}
