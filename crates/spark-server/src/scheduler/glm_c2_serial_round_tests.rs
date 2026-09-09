// SPDX-License-Identifier: AGPL-3.0-only
//! Focused serial continuation using the existing actual dependency fixture.
use super::tests::{context, prefilled};
use super::*;
use crate::scheduler::test_support::{RespRx, test_owned_seq};
use spark_model::{
    model::{
        TransformerModel,
        glm_c2_test_support::{Event, Observer, Wire},
    },
    traits::SequenceState,
};

#[path = "glm_c2_fixture_test_process.rs"]
mod process;
fn isolated(name: &str) -> bool {
    process::isolated(&format!("scheduler::glm_c2_serial::round_tests::{name}"))
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
    fn new(order: [usize; 2]) -> Self {
        let mut head = prefilled(0, order);
        let mut peer = prefilled(1, order);
        let tx = head.install_wire();
        let rx = peer.install_wire();
        let (model, seqs, observer) = head.into_parts();
        let (worker, seqs_peer, peer_observer) = peer.into_parts();
        let (mut active, responses): (Vec<_>, Vec<_>) = seqs
            .into_iter()
            .map(|seq| {
                let first = 5 + seq.slot_idx as u32;
                let (mut a, response) = test_owned_seq(seq, vec![first], 128, None);
                a.finished = false;
                a.min_tokens = 0;
                a.lz_penalty = 0.0;
                (a, response)
            })
            .unzip();
        if order == [1, 0] {
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
        run.step().unwrap();
        run.replay(4);
        run.clear();
        run
    }
    fn clear(&self) {
        self.tx.clear();
        self.observer.clear();
    }
    fn step(&mut self) -> Result<()> {
        step_selected_serial(
            &self.model,
            &mut self.active,
            &self.sched,
            &context(&self.sched),
        )
    }
    fn replay(&mut self, commands: usize) {
        self.rx.queue(&self.tx.packets());
        for _ in 0..commands {
            assert!(self.worker.ep_worker_step(&mut self.slots).unwrap());
        }
        self.rx.assert_drained();
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
        }
    }
    fn close(mut self) {
        self.observer.clear();
        for a in &mut self.active {
            self.model.free_sequence(&mut a.seq).unwrap();
        }
        for seq in &mut self.slots {
            self.worker.free_sequence(seq.as_mut().unwrap()).unwrap();
        }
        drop(self.active);
        drop(self.slots);
        drop(self.responses);
        self.model.teardown().unwrap();
        self.worker.teardown().unwrap();
    }
    fn abandon_failed_mock(mut self) {
        // All no-following-work assertions precede this local fixture teardown.
        // Do not call ordinary free_sequence on an unfinished Produced owner:
        // it correctly refuses. Dropping mock host owners is NOT T2 recovery.
        drop(self.active);
        for seq in &mut self.slots {
            self.worker.free_sequence(seq.as_mut().unwrap()).unwrap();
        }
        drop(self.slots);
        drop(self.responses);
        self.model.teardown().unwrap();
        self.worker.teardown().unwrap();
    }
}

#[test]
fn actual_multiple_rounds_replay_in_both_vector_orders() {
    if isolated("actual_multiple_rounds_replay_in_both_vector_orders") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut run = Run::new(order);
        for _ in 0..3 {
            let before: Vec<_> = run
                .active
                .iter()
                .map(|a| (a.seq.seq_len, a.output_tokens.len()))
                .collect();
            run.step().unwrap();
            let packets = run.tx.packets();
            assert_eq!(packets.len(), 16);
            for slot in 0..2 {
                let start = slot * 8;
                assert_eq!(packets[start], vec![slot as u32]);
                assert_eq!(packets[start + 1], vec![0xfffffff5]);
                assert_eq!(packets[start + 2], vec![5]);
                let accepted = packets[start + 4][0] as usize;
                assert!(accepted <= 4);
                assert_eq!(packets[start + 5], vec![slot as u32]);
                assert_eq!(packets[start + 6], vec![0xffffffe1]);
                let index = run
                    .active
                    .iter()
                    .position(|a| a.seq.slot_idx == slot)
                    .unwrap();
                let a = &run.active[index];
                assert_eq!(a.seq.seq_len, before[index].0 + accepted + 1);
                assert_eq!(a.output_tokens.len(), before[index].1 + accepted + 1);
                assert_eq!(a.output_tokens.last(), Some(&a.last_token));
                assert_eq!(packets[start + 7].last(), Some(&a.last_token));
                assert_eq!(a.pending_drafts.len(), 4);
            }
            run.replay(4);
            run.clear();
        }
        run.close();
    }
}

#[test]
fn early_finish_completes_both_verdicts_without_e1() {
    if isolated("early_finish_completes_both_verdicts_without_e1") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut run = Run::new(order);
        for a in &mut run.active {
            a.remaining = 1;
        }
        let outputs: Vec<_> = run.active.iter().map(|a| a.output_tokens.len()).collect();
        run.step().unwrap();
        let packets = run.tx.packets();
        assert_eq!(packets.len(), 10);
        assert!(!packets.iter().any(|p| p == &[0xffffffe1]));
        for (a, count) in run.active.iter().zip(outputs) {
            assert!(a.finished && a.pending_drafts.is_empty());
            assert_eq!(a.output_tokens.len(), count + 1);
            // This immutable actual Model check requires completed trim+commit,
            // even though output-budget finish deliberately skipped repair/E1.
            run.model
                .glm_paired_execution()
                .unwrap()
                .validate_propose(&a.seq, a.last_token, a.seq.seq_len, 4, None)
                .unwrap();
        }
        run.replay(2);
        run.clear();
        run.step().unwrap();
        assert!(run.tx.packets().is_empty() && run.observer.events().is_empty());
        run.close();
    }
}

#[test]
fn checked_selection_failure_stops_before_accept_peer_or_emission() {
    if isolated("checked_selection_failure_stops_before_accept_peer_or_emission") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut control = Run::new(order);
        for a in &mut control.active {
            a.presence_penalty = -0.25;
        }
        control.step().unwrap();
        let events = control.observer.events();
        let ordinal = events
            .iter()
            .position(|e| *e == Event::Read(80, 7))
            .expect("actual checked full-K5 copy")
            + 1;
        control.replay(4);
        control.close();

        let mut run = Run::new(order);
        for a in &mut run.active {
            a.presence_penalty = -0.25;
        }
        let outputs: Vec<_> = run.active.iter().map(|a| a.output_tokens.clone()).collect();
        let peer = run.active.iter().find(|a| a.seq.slot_idx == 1).unwrap();
        let peer_tokens = peer.seq.tokens.clone();
        let snapshot = run
            .observer
            .snapshot(&run.model, &peer.seq, peer.seq.seq_len + 3)
            .unwrap();
        run.observer.fail_at(ordinal);
        let error = run
            .step()
            .expect_err("checked selection failure must escape driver");
        assert!(format!("{error:#}").contains("injected fixture operation failure"));
        assert_eq!(run.observer.events(), events[..ordinal]);
        let packets = run.tx.packets();
        assert_eq!(
            packets.len(),
            4,
            "F5 only; no accepted count, E1 or peer command"
        );
        assert_eq!(packets[0], vec![0]);
        assert_eq!(packets[1], vec![0xfffffff5]);
        for (a, before) in run.active.iter().zip(outputs) {
            assert_eq!(a.output_tokens, before);
            assert!(!a.finished && a.pending_drafts.len() == 4);
        }
        let peer = run.active.iter().find(|a| a.seq.slot_idx == 1).unwrap();
        assert_eq!(peer.seq.tokens, peer_tokens);
        assert_eq!(
            run.observer.read_snapshot(&snapshot).unwrap(),
            snapshot.initial()
        );
        // Explicit local mock cleanup only; this is not a serving failure policy.
        run.abandon_failed_mock();
    }
}

#[test]
fn invalid_live_peer_profile_refuses_whole_slice_before_wire() {
    if isolated("invalid_live_peer_profile_refuses_whole_slice_before_wire") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        for case in 0..3 {
            let mut run = Run::new(order);
            let index = run.active.iter().position(|a| a.seq.slot_idx == 1).unwrap();
            let position = run.active[index].seq.seq_len;
            match case {
                0 => run.active[index].seq.seq_len = usize::MAX,
                1 => run.active[index].seq.seq_len += 1,
                _ => run.active[index].disable_mtp = true,
            }
            let outputs: Vec<_> = run.active.iter().map(|a| a.output_tokens.clone()).collect();
            assert!(run.step().is_err());
            assert!(run.tx.packets().is_empty() && run.observer.events().is_empty());
            for (a, before) in run.active.iter().zip(outputs) {
                assert_eq!(a.output_tokens, before);
            }
            run.active[index].seq.seq_len = position;
            run.active[index].disable_mtp = false;
            run.close();
        }
    }
}
