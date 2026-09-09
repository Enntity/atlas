// SPDX-License-Identifier: AGPL-3.0-only
//! Actual scheduler/Model/worker pair transactions over the existing byte backend.
//! Packet and detached-owner checks are not GPU attention or NCCL numerics.
use crate::scheduler::{
    ActiveSeq, glm_c2_serial,
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

fn context(sched: &SchedCtx) -> LogitsContext<'_> {
    LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        watchdog: sched.watchdog,
        boundary_mask: None,
        mid_word_mask: None,
        sampling: sched.levers.sampling(),
        timing: sched.timing.clone(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

fn prefilled(rank: usize, order: [usize; 2]) -> Fixture {
    let mut fixture = Fixture::paired_compute(rank);
    fixture.deterministic_logits(true);
    let (model, seqs) = fixture.parts_mut();
    let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
    for owner in order {
        seqs[owner].prompt_len = prompts[owner].len();
        model
            .prefill(&prompts[owner], &mut seqs[owner], 37)
            .unwrap();
    }
    fixture
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
        let (worker, peer_seqs, peer_observer) = peer.into_parts();
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
            slots: peer_seqs.map(Some),
            observer,
            peer_observer,
            tx,
            rx,
            responses,
            sched: SchedCtx::for_test(),
        };
        assert!(
            run.model
                .glm_paired_execution()
                .unwrap()
                .pair_verification_enabled()
        );
        assert!(
            run.worker
                .glm_paired_execution()
                .unwrap()
                .pair_verification_enabled()
        );
        // Cold owners deliberately take the existing scalar bootstrap + E1 path.
        run.step();
        assert_eq!(run.tx.packets().len(), 10);
        run.rx.queue(&run.tx.packets());
        for _ in 0..4 {
            assert!(run.worker.ep_worker_step(&mut run.slots).unwrap());
        }
        run.rx.assert_drained();
        run.compare_owners();
        for a in &mut run.active {
            // Force the real checked full-row pipeline for both owner slices;
            // do not rely only on the raw-argmax no-read shortcut.
            a.presence_penalty = -0.25;
        }
        run.tx.clear();
        run
    }
    fn step(&mut self) {
        glm_c2_serial::step_selected_serial(
            &self.model,
            &mut self.active,
            &self.sched,
            &context(&self.sched),
        )
        .unwrap();
    }
    fn compare_owners(&self) {
        for a in &self.active {
            let peer = self.slots[a.seq.slot_idx].as_ref().unwrap();
            assert_eq!(a.seq.tokens, peer.tokens);
            assert_eq!(a.seq.seq_len, peer.seq_len);
            let rows = self.observer.private_cursor(&self.model, &a.seq).unwrap();
            assert_eq!(rows, a.seq.seq_len + 3);
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
                    .initial(),
            );
        }
    }
    fn round(&mut self) {
        let stats_before = self.sched.stats.glm_c2.snapshot();
        let before: Vec<_> = self
            .active
            .iter()
            .map(|a| {
                (
                    a.seq.seq_len,
                    a.output_tokens.len(),
                    glm_c2_serial::issued(a, self.model.vocab_size()).unwrap(),
                )
            })
            .collect();
        self.observer.clear();
        self.step();
        assert_eq!(
            self.observer
                .events()
                .iter()
                .filter(|event| **event == Event::Read(5 * 8 * 2, 7))
                .count(),
            2,
            "both actual checked five-row BF16 selections must execute",
        );
        let packets = self.tx.packets();
        assert_eq!(packets.len(), 10, "one E6/verdict and two E1 commands");
        assert_eq!(packets[0], [0]);
        assert_eq!(packets[1], [0xffff_ffe6]);
        assert_eq!(packets[2].len(), 26);
        assert_eq!(&packets[2][..4], &[1, 2, 10, 1]);
        assert_eq!(packets[3].len(), 4);
        assert_eq!(&packets[3][..2], &[1, 2]);
        // Actual completed E6/verdict, canonical physical owner histogram,
        // including reversed scheduler vectors. Not an attempted-header count.
        let stats = self.sched.stats.glm_c2.snapshot();
        let bin = packets[3][2] as usize * 5 + packets[3][3] as usize;
        assert_eq!(stats.pair_commits, stats_before.pair_commits + 1);
        for index in 0..25 {
            assert_eq!(
                stats.pair_accept[index],
                stats_before.pair_accept[index] + u64::from(index == bin)
            );
        }
        assert_eq!(stats.serial_commits, 0);
        assert_eq!(stats.serial_accept, [0; 5]);
        assert_eq!(stats.bootstrap_commits, 2);
        for owner in 0..2 {
            let index = self
                .active
                .iter()
                .position(|a| a.seq.slot_idx == owner)
                .unwrap();
            let a = &self.active[index];
            let accepted = packets[3][2 + owner] as usize;
            assert!(accepted <= 4);
            let record = 4 + owner * 11;
            assert_eq!(packets[2][record], owner as u32);
            assert_eq!(packets[2][record + 5], before[index].0 as u32);
            assert_eq!(&packets[2][record + 6..record + 11], &before[index].2);
            let e1 = 4 + owner * 3;
            assert_eq!(packets[e1], [owner as u32]);
            assert_eq!(packets[e1 + 1], [0xffff_ffe1]);
            assert_eq!(packets[e1 + 2].last(), Some(&a.last_token));
            assert_eq!(a.seq.seq_len, before[index].0 + accepted + 1);
            assert_eq!(a.output_tokens.len(), before[index].1 + accepted + 1);
            assert_eq!(a.output_tokens.last(), Some(&a.last_token));
            assert_eq!(a.pending_drafts.len(), 4);
        }
        self.rx.queue(&packets);
        // The actual worker consumes E6 and both verdict counts in ONE command.
        // Stop here, before either E1, and ask the actual capability to validate
        // both next proposals. This requires both detached/committed receipts.
        assert!(self.worker.ep_worker_step(&mut self.slots).unwrap());
        for a in &self.active {
            let peer = self.slots[a.seq.slot_idx].as_ref().unwrap();
            assert_eq!(a.seq.tokens, peer.tokens);
            self.worker
                .glm_paired_execution()
                .unwrap()
                .validate_propose(peer, a.last_token, peer.seq_len, 4, None)
                .unwrap();
        }
        let peer1 = self.slots[1].as_ref().unwrap();
        let cursor = self
            .peer_observer
            .private_cursor(&self.worker, peer1)
            .unwrap();
        let saved = self
            .peer_observer
            .snapshot(&self.worker, peer1, cursor)
            .unwrap();
        assert!(self.worker.ep_worker_step(&mut self.slots).unwrap()); // owner0 E1
        assert_eq!(
            cursor,
            self.peer_observer
                .private_cursor(&self.worker, self.slots[1].as_ref().unwrap())
                .unwrap(),
        );
        let after = self.peer_observer.read_snapshot(&saved).unwrap();
        let slab = after.len() - 1;
        assert_eq!(
            &after[..slab],
            &saved.initial()[..slab],
            "peer1 private K/V unchanged"
        );
        // Snapshot's final span is the actual whole six-row-per-owner slab.
        // Owner0 may repair its half; owner1's detached rows must remain intact.
        assert_eq!(&after[slab][6 * 8192..], &saved.initial()[slab][6 * 8192..]);
        assert!(self.worker.ep_worker_step(&mut self.slots).unwrap()); // owner1 E1
        self.rx.assert_drained();
        self.compare_owners();
        self.tx.clear();
    }
    fn close(mut self) {
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
}

#[test]
fn actual_pair_rounds_replay_with_detached_owners() {
    if process::isolated(
        "scheduler::glm_c2_pair_step::tests::actual_pair_rounds_replay_with_detached_owners",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut run = Run::new(order);
        assert_eq!(
            run.active
                .iter()
                .map(|a| a.seq.slot_idx)
                .collect::<Vec<_>>(),
            order
        );
        for _ in 0..2 {
            run.round();
        }
        run.close();
    }
}
