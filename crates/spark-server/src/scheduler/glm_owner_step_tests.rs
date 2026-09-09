// SPDX-License-Identifier: AGPL-3.0-only
//! Actual scheduler/E7/worker replay over byte-backed Models, not GPU numerics.
use crate::scheduler::{
    ActiveSeq, glm_c2_serial,
    logit_processors::LogitsContext,
    sched_ctx::SchedCtx,
    test_support::{RespRx, test_owned_seq},
};
use spark_model::{
    model::{
        TransformerModel,
        glm_c2_test_support::{Fixture, Observer, Wire},
    },
    traits::{Model, SequenceState},
};

#[path = "glm_c2_fixture_test_process.rs"]
mod process;

fn context(s: &SchedCtx) -> LogitsContext<'_> {
    LogitsContext {
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
    }
}

struct Run {
    model: TransformerModel,
    worker: TransformerModel,
    active: Vec<ActiveSeq>,
    spare: Vec<SequenceState>,
    slots: Vec<Option<SequenceState>>,
    responses: Vec<RespRx>,
    observer: Observer,
    peer_observer: Observer,
    tx: Wire,
    rx: Wire,
    sched: SchedCtx,
}

impl Run {
    fn new(physical: &[usize], reverse: bool) -> Self {
        let prepare = |rank| {
            let mut fixture = Fixture::owner_compute(rank);
            fixture.deterministic_logits(true);
            let wire = fixture.install_wire();
            let (model, initial, observer) = fixture.into_parts();
            let mut seqs = Vec::from(initial);
            for slot in 2..4 {
                let seq = model.alloc_sequence().unwrap();
                assert_eq!(seq.slot_idx, slot);
                seqs.push(seq);
            }
            for seq in &mut seqs {
                let prompt = vec![1 + seq.slot_idx as u32; 4 + seq.slot_idx];
                seq.prompt_len = prompt.len();
                model.prefill(&prompt, seq, 37).unwrap();
            }
            assert!(
                model
                    .glm_paired_execution()
                    .unwrap()
                    .owner_verification_enabled()
            );
            wire.clear();
            observer.clear();
            (model, seqs, observer, wire)
        };
        let (model, seqs, observer, tx) = prepare(0);
        let (worker, peer, peer_observer, rx) = prepare(1);
        let mut active = vec![];
        let mut spare = vec![];
        let mut responses = vec![];
        for seq in seqs {
            if physical.contains(&seq.slot_idx) {
                let first = (5 + seq.slot_idx as u32) % 8;
                let (mut a, response) = test_owned_seq(seq, vec![first], 128, None);
                a.finished = false;
                a.min_tokens = 0;
                a.lz_penalty = 0.0;
                active.push(a);
                responses.push(response);
            } else {
                spare.push(seq);
            }
        }
        if reverse {
            active.reverse();
        }
        Self {
            model,
            worker,
            active,
            spare,
            slots: peer.into_iter().map(Some).collect(),
            responses,
            observer,
            peer_observer,
            tx,
            rx,
            sched: SchedCtx::for_test(),
        }
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
        self.tx.clear();
        self.observer.clear();
    }
    fn cold(&mut self) {
        self.step();
        assert_eq!(self.tx.packets().len(), 5 * self.active.len());
        assert!(!self.tx.packets().iter().any(|p| p == &[0xffff_ffe7]));
        self.replay(2 * self.active.len());
        for a in &mut self.active {
            a.presence_penalty = -0.25;
        }
    }
    fn close(mut self) {
        // Byte fixture disposal only; serving retains these owners until _exit.
        drop(self.active);
        drop(self.spare);
        drop(self.slots);
        drop(self.responses);
        self.model.teardown().unwrap();
        self.worker.teardown().unwrap();
    }
}

#[test]
fn actual_three_four_owner_scheduler_uses_e7_and_replays_two_rounds() {
    if process::isolated(
        "scheduler::glm_owner_step::tests::actual_three_four_owner_scheduler_uses_e7_and_replays_two_rounds",
    ) {
        return;
    }
    for physical in [&[0usize, 2, 3][..], &[0, 1, 2, 3][..]] {
        for reverse in [false, true] {
            let mut r = Run::new(physical, reverse);
            r.cold();
            for _ in 0..2 {
                let before: Vec<_> = physical
                    .iter()
                    .map(|slot| {
                        let a = r.active.iter().find(|a| a.seq.slot_idx == *slot).unwrap();
                        (
                            a.seq.seq_len,
                            a.output_tokens.len(),
                            glm_c2_serial::issued(a, r.model.vocab_size()).unwrap(),
                        )
                    })
                    .collect();
                let spare = r.spare.first().map(|seq| {
                    let rows = r.observer.private_cursor(&r.model, seq).unwrap();
                    (rows, r.observer.snapshot(&r.model, seq, rows).unwrap())
                });
                let stats_before = r.sched.stats.glm_c2.snapshot();
                r.step();
                let packets = r.tx.packets();
                assert_eq!(
                    packets[1],
                    [0xffff_ffe7],
                    "ready actual3/4 cohort must use E7, not pair/scalar fallback"
                );
                assert_eq!(packets.len(), 4 + physical.len() * 3);
                assert_eq!(
                    &packets[2][..4],
                    &[1, physical.len() as u32, (physical.len() * 5) as u32, 1]
                );
                assert_eq!(packets[2].len(), 48);
                assert_eq!(packets[3].len(), 6);
                assert_eq!(&packets[3][..2], &[1, physical.len() as u32]);
                let stats = r.sched.stats.glm_c2.snapshot();
                assert_eq!(stats.pair_commits, stats_before.pair_commits);
                assert_eq!(stats.serial_commits, stats_before.serial_commits);
                for group in 0..2 {
                    assert_eq!(
                        stats.owner_commits[group],
                        stats_before.owner_commits[group] + u64::from(group == physical.len() - 3)
                    );
                    for count in 0..5 {
                        let added = if group == physical.len() - 3 {
                            packets[3][2..2 + physical.len()]
                                .iter()
                                .filter(|&&a| a as usize == count)
                                .count() as u64
                        } else {
                            0
                        };
                        assert_eq!(
                            stats.owner_accept[group][count],
                            stats_before.owner_accept[group][count] + added
                        );
                    }
                }
                let logits = r.model.logits_buffer_ptr();
                let logits_end = logits.offset(physical.len() * 80);
                let reads: Vec<_> = r
                    .observer
                    .read_spans()
                    .into_iter()
                    .filter(|(ptr, _, _)| ptr.0 >= logits.0 && ptr.0 < logits_end.0)
                    .collect();
                let expected: Vec<_> = (0..physical.len())
                    .map(|ordinal| (logits.offset(ordinal * 80), 80, 7))
                    .collect();
                assert_eq!(
                    reads, expected,
                    "all checked full-row selections must run at actual ordinal offsets"
                );
                for (ordinal, &slot) in physical.iter().enumerate() {
                    let a = r.active.iter().find(|a| a.seq.slot_idx == slot).unwrap();
                    let accepted = packets[3][2 + ordinal] as usize;
                    let (base, emitted, tokens) = before[ordinal];
                    assert_eq!(packets[2][4 + ordinal * 11], slot as u32);
                    assert_eq!(packets[2][9 + ordinal * 11], base as u32);
                    assert_eq!(a.seq.seq_len, base + accepted + 1);
                    assert_eq!(&a.seq.tokens[base..], &tokens[..accepted + 1]);
                    assert_eq!(a.output_tokens.len(), emitted + accepted + 1);
                    assert_eq!(
                        &a.output_tokens[emitted..emitted + accepted],
                        &tokens[1..1 + accepted]
                    );
                    assert_eq!(packets[4 + ordinal * 3], [slot as u32]);
                    assert_eq!(packets[5 + ordinal * 3], [0xffff_ffe1]);
                }
                if let Some((rows, saved)) = spare {
                    let seq = &r.spare[0];
                    assert_eq!(r.observer.private_cursor(&r.model, seq).unwrap(), rows);
                    let now = r.observer.read_snapshot(&saved).unwrap();
                    let last = now.len() - 1;
                    assert_eq!(now[..last], saved.initial()[..last]);
                    let range = seq.slot_idx * 6 * 8192..(seq.slot_idx + 1) * 6 * 8192;
                    assert_eq!(now[last][range.clone()], saved.initial()[last][range]);
                }
                // Successful E1 itself requires every owner commit released
                // the actual shared producer; worker repeats the same ordering.
                r.replay(1 + physical.len());
            }
            r.close();
        }
    }
}

#[test]
fn actual_owner_mode_preserves_cold_pair_and_scalar_drain() {
    if process::isolated(
        "scheduler::glm_owner_step::tests::actual_owner_mode_preserves_cold_pair_and_scalar_drain",
    ) {
        return;
    }
    let mut r = Run::new(&[0, 2, 3], true);
    r.cold();
    r.active
        .iter_mut()
        .find(|a| a.seq.slot_idx == 0)
        .unwrap()
        .finished = true;
    r.step();
    assert_eq!(r.tx.packets()[0], [2]);
    assert_eq!(r.tx.packets()[1], [0xffff_ffe6]);
    assert!(!r.tx.packets().iter().any(|p| p == &[0xffff_ffe7]));
    r.replay(3);
    r.active
        .iter_mut()
        .find(|a| a.seq.slot_idx == 2)
        .unwrap()
        .finished = true;
    r.step();
    assert_eq!(r.tx.packets()[0], [3]);
    assert_eq!(r.tx.packets()[1], [0xffff_fff5]);
    assert!(!r.tx.packets().iter().any(|p| p == &[0xffff_ffe7]));
    r.replay(2);
    r.close();
}
