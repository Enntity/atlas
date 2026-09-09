// SPDX-License-Identifier: AGPL-3.0-only
//! Real paired dependency owners and real worker replay, not model arithmetic.
use super::*;
use crate::scheduler::test_support::test_owned_seq;
use spark_model::model::glm_c2_test_support::Fixture;

#[path = "glm_c2_fixture_test_process.rs"]
mod process;

fn isolated(name: &str) -> bool {
    // Shared process helper will accept the complete test path at registration.
    process::isolated(&format!("scheduler::glm_c2_serial::tests::{name}"))
}

pub(super) fn prefilled(rank: usize, order: [usize; 2]) -> Fixture {
    let mut fixture = Fixture::paired(rank);
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

pub(super) fn context(sched: &SchedCtx) -> LogitsContext<'_> {
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

#[test]
fn actual_two_owner_bootstrap_and_worker_replay() {
    if isolated("actual_two_owner_bootstrap_and_worker_replay") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut head = prefilled(0, order);
        let mut peer = prefilled(1, order);
        let tx = head.install_wire();
        let rx = peer.install_wire();
        let (mut model, seqs, observer) = head.into_parts();
        let (mut worker, worker_seqs, worker_observer) = peer.into_parts();
        let mut slots = worker_seqs.map(Some);
        let (mut active, responses): (Vec<_>, Vec<_>) = seqs
            .into_iter()
            .map(|seq| {
                let first = 5 + seq.slot_idx as u32;
                let (mut a, response) = test_owned_seq(seq, vec![first], 32, None);
                a.finished = false;
                a.min_tokens = 0;
                a.lz_penalty = 0.0;
                (a, response)
            })
            .unzip();
        if order == [1, 0] {
            active.reverse();
        }
        let sched = SchedCtx::for_test();
        observer.clear();
        step_selected_serial(&model, &mut active, &sched, &context(&sched)).unwrap();
        let packets = tx.packets();
        assert_eq!(packets.len(), 10); // scalar two words + E1 three, per owner.
        for slot in 0..2 {
            let start = slot * 5;
            assert_eq!(packets[start], vec![slot as u32]);
            assert_eq!(packets[start + 1], vec![5 + slot as u32]);
            assert_eq!(packets[start + 2], vec![slot as u32]);
            assert_eq!(packets[start + 3], vec![0xffffffe1]);
            let a = active.iter().find(|a| a.seq.slot_idx == slot).unwrap();
            assert_eq!(a.seq.seq_len, a.seq.prompt_len + 1);
            assert_eq!(a.output_tokens.len(), 2);
            assert_eq!(a.output_tokens.last(), Some(&a.last_token));
            assert_eq!(a.pending_drafts.len(), 4);
            assert!(
                a.pending_drafts
                    .iter()
                    .all(|t| *t < model.vocab_size() as u32)
            );
            assert_eq!(packets[start + 4].last(), Some(&a.last_token));
            assert_eq!(
                observer.private_cursor(&model, &a.seq).unwrap(),
                a.seq.seq_len + 3
            );
        }
        rx.queue(&packets);
        for _ in 0..4 {
            assert!(worker.ep_worker_step(&mut slots).unwrap());
        }
        rx.assert_drained();
        for a in &active {
            let worker_seq = slots[a.seq.slot_idx].as_ref().unwrap();
            assert_eq!(a.seq.tokens, worker_seq.tokens);
            assert_eq!(a.seq.seq_len, worker_seq.seq_len);
            let head_bytes = observer
                .snapshot(&model, &a.seq, a.seq.seq_len + 3)
                .unwrap();
            let peer_bytes = worker_observer
                .snapshot(&worker, worker_seq, worker_seq.seq_len + 3)
                .unwrap();
            assert_eq!(head_bytes.initial(), peer_bytes.initial());
        }
        for a in &mut active {
            model.free_sequence(&mut a.seq).unwrap();
        }
        for seq in &mut slots {
            worker.free_sequence(seq.as_mut().unwrap()).unwrap();
        }
        drop(active);
        drop(slots);
        drop(responses);
        model.teardown().unwrap();
        worker.teardown().unwrap();
    }
}
