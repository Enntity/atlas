// SPDX-License-Identifier: AGPL-3.0-only
//! Actual cold F0 and first selection, with the existing owned dependency Model.
use super::*;
use crate::scheduler::prefill_a_step_params::build_prefill_in_progress;
use crate::scheduler::types::ResponseSink;
use spark_model::model::glm_c2_test_support::{Event, Fixture};

#[path = "glm_c2_fixture_test_process.rs"]
mod process;

fn prefill(seq: spark_model::traits::SequenceState, prompt: Vec<u32>) -> PrefillInProgress {
    let (tx, _rx) = tokio::sync::oneshot::channel();
    build_prefill_in_progress(
        std::sync::Arc::new(prompt),
        0,
        seq,
        0,
        32,
        0,
        vec![0],
        ResponseSink::Blocking(Some(tx)),
        None,
        std::time::Instant::now(),
        0.0,
        0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.75,
        2,
        vec![],
        false,
        None,
        None,
        0,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
    )
}

#[test]
fn actual_cold_scheduler_producer_and_first_selection_replay() {
    if process::isolated(
        "scheduler::glm_c2_selected_prefill::tests::actual_cold_scheduler_producer_and_first_selection_replay",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut head = Fixture::paired(0);
        let mut peer = Fixture::paired(1);
        head.deterministic_logits(true);
        peer.deterministic_logits(true);
        let tx = head.install_wire();
        let rx = peer.install_wire();
        tx.enable_cold_prefix();
        rx.enable_cold_prefix();
        let (mut model, seqs, observer) = head.into_parts();
        let (mut worker, peer_seqs, _) = peer.into_parts();
        let mut prefills: Vec<_> = seqs
            .into_iter()
            .map(|seq| {
                let prompt = if seq.slot_idx == 0 {
                    vec![1, 2, 3, 4]
                } else {
                    vec![4, 3, 2, 1]
                };
                prefill(seq, prompt)
            })
            .collect();
        let mut slots = peer_seqs.map(Some);
        let sched = SchedCtx::for_test();
        for slot in order {
            tx.clear();
            observer.clear();
            let p = &mut prefills[slot];
            let first = cold(&model, p, &sched).unwrap();
            assert!(first > 0 && first < model.vocab_size() as u32);
            assert_eq!(p.seq.tokens.as_slice(), p.prompt_tokens.as_slice());
            assert_eq!(observer.private_cursor(&model, &p.seq).unwrap(), 3);
            assert!(observer.events().contains(&Event::Target(4, 0, 7)));
            assert_eq!(observer.events().last(), Some(&Event::Health(true)));
            assert_eq!(tx.packets()[..2], [vec![slot as u32], vec![0xfffffff0]]);
            rx.queue(&tx.packets());
            assert!(worker.ep_worker_step(&mut slots).unwrap());
            rx.assert_drained();
            assert_eq!(rx.roots(), tx.roots());
            assert_eq!(slots[slot].as_ref().unwrap().tokens, p.seq.tokens);
        }
        for p in &mut prefills {
            model.free_sequence(&mut p.seq).unwrap();
        }
        for seq in &mut slots {
            worker.free_sequence(seq.as_mut().unwrap()).unwrap();
        }
        drop(prefills);
        drop(slots);
        model.teardown().unwrap();
        worker.teardown().unwrap();
    }
}

#[test]
fn actual_cold_first_tool_opener_tracks_selected_promotion() {
    if process::isolated(
        "scheduler::glm_c2_selected_prefill::tests::actual_cold_first_tool_opener_tracks_selected_promotion",
    ) {
        return;
    }
    for max in [0, 1, 32] {
        for thinking in [false, true] {
            let mut fixture = Fixture::paired(0);
            fixture.deterministic_logits(true);
            let wire = fixture.install_wire();
            wire.enable_cold_prefix();
            let (mut model, seqs, _) = fixture.into_parts();
            for seq in seqs {
                let mut p = prefill(seq, vec![1, 2, 3, 4]);
                p.max_tokens = max;
                p.require_tool_call = true;
                p.tools_present = true;
                p.enable_thinking = thinking;
                let first = cold(&model, &mut p, &SchedCtx::for_test()).unwrap();
                // Bind the fixture's actually selected token as the opener;
                // this checks scheduler state, not tokenizer/model semantics.
                let tokens = crate::scheduler::glm_c2_selected::Tokens {
                    eos: vec![0],
                    think_end: Some(100),
                    think_start: Some(101),
                    tool_start: Some(first),
                    tool_end: Some(102),
                    spontaneous_budget: 0,
                };
                let mut a = promote(p, first, &tokens, model.decode_rollback_ring_slots());
                let visible_opener = max > 0 && !thinking;
                assert_eq!(a.inside_thinking, thinking);
                assert_eq!(a.output_tokens, if max == 0 { vec![] } else { vec![first] });
                assert_eq!(a.finished, max <= 1);
                assert_eq!(a.require_tool_call, !visible_opener);
                assert_eq!(a.tool_call_opened, visible_opener);
                assert_eq!(a.inside_tool_body, visible_opener);
                model.free_sequence(&mut a.seq).unwrap();
            }
            model.teardown().unwrap();
        }
    }
}
