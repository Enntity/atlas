// SPDX-License-Identifier: AGPL-3.0-only
//! Actual dependency-model seam; no selected scheduler activation.
use super::logit_processors::{LogitsContext, SamplingLevers};
use super::verify_pipeline_helper::verify_pick_all_with_pipeline_checked;
use super::{sched_ctx::SchedCtx, test_support::test_owned_seq};
use spark_model::model::glm_c2_test_support::{Event, Fixture};
use spark_model::traits::Model;

#[path = "glm_c2_fixture_test_process.rs"]
mod process;

fn isolated(name: &str) -> bool {
    process::isolated(&format!("scheduler::glm_c2_fixture_tests::{name}"))
}

#[test]
fn actual_paired_dependency_fixture_is_available() {
    if isolated("actual_paired_dependency_fixture_is_available") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        exercise(order);
    }
}

#[test]
fn observer_snapshots_keep_revocation_and_foreign_boundaries() {
    if isolated("observer_snapshots_keep_revocation_and_foreign_boundaries") {
        return;
    }
    let (mut model, mut seqs, observer) = setup(0, [0, 1]).into_parts();
    let (mut other, mut other_seqs, foreign) = setup(1, [0, 1]).into_parts();
    let saved = observer.snapshot(&model, &seqs[0], 3).unwrap();
    let peer = observer.snapshot(&model, &seqs[1], 3).unwrap();
    observer.clear();
    foreign.clear();
    assert!(observer.snapshot(&other, &other_seqs[0], 3).is_err());
    assert!(foreign.read_snapshot(&saved).is_err());
    assert!(observer.snapshot(&model, &seqs[0], 2049).is_err());
    assert!(observer.events().is_empty());
    assert!(foreign.events().is_empty());
    model.free_sequence(&mut seqs[0]).unwrap();
    observer.clear();
    assert!(observer.snapshot(&model, &seqs[0], 0).is_err());
    assert!(observer.private_cursor(&model, &seqs[0]).is_err());
    assert_eq!(observer.read_snapshot(&saved).unwrap(), saved.initial());
    assert_eq!(observer.read_snapshot(&peer).unwrap(), peer.initial());
    assert!(observer.events().is_empty());
    model.free_sequence(&mut seqs[1]).unwrap();
    for seq in &mut other_seqs {
        other.free_sequence(seq).unwrap();
    }
    drop(seqs);
    drop(other_seqs);
    // Retained read-only observer has no cleanup action; actual Model closes the slab.
    model.teardown().unwrap();
    other.teardown().unwrap();
    assert!(observer.read_snapshot(&saved).is_err());
    drop(saved);
    drop(peer);
    drop(observer);
    drop(foreign);
}

#[test]
fn explicit_legacy_fixture_has_no_selected_capability() {
    if isolated("explicit_legacy_fixture_has_no_selected_capability") {
        return;
    }
    for rank in 0..2 {
        let (mut model, mut seqs, observer) = Fixture::legacy(rank).into_parts();
        assert!(model.glm_paired_execution().is_none());
        model.free_sequence(&mut seqs[0]).unwrap();
        drop(seqs);
        drop(observer);
        model.teardown().unwrap();
    }
}

fn setup(rank: usize, order: [usize; 2]) -> Fixture {
    let mut fixture = Fixture::paired(rank);
    fixture.deterministic_logits(true);
    let (model, seqs) = fixture.parts_mut();
    let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
    for owner in order {
        seqs[owner].prompt_len = prompts[owner].len();
        model
            .prefill(&prompts[owner], &mut seqs[owner], 37)
            .unwrap();
        model
            .decode(5 + owner as u32, &mut seqs[owner], 37)
            .unwrap();
    }
    fixture
}

fn exercise(order: [usize; 2]) {
    let mut head = setup(0, order);
    let mut peer = setup(1, order);
    let tx = head.install_wire();
    let rx = peer.install_wire();
    let (mut model, sequences, observer) = head.into_parts();
    let (mut worker, sequences_peer, peer_observer) = peer.into_parts();
    let mut slots = sequences_peer.map(Some);
    let mut active = sequences.map(|seq| {
        let seed = *seq.tokens.last().unwrap();
        let (mut a, response) = test_owned_seq(seq, vec![seed], 32, None);
        a.finished = false;
        a.min_tokens = 0;
        a.lz_penalty = 0.0;
        a.repetition_penalty = 2.0; // Force a real logits read, not the neutral raw fast path.
        (a, response)
    });
    assert_eq!(observer.private_free_blocks(), 0);
    for owner in order {
        tx.clear();
        observer.clear();
        peer_observer.clear();
        let a = &mut active[owner].0;
        let base = a.seq.seq_len;
        let seed = 7 - owner as u32;
        let primed = observer.private_cursor(&model, &a.seq).unwrap();
        assert_eq!(primed, base - 2); // P-1: first E1 still writes the missing tail pair.
        let before = observer.snapshot(&model, &a.seq, primed).unwrap();
        let capability = model
            .glm_paired_execution()
            .expect("real sealed paired capability");
        capability
            .validate_propose(&a.seq, seed, base, 4, None)
            .unwrap();
        assert!(observer.events().is_empty());
        let drafts = capability.propose(&mut a.seq, seed, base, 4, None).unwrap();
        assert_eq!(drafts.len(), 4);
        assert_eq!(
            tx.packets(),
            vec![
                vec![owner as u32],
                vec![0xffffffe1],
                vec![1, 1, 0, 1, 0, base as u32, 4, seed]
            ]
        );
        assert_eq!(
            observer
                .events()
                .iter()
                .filter(|e| matches!(e, Event::Body(_, 7)))
                .count(),
            4
        );
        assert_eq!(observer.private_cursor(&model, &a.seq).unwrap(), base + 3);
        let after = observer.snapshot(&model, &a.seq, base + 3).unwrap();
        assert_eq!(
            &after.initial()[..2 * primed],
            &before.initial()[..2 * primed]
        );
        let slab = before.initial().last().unwrap();
        for (row, hidden_row) in [(primed, 0), (primed + 1, 5)] {
            let start = (owner * 6 + hidden_row) * 8192;
            for kv in 0..2 {
                assert_eq!(after.initial()[2 * row + kv], slab[start..start + 1024]);
            }
        }
        assert!(
            after.initial()[2 * (base - 1)..2 * (base + 3)]
                .iter()
                .all(|row| row.iter().all(|b| *b == row[0]) && row[0] != 0)
        );
        rx.queue(&tx.packets());
        assert!(worker.ep_worker_step(&mut slots).unwrap());
        rx.assert_drained();
        let peer_bytes = peer_observer
            .snapshot(&worker, slots[owner].as_ref().unwrap(), base + 3)
            .unwrap();
        assert_eq!(after.initial(), peer_bytes.initial());

        tx.clear();
        observer.clear();
        let issued: Vec<_> = std::iter::once(seed).chain(drafts).collect();
        capability.validate_verify(&a.seq, &issued).unwrap();
        assert!(observer.events().is_empty());
        let raw = capability.verify(&mut a.seq, &issued).unwrap();
        assert_eq!(raw.len(), 5);
        assert_eq!(
            tx.packets(),
            vec![
                vec![owner as u32],
                vec![0xfffffff5],
                vec![5],
                issued.clone()
            ]
        );
        let tokens_before = a.seq.tokens.clone();
        let length_before = a.seq.seq_len;
        let output_before = a.output_tokens.clone();
        let sched = SchedCtx::for_test();
        let ctx = LogitsContext {
            scratch: &sched.scratch,
            dumps: &sched.dumps,
            stats: sched.stats.clone(),
            watchdog: sched.watchdog,
            boundary_mask: None,
            mid_word_mask: None,
            sampling: SamplingLevers {
                fast_greedy_chat: false,
                fast_greedy_grammar: false,
                dflash_masked_verify: false,
                force_temp_zero: false,
                ..Default::default()
            },
            timing: sched.timing.clone(),
            think_end_token: None,
            think_start_token: None,
            tool_call_start_token: None,
            tool_call_end_token: None,
        };
        observer.clear();
        let selected = verify_pick_all_with_pipeline_checked(&model, &raw, a, &ctx, 0).unwrap();
        assert_eq!(selected.len(), 5);
        assert!(selected.iter().all(|id| *id < 8));
        assert_eq!(observer.events(), vec![Event::Read(5 * 8 * 2, 7)]);
        assert_eq!(a.output_tokens, output_before);
        assert_eq!(a.seq.tokens, tokens_before);
        assert_eq!(a.seq.seq_len, length_before);
        let accepted = issued[1..]
            .iter()
            .zip(&selected)
            .take_while(|(draft, pick)| draft == pick)
            .count();
        model.ep_broadcast_cmd(accepted as u32).unwrap();
        a.seq.tokens.truncate(base + accepted + 1);
        a.seq.seq_len = base + accepted + 1;
        model
            .record_glm_mtp_verified(&mut a.seq, base, &issued, accepted)
            .unwrap();
        model.trim_proposer_state(&mut a.seq, accepted, 0).unwrap();
        model
            .commit_accepted_prefix(&mut a.seq, accepted + 1, 5)
            .unwrap();
        rx.queue(&tx.packets());
        assert!(worker.ep_worker_step(&mut slots).unwrap());
        rx.assert_drained();
        let detached = observer.snapshot(&model, &a.seq, base + 3).unwrap();
        let peer_detached = peer_observer
            .snapshot(&worker, slots[owner].as_ref().unwrap(), base + 3)
            .unwrap();
        assert_eq!(detached.initial(), peer_detached.initial());
        assert_eq!(
            observer.read_snapshot(&detached).unwrap(),
            detached.initial()
        );
    }
    // No active Verification is abandoned. Actual retirement drains both genuine owners.
    for (a, _) in &mut active {
        model.free_sequence(&mut a.seq).unwrap();
    }
    for seq in &mut slots {
        worker.free_sequence(seq.as_mut().unwrap()).unwrap();
    }
    assert_eq!(observer.private_free_blocks(), 256);
    assert_eq!(peer_observer.private_free_blocks(), 256);
    drop(active);
    drop(slots);
    drop(observer);
    drop(peer_observer);
    drop(tx);
    drop(rx);
    model.teardown().unwrap();
    worker.teardown().unwrap();
}
