// SPDX-License-Identifier: AGPL-3.0-only
//! Actual pair/serial ownership and local wire replay; not KDA/MLA arithmetic.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::layer::glm_pair_verify::GlmPairFfn;
use crate::traits::Model;

#[test]
fn actual_model_joint_temporal_verification_appends_both_issued_owners() {
    if flow::isolated(
        "pair_verify_tests::actual_model_joint_temporal_verification_appends_both_issued_owners",
    ) {
        return;
    }
    let mut f = Fixture::new_pair_compute(0);
    f.model
        .initialize_glm_pair_verification(GlmPairFfn::TwoK5)
        .unwrap();
    let mut issued = [[0; 5]; 2];
    for (owner, prompt) in [vec![1, 2, 3, 4], vec![4, 3, 2, 1, 2, 3]]
        .iter()
        .enumerate()
    {
        f.seqs[owner].prompt_len = prompt.len();
        f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
        f.model
            .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
            .unwrap();
        let seed = 7 - owner as u32;
        let position = f.seqs[owner].seq_len;
        let drafts = f
            .model
            .run_mtp_propose_inner(seed, position, 4, &mut f.seqs[owner], None)
            .unwrap();
        issued[owner][0] = seed;
        issued[owner][1..].copy_from_slice(&drafts);
    }
    let base = f.seqs.each_ref().map(|seq| seq.seq_len);
    f.gpu.clear();
    let [s0, s1] = &mut f.seqs;
    let result = f
        .model
        .glm_paired_execution()
        .unwrap()
        .verify_pair([s0, s1], &issued)
        .unwrap();
    for owner in 0..2 {
        assert_eq!(f.seqs[owner].seq_len, base[owner] + 5);
        assert_eq!(&f.seqs[owner].tokens[base[owner]..], &issued[owner]);
        assert!(
            result[owner]
                .iter()
                .all(|&token| (token as usize) < f.model.vocab_size())
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn pair_round(
    head: &mut Fixture,
    worker: &mut Fixture,
    control: &mut Fixture,
    history: &[flow::History; 2],
    tx: &wire::Wire,
    rx: &wire::Wire,
    accepted: [usize; 2],
    order: [usize; 2],
) {
    let tokens: [[u32; 5]; 2] =
        std::array::from_fn(|owner| history[owner].issued.as_slice().try_into().unwrap());
    tx.clear();
    head.gpu.clear();
    let [s0, s1] = &mut head.seqs;
    let predictions = head
        .model
        .glm_paired_execution()
        .unwrap()
        .verify_pair([s0, s1], &tokens)
        .unwrap();
    for owner in 0..2 {
        let rows = flow::normalized(&history[owner]);
        assert_eq!(
            predictions[owner],
            std::array::from_fn(|row| u32::from(rows[row][0] % 8))
        );
        assert_eq!(head.seqs[owner].seq_len, history[owner].base + 5);
        assert_eq!(
            &head.seqs[owner].tokens[history[owner].base..],
            &tokens[owner]
        );
        for (row, expected) in rows.iter().enumerate() {
            assert_eq!(
                head.gpu.read_span(
                    head.model
                        .buffers
                        .norm_output()
                        .offset((owner * 5 + row) * ROW_BYTES),
                    ROW_BYTES
                ),
                *expected,
                "actual final normalized owner/row placement"
            );
        }
    }
    let [s0, s1] = &mut head.seqs;
    head.model
        .glm_paired_execution()
        .unwrap()
        .finish_verify_pair([s0, s1], &tokens, accepted)
        .unwrap();
    let packets = tx.packets();
    assert_eq!(packets.len(), 4);
    assert_eq!(packets[0], [0]);
    assert_eq!(packets[1], [0xffff_ffe6]);
    assert_eq!(packets[2].len(), 26);
    assert_eq!(&packets[2][..4], &[1, 2, 10, 1]);
    assert_eq!(packets[3], [1, 2, accepted[0] as u32, accepted[1] as u32]);
    for owner in 0..2 {
        assert_eq!(packets[2][4 + owner * 11], owner as u32);
        assert_eq!(packets[2][9 + owner * 11], history[owner].base as u32);
        assert_eq!(
            &packets[2][10 + owner * 11..15 + owner * 11],
            &tokens[owner]
        );
        flow::detached(head, owner, &history[owner], accepted[owner]);
    }
    rx.queue(&packets);
    assert!(wire::worker(worker).unwrap());
    rx.done();
    for owner in order {
        // Reuse the independently exercised serial producer/repair oracle.
        flow::head_verdict(control, owner, &history[owner], accepted[owner]);
        flow::acknowledge(control, owner, accepted[owner], false);
        flow::detached(worker, owner, &history[owner], accepted[owner]);
        wire::same_private(head, worker, owner);
        wire::same_private(head, control, owner);
        assert_eq!(
            head.seqs[owner].seq_len,
            history[owner].base + accepted[owner] + 1
        );
        assert_eq!(worker.seqs[owner].tokens, head.seqs[owner].tokens);
        assert_eq!(control.seqs[owner].tokens, head.seqs[owner].tokens);
    }
}

#[allow(clippy::too_many_arguments)]
fn repropose(
    head: &mut Fixture,
    worker: &mut Fixture,
    control: &mut Fixture,
    history: &mut [flow::History; 2],
    tx: &wire::Wire,
    rx: &wire::Wire,
    accepted: [usize; 2],
    order: [usize; 2],
) {
    for owner in order {
        // This checks accepted shifted EH pairs, canonical K/V, rejected-tail
        // replacement, actual draft feedback, and untouched peer storage.
        flow::continue_owner(control, owner, &mut history[owner], accepted[owner]);
        let h = &history[owner];
        let peer = 1 - owner;
        let peer_rows = flow::private(&head.seqs[peer]).seq_len;
        let peer_bytes = flow::bytes(head, peer, peer_rows);
        let peer_slab: Vec<_> = (0..6).map(|row| flow::slab(head, peer, row)).collect();
        tx.clear();
        head.gpu.clear();
        worker.gpu.clear();
        let drafts = head
            .model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut head.seqs[owner], h.issued[0], h.base, 4, None)
            .unwrap();
        assert_eq!(drafts, h.issued[1..]);
        let packets = tx.packets();
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0], [owner as u32]);
        assert_eq!(packets[1], [0xffff_ffe1]);
        assert_eq!(packets[2].len(), 8);
        assert_eq!(&packets[2][5..], &[h.base as u32, 4, h.issued[0]]);
        rx.queue(&packets);
        assert!(wire::worker(worker).unwrap());
        rx.done();
        wire::same_private(head, worker, owner);
        wire::same_private(head, control, owner);
        assert_eq!(head.gpu.eh_pairs(), control.gpu.eh_pairs());
        assert_eq!(worker.gpu.eh_pairs(), control.gpu.eh_pairs());
        assert_eq!(flow::bytes(head, peer, peer_rows), peer_bytes);
        assert_eq!(
            (0..6)
                .map(|row| flow::slab(head, peer, row))
                .collect::<Vec<_>>(),
            peer_slab
        );
    }
}

#[test]
fn all25_pair_verdicts_actual_worker_e1_and_next_pair() {
    if flow::isolated("pair_verify_tests::all25_pair_verdicts_actual_worker_e1_and_next_pair") {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        for a in 0..5 {
            for b in 0..5 {
                let (mut head, _) = flow::prepare_pair(0, order);
                let (mut worker, _) = flow::prepare_pair(1, order);
                let (mut control, mut history) = flow::prepare_pair(0, order);
                for f in [&mut head, &mut worker] {
                    f.model
                        .initialize_glm_pair_verification(GlmPairFfn::TwoK5)
                        .unwrap();
                }
                let tx = wire::Wire::install(&mut head, 0);
                let rx = wire::Wire::install(&mut worker, 1);
                for (round, accepted) in [[a, b], [b, a]].into_iter().enumerate() {
                    let turn = if round == 0 {
                        order
                    } else {
                        [order[1], order[0]]
                    };
                    pair_round(
                        &mut head,
                        &mut worker,
                        &mut control,
                        &history,
                        &tx,
                        &rx,
                        accepted,
                        turn,
                    );
                    repropose(
                        &mut head,
                        &mut worker,
                        &mut control,
                        &mut history,
                        &tx,
                        &rx,
                        accepted,
                        [turn[1], turn[0]],
                    );
                }
            }
        }
    }
}
