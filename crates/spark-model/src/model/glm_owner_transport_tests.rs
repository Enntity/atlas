// SPDX-License-Identifier: AGPL-3.0-only
//! Actual E7/E1 replay and retained physical owners; no CUDA arithmetic claim.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use crate::model::glm_owner_wire::{Bounds, Mode, Packet};

fn prepared(rank: usize) -> (Four, [flow::History; 4]) {
    let (mut four, history) = Four::prepare_fixture(Fixture::new_owner_compute(rank));
    four.f
        .model
        .initialize_glm_owner_verification(Mode::OwnersJoint)
        .unwrap();
    (four, history)
}

#[test]
fn actual_e7_three_four_verdict_and_e1_replay() {
    if flow::isolated(
        "pair_group_tests::owner_transport::actual_e7_three_four_verdict_and_e1_replay",
    ) {
        return;
    }
    for physical in [&[0usize, 2, 3][..], &[0, 1, 2, 3][..]] {
        let (mut head, histories) = prepared(0);
        let (mut worker, _) = prepared(1);
        let tx = Wire::install(&mut head.f, 0);
        let rx = Wire::install(&mut worker.f, 1);
        let shape = GlmOwnerBatchShape::new(physical.len()).unwrap();
        let tokens: Vec<[u32; 5]> = physical
            .iter()
            .map(|&slot| histories[slot].issued.as_slice().try_into().unwrap())
            .collect();
        let accepted = &[0usize, 4, 2, 1][..physical.len()];
        let peer =
            (physical.len() == 3).then(|| (peer_snapshot(&head, 1), peer_snapshot(&worker, 1)));
        let mut selected: Vec<_> = head
            .states
            .iter_mut()
            .filter(|s| physical.contains(&s.slot_idx))
            .collect();
        let predictions = head
            .f
            .model
            .owner_send_verify(shape, &mut selected, &tokens)
            .expect("actual issued owners must execute E7");
        head.f
            .model
            .owner_send_verdict(shape, &mut selected, &tokens, accepted)
            .unwrap();
        let packets = tx.packets();
        assert_eq!(packets.len(), 4);
        assert_eq!(packets[0], [0]);
        assert_eq!(packets[1], [0xffff_ffe7]);
        assert_eq!(packets[2].len(), 48);
        assert_eq!(
            &packets[2][..4],
            &[1, physical.len() as u32, shape.rows() as u32, 1]
        );
        let decoded = Packet::decode(
            packets[2].as_slice().try_into().unwrap(),
            Bounds {
                capacity: 4,
                vocab_size: 8,
                context_tokens: 2048,
            },
        )
        .unwrap();
        for (ordinal, &slot) in physical.iter().enumerate() {
            let record = decoded.owners[ordinal].unwrap();
            assert_eq!(record.slot, slot as u32);
            assert_eq!(record.base, histories[slot].base as u32);
            assert_eq!(record.tokens, tokens[ordinal]);
        }
        let expected_verdict = if physical.len() == 3 {
            [1, 3, 0, 4, 2, 0]
        } else {
            [1, 4, 0, 4, 2, 1]
        };
        assert_eq!(packets[3], expected_verdict);
        worker.replay(&rx, &packets);
        for (ordinal, &slot) in physical.iter().enumerate().rev() {
            let h = &histories[slot];
            let normalized = flow::normalized(h);
            assert_eq!(
                predictions[ordinal],
                std::array::from_fn(|row| u32::from(normalized[row][0] % 8))
            );
            let end = h.base + accepted[ordinal] + 1;
            for f in [&head, &worker] {
                assert_eq!(f.states[slot].seq_len, end);
                assert_eq!(
                    &f.states[slot].tokens[h.base..],
                    &tokens[ordinal][..accepted[ordinal] + 1]
                );
                assert_eq!(f.slab_row(slot, 5), normalized[accepted[ordinal]]);
                for row in 0..accepted[ordinal] {
                    assert_eq!(f.slab_row(slot, row + 1), normalized[row]);
                }
            }
            tx.clear();
            let seed = predictions[ordinal][accepted[ordinal]];
            let drafts = head
                .f
                .model
                .glm_paired_execution()
                .unwrap()
                .propose(&mut head.states[slot], seed, end, 4, None)
                .unwrap();
            let packets = tx.packets();
            assert_eq!(packets.len(), 3);
            assert_eq!(packets[0], [slot as u32]);
            assert_eq!(packets[1], [0xffff_ffe1]);
            worker.replay(&rx, &packets);
            let mut next = flow::History {
                base: end,
                issued: std::iter::once(seed).chain(drafts).collect(),
                canonical: h.canonical.clone(),
                bonus: normalized[accepted[ordinal]].clone(),
            };
            next.canonical.push(h.bonus[..1024].to_vec());
            next.canonical.extend(
                normalized
                    .iter()
                    .take(accepted[ordinal])
                    .map(|row| row[..1024].to_vec()),
            );
            head.assert_private(slot, &next);
            worker.assert_private(slot, &next);
        }
        assert!(
            predictions[physical.len()..]
                .iter()
                .all(|row| *row == [0; 5])
        );
        if let Some((before_head, before_worker)) = peer {
            assert_eq!(peer_snapshot(&head, 1), before_head);
            assert_eq!(peer_snapshot(&worker, 1), before_worker);
        }
    }
}

#[test]
fn last_owner_attempt_corruption_refuses_before_worker_target() {
    if flow::isolated(
        "pair_group_tests::owner_transport::last_owner_attempt_corruption_refuses_before_worker_target",
    ) {
        return;
    }
    let (mut head, histories) = prepared(0);
    let (mut worker, _) = prepared(1);
    let tx = Wire::install(&mut head.f, 0);
    let rx = Wire::install(&mut worker.f, 1);
    let shape = GlmOwnerBatchShape::new(4).unwrap();
    let tokens: [[u32; 5]; 4] = histories
        .each_ref()
        .map(|h| h.issued.as_slice().try_into().unwrap());
    let before = std::array::from_fn::<_, 4, _>(|slot| peer_snapshot(&worker, slot));
    let free = worker.f.model.kv_cache.lock().num_free_blocks();
    let private_free = worker.f.head.paired_test_free_blocks();
    head.f
        .model
        .owner_send_verify(shape, &mut head.states.each_mut(), &tokens)
        .unwrap();
    let mut packets = tx.packets();
    assert_eq!(packets.len(), 3);
    packets[2][41] ^= 1; // Fourth attempt high word: nonzero identity, wrong actual receipt.
    rx.queue(&packets);
    worker.f.gpu.clear();
    assert!(worker.receive().is_err());
    rx.done();
    assert!(
        !worker.f.gpu.trace().iter().any(|event| matches!(
            event,
            Event::Kernel(_, _, _)
                | Event::Target(_, _, _)
                | Event::Body(_, _)
                | Event::Kv(_, _)
                | Event::Copy(_, _, _, _)
                | Event::Memset(_, _, _)
                | Event::Alloc(_, _)
                | Event::Free(_)
        )),
        "last owner mismatch must precede every target writer"
    );
    assert_eq!(worker.f.model.kv_cache.lock().num_free_blocks(), free);
    assert_eq!(worker.f.head.paired_test_free_blocks(), private_free);
    for (slot, snapshot) in before.iter().enumerate() {
        let seq = &worker.states[slot];
        assert_eq!(seq.seq_len, snapshot.target_len);
        assert_eq!(seq.tokens, snapshot.target_tokens);
        assert_eq!(seq.block_table, snapshot.target_blocks);
        for (ptr, bytes) in &snapshot.spans {
            assert_eq!(
                &worker.f.gpu.read_live_span(*ptr, bytes.len()).unwrap(),
                bytes
            );
        }
    }
    worker.f.gpu.clear();
    for seq in &mut worker.states {
        assert!(
            worker
                .f
                .model
                .run_mtp_propose_inner(7, seq.seq_len, 4, seq, None)
                .is_err()
        );
    }
    assert!(
        worker.f.gpu.trace().is_empty(),
        "terminal refusal covers all owners"
    );
}
