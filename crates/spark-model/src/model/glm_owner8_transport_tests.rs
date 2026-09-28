// SPDX-License-Identifier: AGPL-3.0-only
//! Actual E8/E7/E1 and retained eight-slot owners; no CUDA arithmetic claim.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use crate::model::glm_owner_wire::Mode;

fn prepared(rank: usize) -> (Eight, [flow::History; 8]) {
    let (mut f, histories) = Eight::prepared(rank);
    f.f.model
        .initialize_glm_owner_verification(Mode::OwnersJoint)
        .unwrap();
    (f, histories)
}

#[test]
fn actual_e8_and_high_slot_e7_verdict_then_reverse_e1_replay() {
    if flow::isolated(
        "pair_group_tests::owner8_transport::actual_e8_and_high_slot_e7_verdict_then_reverse_e1_replay",
    ) {
        return;
    }
    let mut cohorts: Vec<Vec<usize>> = (5..=8)
        .map(|count| (0..count - 1).chain(std::iter::once(7)).collect())
        .collect();
    cohorts.extend([vec![0, 4, 7], vec![1, 4, 6, 7]]);
    for physical in cohorts {
        let count = physical.len();
        let (mut head, histories) = prepared(0);
        let (mut worker, _) = prepared(1);
        let tx = Wire::install(&mut head.f, 0);
        let rx = Wire::install(&mut worker.f, 1);
        let shape = GlmOwnerBatchShape::new(count).unwrap();
        let tokens: Vec<[u32; 5]> = physical
            .iter()
            .map(|&slot| histories[slot].issued.as_slice().try_into().unwrap())
            .collect();
        let accepted: Vec<_> = (0..count).map(|i| [0, 4, 2, 1, 3][i % 5]).collect();
        let peers: Vec<_> = (0..8)
            .filter(|slot| !physical.contains(slot))
            .map(|slot| {
                (
                    slot,
                    peer_snapshot(&head, slot),
                    peer_snapshot(&worker, slot),
                )
            })
            .collect();
        let mut selected: Vec<_> = head
            .states
            .iter_mut()
            .filter(|seq| physical.contains(&seq.slot_idx))
            .collect();
        let predictions = head
            .f
            .model
            .owner_send_verify(shape, &mut selected, &tokens)
            .expect("actual retained owners must execute their selected E8/E7 transport");
        head.f
            .model
            .owner_send_verdict(shape, &mut selected, &tokens, &accepted)
            .unwrap();
        let packets = tx.packets();
        let wide = count > 4;
        assert_eq!(packets.len(), 4);
        assert_eq!(packets[0], [0]);
        assert_eq!(packets[1], [if wide { 0xffff_ffe8 } else { 0xffff_ffe7 }]);
        assert_eq!(packets[2].len(), if wide { 92 } else { 48 });
        assert_eq!(&packets[2][..4], &[1, count as u32, (count * 5) as u32, 1]);
        assert!(packets[2][4 + count * 11..].iter().all(|&word| word == 0));
        let mut verdict = vec![0; if wide { 10 } else { 6 }];
        verdict[..2].copy_from_slice(&[1, count as u32]);
        for (i, &value) in accepted.iter().enumerate() {
            verdict[i + 2] = value as u32;
        }
        assert_eq!(packets[3], verdict);
        for (ordinal, &slot) in physical.iter().enumerate() {
            let start = 4 + ordinal * 11;
            assert_eq!(packets[2][start], slot as u32);
            assert_eq!(packets[2][start + 5], histories[slot].base as u32);
            assert_eq!(&packets[2][start + 6..start + 11], &tokens[ordinal]);
        }
        worker.replay(&rx, &packets);
        assert!(predictions[count..].iter().all(|row| *row == [0; 5]));
        // Every commit must release its actual receipt before any next E1.
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
        for (slot, before_head, before_worker) in peers {
            assert_eq!(peer_snapshot(&head, slot), before_head);
            assert_eq!(peer_snapshot(&worker, slot), before_worker);
        }
    }
}

#[test]
fn eighth_attempt_corruption_is_prewrite_and_terminal_for_all_eight() {
    if flow::isolated(
        "pair_group_tests::owner8_transport::eighth_attempt_corruption_is_prewrite_and_terminal_for_all_eight",
    ) {
        return;
    }
    let (mut head, histories) = prepared(0);
    let (mut worker, _) = prepared(1);
    let tx = Wire::install(&mut head.f, 0);
    let rx = Wire::install(&mut worker.f, 1);
    let shape = GlmOwnerBatchShape::new(8).unwrap();
    let tokens: [[u32; 5]; 8] = histories
        .each_ref()
        .map(|h| h.issued.as_slice().try_into().unwrap());
    let before = std::array::from_fn::<_, 8, _>(|slot| peer_snapshot(&worker, slot));
    let free = worker.f.model.kv_cache.lock().num_free_blocks();
    let private_free = worker.f.head.paired_test_free_blocks();
    head.f
        .model
        .owner_send_verify(shape, &mut head.states.each_mut(), &tokens)
        .unwrap();
    let mut packets = tx.packets();
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[1], [0xffff_ffe8]);
    assert_eq!(packets[2].len(), 92);
    packets[2][85] ^= 1; // Eighth attempt high word: well-formed, wrong actual receipt.
    rx.queue(&packets);
    worker.f.gpu.clear();
    let error = worker.receive().unwrap_err();
    assert!(
        format!("{error:#}").contains("actual mode/owner/generation/attempt/token mismatch"),
        "{error:#}"
    );
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
        "eighth mismatch must precede every target writer"
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
        "terminal refusal covers all eight owners"
    );
}
