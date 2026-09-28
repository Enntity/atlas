// SPDX-License-Identifier: AGPL-3.0-only
//! Actual high-slot E6/E1 owners; fixed packet ABI and byte ownership, not CUDA numerics.
use super::*;
use crate::layers::glm5_mtp::Glm5MtpHead;

#[test]
fn fixed_pair_membership_through_eight_requires_complete_canonical_group() {
    for capacity in 2..=8 {
        for first in (0..8).step_by(2) {
            assert_eq!(
                Glm5MtpHead::validate_fixed_pair_slots([first, first + 1], capacity).is_ok(),
                first + 1 < capacity,
                "physical group {first} at actual capacity {capacity}"
            );
        }
        for slots in [
            [0, 0],
            [1, 0],
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 6],
            [6, 6],
            [8, 9],
        ] {
            assert!(Glm5MtpHead::validate_fixed_pair_slots(slots, capacity).is_err());
        }
    }
    for capacity in [0, 1, 9, usize::MAX] {
        assert!(Glm5MtpHead::validate_fixed_pair_slots([0, 1], capacity).is_err());
    }
}

#[test]
fn actual_high_physical_pairs_preserve_e6_and_unselected_owners() {
    if flow::isolated(
        "pair_group_tests::owner8_pair::actual_high_physical_pairs_preserve_e6_and_unselected_owners",
    ) {
        return;
    }
    let (mut head, mut histories) = Eight::prepared(0);
    let (mut worker, _) = Eight::prepared(1);
    let tx = Wire::install(&mut head.f, 0);
    let rx = Wire::install(&mut worker.f, 1);
    for (base, accepted) in [(4, [0usize, 4]), (6, [4, 0])] {
        let peers: Vec<_> = (0..8)
            .filter(|slot| *slot != base && *slot != base + 1)
            .collect();
        let before_head: Vec<_> = peers
            .iter()
            .map(|&slot| peer_snapshot(&head, slot))
            .collect();
        let before_worker: Vec<_> = peers
            .iter()
            .map(|&slot| peer_snapshot(&worker, slot))
            .collect();
        let tokens: [[u32; 5]; 2] =
            std::array::from_fn(|i| histories[base + i].issued.as_slice().try_into().unwrap());
        let normalized = [
            flow::normalized(&histories[base]),
            flow::normalized(&histories[base + 1]),
        ];
        tx.clear();
        let [s0, s1] =
            <&mut [SequenceState; 2]>::try_from(&mut head.states[base..base + 2]).unwrap();
        let predictions = head
            .f
            .model
            .glm_paired_execution()
            .unwrap()
            .verify_pair([s0, s1], &tokens)
            .expect("actual high physical owner group must retain the fixed E6 path");
        let [s0, s1] =
            <&mut [SequenceState; 2]>::try_from(&mut head.states[base..base + 2]).unwrap();
        head.f
            .model
            .glm_paired_execution()
            .unwrap()
            .finish_verify_pair([s0, s1], &tokens, accepted)
            .unwrap();
        let packets = tx.packets();
        assert_eq!(packets.len(), 4);
        assert_eq!(packets[0], [base as u32]);
        assert_eq!(packets[1], [0xffff_ffe6]);
        assert_eq!(packets[2].len(), 26, "original fixed E6 payload extent");
        assert_eq!(&packets[2][..4], &[1, 2, 10, 1]);
        for i in 0..2 {
            let start = 4 + 11 * i;
            assert_eq!(packets[2][start], (base + i) as u32);
            assert_ne!(
                u64::from(packets[2][start + 1]) | (u64::from(packets[2][start + 2]) << 32),
                0
            );
            assert_ne!(
                u64::from(packets[2][start + 3]) | (u64::from(packets[2][start + 4]) << 32),
                0
            );
            assert_eq!(packets[2][start + 5], histories[base + i].base as u32);
            assert_eq!(&packets[2][start + 6..start + 11], &tokens[i]);
        }
        assert_eq!(packets[3], [1, 2, accepted[0] as u32, accepted[1] as u32]);
        worker.replay(&rx, &packets);
        for i in [1, 0] {
            let slot = base + i;
            let h = &mut histories[slot];
            let end = h.base + accepted[i] + 1;
            assert_eq!(
                predictions[i],
                std::array::from_fn(|row| u32::from(normalized[i][row][0] % 8))
            );
            for f in [&head, &worker] {
                assert_eq!(f.states[slot].seq_len, end);
                assert_eq!(
                    &f.states[slot].tokens[h.base..],
                    &tokens[i][..accepted[i] + 1]
                );
                assert_eq!(f.slab_row(slot, 5), normalized[i][accepted[i]]);
                for row in 0..accepted[i] {
                    assert_eq!(f.slab_row(slot, row + 1), normalized[i][row]);
                }
            }
            let seed = predictions[i][accepted[i]];
            tx.clear();
            head.f.gpu.clear();
            worker.f.gpu.clear();
            let drafts = head
                .f
                .model
                .glm_paired_execution()
                .unwrap()
                .propose(&mut head.states[slot], seed, end, 4, None)
                .unwrap();
            assert_eq!(drafts.len(), 4);
            let packets = tx.packets();
            assert_eq!(packets.len(), 3);
            assert_eq!(packets[0], [slot as u32]);
            assert_eq!(packets[1], [0xffff_ffe1]);
            assert_eq!(&packets[2][5..], &[end as u32, 4, seed]);
            worker.replay(&rx, &packets);
            let mut expected_eh: Vec<_> = h.issued[1..1 + accepted[i]]
                .iter()
                .zip(&normalized[i])
                .map(|(token, hidden)| (*token as u8, hidden[0]))
                .collect();
            expected_eh.extend(
                std::iter::once(seed)
                    .chain(drafts.iter().take(3).copied())
                    .map(|token| (token as u8, normalized[i][accepted[i]][0])),
            );
            assert_eq!(head.f.gpu.eh_pairs(), expected_eh);
            assert_eq!(worker.f.gpu.eh_pairs(), expected_eh);
            h.canonical.push(h.bonus[..1024].to_vec());
            h.canonical.extend(
                normalized[i]
                    .iter()
                    .take(accepted[i])
                    .map(|row| row[..1024].to_vec()),
            );
            h.base = end;
            h.bonus = normalized[i][accepted[i]].clone();
            h.issued = std::iter::once(seed).chain(drafts).collect();
            head.assert_private(slot, h);
            worker.assert_private(slot, h);
        }
        for (i, &slot) in peers.iter().enumerate() {
            assert_eq!(peer_snapshot(&head, slot), before_head[i]);
            assert_eq!(peer_snapshot(&worker, slot), before_worker[i]);
        }
    }
}
