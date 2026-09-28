// SPDX-License-Identifier: AGPL-3.0-only
//! Actual capacity-eight scheduler/worker replay, not native collective numerics.
use super::*;

#[test]
fn actual_c5_c8_and_sparse_high_cohorts_use_exact_owner_transport() {
    if process::isolated(
        "scheduler::glm_owner_step::tests::owner8::actual_c5_c8_and_sparse_high_cohorts_use_exact_owner_transport",
    ) {
        return;
    }
    for physical in [
        &[0usize, 2, 4, 6, 7][..],
        &[0, 1, 3, 4, 6, 7][..],
        &[1, 2, 3, 4, 5, 6, 7][..],
        &[0, 1, 2, 3, 4, 5, 6, 7][..],
        &[1, 5, 7][..],
        &[0, 2, 5, 7][..],
    ] {
        for reverse in [false, true] {
            let mut r = Run::with_capacity(8, physical, reverse);
            r.cold();
            for _ in 0..2 {
                owner_round(&mut r, physical);
            }
            r.close();
        }
    }
}

fn owner_round(r: &mut Run, physical: &[usize]) {
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
    let spare: Vec<_> = r
        .spare
        .iter()
        .map(|seq| {
            let rows = r.observer.private_cursor(&r.model, seq).unwrap();
            (
                seq.slot_idx,
                rows,
                r.observer.snapshot(&r.model, seq, rows).unwrap(),
            )
        })
        .collect();
    let stats_before = r.sched.stats.glm_c2.snapshot();
    r.step();
    let packets = r.tx.packets();
    let (command, payload, verdict) = if physical.len() >= 5 {
        (0xffff_ffe8, 92, 10)
    } else {
        (0xffff_ffe7, 48, 6)
    };
    assert_eq!(
        packets[1],
        [command],
        "ready cohort must not silently fall back"
    );
    assert_eq!(packets.len(), 4 + physical.len() * 3);
    assert_eq!(packets[2].len(), payload);
    assert_eq!(packets[3].len(), verdict);
    assert_eq!(
        &packets[2][..4],
        &[1, physical.len() as u32, (physical.len() * 5) as u32, 1]
    );
    assert_eq!(&packets[3][..2], &[1, physical.len() as u32]);
    assert!(
        packets[2][4 + physical.len() * 11..]
            .iter()
            .all(|&v| v == 0)
    );
    assert!(packets[3][2 + physical.len()..].iter().all(|&v| v == 0));
    let stats = r.sched.stats.glm_c2.snapshot();
    assert_eq!(stats.pair_commits, stats_before.pair_commits);
    assert_eq!(stats.serial_commits, stats_before.serial_commits);
    assert!(stats.owner_commits.len() >= 6);
    for group in 0..6 {
        let selected = group == physical.len() - 3;
        assert_eq!(
            stats.owner_commits[group],
            stats_before.owner_commits[group] + u64::from(selected)
        );
        for count in 0..5 {
            let added = if selected {
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
        "checked sampling must use packed owner ordinal"
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
    for (slot, rows, saved) in spare {
        let seq = r.spare.iter().find(|seq| seq.slot_idx == slot).unwrap();
        assert_eq!(r.observer.private_cursor(&r.model, seq).unwrap(), rows);
        let now = r.observer.read_snapshot(&saved).unwrap();
        let last = now.len() - 1;
        assert_eq!(now[..last], saved.initial()[..last]);
        let range = slot * 6 * 8192..(slot + 1) * 6 * 8192;
        assert_eq!(now[last][range.clone()], saved.initial()[last][range]);
    }
    r.replay(1 + physical.len());
}

#[test]
fn actual_e8_to_e7_to_high_pair_to_slot7_scalar_drain() {
    if process::isolated(
        "scheduler::glm_owner_step::tests::owner8::actual_e8_to_e7_to_high_pair_to_slot7_scalar_drain",
    ) {
        return;
    }
    let mut r = Run::with_capacity(8, &[0, 1, 2, 3, 4, 5, 6, 7], true);
    r.cold();
    owner_round(&mut r, &[0, 1, 2, 3, 4, 5, 6, 7]);
    for a in &mut r.active {
        if ![3, 6, 7].contains(&a.seq.slot_idx) {
            a.finished = true;
        }
    }
    owner_round(&mut r, &[3, 6, 7]);
    r.active
        .iter_mut()
        .find(|a| a.seq.slot_idx == 3)
        .unwrap()
        .finished = true;
    r.step();
    assert_eq!(r.tx.packets()[0], [6]);
    assert_eq!(r.tx.packets()[1], [0xffff_ffe6]);
    assert!(
        !r.tx
            .packets()
            .iter()
            .any(|p| p == &[0xffff_ffe7] || p == &[0xffff_ffe8])
    );
    r.replay(3);
    r.active
        .iter_mut()
        .find(|a| a.seq.slot_idx == 6)
        .unwrap()
        .finished = true;
    r.step();
    assert_eq!(r.tx.packets()[0], [7]);
    assert_eq!(r.tx.packets()[1], [0xffff_fff5]);
    assert!(
        !r.tx
            .packets()
            .iter()
            .any(|p| p == &[0xffff_ffe6] || p == &[0xffff_ffe7] || p == &[0xffff_ffe8])
    );
    r.replay(2);
    assert_eq!(
        r.active.iter().find(|a| !a.finished).unwrap().seq.slot_idx,
        7
    );
    r.close();
}
