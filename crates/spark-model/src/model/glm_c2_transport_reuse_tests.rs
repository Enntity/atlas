// SPDX-License-Identifier: AGPL-3.0-only
//! Retained real wire identities cannot authorize a newly allocated owner.
use super::{
    fixture::*, transport_boundary_tests::terminal, transport_test_fixture as wire,
    verdict_continuation_tests as flow,
};
use crate::traits::{Model, SequenceState};

fn retire_and_bootstrap(f: &mut Fixture, owner: usize) -> SequenceState {
    for i in 0..2 {
        let guard = f.model.ssm_pool.claim_guarded().unwrap();
        assert_eq!(guard.idx(), Some(i));
        f.seqs[i].ssm_slot = Some(guard);
    }
    let original: std::collections::BTreeSet<_> = flow::private(&f.seqs[owner])
        .block_table
        .iter()
        .copied()
        .collect();
    f.model.free_sequence(&mut f.seqs[owner]).unwrap();
    let new = f.model.alloc_sequence().unwrap();
    assert_eq!(new.slot_idx, owner);
    assert_eq!(
        flow::private(&new)
            .block_table
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        original
    );
    let old = std::mem::replace(&mut f.seqs[owner], new);
    let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
    f.model
        .prefill(&prompts[owner], &mut f.seqs[owner], CALLER)
        .unwrap();
    f.model
        .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
        .unwrap();
    old
}

#[test]
fn reused_owner_accepts_new_generation_packet_but_rejects_real_retained_old_packet() {
    if flow::isolated(
        "transport_reuse_tests::reused_owner_accepts_new_generation_packet_but_rejects_real_retained_old_packet",
    ) {
        return;
    }
    for owner in 0..2 {
        let mut head = wire::bootstrapped(0, [1, 0]);
        let original_comm = head.model.comm.clone();
        let tx = wire::Wire::install(&mut head, 0);
        let position = head.seqs[owner].seq_len;
        head.model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut head.seqs[owner], 7, position, 4, None)
            .unwrap();
        let old_packet = tx.packets();
        assert_eq!(&old_packet[2][1..5], &[1, 0, 1, 0]);
        head.model.comm = original_comm;
        let mut old = retire_and_bootstrap(&mut head, owner);
        head.model.comm = Some(tx.clone());
        tx.clear();
        head.gpu.clear();
        assert!(
            head.model
                .glm_paired_execution()
                .unwrap()
                .propose(&mut old, 7, position, 4, None)
                .is_err()
        );
        assert!(tx.packets().is_empty());
        assert!(head.gpu.trace().is_empty());
        head.model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut head.seqs[owner], 7, position, 4, None)
            .unwrap();
        let new_packet = tx.packets();
        assert_eq!(&new_packet[2][1..5], &[2, 0, 1, 0]);
        assert_eq!(&new_packet[2][5..], &old_packet[2][5..]);
        for stale in [false, true] {
            let mut worker = wire::bootstrapped(1, [1, 0]);
            // A real first proposal consumes generation1 before actual retirement.
            worker
                .model
                .run_mtp_propose_inner(7, position, 4, &mut worker.seqs[owner], None)
                .unwrap();
            let _old = retire_and_bootstrap(&mut worker, owner);
            let rx = wire::Wire::install(&mut worker, 1);
            let before = worker.gpu.read_span(worker.gpu.slab(), SLAB_BYTES);
            rx.queue(if stale { &old_packet } else { &new_packet });
            worker.gpu.clear();
            let result = wire::worker(&mut worker);
            rx.done();
            if stale {
                assert!(format!("{:#}", result.unwrap_err()).contains("generation/attempt"));
                assert_eq!(worker.gpu.read_span(worker.gpu.slab(), SLAB_BYTES), before);
                assert!(!worker.gpu.trace().iter().any(|e| matches!(
                    e,
                    Event::Body(_, _) | Event::Kv(_, _) | Event::Kernel(_, _, _)
                )));
                terminal(&mut worker);
            } else {
                assert!(result.unwrap());
                wire::same_private(&head, &worker, owner);
            }
        }
    }
}
