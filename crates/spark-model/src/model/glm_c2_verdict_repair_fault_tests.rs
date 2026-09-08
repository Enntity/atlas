// SPDX-License-Identifier: AGPL-3.0-only
//! Detached repair failures quarantine one owner, not the released producer lease.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::speculative::DraftProposer;
use crate::traits::Model;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
enum Boundary {
    RepairWriter,
    FirstBody,
    LastBody,
}

fn expected(f: &Fixture, owner: usize, h: &flow::History, a: usize, boundary: Boundary) -> Event {
    match boundary {
        Boundary::RepairWriter => {
            assert!(a > 0);
            let blocks = &flow::private(&f.seqs[owner]).block_table;
            Event::Kv(
                (h.base..h.base + a)
                    .map(|row| i64::from(blocks[row / 16]) * 16 + (row % 16) as i64)
                    .collect(),
                DEFAULT,
            )
        }
        Boundary::FirstBody => Event::Body(h.base + a, DEFAULT),
        Boundary::LastBody => Event::Body(h.base + a + 3, DEFAULT),
    }
}

fn prepared(rank: usize, owner: usize, a: usize) -> (Fixture, [flow::History; 2]) {
    let (mut f, histories) = flow::prepare(rank, [1 - owner, owner]);
    flow::head_verdict(&mut f, owner, &histories[owner], a);
    flow::detached(&f, owner, &histories[owner], a);
    flow::acknowledge(&mut f, owner, a, owner == 0);
    (f, histories)
}

fn control(rank: usize, owner: usize, a: usize, boundary: Boundary) -> usize {
    let (mut f, mut histories) = prepared(rank, owner, a);
    let event = expected(&f, owner, &histories[owner], a, boundary);
    flow::continue_owner(&mut f, owner, &mut histories[owner], a);
    let found: Vec<_> = f
        .gpu
        .trace()
        .iter()
        .enumerate()
        .filter_map(|(i, e)| (*e == event).then_some(i + 1))
        .collect();
    assert_eq!(
        found.len(),
        1,
        "successful complete repair/reproposal event {event:?}"
    );
    found[0]
}

fn peer_continues(f: &mut Fixture, owner: usize, h: &flow::History) {
    flow::head_verdict(f, owner, h, 2);
    flow::detached(f, owner, h, 2);
    flow::acknowledge(f, owner, 2, false);
    let rows = flow::normalized(h);
    let seed = u32::from(rows[2][0] % 8);
    let next_base = h.base + 3;
    // Call real ownership/repair/body path directly: shared continue helper
    // inspects both live peers, while the other state is intentionally failed.
    let drafts = f
        .model
        .run_mtp_propose_inner(seed, next_base, 4, &mut f.seqs[owner], None)
        .unwrap();
    assert_eq!(drafts.len(), 4);
    let mut canonical = h.canonical.clone();
    canonical.push(h.bonus[..1024].to_vec());
    canonical.extend(rows.iter().take(2).map(|row| row[..1024].to_vec()));
    assert_eq!(canonical.len(), next_base - 1);
    let mut all = canonical;
    all.extend((0..4).map(|_| rows[2][..1024].to_vec()));
    assert_eq!(flow::private(&f.seqs[owner]).seq_len, next_base + 3);
    assert_eq!(flow::bytes(f, owner, next_base + 3), all);
    let issued: Vec<_> = std::iter::once(seed).chain(drafts).collect();
    assert_eq!(
        f.model
            .decode_verify_graphed_kgamma(&issued, &mut f.seqs[owner], CALLER)
            .unwrap()
            .len(),
        5,
        "healthy peer's next actual issued K5 remains usable"
    );
    f.seqs[owner].seq_len = next_base + 1;
    f.seqs[owner].tokens.truncate(next_base + 1);
    f.model
        .record_glm_mtp_verified(&mut f.seqs[owner], next_base, &issued, 0)
        .unwrap();
}

fn fault(rank: usize, owner: usize, a: usize, boundary: Boundary) {
    let ordinal = control(rank, owner, a, boundary);
    let (mut f, histories) = prepared(rank, owner, a);
    let peer = 1 - owner;
    let peer_len = flow::private(&f.seqs[peer]).seq_len;
    let peer_kv = flow::bytes(&f, peer, peer_len);
    let peer_blocks = flow::private(&f.seqs[peer]).block_table.clone();
    let peer_tokens = f.seqs[peer].tokens.clone();
    let peer_slab = f
        .gpu
        .read_span(f.gpu.slab().offset(peer * 6 * ROW_BYTES), 6 * ROW_BYTES);
    let event = expected(&f, owner, &histories[owner], a, boundary);
    let rows = flow::normalized(&histories[owner]);
    let seed = u32::from(rows[a][0] % 8);
    let next_base = histories[owner].base + a + 1;
    f.gpu.clear();
    f.gpu.fail.store(ordinal, Ordering::Relaxed);
    // Do not use flow::continue_owner here: it clears the fault injector.
    let error = f
        .model
        .run_mtp_propose_inner(seed, next_base, 4, &mut f.seqs[owner], None)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("injected fixture operation failure"),
        "{error:#}"
    );
    assert_eq!(f.gpu.trace().get(ordinal - 1), Some(&event));
    assert_eq!(flow::bytes(&f, peer, peer_len), peer_kv);
    assert_eq!(flow::private(&f.seqs[peer]).block_table, peer_blocks);
    assert_eq!(f.seqs[peer].tokens, peer_tokens);
    assert_eq!(
        f.gpu
            .read_span(f.gpu.slab().offset(peer * 6 * ROW_BYTES), 6 * ROW_BYTES),
        peer_slab
    );
    f.gpu.fail.store(usize::MAX, Ordering::Relaxed);
    f.model.gpu.synchronize(DEFAULT).unwrap();
    f.gpu.clear();
    assert!(
        f.model
            .run_mtp_propose_inner(seed, next_base, 4, &mut f.seqs[owner], None)
            .is_err()
    );
    assert!(
        f.model
            .decode_verify_graphed_kgamma(&histories[owner].issued, &mut f.seqs[owner], CALLER)
            .is_err()
    );
    assert!(
        f.gpu.trace().is_empty(),
        "failed detached owner cannot retry after good completion"
    );
    let failed_slab = f
        .gpu
        .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES);
    peer_continues(&mut f, peer, &histories[peer]);
    assert_eq!(
        f.gpu
            .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
        failed_slab
    );
    assert!(f.model.free_sequence(&mut f.seqs[owner]).is_err());
    assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
    f.model.free_sequence(&mut f.seqs[peer]).unwrap();
    let mut replacement = f.head.alloc_state(f.model.gpu.as_ref()).unwrap();
    let actual = replacement
        .as_any()
        .downcast_ref::<crate::layers::glm5_mtp::Glm5MtpProposerState>()
        .unwrap();
    assert_eq!(
        actual
            .block_table
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        peer_blocks.into_iter().collect()
    );
    assert!(
        f.head.alloc_state(f.model.gpu.as_ref()).is_err(),
        "failed reserve was not recycled"
    );
    f.head
        .free_state(f.model.gpu.as_ref(), replacement.as_mut())
        .unwrap();
}

#[test]
fn actual_accepted_writer_failure_keeps_detached_peer_usable() {
    if flow::isolated(
        "verdict_repair_fault_tests::actual_accepted_writer_failure_keeps_detached_peer_usable",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for a in [1, 4] {
                fault(rank, owner, a, Boundary::RepairWriter);
            }
        }
    }
}

#[test]
fn first_and_last_reproposal_body_failure_keeps_detached_peer_usable() {
    if flow::isolated(
        "verdict_repair_fault_tests::first_and_last_reproposal_body_failure_keeps_detached_peer_usable",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for a in [0, 4] {
                for boundary in [Boundary::FirstBody, Boundary::LastBody] {
                    fault(rank, owner, a, boundary);
                }
            }
        }
    }
}
