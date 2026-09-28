// SPDX-License-Identifier: AGPL-3.0-only
//! Actual selected Model churn; byte sentinels are ownership, not CUDA numerics.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::traits::{Model, SequenceState};
use std::collections::BTreeSet;

fn blocks(seq: &SequenceState) -> BTreeSet<u32> {
    flow::private(seq).block_table.iter().copied().collect()
}

struct Peer {
    tokens: Vec<u32>,
    target: usize,
    private: usize,
    blocks: BTreeSet<u32>,
    kv: Vec<Vec<u8>>,
    slab: Vec<u8>,
}

impl Peer {
    fn snapshot(f: &Fixture, owner: usize) -> Self {
        let private = flow::private(&f.seqs[owner]).seq_len;
        Self {
            tokens: f.seqs[owner].tokens.clone(),
            target: f.seqs[owner].seq_len,
            private,
            blocks: blocks(&f.seqs[owner]),
            kv: flow::bytes(f, owner, private),
            slab: f
                .gpu
                .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
        }
    }

    fn unchanged(&self, f: &Fixture, owner: usize) {
        assert_eq!(f.seqs[owner].tokens, self.tokens);
        assert_eq!(f.seqs[owner].seq_len, self.target);
        assert_eq!(flow::private(&f.seqs[owner]).seq_len, self.private);
        assert_eq!(blocks(&f.seqs[owner]), self.blocks);
        assert_eq!(flow::bytes(f, owner, self.private), self.kv);
        assert_eq!(
            f.gpu
                .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
            self.slab
        );
    }
}

fn fresh_history(f: &mut Fixture, owner: usize, round: usize) -> flow::History {
    let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
    let prompt = &prompts[owner];
    let bootstrap = 3 + ((owner + round) % 3) as u32;
    let seed = 7 - owner as u32;
    assert_eq!(f.seqs[owner].slot_idx, owner);
    assert_eq!(f.seqs[owner].ssm_slot_idx(), Some(owner));
    assert!(
        f.seqs[owner]
            .ssm_slot
            .as_ref()
            .unwrap()
            .belongs_to(&f.model.ssm_pool)
    );
    assert!(f.seqs[owner].tokens.is_empty());
    f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
    f.model
        .decode(bootstrap, &mut f.seqs[owner], CALLER)
        .unwrap();
    let base = f.seqs[owner].seq_len;
    let drafts = f
        .model
        .run_mtp_propose_inner(seed, base, 4, &mut f.seqs[owner], None)
        .unwrap();
    assert_eq!(drafts.len(), 4);
    let canonical: Vec<_> = prompt
        .iter()
        .enumerate()
        .map(|(row, token)| vec![*token as u8 + 0x20 + row as u8; 1024])
        .collect();
    assert_eq!(flow::bytes(f, owner, base - 1), canonical);
    let bonus = flow::slab(f, owner, 5);
    assert_eq!(
        bonus,
        vec![bootstrap as u8 + 0x20 + prompt.len() as u8; ROW_BYTES]
    );
    let mut written = canonical.clone();
    written.extend((0..4).map(|_| bonus[..1024].to_vec()));
    assert_eq!(flow::bytes(f, owner, base + 3), written);
    flow::History {
        base,
        issued: std::iter::once(seed).chain(drafts).collect(),
        canonical,
        bonus,
    }
}

#[test]
fn both_actual_retirement_orders_reallocate_then_reverse_prime_and_verify() {
    if flow::isolated(
        "aligned_churn_tests::both_actual_retirement_orders_reallocate_then_reverse_prime_and_verify",
    ) {
        return;
    }
    for rank in 0..2 {
        for retirement in [[0, 1], [1, 0]] {
            let (mut f, mut history) = flow::prepare(rank, retirement);
            let reserve_union: BTreeSet<_> = blocks(&f.seqs[0])
                .union(&blocks(&f.seqs[1]))
                .copied()
                .collect();
            assert_eq!(reserve_union.len(), 256);
            for round in 0..3 {
                let accepted = if round % 2 == 0 { [0, 4] } else { [4, 0] };
                for owner in retirement {
                    flow::head_verdict(&mut f, owner, &history[owner], accepted[owner]);
                    flow::acknowledge(&mut f, owner, accepted[owner], owner == 0);
                }
                let peer = Peer::snapshot(&f, retirement[1]);
                assert!(
                    f.seqs[0]
                        .block_table
                        .iter()
                        .all(|block| !f.seqs[1].block_table.contains(block))
                );
                // Model owns a padding block outside both sequences. Derive
                // reusable target budget without pretending it is seq-owned.
                let target_reusable = f.model.kv_cache.lock().num_free_blocks()
                    + f.seqs
                        .iter()
                        .map(|seq| seq.block_table.len())
                        .sum::<usize>();
                // Separate actual calls model retirement across ticks. No
                // free-list reordering or delayed idle owner stash is supplied.
                f.model.free_sequence(&mut f.seqs[retirement[0]]).unwrap();
                assert_eq!(f.head.paired_test_free_blocks(), 128);
                peer.unchanged(&f, retirement[1]);
                f.model.free_sequence(&mut f.seqs[retirement[1]]).unwrap();
                assert_eq!(f.head.paired_test_free_blocks(), 256);
                assert_eq!(f.model.kv_cache.lock().num_free_blocks(), target_reusable);
                for owner in 0..2 {
                    let replacement = f
                        .model
                        .alloc_sequence()
                        .expect("actual private candidate must select its matching free target");
                    assert_eq!(replacement.slot_idx, owner);
                    assert_eq!(replacement.ssm_slot_idx(), Some(owner));
                    assert_eq!(f.head.paired_test_free_blocks(), (1 - owner) * 128);
                    let mut old = std::mem::replace(&mut f.seqs[owner], replacement);
                    f.gpu.clear();
                    f.model.free_sequence(&mut old).unwrap();
                    assert!(f.gpu.trace().is_empty());
                }
                let first = blocks(&f.seqs[0]);
                let second = blocks(&f.seqs[1]);
                assert!(first.is_disjoint(&second));
                assert_eq!(
                    first.union(&second).copied().collect::<BTreeSet<_>>(),
                    reserve_union
                );
                assert!(f.model.alloc_sequence().is_err());
                for owner in retirement.into_iter().rev() {
                    history[owner] = fresh_history(&mut f, owner, round);
                }
                // Actual subsequent K5/repair verifies both newly minted live
                // owners, not merely their public integer slot fields.
                for owner in retirement.into_iter().rev() {
                    flow::head_verdict(&mut f, owner, &history[owner], accepted[owner]);
                    flow::acknowledge(&mut f, owner, accepted[owner], owner == 1);
                    flow::continue_owner(&mut f, owner, &mut history[owner], accepted[owner]);
                }
            }
        }
    }
}

#[test]
fn alternating_actual_single_owner_reuse_preserves_live_peer() {
    if flow::isolated(
        "aligned_churn_tests::alternating_actual_single_owner_reuse_preserves_live_peer",
    ) {
        return;
    }
    for rank in 0..2 {
        for first in 0..2 {
            let (mut f, mut history) = flow::prepare(rank, [first, 1 - first]);
            for (round, owner) in [first, 1 - first, first, 1 - first].into_iter().enumerate() {
                let peer = 1 - owner;
                let saved = Peer::snapshot(&f, peer);
                let returned = blocks(&f.seqs[owner]);
                let accepted = if round % 2 == 0 { 0 } else { 4 };
                flow::head_verdict(&mut f, owner, &history[owner], accepted);
                flow::acknowledge(&mut f, owner, accepted, true);
                f.model.free_sequence(&mut f.seqs[owner]).unwrap();
                assert_eq!(f.head.paired_test_free_blocks(), 128);
                saved.unchanged(&f, peer);
                f.seqs[owner] = f.model.alloc_sequence().unwrap();
                assert_eq!(f.seqs[owner].slot_idx, owner);
                assert_eq!(blocks(&f.seqs[owner]), returned);
                assert!(returned.is_disjoint(&blocks(&f.seqs[peer])));
                history[owner] = fresh_history(&mut f, owner, round);
                saved.unchanged(&f, peer);
            }
        }
    }
}
