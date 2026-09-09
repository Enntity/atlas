// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Pending retirement and reallocation. Old freed objects are retained;
//! no duplicate live Lease, Ready value, or generation token is manufactured.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::traits::{Model, SequenceState};
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

fn blocks(seq: &SequenceState) -> BTreeSet<u32> {
    flow::private(seq).block_table.iter().copied().collect()
}

struct Snapshot {
    tokens: Vec<u32>,
    target: usize,
    private: usize,
    blocks: BTreeSet<u32>,
    kv: Vec<Vec<u8>>,
    slab: Vec<u8>,
}
impl Snapshot {
    fn new(f: &Fixture, owner: usize) -> Self {
        let seq = &f.seqs[owner];
        let private = flow::private(seq).seq_len;
        Self {
            tokens: seq.tokens.clone(),
            target: seq.seq_len,
            private,
            blocks: blocks(seq),
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

fn new_history(f: &mut Fixture, owner: usize) -> flow::History {
    let prompt = [2, 7, 3, 6, 1];
    assert_eq!(f.seqs[owner].seq_len, 0);
    assert!(f.seqs[owner].tokens.is_empty());
    f.model
        .prefill(&prompt, &mut f.seqs[owner], CALLER)
        .unwrap();
    f.model.decode(3, &mut f.seqs[owner], CALLER).unwrap();
    let base = f.seqs[owner].seq_len;
    assert_eq!(base, 6);
    let seed = 2;
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
    assert_eq!(bonus, vec![3 + 0x20 + prompt.len() as u8; ROW_BYTES]);
    let mut expected = canonical.clone();
    expected.extend((0..4).map(|_| bonus[..1024].to_vec()));
    assert_eq!(flow::bytes(f, owner, base + 3), expected);
    flow::History {
        base,
        issued: std::iter::once(seed).chain(drafts).collect(),
        canonical,
        bonus,
    }
}

fn stale_calls(
    f: &Fixture,
    old: &mut SequenceState,
    old_base: usize,
    old_issued: &[u32],
    accepted: usize,
) {
    f.gpu.clear();
    assert!(
        f.model
            .record_glm_mtp_verified(old, old_base, old_issued, accepted)
            .is_err()
    );
    assert!(
        f.gpu.trace().is_empty(),
        "old detached verdict must not steal current Produced lease"
    );
    assert!(f.model.trim_proposer_state(old, accepted, 0).is_err());
    assert!(
        f.model
            .commit_accepted_prefix(old, accepted + 1, 5)
            .is_err()
    );
    assert!(
        f.model
            .decode_verify_graphed_kgamma(old_issued, old, CALLER)
            .is_err()
    );
    assert!(
        f.model
            .run_mtp_propose_inner(1, old.seq_len, 4, old, None)
            .is_err()
    );
    assert!(f.model.decode(1, old, CALLER).is_err());
    assert!(
        f.gpu.trace().is_empty(),
        "freed object cannot launch after its slot is reallocated"
    );
}

#[test]
fn pending_retirement_actual_model_reallocation_and_peer_continuation() {
    if flow::isolated(
        "verdict_lifecycle_tests::pending_retirement_actual_model_reallocation_and_peer_continuation",
    ) {
        return;
    }
    for rank in 0..2 {
        for victim in 0..2 {
            for retired_acceptance in [0, 4] {
                let peer = 1 - victim;
                let (mut f, mut histories) = flow::prepare(rank, [victim, peer]);
                // Fixture owners now originate from actual Model allocation.
                assert_eq!(f.model.ssm_pool.max_slots, 2);
                for owner in 0..2 {
                    let guard = f.seqs[owner].ssm_slot.as_ref().unwrap();
                    assert_eq!(guard.idx(), Some(owner));
                    assert!(guard.belongs_to(&f.model.ssm_pool));
                }
                assert!(f.model.ssm_pool.claim_guarded().is_err());
                let original = [blocks(&f.seqs[0]), blocks(&f.seqs[1])];
                assert_eq!(original[0].len(), 128);
                assert_eq!(original[1].len(), 128);
                assert!(original[0].is_disjoint(&original[1]));
                let accepted = if victim == 0 {
                    [retired_acceptance, 2]
                } else {
                    [2, retired_acceptance]
                };
                for owner in [victim, peer] {
                    flow::head_verdict(&mut f, owner, &histories[owner], accepted[owner]);
                    flow::detached(&f, owner, &histories[owner], accepted[owner]);
                }
                let old_base = histories[victim].base;
                let old_issued = histories[victim].issued.clone();
                let peer_before = Snapshot::new(&f, peer);
                let capture = f.model.mtp_prefill_capture_gen.load(Ordering::Relaxed);
                f.model.free_sequence(&mut f.seqs[victim]).unwrap();
                peer_before.unchanged(&f, peer);
                assert!(f.model.ssm_pool.slot_is_free(victim));
                assert!(!f.model.ssm_pool.slot_is_free(peer));
                let replacement = f.model.alloc_sequence().unwrap();
                assert_eq!(replacement.slot_idx, victim);
                assert_eq!(replacement.ssm_slot.as_ref().unwrap().idx(), Some(victim));
                assert_eq!(
                    blocks(&replacement),
                    original[victim],
                    "exact reserve, possibly reordered"
                );
                let mut old = std::mem::replace(&mut f.seqs[victim], replacement);
                histories[victim] = new_history(&mut f, victim);
                assert!(f.model.mtp_prefill_capture_gen.load(Ordering::Relaxed) > capture);
                peer_before.unchanged(&f, peer);

                // Keep a real new-generation verification Produced while stale old
                // requests attempt to consume it. The current producer must survive.
                let h = &histories[victim];
                let predictions = f
                    .model
                    .decode_verify_graphed_kgamma(&h.issued, &mut f.seqs[victim], CALLER)
                    .unwrap();
                assert_eq!(predictions.len(), 5);
                let replacement_before = Snapshot::new(&f, victim);
                stale_calls(&f, &mut old, old_base, &old_issued, retired_acceptance);
                peer_before.unchanged(&f, peer);
                replacement_before.unchanged(&f, victim);
                f.model.free_sequence(&mut old).unwrap(); // Repeat old free cannot return the replacement slot.
                assert!(!f.model.ssm_pool.slot_is_free(victim));
                f.gpu.clear();
                assert!(f.model.alloc_sequence().is_err());
                assert!(
                    f.gpu.trace().is_empty(),
                    "no third lease or double-returned slot"
                );
                f.seqs[victim].seq_len = h.base + 2;
                f.seqs[victim].tokens.truncate(h.base + 2);
                f.model
                    .record_glm_mtp_verified(&mut f.seqs[victim], h.base, &h.issued, 1)
                    .unwrap();
                flow::detached(&f, victim, h, 1);
                flow::acknowledge(&mut f, victim, 1, true);
                flow::continue_owner(&mut f, victim, &mut histories[victim], 1);
                peer_before.unchanged(&f, peer);
                flow::acknowledge(&mut f, peer, 2, false);
                flow::continue_owner(&mut f, peer, &mut histories[peer], 2);

                for (round, a) in [[0, 4], [4, 0], [2, 3]].into_iter().enumerate() {
                    let order = if round % 2 == 0 {
                        [peer, victim]
                    } else {
                        [victim, peer]
                    };
                    for owner in order {
                        flow::head_verdict(&mut f, owner, &histories[owner], a[owner]);
                        flow::detached(&f, owner, &histories[owner], a[owner]);
                    }
                    for owner in order.into_iter().rev() {
                        flow::acknowledge(&mut f, owner, a[owner], owner == 0);
                        flow::continue_owner(&mut f, owner, &mut histories[owner], a[owner]);
                        assert_eq!(blocks(&f.seqs[owner]), original[owner]);
                    }
                }
                // Current selected allocation requires aligned private-lowest
                // and target-LIFO indices; arbitrary two-idle churn is B2 work.
                for owner in [1, 0] {
                    f.model.free_sequence(&mut f.seqs[owner]).unwrap();
                }
                let mut next = [
                    f.model.alloc_sequence().unwrap(),
                    f.model.alloc_sequence().unwrap(),
                ];
                assert!(blocks(&next[0]).is_disjoint(&blocks(&next[1])));
                let union: BTreeSet<_> =
                    blocks(&next[0]).union(&blocks(&next[1])).copied().collect();
                assert_eq!(union, original[0].union(&original[1]).copied().collect());
                assert!(f.model.alloc_sequence().is_err());
                for seq in &mut next {
                    f.model.free_sequence(seq).unwrap();
                }
            }
        }
    }
}
