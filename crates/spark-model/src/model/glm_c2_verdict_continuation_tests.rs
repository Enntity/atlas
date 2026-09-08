// SPDX-License-Identifier: AGPL-3.0-only
//! Real producer/repair paths; dense sentinel ownership, not attention numerics.
use super::fixture::*;
use crate::layers::glm5_mtp::Glm5MtpProposerState;
use crate::traits::{Model, SequenceState};
use std::sync::atomic::Ordering;

pub(super) fn isolated(name: &str) -> bool {
    if std::env::var("ATLAS_C2_VERDICT_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        &format!("model::glm_c2_handoff_tests::{name}"),
        "--nocapture",
    ])
    .env("ATLAS_C2_VERDICT_TEST_CHILD", "1")
    .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
    .env("ATLAS_GLM_MTP_REPAIR", "0")
    .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
    .env("ATLAS_GLM_MTP_ALL_GATHER", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0")
    .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1");
    for key in [
        "ATLAS_NO_MTP_EAGER_DRAFTER",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_CARRY_DRAFTER",
        "ATLAS_MTP_ACCEPT_DEBUG",
        "ATLAS_GLM_MTP_FUSED_EH_NORM",
        "ATLAS_GLM_MTP_SERIAL_PREFILL",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_GLM_MTP_PROFILE",
    ] {
        cmd.env_remove(key);
    }
    assert!(
        cmd.status().unwrap().success(),
        "actual verdict child failed: {name}"
    );
    true
}

pub(super) fn private(seq: &SequenceState) -> &Glm5MtpProposerState {
    seq.proposer_state
        .as_ref()
        .unwrap()
        .as_any()
        .downcast_ref::<Glm5MtpProposerState>()
        .unwrap()
}

pub(super) fn bytes(f: &Fixture, owner: usize, rows: usize) -> Vec<Vec<u8>> {
    f.head
        .paired_test_kv_rows(
            f.seqs[owner].proposer_state.as_ref().unwrap().as_ref(),
            f.model.gpu.as_ref(),
            rows,
        )
        .unwrap()
        .into_iter()
        .map(|(k, v)| {
            let expected = f.gpu.read_span(k, 1024);
            assert_eq!(
                f.gpu.read_span(v, 1024),
                expected,
                "K and V must both be written"
            );
            expected
        })
        .collect()
}

pub(super) fn slab(f: &Fixture, owner: usize, row: usize) -> Vec<u8> {
    f.gpu.read_span(
        f.gpu.slab().offset((owner * 6 + row) * ROW_BYTES),
        ROW_BYTES,
    )
}

pub(super) struct History {
    pub base: usize,
    pub issued: Vec<u32>,
    pub canonical: Vec<Vec<u8>>,
    pub bonus: Vec<u8>,
}

pub(super) fn prepare(rank: usize, order: [usize; 2]) -> (Fixture, [History; 2]) {
    let mut f = Fixture::new(rank);
    f.gpu.deterministic_logits.store(true, Ordering::Relaxed);
    f.gpu.write_span(f.gpu.slab(), &vec![0xa5; SLAB_BYTES]);
    let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
    for owner in order {
        f.seqs[owner].prompt_len = prompts[owner].len();
        f.model
            .prefill(&prompts[owner], &mut f.seqs[owner], CALLER)
            .unwrap();
        f.model
            .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
            .unwrap();
    }
    let mut histories: [Option<History>; 2] = [None, None];
    for owner in order {
        let base = f.seqs[owner].seq_len;
        let seed = 7 - owner as u32;
        let drafts = f
            .model
            .run_mtp_propose_inner(seed, base, 4, &mut f.seqs[owner], None)
            .unwrap();
        assert_eq!(drafts.len(), 4);
        let expected: Vec<_> = prompts[owner]
            .iter()
            .enumerate()
            .map(|(row, token)| {
                vec![(*token as u8).wrapping_add(0x20).wrapping_add(row as u8); 1024]
            })
            .collect();
        assert_eq!(
            bytes(&f, owner, base - 1),
            expected,
            "real primer and bootstrap"
        );
        let bonus = slab(&f, owner, 5);
        assert_eq!(
            bonus,
            vec![5 + owner as u8 + 0x20 + prompts[owner].len() as u8; ROW_BYTES]
        );
        let mut all = expected.clone();
        all.extend((0..4).map(|_| bonus[..1024].to_vec()));
        assert_eq!(private(&f.seqs[owner]).seq_len, base + 3);
        assert_eq!(
            bytes(&f, owner, base + 3),
            all,
            "four actual private body writes"
        );
        let mut issued = vec![seed];
        issued.extend(drafts);
        histories[owner] = Some(History {
            base,
            issued,
            canonical: expected,
            bonus,
        });
    }
    assert_ne!(
        histories[0].as_ref().unwrap().base,
        histories[1].as_ref().unwrap().base
    );
    assert!(
        private(&f.seqs[0])
            .block_table
            .iter()
            .all(|block| !private(&f.seqs[1]).block_table.contains(block))
    );
    (f, histories.map(Option::unwrap))
}

pub(super) fn normalized(h: &History) -> Vec<Vec<u8>> {
    h.issued
        .iter()
        .enumerate()
        .map(|(row, token)| {
            vec![
                (*token as u8)
                    .wrapping_add(0x20)
                    .wrapping_add((h.base + row) as u8);
                ROW_BYTES
            ]
        })
        .collect()
}

pub(super) fn head_verdict(f: &mut Fixture, owner: usize, h: &History, accepted: usize) {
    let predictions = f
        .model
        .decode_verify_graphed_kgamma(&h.issued, &mut f.seqs[owner], CALLER)
        .unwrap();
    let rows = normalized(h);
    assert_eq!(
        predictions,
        rows.iter().map(|r| u32::from(r[0] % 8)).collect::<Vec<_>>()
    );
    assert_eq!(f.seqs[owner].seq_len, h.base + 5);
    assert_eq!(&f.seqs[owner].tokens[h.base..], h.issued);
    f.seqs[owner].tokens.truncate(h.base + accepted + 1);
    f.seqs[owner].seq_len = h.base + accepted + 1;
    f.model
        .record_glm_mtp_verified(&mut f.seqs[owner], h.base, &h.issued, accepted)
        .unwrap();
}

pub(super) fn detached(f: &Fixture, owner: usize, h: &History, accepted: usize) {
    let rows = normalized(h);
    assert_eq!(
        slab(f, owner, 5),
        rows[accepted],
        "owned actual target bonus"
    );
    for (row, expected) in rows.iter().take(accepted).enumerate() {
        assert_eq!(
            slab(f, owner, row + 1),
            *expected,
            "owned actual accepted target row"
        );
    }
    assert_eq!(
        private(&f.seqs[owner]).seq_len,
        h.base + 3,
        "record/ack must not pretend that canonical private repair ran"
    );
    assert_eq!(bytes(f, owner, h.base - 1), h.canonical);
    assert_eq!(
        bytes(f, owner, h.base)[h.base - 1],
        h.bonus[..1024],
        "keep_seed is actual data"
    );
}

pub(super) fn acknowledge(f: &mut Fixture, owner: usize, a: usize, commit_first: bool) {
    if commit_first {
        f.model
            .commit_accepted_prefix(&mut f.seqs[owner], a + 1, 5)
            .unwrap();
        f.model
            .trim_proposer_state(&mut f.seqs[owner], a, 0)
            .unwrap();
    } else {
        f.model
            .trim_proposer_state(&mut f.seqs[owner], a, 0)
            .unwrap();
        f.model
            .commit_accepted_prefix(&mut f.seqs[owner], a + 1, 5)
            .unwrap();
    }
}

pub(super) fn continue_owner(f: &mut Fixture, owner: usize, h: &mut History, accepted: usize) {
    continue_with_seed(f, owner, h, accepted, None);
}

fn continue_with_seed(
    f: &mut Fixture,
    owner: usize,
    h: &mut History,
    accepted: usize,
    selected: Option<u32>,
) {
    let rows = normalized(h);
    let next_base = h.base + accepted + 1;
    h.canonical.push(h.bonus[..1024].to_vec());
    h.canonical
        .extend(rows.iter().take(accepted).map(|row| row[..1024].to_vec()));
    let bonus = rows[accepted].clone();
    let seed = selected.unwrap_or(u32::from(bonus[0] % 8));
    assert!(seed < 8);
    assert_eq!(h.canonical.len(), next_base - 1);
    let peer = 1 - owner;
    let peer_rows = private(&f.seqs[peer]).seq_len;
    let peer_bytes = bytes(f, peer, peer_rows);
    let peer_slab = f
        .gpu
        .read_span(f.gpu.slab().offset(peer * 6 * ROW_BYTES), 6 * ROW_BYTES);
    f.gpu.clear();
    let drafts = f
        .model
        .run_mtp_propose_inner(seed, next_base, 4, &mut f.seqs[owner], None)
        .unwrap();
    assert_eq!(drafts.len(), 4);
    let mut pairs: Vec<_> = h.issued[1..1 + accepted]
        .iter()
        .zip(rows.iter())
        .map(|(token, hidden)| (*token as u8, hidden[0]))
        .collect();
    pairs.extend(
        std::iter::once(seed)
            .chain(drafts.iter().take(3).copied())
            .map(|token| (token as u8, bonus[0])),
    );
    assert_eq!(
        f.gpu.eh_pairs(),
        pairs,
        "actual EH boundary must see shifted repair token/hidden pairs, then selected seed and real draft feedback"
    );
    let trace = f.gpu.trace();
    let kv_counts: Vec<_> = trace
        .iter()
        .filter_map(|e| match e {
            Event::Kv(slots, DEFAULT) => Some(slots.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        kv_counts,
        if accepted == 0 {
            vec![]
        } else {
            vec![accepted]
        },
        "only accepted pairs use the real KV-only writer; a=0 writes none"
    );
    assert_eq!(
        trace
            .iter()
            .filter(|e| matches!(e, Event::Body(_, DEFAULT)))
            .count(),
        4
    );
    assert_eq!(
        bytes(f, owner, next_base - 1),
        h.canonical,
        "all repaired canonical K/V bytes"
    );
    let mut expected = h.canonical.clone();
    expected.extend((0..4).map(|_| bonus[..1024].to_vec()));
    assert_eq!(private(&f.seqs[owner]).seq_len, next_base + 3);
    assert_eq!(
        bytes(f, owner, next_base + 3),
        expected,
        "rejected suffix is not retained; next four actual writes start at the canonical end"
    );
    assert_eq!(bytes(f, peer, peer_rows), peer_bytes);
    assert_eq!(
        f.gpu
            .read_span(f.gpu.slab().offset(peer * 6 * ROW_BYTES), 6 * ROW_BYTES),
        peer_slab
    );
    h.base = next_base;
    h.issued = std::iter::once(seed).chain(drafts).collect();
    h.bonus = bonus;
}

#[test]
fn all25_head_verdicts_two_ranks_two_orders_actual_continuation() {
    if isolated(
        "verdict_continuation_tests::all25_head_verdicts_two_ranks_two_orders_actual_continuation",
    ) {
        return;
    }
    for rank in 0..2 {
        for order in [[0, 1], [1, 0]] {
            for a in 0..5 {
                for b in 0..5 {
                    let accepted = [a, b];
                    let (mut f, mut histories) = prepare(rank, order);
                    for owner in order {
                        head_verdict(&mut f, owner, &histories[owner], accepted[owner]);
                        detached(&f, owner, &histories[owner], accepted[owner]);
                    }
                    // Peer K5 really overwrote normalized scratch after first detachment.
                    let first = order[0];
                    assert_ne!(
                        f.gpu.read_span(f.model.buffers.norm_output(), ROW_BYTES),
                        normalized(&histories[first])[0]
                    );
                    for owner in order.into_iter().rev() {
                        detached(&f, owner, &histories[owner], accepted[owner]);
                        acknowledge(&mut f, owner, accepted[owner], owner == 0);
                        continue_owner(&mut f, owner, &mut histories[owner], accepted[owner]);
                    }
                }
            }
        }
    }
}

#[test]
fn multi_round_zero_full_asymmetric_after_peer_scratch_overwrite() {
    if isolated(
        "verdict_continuation_tests::multi_round_zero_full_asymmetric_after_peer_scratch_overwrite",
    ) {
        return;
    }
    for rank in 0..2 {
        for first_order in [[0, 1], [1, 0]] {
            let (mut f, mut histories) = prepare(rank, first_order);
            for (round, accepted) in [[0, 4], [4, 0], [1, 3], [3, 1], [4, 4], [0, 0]]
                .into_iter()
                .enumerate()
            {
                let order = if round % 2 == 0 {
                    first_order
                } else {
                    [first_order[1], first_order[0]]
                };
                for owner in order {
                    head_verdict(&mut f, owner, &histories[owner], accepted[owner]);
                    detached(&f, owner, &histories[owner], accepted[owner]);
                }
                for owner in order.into_iter().rev() {
                    detached(&f, owner, &histories[owner], accepted[owner]);
                    acknowledge(&mut f, owner, accepted[owner], round % 2 == 0);
                    continue_owner(&mut f, owner, &mut histories[owner], accepted[owner]);
                }
            }
        }
    }
}

#[test]
fn selected_nonraw_bonus_is_consumed_and_sealed_by_next_actual_k5() {
    if isolated(
        "verdict_continuation_tests::selected_nonraw_bonus_is_consumed_and_sealed_by_next_actual_k5",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let peer = 1 - owner;
            let (mut f, mut histories) = prepare(rank, [owner, peer]);
            for who in [owner, peer] {
                let accepted = if who == owner { 4 } else { 0 };
                head_verdict(&mut f, who, &histories[who], accepted);
                detached(&f, who, &histories[who], accepted);
                acknowledge(&mut f, who, accepted, who == owner);
            }
            continue_owner(&mut f, peer, &mut histories[peer], 0);
            let raw = u32::from(normalized(&histories[owner])[4][0] % 8);
            let selected = (raw + 1) % 8;
            assert_ne!(selected, raw);
            continue_with_seed(&mut f, owner, &mut histories[owner], 4, Some(selected));
            assert_eq!(histories[owner].issued[0], selected);
            // No public receipt is minted: the actual next K5 accepts exactly the
            // seed and four IDs returned by the preceding actual proposal.
            head_verdict(&mut f, owner, &histories[owner], 3);
            detached(&f, owner, &histories[owner], 3);
            acknowledge(&mut f, owner, 3, false);
            continue_owner(&mut f, owner, &mut histories[owner], 3);
        }
    }
}
