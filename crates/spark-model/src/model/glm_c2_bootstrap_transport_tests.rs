// SPDX-License-Identifier: AGPL-3.0-only
//! Real prefill owners and scalar command replay; no native collective proof.
use super::verdict_continuation_tests as flow;
use super::{fixture::*, isolated, transport_test_fixture as wire};
use crate::traits::Model;
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::Ordering;

pub(super) fn prepared(rank: usize, order: [usize; 2]) -> Fixture {
    let mut f = Fixture::new(rank);
    f.gpu.write_span(f.gpu.slab(), &vec![0xa5; SLAB_BYTES]);
    let prompts = [[1, 2, 3, 4], [4, 3, 2, 1]];
    for owner in order {
        f.model
            .prefill(&prompts[owner], &mut f.seqs[owner], CALLER)
            .unwrap();
        assert_eq!(f.seqs[owner].seq_len, 4);
        assert_eq!(flow::private(&f.seqs[owner]).seq_len, 3);
    }
    assert_ne!(flow::slab(&f, 0, 0), flow::slab(&f, 1, 0));
    f.gpu.clear();
    f
}

pub(super) fn execute(f: &mut Fixture, owner: usize, rank: usize) -> anyhow::Result<()> {
    if rank == 0 {
        f.model
            .glm_paired_execution()
            .unwrap()
            .bootstrap(&mut f.seqs[owner], 5 + owner as u32)?;
    } else {
        assert!(wire::worker(f)?);
    }
    Ok(())
}

pub(super) fn bonus(f: &Fixture, owner: usize) -> Event {
    Event::Copy(
        f.model.buffers.norm_output(),
        f.gpu.slab().offset((owner * 6 + 5) * ROW_BYTES),
        ROW_BYTES,
        DEFAULT,
    )
}

pub(super) struct Snapshot {
    slab: Vec<u8>,
    rows: Vec<(DevicePtr, Vec<u8>)>,
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    private_blocks: Vec<u32>,
    position: usize,
}

impl Snapshot {
    pub(super) fn new(f: &Fixture, owner: usize) -> Self {
        let seq = &f.seqs[owner];
        let rows = f
            .head
            .paired_test_kv_rows(
                seq.proposer_state.as_ref().unwrap().as_ref(),
                f.model.gpu.as_ref(),
                3,
            )
            .unwrap()
            .into_iter()
            .flat_map(|(k, v)| [k, v])
            .map(|p| (p, f.gpu.read_span(p, 1024)))
            .collect();
        Self {
            slab: f
                .gpu
                .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
            rows,
            tokens: seq.tokens.clone(),
            blocks: seq.block_table.clone(),
            private_blocks: flow::private(seq).block_table.clone(),
            position: seq.seq_len,
        }
    }
    pub(super) fn unchanged(&self, f: &Fixture, owner: usize) {
        let seq = &f.seqs[owner];
        assert_eq!(
            f.gpu
                .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
            self.slab
        );
        assert_eq!(seq.tokens, self.tokens);
        assert_eq!(seq.block_table, self.blocks);
        assert_eq!(seq.seq_len, self.position);
        assert_eq!(flow::private(seq).block_table, self.private_blocks);
        assert_eq!(flow::private(seq).seq_len, 3);
        // Saved physical owners remain observable even after capability quarantine.
        for (p, bytes) in &self.rows {
            assert_eq!(f.gpu.read_span(*p, 1024), *bytes);
        }
    }
}

#[test]
fn immutable_bootstrap_checks_do_not_claim_then_real_send_publishes() {
    if isolated(
        "bootstrap_transport_tests::immutable_bootstrap_checks_do_not_claim_then_real_send_publishes",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut f = prepared(0, order);
        let tx = wire::Wire::install(&mut f, 0);
        let before = [Snapshot::new(&f, 0), Snapshot::new(&f, 1)];
        let free = f.model.kv_cache.lock().num_free_blocks();
        for _ in 0..3 {
            for owner in order {
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .validate_bootstrap(&f.seqs[owner], 5 + owner as u32)
                    .unwrap();
            }
        }
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.model.kv_cache.lock().num_free_blocks(), free);
        for owner in 0..2 {
            before[owner].unchanged(&f, owner);
        }
        for owner in order {
            tx.clear();
            execute(&mut f, owner, 0).unwrap();
            assert_eq!(
                tx.packets(),
                vec![vec![owner as u32], vec![5 + owner as u32]]
            );
            assert_eq!(f.seqs[owner].seq_len, 5);
            assert_eq!(f.seqs[owner].tokens.last(), Some(&(5 + owner as u32)));
            assert_eq!(flow::private(&f.seqs[owner]).seq_len, 3);
            assert_eq!(
                flow::slab(&f, owner, 5),
                vec![5 + owner as u8 + 0x24; ROW_BYTES]
            );
            assert!(f.gpu.trace().contains(&Event::Target(1, 4, DEFAULT)));
            assert!(f.gpu.trace().contains(&bonus(&f, owner)));
            f.gpu.clear();
            tx.clear();
            assert!(
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .bootstrap(&mut f.seqs[owner], 7)
                    .is_err()
            );
            assert!(tx.packets().is_empty());
            assert!(f.gpu.trace().is_empty());
        }
    }
}

#[test]
fn actual_scalar_head_packets_replay_on_worker_in_both_orders() {
    if isolated(
        "bootstrap_transport_tests::actual_scalar_head_packets_replay_on_worker_in_both_orders",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut head = prepared(0, order);
        let mut worker = prepared(1, order);
        let tx = wire::Wire::install(&mut head, 0);
        let rx = wire::Wire::install(&mut worker, 1);
        assert!(
            worker
                .model
                .glm_paired_execution()
                .unwrap()
                .validate_bootstrap(&worker.seqs[order[0]], 5 + order[0] as u32)
                .is_err()
        );
        assert!(
            worker
                .model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut worker.seqs[order[0]], 5 + order[0] as u32)
                .is_err()
        );
        assert!(rx.packets().is_empty());
        assert!(worker.gpu.trace().is_empty());
        for owner in order {
            tx.clear();
            execute(&mut head, owner, 0).unwrap();
            let packets = tx.packets();
            assert_eq!(packets, vec![vec![owner as u32], vec![5 + owner as u32]]);
            rx.queue(&packets);
            execute(&mut worker, owner, 1).unwrap();
            rx.done();
            assert_eq!(rx.packets(), packets);
            assert_eq!(head.seqs[owner].tokens, worker.seqs[owner].tokens);
            assert_eq!(head.seqs[owner].seq_len, worker.seqs[owner].seq_len);
            wire::same_private(&head, &worker, owner);
            assert!(worker.gpu.trace().contains(&Event::Target(1, 4, DEFAULT)));
        }
    }
}

#[test]
fn restored_profile_and_owner_negatives_refuse_before_headers() {
    if isolated(
        "bootstrap_transport_tests::restored_profile_and_owner_negatives_refuse_before_headers",
    ) {
        return;
    }
    for owner in 0..2 {
        for case in 0..10 {
            let mut f = prepared(0, [1, 0]);
            let tx = wire::Wire::install(&mut f, 0);
            let original_slot = f.seqs[owner].slot_idx;
            let generation = f.seqs[owner].mtp_capture_gen;
            let blocks = f.seqs[owner].block_table.clone();
            let mut token = 5 + owner as u32;
            match case {
                0 => token = 8,
                1 => f.model.ep_protocol_v2 = false,
                2 => f.model.levers.max_decode_seqs = 1,
                3 => f.gpu.capturing.store(true, Ordering::Relaxed),
                4 => f.seqs[owner].slot_idx = 1 - owner,
                5 => f.seqs[owner].mtp_capture_gen += 1,
                6 => f.seqs[owner].seq_len += 1,
                7 => f.seqs[owner].prompt_len -= 1,
                8 => f.seqs[owner].block_table.clear(),
                9 => f.seqs[owner].block_table.push(blocks[0]),
                _ => unreachable!(),
            }
            let slab = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            assert!(
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .bootstrap(&mut f.seqs[owner], token)
                    .is_err(),
                "case{case}"
            );
            assert!(tx.packets().is_empty(), "case{case}");
            assert!(f.gpu.trace().is_empty(), "case{case}");
            assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), slab);
            f.model.ep_protocol_v2 = true;
            f.model.levers.max_decode_seqs = 2;
            f.gpu.capturing.store(false, Ordering::Relaxed);
            f.seqs[owner].slot_idx = original_slot;
            f.seqs[owner].mtp_capture_gen = generation;
            f.seqs[owner].seq_len = 4;
            f.seqs[owner].prompt_len = 4;
            f.seqs[owner].block_table = blocks;
            execute(&mut f, owner, 0).unwrap();
        }
    }
}

#[test]
fn scalar_block_boundaries_use_one_row_and_later_proposal_keeps_k5_budget() {
    if isolated(
        "bootstrap_transport_tests::scalar_block_boundaries_use_one_row_and_later_proposal_keeps_k5_budget",
    ) {
        return;
    }
    for prompt_len in [15, 16, 17] {
        let mut f = Fixture::new(0);
        // Use a real larger arena for this actual complete prompt; no sizing guard bypass.
        let arena = spark_runtime::buffers::BufferArena::new(
            &f.model.config,
            32,
            2044,
            16,
            1,
            f.model.gpu.as_ref(),
        )
        .unwrap();
        let mut old = std::mem::replace(&mut f.model.buffers, arena);
        old.release(f.model.gpu.as_ref()).unwrap();
        let tokens: Vec<_> = (0..prompt_len).map(|i| (i % 8) as u32).collect();
        f.seqs[0].prompt_len = prompt_len;
        f.model.prefill(&tokens, &mut f.seqs[0], CALLER).unwrap();
        assert_eq!(f.seqs[0].seq_len, prompt_len);
        assert_eq!(flow::private(&f.seqs[0]).seq_len, prompt_len - 1);
        let tx = wire::Wire::install(&mut f, 0);
        let mut held = Vec::new();
        {
            let mut cache = f.model.kv_cache.lock();
            while cache.num_free_blocks() > 0 {
                held.push(cache.alloc_block().unwrap());
            }
        }
        let before = f.seqs[0].block_table.clone();
        f.gpu.clear();
        let validation = f
            .model
            .glm_paired_execution()
            .unwrap()
            .validate_bootstrap(&f.seqs[0], 5);
        assert_eq!(validation.is_ok(), prompt_len != 16);
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.seqs[0].block_table, before);
        if prompt_len == 16 {
            assert!(
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .bootstrap(&mut f.seqs[0], 5)
                    .is_err()
            );
            assert!(tx.packets().is_empty());
            assert!(f.gpu.trace().is_empty());
            f.model.kv_cache.lock().free_block(held.pop().unwrap());
        }
        execute(&mut f, 0, 0).unwrap();
        assert_eq!(f.seqs[0].seq_len, prompt_len + 1);
        assert_eq!(flow::private(&f.seqs[0]).seq_len, prompt_len - 1);
        tx.clear();
        f.gpu.clear();
        // P15 scalar fits its old block, but first K5 at position16 needs another.
        let proposal = f.model.glm_paired_execution().unwrap().validate_propose(
            &f.seqs[0],
            7,
            prompt_len + 1,
            4,
            None,
        );
        assert_eq!(proposal.is_ok(), prompt_len != 15);
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        f.model.kv_cache.lock().free_blocks(&held);
        f.model
            .glm_paired_execution()
            .unwrap()
            .validate_propose(&f.seqs[0], 7, prompt_len + 1, 4, None)
            .unwrap();
    }
}

#[test]
fn missing_retired_and_peer_produced_owners_refuse_without_wire() {
    if flow::isolated(
        "bootstrap_transport_tests::missing_retired_and_peer_produced_owners_refuse_without_wire",
    ) {
        return;
    }
    for owner in 0..2 {
        let mut unprimed = Fixture::new(0);
        let wire = wire::Wire::install(&mut unprimed, 0);
        assert!(
            unprimed
                .model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut unprimed.seqs[owner], 5)
                .is_err()
        );
        assert!(wire.packets().is_empty());
        assert!(unprimed.gpu.trace().is_empty());
        let mut f = prepared(0, [0, 1]);
        let tx = wire::Wire::install(&mut f, 0);
        let state = f.seqs[owner].proposer_state.take().unwrap();
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut f.seqs[owner], 5)
                .is_err()
        );
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        f.seqs[owner].proposer_state = Some(state);
        let peer = 1 - owner;
        execute(&mut f, peer, 0).unwrap();
        let drafts = f
            .model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut f.seqs[peer], 7, 5, 4, None)
            .unwrap();
        let mut issued = vec![7];
        issued.extend(drafts);
        f.model
            .glm_paired_execution()
            .unwrap()
            .verify(&mut f.seqs[peer], &issued)
            .unwrap();
        tx.clear();
        f.gpu.clear();
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut f.seqs[owner], 5)
                .is_err()
        );
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());

        let mut f = prepared(0, [0, 1]);
        let tx = wire::Wire::install(&mut f, 0);
        f.model.free_sequence(&mut f.seqs[owner]).unwrap();
        f.gpu.clear();
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut f.seqs[owner], 5)
                .is_err()
        );
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
    }
}
