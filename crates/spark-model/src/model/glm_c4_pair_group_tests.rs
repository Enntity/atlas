// SPDX-License-Identifier: AGPL-3.0-only
//! Actual four-owner Model/E6/E1 ownership; byte sentinels, not native arithmetic.
use super::{fixture::*, transport_test_fixture::Wire, verdict_continuation_tests as flow};
use crate::layer::glm_pair_verify::GlmPairFfn;
use crate::speculative::DraftProposer;
use crate::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::Ordering;

struct Four {
    f: Fixture,
    states: [SequenceState; 4],
}

impl Four {
    fn prepared(rank: usize) -> (Self, [flow::History; 4]) {
        let mut f = Fixture::new_pair_compute_with_owner_capacity(rank, 4);
        let extra2 = f
            .model
            .alloc_sequence()
            .expect("actual target/private owner2");
        let extra3 = f
            .model
            .alloc_sequence()
            .expect("actual target/private owner3");
        assert_eq!((extra2.slot_idx, extra3.slot_idx), (2, 3));
        // Move the genuine first two states, leaving inert placeholders only
        // in the old two-slot fixture container. They never enter a Model call.
        let [s0, s1] =
            std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only));
        let mut states = [s0, s1, extra2, extra3];
        f.model
            .initialize_glm_pair_verification(GlmPairFfn::TwoK5)
            .unwrap();
        f.gpu.deterministic_logits.store(true, Ordering::Relaxed);
        f.gpu
            .write_span(f.gpu.slab_for_owners(4), &vec![0xa5; 4 * 6 * ROW_BYTES]);
        let prompts = [
            vec![1, 2, 3, 4],
            vec![6, 5, 4, 3, 2, 1],
            vec![2, 4, 6, 1, 3],
            vec![7, 6, 5, 4, 3, 2, 1],
        ];
        let histories = std::array::from_fn(|owner| {
            let seq = &mut states[owner];
            seq.prompt_len = prompts[owner].len();
            f.model.prefill(&prompts[owner], seq, CALLER).unwrap();
            let token = (5 + owner as u32) % 8;
            f.model.decode(token, seq, CALLER).unwrap();
            let base = seq.seq_len;
            let seed = 7 - owner as u32;
            let drafts = f
                .model
                .run_mtp_propose_inner(seed, base, 4, seq, None)
                .unwrap();
            assert_eq!(drafts.len(), 4);
            let mut issued = vec![seed];
            issued.extend(drafts);
            let canonical = prompts[owner]
                .iter()
                .enumerate()
                .map(|(row, token)| {
                    vec![(*token as u8).wrapping_add(0x20).wrapping_add(row as u8); 1024]
                })
                .collect();
            let bonus = vec![
                (token as u8)
                    .wrapping_add(0x20)
                    .wrapping_add(prompts[owner].len() as u8);
                ROW_BYTES
            ];
            flow::History {
                base,
                issued,
                canonical,
                bonus,
            }
        });
        let result = Self { f, states };
        for (owner, h) in histories.iter().enumerate() {
            result.assert_private(owner, h);
            for peer in 0..owner {
                assert!(
                    result.states[owner]
                        .block_table
                        .iter()
                        .all(|block| !result.states[peer].block_table.contains(block))
                );
            }
        }
        (result, histories)
    }

    fn rows(&self, owner: usize, rows: usize) -> Vec<Vec<u8>> {
        self.f
            .head
            .paired_test_kv_rows(
                self.states[owner].proposer_state.as_deref().unwrap(),
                self.f.model.gpu.as_ref(),
                rows,
            )
            .unwrap()
            .into_iter()
            .map(|(k, v)| {
                let bytes = self.f.gpu.read_span(k, 1024);
                assert_eq!(self.f.gpu.read_span(v, 1024), bytes);
                bytes
            })
            .collect()
    }

    fn slab_row(&self, owner: usize, row: usize) -> Vec<u8> {
        self.f.gpu.read_span(
            self.f
                .gpu
                .slab_for_owners(4)
                .offset((owner * 6 + row) * ROW_BYTES),
            ROW_BYTES,
        )
    }

    fn assert_private(&self, owner: usize, h: &flow::History) {
        assert_eq!(flow::private(&self.states[owner]).seq_len, h.base + 3);
        let mut expected = h.canonical.clone();
        expected.extend((0..4).map(|_| h.bonus[..1024].to_vec()));
        assert_eq!(self.rows(owner, h.base + 3), expected);
        assert_eq!(self.slab_row(owner, 5), h.bonus);
    }

    fn replay(&mut self, wire: &Wire, packets: &[Vec<u32>]) {
        wire.queue(packets);
        assert!(self.receive().unwrap());
        wire.done();
        assert_eq!(wire.packets(), packets);
    }

    fn receive(&mut self) -> anyhow::Result<bool> {
        let mut slots = std::mem::replace(
            &mut self.states,
            std::array::from_fn(SequenceState::host_only),
        )
        .map(Some);
        let result = self.f.model.ep_worker_step(&mut slots);
        self.states = slots.map(Option::unwrap);
        result
    }
}

#[derive(Debug, PartialEq, Eq)]
struct PeerSnapshot {
    slot: usize,
    target_len: usize,
    target_tokens: Vec<u32>,
    target_blocks: Vec<u32>,
    capture_generation: u64,
    private_len: usize,
    private_blocks: Vec<u32>,
    spans: Vec<(DevicePtr, Vec<u8>)>,
}

fn peer_snapshot(f: &Four, owner: usize) -> PeerSnapshot {
    let seq = &f.states[owner];
    let private = flow::private(seq);
    let pointers =
        f.f.head
            .paired_test_kv_rows(
                seq.proposer_state.as_deref().unwrap(),
                f.f.model.gpu.as_ref(),
                private.block_table.len() * 16,
            )
            .unwrap();
    let mut spans = Vec::new();
    // Full reserved private blocks, not only the current sequence prefix.
    for (k, v) in pointers.into_iter().step_by(16) {
        for ptr in [k, v] {
            spans.push((ptr, f.f.gpu.read_live_span(ptr, 16 * 1024).unwrap()));
        }
    }
    let slab = f.f.gpu.slab_for_owners(4).offset(owner * 6 * ROW_BYTES);
    spans.push((slab, f.f.gpu.read_live_span(slab, 6 * ROW_BYTES).unwrap()));
    let cache = f.f.model.kv_cache.lock();
    for block in &seq.block_table {
        for (ptr, bytes) in [
            (
                cache.k_cache_ptr(0, *block),
                cache.k_block_stride_bytes_for_layer(0),
            ),
            (
                cache.v_cache_ptr(0, *block),
                cache.v_block_stride_bytes_for_layer(0),
            ),
        ] {
            spans.push((ptr, f.f.gpu.read_live_span(ptr, bytes).unwrap()));
        }
    }
    PeerSnapshot {
        slot: seq.slot_idx,
        target_len: seq.seq_len,
        target_tokens: seq.tokens.clone(),
        target_blocks: seq.block_table.clone(),
        capture_generation: seq.mtp_capture_gen,
        private_len: private.seq_len,
        private_blocks: private.block_table.clone(),
        spans,
    }
}

#[test]
fn actual_four_owner_groups_preserve_peers_through_e6_verdict_and_e1() {
    if flow::isolated(
        "pair_group_tests::actual_four_owner_groups_preserve_peers_through_e6_verdict_and_e1",
    ) {
        return;
    }
    let (mut head, mut histories) = Four::prepared(0);
    let (mut worker, _) = Four::prepared(1);
    let tx = Wire::install(&mut head.f, 0);
    let rx = Wire::install(&mut worker.f, 1);
    for (base, accepted) in [(2, [0, 4]), (0, [4, 0]), (2, [4, 0])] {
        let peers = [2 - base, 3 - base];
        let saved_head = peers.map(|owner| peer_snapshot(&head, owner));
        let saved_worker = peers.map(|owner| peer_snapshot(&worker, owner));
        let issued: [[u32; 5]; 2] = std::array::from_fn(|ordinal| {
            histories[base + ordinal]
                .issued
                .as_slice()
                .try_into()
                .unwrap()
        });
        let normalized = [
            flow::normalized(&histories[base]),
            flow::normalized(&histories[base + 1]),
        ];
        tx.clear();
        head.f.gpu.clear();
        let [s0, s1] =
            <&mut [SequenceState; 2]>::try_from(&mut head.states[base..base + 2]).unwrap();
        let prediction = head
            .f
            .model
            .glm_paired_execution()
            .unwrap()
            .verify_pair([s0, s1], &issued)
            .unwrap();
        for ordinal in 0..2 {
            assert_eq!(
                prediction[ordinal],
                std::array::from_fn(|row| u32::from(normalized[ordinal][row][0] % 8))
            );
            for row in 0..5 {
                assert_eq!(
                    head.f.gpu.read_span(
                        head.f
                            .model
                            .buffers
                            .norm_output()
                            .offset((ordinal * 5 + row) * ROW_BYTES),
                        ROW_BYTES
                    ),
                    normalized[ordinal][row]
                );
            }
        }
        let [s0, s1] =
            <&mut [SequenceState; 2]>::try_from(&mut head.states[base..base + 2]).unwrap();
        head.f
            .model
            .glm_paired_execution()
            .unwrap()
            .finish_verify_pair([s0, s1], &issued, accepted)
            .unwrap();
        let packets = tx.packets();
        assert_eq!(packets.len(), 4);
        assert_eq!(packets[0], [base as u32]);
        assert_eq!(packets[1], [0xffff_ffe6]);
        assert_eq!(&packets[2][..4], &[1, 2, 10, 1]);
        for ordinal in 0..2 {
            let owner = base + ordinal;
            assert_eq!(packets[2][4 + ordinal * 11], owner as u32);
            assert_eq!(packets[2][9 + ordinal * 11], histories[owner].base as u32);
            assert_eq!(
                &packets[2][10 + ordinal * 11..15 + ordinal * 11],
                &issued[ordinal]
            );
        }
        assert_eq!(packets[3], [1, 2, accepted[0] as u32, accepted[1] as u32]);
        worker.replay(&rx, &packets);
        for ordinal in 0..2 {
            let owner = base + ordinal;
            for f in [&head, &worker] {
                assert_eq!(
                    f.states[owner].seq_len,
                    histories[owner].base + accepted[ordinal] + 1
                );
                assert_eq!(
                    &f.states[owner].tokens[histories[owner].base..],
                    &issued[ordinal][..accepted[ordinal] + 1],
                    "physical owner's exact committed target prefix"
                );
                assert_eq!(f.slab_row(owner, 5), normalized[ordinal][accepted[ordinal]]);
                for row in 0..accepted[ordinal] {
                    assert_eq!(f.slab_row(owner, row + 1), normalized[ordinal][row]);
                }
            }
            assert_eq!(head.states[owner].tokens, worker.states[owner].tokens);
        }
        for ordinal in [1, 0] {
            let owner = base + ordinal;
            let h = &mut histories[owner];
            let seed = prediction[ordinal][accepted[ordinal]];
            let next_base = h.base + accepted[ordinal] + 1;
            tx.clear();
            head.f.gpu.clear();
            worker.f.gpu.clear();
            let drafts = head
                .f
                .model
                .glm_paired_execution()
                .unwrap()
                .propose(&mut head.states[owner], seed, next_base, 4, None)
                .unwrap();
            assert_eq!(drafts.len(), 4);
            let mut eh: Vec<_> = h.issued[1..1 + accepted[ordinal]]
                .iter()
                .zip(&normalized[ordinal])
                .map(|(token, hidden)| (*token as u8, hidden[0]))
                .collect();
            let bonus = normalized[ordinal][accepted[ordinal]].clone();
            eh.extend(
                std::iter::once(seed)
                    .chain(drafts.iter().take(3).copied())
                    .map(|token| (token as u8, bonus[0])),
            );
            let packets = tx.packets();
            assert_eq!(packets.len(), 3);
            assert_eq!(packets[0], [owner as u32]);
            assert_eq!(packets[1], [0xffff_ffe1]);
            assert_eq!(&packets[2][5..], &[next_base as u32, 4, seed]);
            worker.replay(&rx, &packets);
            assert_eq!(head.f.gpu.eh_pairs(), eh);
            assert_eq!(worker.f.gpu.eh_pairs(), eh);
            h.canonical.push(h.bonus[..1024].to_vec());
            h.canonical.extend(
                normalized[ordinal]
                    .iter()
                    .take(accepted[ordinal])
                    .map(|row| row[..1024].to_vec()),
            );
            h.base = next_base;
            h.bonus = bonus;
            h.issued = std::iter::once(seed).chain(drafts).collect();
            head.assert_private(owner, h);
            worker.assert_private(owner, h);
        }
        for (index, owner) in peers.into_iter().enumerate() {
            assert_eq!(peer_snapshot(&head, owner), saved_head[index]);
            assert_eq!(peer_snapshot(&worker, owner), saved_worker[index]);
        }
    }
    // Final negative only: the actual mismatch poisons the entire worker pool.
    // Resolve raw backing before refusal; never query a terminal lease afterward.
    let saved = std::array::from_fn::<_, 4, _>(|owner| peer_snapshot(&worker, owner));
    let free = worker.f.model.kv_cache.lock().num_free_blocks();
    let private_free = worker.f.head.paired_test_free_blocks();
    tx.clear();
    let issued: [[u32; 5]; 2] =
        std::array::from_fn(|ordinal| histories[2 + ordinal].issued.as_slice().try_into().unwrap());
    let [s0, s1] = <&mut [SequenceState; 2]>::try_from(&mut head.states[2..4]).unwrap();
    head.f
        .model
        .glm_paired_execution()
        .unwrap()
        .verify_pair([s0, s1], &issued)
        .unwrap();
    let mut packets = tx.packets();
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[0], [2]);
    packets[0] = vec![0]; // Valid other group, wrong for these actual owner records.
    rx.queue(&packets);
    worker.f.gpu.clear();
    let error = worker.receive().unwrap_err();
    assert!(format!("{error:#}").contains("paired"), "{error:#}");
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
        "wrong physical group must refuse before worker target writers"
    );
    assert_eq!(worker.f.model.kv_cache.lock().num_free_blocks(), free);
    assert_eq!(worker.f.head.paired_test_free_blocks(), private_free);
    for (owner, before) in saved.iter().enumerate() {
        let seq = &worker.states[owner];
        assert_eq!(seq.seq_len, before.target_len);
        assert_eq!(seq.tokens, before.target_tokens);
        assert_eq!(seq.block_table, before.target_blocks);
        assert_eq!(flow::private(seq).seq_len, before.private_len);
        assert_eq!(flow::private(seq).block_table, before.private_blocks);
        for (ptr, bytes) in &before.spans {
            assert_eq!(
                &worker.f.gpu.read_live_span(*ptr, bytes.len()).unwrap(),
                bytes
            );
        }
    }
    worker.f.gpu.clear();
    for seq in &mut worker.states {
        let position = seq.seq_len;
        assert!(
            worker
                .f
                .model
                .run_mtp_propose_inner(7, position, 4, seq, None)
                .is_err()
        );
        assert!(worker.f.model.decode(1, seq, CALLER).is_err());
    }
    assert!(
        worker
            .f
            .head
            .alloc_state(worker.f.model.gpu.as_ref())
            .is_err()
    );
    assert!(
        worker.f.gpu.trace().is_empty(),
        "all four failed-session owners refuse without work"
    );
}
