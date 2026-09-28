// SPDX-License-Identifier: AGPL-3.0-only
//! Actual K5 producer and verdict ownership; no manually minted producer rows.
use super::{fixture::*, isolated};
use crate::traits::Model;

fn prepared(rank: usize) -> (Fixture, [[u32; 5]; 2]) {
    let mut f = Fixture::new(rank);
    let mut inputs = [[0; 5]; 2];
    for (owner, prompt) in [[1, 2, 3, 4], [4, 3, 2, 1]].iter().enumerate() {
        f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
        f.model
            .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
            .unwrap();
        let seed = 7 - owner as u32;
        let drafts = f
            .model
            .run_mtp_propose_inner(seed, 5, 4, &mut f.seqs[owner], None)
            .unwrap();
        inputs[owner][0] = seed;
        inputs[owner][1..].copy_from_slice(&drafts);
    }
    (f, inputs)
}

#[test]
fn actual_k5_refuses_unissued_tokens_before_target_work() {
    if isolated("verdict_producer_tests::actual_k5_refuses_unissued_tokens_before_target_work") {
        return;
    }
    for rank in 0..2 {
        let (mut f, inputs) = prepared(rank);
        let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        f.gpu.clear();
        let result = f
            .model
            .decode_verify_graphed_kgamma(&inputs[1], &mut f.seqs[0], CALLER);
        assert!(
            result.is_err(),
            "actual wrapper accepted peer-issued K5 inputs"
        );
        assert!(
            f.gpu.trace().is_empty(),
            "refusal must precede any backend operation"
        );
        assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
        assert_eq!(f.seqs[0].seq_len, 5);
    }
}

#[test]
fn actual_k5_output_lease_blocks_peer_verification_until_record() {
    if isolated(
        "verdict_producer_tests::actual_k5_output_lease_blocks_peer_verification_until_record",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, inputs) = prepared(rank);
        let prediction = f
            .model
            .decode_verify_graphed_kgamma(&inputs[0], &mut f.seqs[0], CALLER)
            .unwrap();
        assert_eq!(prediction.len(), 5);
        let normalized = f
            .gpu
            .read_span(f.model.buffers.norm_output(), 5 * ROW_BYTES);
        f.gpu.clear();
        let result = f
            .model
            .decode_verify_graphed_kgamma(&inputs[1], &mut f.seqs[1], CALLER);
        assert!(result.is_err(), "peer overwrote a pending actual K5 output");
        assert!(f.gpu.trace().is_empty());
        assert_eq!(
            f.gpu
                .read_span(f.model.buffers.norm_output(), 5 * ROW_BYTES),
            normalized
        );
    }
}

#[test]
fn actual_k5_verdict_detaches_only_accepted_rows_and_bonus() {
    if isolated("verdict_producer_tests::actual_k5_verdict_detaches_only_accepted_rows_and_bonus") {
        return;
    }
    for rank in 0..2 {
        for accepted in 0..=4 {
            let (mut f, inputs) = prepared(rank);
            let slab = f.gpu.slab();
            let before = f.gpu.read_span(slab, SLAB_BYTES);
            let predictions = f
                .model
                .decode_verify_graphed_kgamma(&inputs[0], &mut f.seqs[0], CALLER)
                .unwrap();
            assert_eq!(predictions.len(), 5);
            let rows = f
                .gpu
                .read_span(f.model.buffers.norm_output(), 5 * ROW_BYTES);
            f.seqs[0].seq_len = 6 + accepted;
            f.seqs[0].tokens.truncate(6 + accepted);
            f.gpu.clear();
            f.model
                .record_glm_mtp_verified(&mut f.seqs[0], 5, &inputs[0], accepted)
                .unwrap();
            let after = f.gpu.read_span(slab, SLAB_BYTES);
            assert_eq!(&after[..ROW_BYTES], &before[..ROW_BYTES]);
            assert_eq!(
                &after[ROW_BYTES..(1 + accepted) * ROW_BYTES],
                &rows[..accepted * ROW_BYTES]
            );
            assert_eq!(
                &after[(1 + accepted) * ROW_BYTES..5 * ROW_BYTES],
                &before[(1 + accepted) * ROW_BYTES..5 * ROW_BYTES]
            );
            assert_eq!(
                &after[5 * ROW_BYTES..6 * ROW_BYTES],
                &rows[accepted * ROW_BYTES..(accepted + 1) * ROW_BYTES]
            );
            assert_eq!(&after[6 * ROW_BYTES..], &before[6 * ROW_BYTES..]);
            let copies: Vec<_> = f
                .gpu
                .trace()
                .into_iter()
                .filter(|e| matches!(e, Event::Copy(..)))
                .collect();
            assert_eq!(copies.len(), usize::from(accepted > 0) + 1);
            assert!(
                f.gpu
                    .trace()
                    .iter()
                    .all(|e| matches!(e, Event::Copy(..) | Event::Sync(DEFAULT)))
            );
            f.gpu.clear();
            assert!(
                f.model
                    .record_glm_mtp_verified(&mut f.seqs[0], 5, &inputs[0], accepted)
                    .is_err()
            );
            assert!(
                f.gpu.trace().is_empty(),
                "duplicate record must not copy again"
            );
        }
    }
}

#[test]
fn historical_prefix_mutation_refuses_begin_and_record_without_consuming_receipt() {
    if isolated(
        "verdict_producer_tests::historical_prefix_mutation_refuses_begin_and_record_without_consuming_receipt",
    ) {
        return;
    }
    for rank in 0..2 {
        for after_verify in [false, true] {
            let (mut f, inputs) = prepared(rank);
            if after_verify {
                f.model
                    .decode_verify_graphed_kgamma(&inputs[0], &mut f.seqs[0], CALLER)
                    .unwrap();
                f.seqs[0].seq_len = 7;
                f.seqs[0].tokens.truncate(7);
            }
            let original = f.seqs[0].tokens[0];
            f.seqs[0].tokens[0] = (original + 1) % 8;
            f.gpu.clear();
            let result = if after_verify {
                f.model
                    .record_glm_mtp_verified(&mut f.seqs[0], 5, &inputs[0], 1)
            } else {
                f.model
                    .decode_verify_graphed_kgamma(&inputs[0], &mut f.seqs[0], CALLER)
                    .map(|_| ())
            };
            assert!(
                result.is_err(),
                "changed historical prefix still authorized actual producer/record"
            );
            assert!(f.gpu.trace().is_empty());
            f.seqs[0].tokens[0] = original;
            if after_verify {
                f.model
                    .record_glm_mtp_verified(&mut f.seqs[0], 5, &inputs[0], 1)
                    .unwrap();
            } else {
                f.model
                    .decode_verify_graphed_kgamma(&inputs[0], &mut f.seqs[0], CALLER)
                    .unwrap();
            }
        }
    }
}

#[test]
fn actual_k5_rejects_missing_or_invalid_historical_attention_map_before_work() {
    if isolated(
        "verdict_producer_tests::actual_k5_rejects_missing_or_invalid_historical_attention_map_before_work",
    ) {
        return;
    }
    for rank in 0..2 {
        for invalid in 0..3 {
            let (mut f, inputs) = prepared(rank);
            match invalid {
                0 => f.seqs[0].block_table[0] = 256,
                1 => f.seqs[0].block_table.clear(),
                _ => f.model.max_blocks_per_seq = 0,
            }
            f.gpu.clear();
            let result = f
                .model
                .decode_verify_graphed_kgamma(&inputs[0], &mut f.seqs[0], CALLER);
            assert!(result.is_err(), "invalid target attention mapping accepted");
            assert!(
                f.gpu.trace().is_empty(),
                "existing invalid prefix must fail before allocation or target work"
            );
        }
    }
}
