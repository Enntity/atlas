// SPDX-License-Identifier: AGPL-3.0-only
//! Actual acknowledgements and selected dense producer preflight boundaries.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::traits::Model;
use std::sync::atomic::Ordering;

#[test]
fn wrong_missing_and_duplicate_acknowledgements_cannot_start_owned_repair() {
    if flow::isolated(
        "verdict_preflight_tests::wrong_missing_and_duplicate_acknowledgements_cannot_start_owned_repair",
    ) {
        return;
    }
    for rank in 0..2 {
        for accepted in 0..5 {
            let (mut f, mut history) = flow::prepare(rank, [0, 1]);
            flow::head_verdict(&mut f, 0, &history[0], accepted);
            let position = f.seqs[0].seq_len;
            f.gpu.clear();
            assert!(
                f.model
                    .run_mtp_propose_inner(1, position, 4, &mut f.seqs[0], None)
                    .is_err()
            );
            assert!(
                f.model
                    .trim_proposer_state(&mut f.seqs[0], (accepted + 1) % 5, 0)
                    .is_err()
            );
            assert!(
                f.model
                    .commit_accepted_prefix(&mut f.seqs[0], accepted + 1, 4)
                    .is_err()
            );
            assert!(
                f.model
                    .commit_accepted_prefix(&mut f.seqs[0], accepted + 2, 5)
                    .is_err()
            );
            assert!(f.gpu.trace().is_empty());
            f.model
                .trim_proposer_state(&mut f.seqs[0], accepted, 0)
                .unwrap();
            assert!(
                f.model
                    .run_mtp_propose_inner(1, position, 4, &mut f.seqs[0], None)
                    .is_err()
            );
            assert!(
                f.gpu.trace().is_empty(),
                "trim alone cannot start private repair"
            );
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], accepted + 1, 5)
                .unwrap();
            f.gpu.clear();
            assert!(
                f.model
                    .commit_accepted_prefix(&mut f.seqs[0], accepted + 1, 5)
                    .is_err()
            );
            assert!(
                f.model
                    .trim_proposer_state(&mut f.seqs[0], accepted, 0)
                    .is_err()
            );
            assert!(f.gpu.trace().is_empty());
            flow::continue_owner(&mut f, 0, &mut history[0], accepted);
        }
    }
}

#[test]
fn actual_active_prefix_cache_and_capture_refuse_before_selected_producers() {
    if flow::isolated(
        "verdict_preflight_tests::actual_active_prefix_cache_and_capture_refuse_before_selected_producers",
    ) {
        return;
    }
    for rank in 0..2 {
        let mut cold = Fixture::new(rank);
        cold.model.prefix_cache = Box::new(spark_runtime::radix_tree::RadixTree::new());
        cold.gpu.clear();
        assert!(
            cold.model
                .prefill(&[1, 2, 3, 4], &mut cold.seqs[0], CALLER)
                .is_err()
        );
        assert!(
            cold.gpu.trace().is_empty(),
            "cold sequence fields do not disable actual prefix lookup"
        );
        let (mut f, history) = flow::prepare(rank, [0, 1]);
        f.model.prefix_cache = Box::new(spark_runtime::radix_tree::RadixTree::new());
        f.gpu.clear();
        assert!(
            f.model
                .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
        f.model.prefix_cache = Box::new(spark_runtime::prefix_cache::NoPrefixCaching);
        f.gpu.capturing.store(true, Ordering::Relaxed);
        assert!(
            f.model
                .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
        f.gpu.capturing.store(false, Ordering::Relaxed);
        f.model
            .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
            .unwrap();
    }
}

#[test]
fn historical_block_is_checked_even_when_all_five_write_slots_are_valid() {
    if flow::isolated(
        "verdict_preflight_tests::historical_block_is_checked_even_when_all_five_write_slots_are_valid",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, mut history) = flow::prepare(rank, [0, 1]);
        for _ in 0..3 {
            flow::head_verdict(&mut f, 0, &history[0], 4);
            flow::acknowledge(&mut f, 0, 4, false);
            flow::continue_owner(&mut f, 0, &mut history[0], 4);
        }
        assert_eq!(history[0].base, 20);
        assert!(f.seqs[0].block_table.len() >= 2);
        let prior = f.seqs[0].block_table[0];
        f.seqs[0].block_table[0] = 256;
        assert!(
            (20..25).all(|position| f.seqs[0].physical_block_for(position / 16).unwrap() < 256)
        );
        f.gpu.clear();
        assert!(
            f.model
                .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
        f.seqs[0].block_table[0] = prior;
        f.model
            .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
            .unwrap();
    }
}

#[test]
fn selected_k5_refuses_unsupported_width_and_actual_hss_cache_before_work() {
    if flow::isolated(
        "verdict_preflight_tests::selected_k5_refuses_unsupported_width_and_actual_hss_cache_before_work",
    ) {
        return;
    }
    for width in [0, 1, 2, 3, 4, 6, 32] {
        let (mut f, _) = flow::prepare(1, [0, 1]);
        f.gpu.clear();
        assert!(
            f.model
                .decode_verify_graphed_kgamma(&vec![1; width], &mut f.seqs[0], CALLER)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
    }
    let (mut f, history) = flow::prepare(1, [0, 1]);
    let config = {
        let cache = f.model.kv_cache.lock();
        let c = cache.config();
        spark_runtime::kv_cache::KvCacheConfig {
            block_size: c.block_size,
            num_kv_heads: c.num_kv_heads,
            head_dim: c.head_dim,
            num_layers: c.num_layers,
            dtype: c.dtype,
            layer_dtypes: c.layer_dtypes.clone(),
            layer_dims: c.layer_dims.clone(),
            cache_blocks_per_seq: Some(1),
        }
    };
    let replacement =
        spark_runtime::kv_cache::PagedKvCache::new(config, 256, f.model.gpu.as_ref()).unwrap();
    *f.model.kv_cache.lock() = replacement;
    f.gpu.clear();
    assert!(
        f.model
            .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
            .is_err()
    );
    assert!(f.gpu.trace().is_empty());
}
