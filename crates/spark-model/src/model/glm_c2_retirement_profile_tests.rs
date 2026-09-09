// SPDX-License-Identifier: AGPL-3.0-only
//! Unsupported selected cleanup never borrows authority from a slot number.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::speculative::DraftProposer;
use crate::traits::Model;

#[test]
fn missing_private_state_or_unsupported_cleanup_profile_refuses_before_work() {
    if flow::isolated(
        "retirement_profile_tests::missing_private_state_or_unsupported_cleanup_profile_refuses_before_work",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for case in 0..5 {
                let mut f = Fixture::new(rank);
                let removed = if case == 0 {
                    f.seqs[owner].proposer_state.take()
                } else {
                    None
                };
                match case {
                    1 => f.seqs[owner].adapter_id = 1,
                    2 => f.seqs[owner].cached_prefix_tokens = 1,
                    3 => f.seqs[owner].prefix_ref_tokens = vec![1],
                    4 => f.seqs[owner].disk_last_offloaded_per_layer[0] = 1,
                    _ => {}
                }
                let free = f.head.paired_test_free_blocks();
                f.gpu.clear();
                assert!(
                    f.model.free_sequence(&mut f.seqs[owner]).is_err(),
                    "case {case}"
                );
                assert!(f.gpu.trace().is_empty());
                assert!(f.seqs[owner].ssm_slot_idx().is_none());
                assert!(!f.model.ssm_pool.slot_is_free(owner));
                assert_eq!(f.head.paired_test_free_blocks(), free);
                assert!(f.model.alloc_sequence().is_err());
                assert!(f.gpu.trace().is_empty());
                drop(removed);
            }
        }
    }
}

#[test]
fn no_private_lease_with_remaining_target_owner_refuses_before_cleanup() {
    if flow::isolated(
        "retirement_profile_tests::no_private_lease_with_remaining_target_owner_refuses_before_cleanup",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let mut f = Fixture::new(rank);
            // Real private free clears the lease, but cannot authorize Model
            // cleanup of the independently retained genuine target guard.
            f.head
                .free_state(
                    f.model.gpu.as_ref(),
                    f.seqs[owner].proposer_state.as_mut().unwrap().as_mut(),
                )
                .unwrap();
            assert!(f.seqs[owner].ssm_slot_idx().is_some());
            let free = f.head.paired_test_free_blocks();
            f.gpu.clear();
            assert!(f.model.free_sequence(&mut f.seqs[owner]).is_err());
            assert!(f.gpu.trace().is_empty());
            assert!(f.seqs[owner].ssm_slot_idx().is_none());
            assert!(!f.model.ssm_pool.slot_is_free(owner));
            assert_eq!(f.head.paired_test_free_blocks(), free);
            assert!(f.model.alloc_sequence().is_err());
            assert!(f.gpu.trace().is_empty());
        }
    }
}
