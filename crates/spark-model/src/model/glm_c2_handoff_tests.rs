// SPDX-License-Identifier: AGPL-3.0-only
//! Actual model producers and actual paired head; numerical kernels are recorded.
#[path = "glm_c2_aligned_churn_tests.rs"]
mod aligned_churn_tests;
#[path = "glm_c2_allocation_ownership_tests.rs"]
mod allocation_ownership_tests;
#[path = "glm_c2_bootstrap_transport_fault_tests.rs"]
mod bootstrap_transport_fault_tests;
#[path = "glm_c2_bootstrap_transport_tests.rs"]
mod bootstrap_transport_tests;
#[path = "glm_c2_handoff_cleanup_tests.rs"]
mod cleanup_tests;
#[path = "glm_c2_cold_prefill_transport_tests.rs"]
mod cold_prefill_transport_tests;
#[path = "glm_c2_handoff_decode_fault_tests.rs"]
mod decode_fault_tests;
#[path = "glm_c2_eager_bootstrap_error_tests.rs"]
mod eager_bootstrap_error_tests;
#[path = "glm_c2_f1_ownership_tests.rs"]
mod f1_ownership_tests;
use super::glm_c2_test_support::fixture;
#[path = "glm_c2_legacy_boundary_tests.rs"]
mod legacy_boundary_tests;
#[path = "glm_c4_pair_group_tests.rs"]
mod pair_group_tests;
#[path = "glm_c2_pair_producer_tests.rs"]
mod pair_producer_tests;
#[path = "glm_c2_pair_verify_tests.rs"]
mod pair_verify_tests;
#[path = "glm_c2_predispatch_tests.rs"]
mod predispatch_tests;
#[path = "glm_c2_handoff_preflight_tests.rs"]
mod preflight_tests;
#[path = "glm_c2_quiescence_tests.rs"]
mod quiescence_tests;
#[path = "glm_c2_retirement_fault_tests.rs"]
mod retirement_fault_tests;
#[path = "glm_c2_retirement_guard_tests.rs"]
mod retirement_guard_tests;
#[path = "glm_c2_retirement_profile_tests.rs"]
mod retirement_profile_tests;
#[path = "glm_c2_transport_boundary_tests.rs"]
mod transport_boundary_tests;
#[path = "glm_c2_transport_continuation_tests.rs"]
mod transport_continuation_tests;
#[path = "glm_c2_transport_fault_tests.rs"]
mod transport_fault_tests;
#[path = "glm_c2_transport_legacy_tests.rs"]
mod transport_legacy_tests;
#[path = "glm_c2_transport_reuse_tests.rs"]
mod transport_reuse_tests;
#[path = "glm_c2_transport_test_fixture.rs"]
mod transport_test_fixture;
#[path = "glm_c2_transport_tests.rs"]
mod transport_tests;
#[path = "glm_c2_verdict_continuation_tests.rs"]
mod verdict_continuation_tests;
#[path = "glm_c2_verdict_escape_tests.rs"]
mod verdict_escape_tests;
#[path = "glm_c2_verdict_fault_tests.rs"]
mod verdict_fault_tests;
#[path = "glm_c2_verdict_lifecycle_tests.rs"]
mod verdict_lifecycle_tests;
#[path = "glm_c2_verdict_preflight_tests.rs"]
mod verdict_preflight_tests;
#[path = "glm_c2_verdict_producer_tests.rs"]
mod verdict_producer_tests;
#[path = "glm_c2_verdict_repair_fault_tests.rs"]
mod verdict_repair_fault_tests;
#[path = "glm_c2_verdict_ssm_tests.rs"]
mod verdict_ssm_tests;
#[path = "glm_c2_verdict_worker_tests.rs"]
mod verdict_worker_tests;
#[path = "glm_c2_handoff_worker_tests.rs"]
mod worker_tests;
#[path = "glm_c2_handoff_writer_fault_tests.rs"]
mod writer_fault_tests;
use crate::speculative::DraftProposer;
use crate::traits::Model;
use fixture::*;

fn isolated(name: &str) -> bool {
    if std::env::var("ATLAS_C2_HANDOFF_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        &format!("model::glm_c2_handoff_tests::{name}"),
        "--nocapture",
    ])
    .env("ATLAS_C2_HANDOFF_TEST_CHILD", "1")
    .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
    .env("ATLAS_GLM_MTP_REPAIR", "0")
    .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
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
        "actual handoff hook child failed: {name}"
    );
    true
}

#[test]
fn actual_eager_model_hook_detaches_each_prompt_tail_before_peer_capture() {
    if isolated("actual_eager_model_hook_detaches_each_prompt_tail_before_peer_capture") {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        let slab = f.gpu.slab();
        f.gpu.write_span(slab, &vec![0xa5; SLAB_BYTES]);
        let mut expected = Vec::new();
        for (owner, prompt) in [[1, 2, 3, 4], [4, 3, 2, 1]].iter().enumerate() {
            f.seqs[owner].prompt_len = 0; // Actual alloc_sequence/worker cold metadata.
            f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
            let state = f.seqs[owner]
                .proposer_state
                .as_ref()
                .unwrap()
                .as_any()
                .downcast_ref::<crate::layers::glm5_mtp::Glm5MtpProposerState>()
                .unwrap();
            assert_eq!(state.seq_len, 3, "actual P-1 KV writer must have completed");
            let tail = f
                .gpu
                .read_span(f.model.mtp_prefill_hidden.offset(3 * ROW_BYTES), ROW_BYTES);
            assert_ne!(tail, vec![0xa5; ROW_BYTES]);
            assert!(
                f.gpu
                    .read_span(slab.offset(owner * 6 * ROW_BYTES), ROW_BYTES)
                    == tail,
                "actual eager wrapper must detach owner{owner} tail, rank{rank}"
            );
            expected.push(tail);
        }
        for (owner, bytes) in expected.iter().enumerate() {
            assert_eq!(
                f.gpu
                    .read_span(slab.offset(owner * 6 * ROW_BYTES), ROW_BYTES),
                *bytes
            );
        }
        assert_ne!(expected[0], expected[1]);
    }
}

#[test]
fn actual_decode_publishes_bootstrap_hidden_before_another_producer_runs() {
    if isolated("actual_decode_publishes_bootstrap_hidden_before_another_producer_runs") {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        let slab = f.gpu.slab();
        f.gpu.write_span(slab, &vec![0xa5; SLAB_BYTES]);
        for owner in 0..2 {
            f.model
                .prefill(&[1 + owner as u32, 2, 3, 4], &mut f.seqs[owner], CALLER)
                .unwrap();
        }
        let mut expected = Vec::new();
        for owner in 0..2 {
            f.model
                .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
                .unwrap();
            assert_eq!(f.seqs[owner].seq_len, 5);
            let row = f.gpu.read_span(f.model.buffers.norm_output(), ROW_BYTES);
            assert!(
                f.gpu
                    .read_span(slab.offset((owner * 6 + 5) * ROW_BYTES), ROW_BYTES)
                    == row,
                "actual outer decode must publish owner{owner} H[P], rank{rank}"
            );
            expected.push(row);
        }
        assert_ne!(expected[0], expected[1]);
        for (owner, row) in expected.iter().enumerate() {
            assert!(
                f.gpu
                    .read_span(slab.offset((owner * 6 + 5) * ROW_BYTES), ROW_BYTES)
                    == *row
            );
        }
    }
}

#[test]
fn actual_propose_bootstraps_missing_pair_and_reads_owned_bonus() {
    if isolated("actual_propose_bootstraps_missing_pair_and_reads_owned_bonus") {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        let slab = f.gpu.slab();
        for (owner, prompt) in [[1, 2, 3, 4], [4, 3, 2, 1]].iter().enumerate() {
            f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
            f.model
                .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
                .unwrap();
        }
        let tail: Vec<_> = (0..2)
            .map(|owner| f.gpu.read_span(slab.offset(owner * 6 * ROW_BYTES), 1024))
            .collect();
        assert_ne!(
            tail[0], tail[1],
            "different owners must have distinguishable H[P-1]"
        );
        let rows: Vec<_> = f
            .seqs
            .iter()
            .map(|seq| {
                f.head
                    .paired_test_kv_rows(
                        seq.proposer_state.as_ref().unwrap().as_ref(),
                        f.model.gpu.as_ref(),
                        4,
                    )
                    .unwrap()
            })
            .collect();
        let primed: Vec<Vec<_>> = rows
            .iter()
            .map(|owner| {
                owner[..3]
                    .iter()
                    .map(|&(k, v)| (f.gpu.read_span(k, 1024), f.gpu.read_span(v, 1024)))
                    .collect()
            })
            .collect();
        f.gpu
            .write_span(f.model.mtp_hidden_save, &vec![0xee; ROW_BYTES]);
        f.gpu
            .write_span(f.model.mtp_prefill_hidden, &vec![0xdd; 4 * ROW_BYTES]);
        for owner in [1, 0] {
            f.gpu.clear();
            let drafts = f
                .model
                .run_mtp_propose_inner(7, 5, 4, &mut f.seqs[owner], None)
                .unwrap();
            assert_eq!(drafts.len(), 4);
            let state = f.seqs[owner]
                .proposer_state
                .as_ref()
                .unwrap()
                .as_any()
                .downcast_ref::<crate::layers::glm5_mtp::Glm5MtpProposerState>()
                .unwrap();
            assert_eq!(
                state.seq_len, 8,
                "actual proposal needs P cached pairs before four drafts"
            );
            let meta = f.model.buffers.scratch().offset(49152);
            assert_eq!(
                f.gpu
                    .trace()
                    .iter()
                    .filter(|event| matches!(event,Event::Upload(ptr,768,DEFAULT) if *ptr == meta))
                    .count(),
                4,
                "actual body metadata must carry all128 reserved blocks for each draft"
            );
            let map = f.gpu.read_span(meta.offset(256), 512);
            let blocks: Vec<_> = map
                .chunks_exact(4)
                .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(blocks, state.block_table);
            let expected = slab.offset((owner * 6 + 5) * ROW_BYTES);
            assert!(
                f.gpu.trace().iter().any(|event| matches!(event,
                Event::Kernel(name, ptrs, DEFAULT) if name == "rms_norm_vanilla"
                    && ptrs.first() == Some(&expected))),
                "owned bonus must feed actual body"
            );
            assert!(
                f.gpu.trace().iter().any(|event| matches!(event,
                Event::Kv(slots, DEFAULT) if slots.len() == 1)),
                "actual bootstrap KV pair missing"
            );
            let (k, v) = rows[owner][3];
            assert_eq!(
                f.gpu.read_span(k, 1024),
                tail[owner],
                "missing pair K must use owned H[P-1]"
            );
            assert_eq!(
                f.gpu.read_span(v, 1024),
                tail[owner],
                "missing pair V must use owned H[P-1]"
            );
            for peer in 0..2 {
                for row in 0..3 {
                    let (k, v) = rows[peer][row];
                    assert_eq!(
                        (f.gpu.read_span(k, 1024), f.gpu.read_span(v, 1024)),
                        primed[peer][row]
                    );
                }
            }
        }
    }
}

#[test]
fn retired_slots_can_prime_in_reverse_target_slot_order() {
    if isolated("retired_slots_can_prime_in_reverse_target_slot_order") {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        for owner in 0..2 {
            f.model
                .prefill(&[1, 2, 3, 4], &mut f.seqs[owner], CALLER)
                .unwrap();
        }
        for owner in [1, 0] {
            f.model.free_sequence(&mut f.seqs[owner]).unwrap();
        }
        for owner in 0..2 {
            f.seqs[owner] = f.model.alloc_sequence().unwrap();
            assert_eq!(f.seqs[owner].slot_idx, owner);
        }
        for owner in [1, 0] {
            f.model
                .prefill(&[4, 3, 2, 1], &mut f.seqs[owner], CALLER)
                .unwrap();
        }
    }
}

#[test]
fn failed_retirement_cannot_restore_a_corrupted_mutable_block_view() {
    if isolated("failed_retirement_cannot_restore_a_corrupted_mutable_block_view") {
        return;
    }
    use crate::layers::glm5_mtp::Glm5MtpProposerState;
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        for owner in 0..2 {
            f.model
                .prefill(&[1, 2, 3, 4], &mut f.seqs[owner], CALLER)
                .unwrap();
        }
        let old = f.seqs[0]
            .proposer_state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .unwrap()
            .block_table
            .clone();
        let peer = f.seqs[1]
            .proposer_state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .unwrap()
            .block_table
            .clone();
        f.model.decode(6, &mut f.seqs[1], CALLER).unwrap();
        let bytes = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        f.seqs[0]
            .proposer_state
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .unwrap()
            .block_table = peer;
        f.gpu.clear();
        assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
        assert!(f.gpu.trace().is_empty());
        f.seqs[0]
            .proposer_state
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .unwrap()
            .block_table = old;
        assert!(
            f.model.decode(5, &mut f.seqs[0], CALLER).is_err(),
            "restoring a public view must not resurrect failed retirement"
        );
        assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
        assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), bytes);
        assert!(f.model.decode(6, &mut f.seqs[1], CALLER).is_err());
        assert!(f.gpu.trace().is_empty());
    }
}

#[test]
fn gate_one_refuses_verdicts_without_owned_accepted_row_consumption() {
    if isolated("gate_one_refuses_verdicts_without_owned_accepted_row_consumption") {
        return;
    }
    use crate::speculative::DraftProposer;
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        f.model
            .prefill(&[1, 2, 3, 4], &mut f.seqs[0], CALLER)
            .unwrap();
        f.model.decode(5, &mut f.seqs[0], CALLER).unwrap();
        f.model
            .run_mtp_propose_inner(7, 5, 4, &mut f.seqs[0], None)
            .unwrap();
        f.gpu.clear();
        assert!(
            f.head
                .after_verify(
                    0,
                    None,
                    f.seqs[0].proposer_state.as_mut().unwrap().as_mut(),
                    DEFAULT
                )
                .is_err(),
            "Gate1 must not silently trim a selected private cache"
        );
        assert!(
            f.model
                .record_glm_mtp_verified_impl(&mut f.seqs[0], 5, &[1, 2, 3, 4, 5], 0)
                .is_err(),
            "Gate1 must not silently ignore selected verdict"
        );
        assert!(f.gpu.trace().is_empty());
    }
}
