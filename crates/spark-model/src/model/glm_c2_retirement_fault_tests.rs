// SPDX-License-Identifier: AGPL-3.0-only
//! Real cleanup-call fault boundaries; mock graph lifetime, not CUDA execution.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::model::ssm_pool::SsmStatePool;
use crate::speculative::DraftProposer;
use crate::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::{Arc, atomic::Ordering};

fn isolated(name: &str) -> bool {
    if std::env::var("ATLAS_RETIRE_FAULT_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        &format!("model::glm_c2_handoff_tests::retirement_fault_tests::{name}"),
        "--nocapture",
    ])
    .env("ATLAS_RETIRE_FAULT_CHILD", "1")
    .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
    .env("ATLAS_GLM_MTP_REPAIR", "0")
    .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
    .env("ATLAS_GLM_MTP_ALL_GATHER", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0")
    .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1")
    .env("ATLAS_EP_GRAPHS", "0")
    .env("ATLAS_GDN_DECODE_GRAPH", "0")
    .env("ATLAS_GLM_TP_VERIFY_GRAPH", "1");
    for key in [
        "ATLAS_DEBUG_NO_GRAPH",
        "ATLAS_DFLASH_DEBUG_NO_GRAPH",
        "ATLAS_GLM_MULTI_SEQ_SPARSE",
        "ATLAS_NO_MTP_EAGER_DRAFTER",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_CARRY_DRAFTER",
        "ATLAS_MTP_ACCEPT_DEBUG",
        "ATLAS_GLM_MTP_FUSED_EH_NORM",
        "ATLAS_GLM_MTP_SERIAL_PREFILL",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_GLM_MTP_PROFILE",
        "ATLAS_GLM_VERIFY_PROFILE",
        "ATLAS_SSM_SAVE_DUMP",
        "ATLAS_DIAG_GEMMA4",
    ] {
        cmd.env_remove(key);
    }
    assert!(
        cmd.status().unwrap().success(),
        "actual cleanup child: {name}"
    );
    true
}

struct Prepared {
    f: Fixture,
    metadata: [[DevicePtr; 2]; 2],
    graphs: [u64; 2],
}

fn prepared(rank: usize, owner: usize) -> Prepared {
    let (mut f, history) = flow::prepare(rank, [owner, 1 - owner]);
    f.gpu.capture_handles.store(true, Ordering::Relaxed);
    let mut graphs = [0; 2];
    for who in [owner, 1 - owner] {
        f.gpu.clear();
        flow::head_verdict(&mut f, who, &history[who], 4);
        flow::acknowledge(&mut f, who, 4, false);
        graphs[who] = f.model.verify_kgamma_graph.lock()[&(who, 5)].0;
        assert_ne!(graphs[who], 0);
        assert!(f.gpu.trace().contains(&Event::BeginCapture(DEFAULT)));
        assert!(f.gpu.trace().contains(&Event::EndCapture(DEFAULT)));
        assert!(
            f.gpu
                .trace()
                .contains(&Event::LaunchGraph(graphs[who], DEFAULT))
        );
    }
    assert_ne!(graphs[0], graphs[1]);
    let bs = f.model.kv_cache.lock().block_size();
    let metadata = std::array::from_fn(|who| {
        // The real production allocation helper owns both pointers; no fake
        // graph or metadata handles are inserted into the sequence by tests.
        let tokens = f.seqs[who].seq_len;
        let meta = f
            .model
            .ensure_chunked_prefill_meta(&mut f.seqs[who], tokens, bs)
            .unwrap();
        assert!(!meta.block_table.is_null() && !meta.seq_len.is_null());
        [meta.block_table, meta.seq_len]
    });

    // Like the existing cleanup fixture, attach a real tiny SSM pool AFTER
    // target/primer/graph work. This exercises cleanup bytes and true guards,
    // without pretending the sentinel target layer is a KDA numerical oracle.
    f.model.gpu.synchronize(DEFAULT).unwrap();
    let mut cfg = f.model.config.clone();
    cfg.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
    cfg.linear_num_key_heads = 1;
    cfg.linear_num_value_heads = 1;
    cfg.linear_key_head_dim = 2;
    cfg.linear_value_head_dim = 2;
    cfg.linear_conv_kernel_dim = 2;
    cfg.mamba_num_heads = 0;
    cfg.mamba_head_dim = 0;
    let pool = Arc::new(
        SsmStatePool::new(
            &cfg,
            2,
            false,
            3,
            4,
            false,
            crate::ssm_reserve::SsmRollbackMode::Snapshot,
            f.model.gpu.as_ref(),
        )
        .unwrap(),
    );
    assert_eq!((pool.h_stored_bytes, pool.conv_bytes), (16, 48));
    for who in 0..2 {
        let guard = pool.claim_guarded().unwrap();
        assert_eq!(guard.idx(), Some(who));
        f.seqs[who].ssm_slot = Some(guard);
        f.gpu
            .write_span(pool.h_state(0, who), &[0x31 + who as u8; 16]);
        f.gpu
            .write_span(pool.conv_state(0, who), &[0x51 + who as u8; 48]);
    }
    f.model.ssm_pool = pool;
    f.gpu.clear();
    Prepared {
        f,
        metadata,
        graphs,
    }
}

#[derive(Clone, Copy)]
enum Fault {
    Wait,
    HZero,
    ConvZero,
    ZeroCompletion,
    Graph,
    MetaBlocks,
    MetaLength,
    PrivateCompletion,
}

fn expected(p: &Prepared, owner: usize, fault: Fault) -> Event {
    match fault {
        Fault::Wait => Event::WaitEvent(DEFAULT, p.f.model.secondary_event),
        Fault::HZero => Event::Memset(p.f.model.ssm_pool.h_state(0, owner), 16, DEFAULT),
        Fault::ConvZero => Event::Memset(p.f.model.ssm_pool.conv_state(0, owner), 48, DEFAULT),
        Fault::ZeroCompletion | Fault::PrivateCompletion => Event::Sync(DEFAULT),
        Fault::Graph => Event::DestroyGraph(p.graphs[owner]),
        Fault::MetaBlocks => Event::Free(p.metadata[owner][0]),
        Fault::MetaLength => Event::Free(p.metadata[owner][1]),
    }
}

fn fault_matrix(name: &str, fault: Fault) {
    if isolated(name) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let peer = 1 - owner;
            let mut control = prepared(rank, owner);
            let event = expected(&control, owner, fault);
            control
                .f
                .model
                .free_sequence(&mut control.f.seqs[owner])
                .unwrap();
            let events = control.f.gpu.trace();
            let candidates: Vec<_> = events
                .iter()
                .enumerate()
                .filter(|(_, e)| **e == event)
                .map(|(i, _)| i + 1)
                .collect();
            let ordinal = if matches!(fault, Fault::PrivateCompletion) {
                assert_eq!(
                    candidates.len(),
                    2,
                    "actual zero and paired free_state completion"
                );
                candidates[1]
            } else if matches!(fault, Fault::ZeroCompletion) {
                assert_eq!(candidates.len(), 2);
                candidates[0]
            } else {
                assert_eq!(candidates.len(), 1, "actual cleanup control: {events:?}");
                candidates[0]
            };

            let mut p = prepared(rank, owner);
            let actual_event = expected(&p, owner, fault);
            let rows = flow::private(&p.f.seqs[peer]).seq_len;
            let peer_ptrs =
                p.f.head
                    .paired_test_kv_rows(
                        p.f.seqs[peer].proposer_state.as_ref().unwrap().as_ref(),
                        p.f.model.gpu.as_ref(),
                        rows,
                    )
                    .unwrap();
            let peer_kv: Vec<_> = peer_ptrs
                .iter()
                .map(|(k, v)| (p.f.gpu.read_span(*k, 1024), p.f.gpu.read_span(*v, 1024)))
                .collect();
            let private_blocks = flow::private(&p.f.seqs[owner]).block_table.clone();
            let peer_blocks = flow::private(&p.f.seqs[peer]).block_table.clone();
            let target_blocks = p.f.seqs[owner].block_table.clone();
            let free_target = p.f.model.kv_cache.lock().num_free_blocks();
            let free_private = p.f.head.paired_test_free_blocks();
            let peer_tokens = p.f.seqs[peer].tokens.clone();
            let slab = p.f.gpu.read_span(p.f.gpu.slab(), SLAB_BYTES);
            let peer_h = p.f.gpu.read_span(p.f.model.ssm_pool.h_state(0, peer), 16);
            let peer_conv =
                p.f.gpu
                    .read_span(p.f.model.ssm_pool.conv_state(0, peer), 48);
            p.f.gpu.clear();
            p.f.gpu.fail.store(ordinal, Ordering::Relaxed);
            let result = p.f.model.free_sequence(&mut p.f.seqs[owner]);
            assert!(result.is_err(), "actual cleanup fault was swallowed");
            assert!(
                format!("{:#}", result.unwrap_err()).contains("injected fixture operation failure")
            );
            let events = p.f.gpu.trace();
            assert_eq!(events.get(ordinal - 1), Some(&actual_event));
            assert_eq!(
                events.len(),
                ordinal,
                "backend work continued after failed {actual_event:?}: {events:?}"
            );
            assert!(p.f.seqs[owner].ssm_slot_idx().is_none());
            assert!(!p.f.model.ssm_pool.slot_is_free(owner));
            assert!(!p.f.model.ssm_pool.slot_is_free(peer));
            assert_eq!(p.f.model.kv_cache.lock().num_free_blocks(), free_target);
            assert_eq!(p.f.head.paired_test_free_blocks(), free_private);
            assert_eq!(p.f.seqs[peer].tokens, peer_tokens);
            assert_eq!(p.f.seqs[owner].block_table, target_blocks);
            assert_eq!(flow::private(&p.f.seqs[owner]).block_table, private_blocks);
            assert_eq!(flow::private(&p.f.seqs[peer]).block_table, peer_blocks);
            assert_eq!(p.f.gpu.read_span(p.f.gpu.slab(), SLAB_BYTES), slab);
            assert_eq!(
                p.f.gpu.read_span(p.f.model.ssm_pool.h_state(0, peer), 16),
                peer_h
            );
            assert_eq!(
                p.f.gpu
                    .read_span(p.f.model.ssm_pool.conv_state(0, peer), 48),
                peer_conv
            );
            for ((k, v), (kb, vb)) in peer_ptrs.iter().zip(peer_kv) {
                assert_eq!(p.f.gpu.read_span(*k, 1024), kb);
                assert_eq!(p.f.gpu.read_span(*v, 1024), vb);
            }
            assert_eq!(
                p.f.model.verify_kgamma_graph.lock()[&(peer, 5)].0,
                p.graphs[peer]
            );
            let peer_meta = p.f.seqs[peer].chunked_prefill_meta.as_ref().unwrap();
            assert_eq!([peer_meta.block_table, peer_meta.seq_len], p.metadata[peer]);
            p.f.gpu.clear();
            p.f.model.gpu.synchronize(DEFAULT).unwrap();
            p.f.gpu.clear();
            assert!(p.f.model.free_sequence(&mut p.f.seqs[owner]).is_err());
            assert!(p.f.head.alloc_state(p.f.model.gpu.as_ref()).is_err());
            assert!(
                p.f.gpu.trace().is_empty(),
                "later completion reopened cleanup"
            );
            let failed = std::mem::replace(&mut p.f.seqs[owner], SequenceState::host_only(owner));
            drop(failed);
            assert_eq!(p.f.head.paired_test_free_blocks(), free_private);
            assert!(
                !p.f.model.ssm_pool.slot_is_free(owner),
                "failed owner Drop recycled target slot"
            );
            assert!(p.f.model.ssm_pool.claim_guarded().is_err());
            assert!(
                p.f.gpu.trace().is_empty(),
                "failed owner Drop performed backend cleanup"
            );
        }
    }
}

#[test]
fn wait_failure_stops_retirement() {
    fault_matrix("wait_failure_stops_retirement", Fault::Wait);
}
#[test]
fn h_zero_failure_stops_retirement() {
    fault_matrix("h_zero_failure_stops_retirement", Fault::HZero);
}
#[test]
fn conv_zero_failure_stops_retirement() {
    fault_matrix("conv_zero_failure_stops_retirement", Fault::ConvZero);
}
#[test]
fn zero_completion_failure_stops_retirement() {
    fault_matrix(
        "zero_completion_failure_stops_retirement",
        Fault::ZeroCompletion,
    );
}
#[test]
fn graph_destroy_failure_stops_retirement() {
    fault_matrix("graph_destroy_failure_stops_retirement", Fault::Graph);
}
#[test]
fn metadata_blocks_free_failure_stops_retirement() {
    fault_matrix(
        "metadata_blocks_free_failure_stops_retirement",
        Fault::MetaBlocks,
    );
}
#[test]
fn metadata_length_free_failure_stops_retirement() {
    fault_matrix(
        "metadata_length_free_failure_stops_retirement",
        Fault::MetaLength,
    );
}
#[test]
fn private_completion_failure_stops_retirement() {
    fault_matrix(
        "private_completion_failure_stops_retirement",
        Fault::PrivateCompletion,
    );
}
