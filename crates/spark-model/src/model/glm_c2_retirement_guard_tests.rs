// SPDX-License-Identifier: AGPL-3.0-only
//! Actual retirement authority and replacement graph lifetime, not CUDA replay numerics.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::model::ssm_pool::SsmStatePool;
use crate::traits::Model;
use spark_runtime::gpu::GraphHandle;
use std::sync::{Arc, atomic::Ordering};

fn isolated(name: &str, graphs: bool) -> bool {
    if std::env::var("ATLAS_RETIRE_GUARD_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        &format!("model::glm_c2_handoff_tests::retirement_guard_tests::{name}"),
        "--nocapture",
    ])
    .env("ATLAS_RETIRE_GUARD_CHILD", "1")
    .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
    .env("ATLAS_GLM_MTP_REPAIR", "0")
    .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
    .env("ATLAS_GLM_MTP_ALL_GATHER", "1")
    .env("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0")
    .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1")
    .env("ATLAS_EP_GRAPHS", "0")
    .env("ATLAS_GDN_DECODE_GRAPH", "0")
    .env("ATLAS_GLM_TP_VERIFY_GRAPH", if graphs { "1" } else { "0" });
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
        "actual retirement child: {name}"
    );
    true
}

#[derive(Clone, Copy)]
enum BadGuard {
    Missing,
    Foreign,
    Swapped,
}

fn invalid_guard_refuses(case: BadGuard) {
    for rank in 0..2 {
        for owner in 0..2 {
            let peer = 1 - owner;
            let (mut f, _) = flow::prepare(rank, [owner, peer]);
            let original = f.seqs[owner].ssm_slot.take().unwrap();
            assert_eq!(original.idx(), Some(owner));
            assert!(original.belongs_to(&f.model.ssm_pool));
            let foreign = if matches!(case, BadGuard::Foreign) {
                Some(Arc::new(
                    SsmStatePool::new(
                        &f.model.config,
                        2,
                        true,
                        5,
                        4,
                        false,
                        crate::ssm_reserve::SsmRollbackMode::Snapshot,
                        f.model.gpu.as_ref(),
                    )
                    .unwrap(),
                ))
            } else {
                None
            };
            // Keep every removed genuine guard alive across the call. No
            // unrelated RAII return may make a foreign index look available.
            let mut foreign_other = None;
            match case {
                BadGuard::Missing => {}
                BadGuard::Foreign => {
                    let pool = foreign.as_ref().unwrap();
                    if owner == 1 {
                        foreign_other = Some(pool.claim_guarded().unwrap());
                    }
                    let guard = pool.claim_guarded().unwrap();
                    assert_eq!(guard.idx(), Some(owner));
                    assert!(!guard.belongs_to(&f.model.ssm_pool));
                    f.seqs[owner].ssm_slot = Some(guard);
                }
                BadGuard::Swapped => {
                    let guard = f.seqs[peer].ssm_slot.take().unwrap();
                    assert_eq!(guard.idx(), Some(peer));
                    assert!(guard.belongs_to(&f.model.ssm_pool));
                    f.seqs[owner].ssm_slot = Some(guard);
                }
            }
            let peer_rows = flow::private(&f.seqs[peer]).seq_len;
            let pointers = f
                .head
                .paired_test_kv_rows(
                    f.seqs[peer].proposer_state.as_ref().unwrap().as_ref(),
                    f.model.gpu.as_ref(),
                    peer_rows,
                )
                .unwrap();
            let kv: Vec<_> = pointers
                .iter()
                .map(|(k, v)| (f.gpu.read_span(*k, 1024), f.gpu.read_span(*v, 1024)))
                .collect();
            let peer_blocks = flow::private(&f.seqs[peer]).block_table.clone();
            let peer_tokens = f.seqs[peer].tokens.clone();
            let slab = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            let free_target = f.model.kv_cache.lock().num_free_blocks();
            f.gpu.clear();
            let result = f.model.free_sequence(&mut f.seqs[owner]);
            assert!(
                result.is_err(),
                "invalid live target guard authorized retirement"
            );
            assert!(
                f.gpu.trace().is_empty(),
                "refusal must precede target zero/wait or graph cleanup: {:?}",
                f.gpu.trace()
            );
            assert_eq!(original.idx(), Some(owner));
            assert!(f.seqs[owner].ssm_slot_idx().is_none());
            for slot in 0..2 {
                assert!(!f.model.ssm_pool.slot_is_free(slot));
            }
            assert_eq!(f.model.kv_cache.lock().num_free_blocks(), free_target);
            assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), slab);
            assert_eq!(f.seqs[peer].tokens, peer_tokens);
            assert_eq!(flow::private(&f.seqs[peer]).block_table, peer_blocks);
            assert_eq!(flow::private(&f.seqs[peer]).seq_len, peer_rows);
            for ((k, v), (kb, vb)) in pointers.iter().zip(kv) {
                assert_eq!(f.gpu.read_span(*k, 1024), kb);
                assert_eq!(f.gpu.read_span(*v, 1024), vb);
            }
            if let Some(pool) = foreign.as_ref() {
                assert!(
                    !pool.slot_is_free(owner),
                    "foreign guard must be neutralized"
                );
            }
            // Explicit lifetime witnesses; these Drops happen only after all
            // no-release assertions, not inside the operation under test.
            drop(foreign_other);
            drop(original);
        }
    }
}

#[test]
fn live_missing_target_guard_refuses_before_cleanup() {
    if !isolated("live_missing_target_guard_refuses_before_cleanup", false) {
        invalid_guard_refuses(BadGuard::Missing);
    }
}

#[test]
fn live_foreign_pool_target_guard_refuses_before_cleanup() {
    if !isolated(
        "live_foreign_pool_target_guard_refuses_before_cleanup",
        false,
    ) {
        invalid_guard_refuses(BadGuard::Foreign);
    }
}

#[test]
fn live_swapped_peer_target_guard_refuses_before_cleanup() {
    if !isolated(
        "live_swapped_peer_target_guard_refuses_before_cleanup",
        false,
    ) {
        invalid_guard_refuses(BadGuard::Swapped);
    }
}

fn graph(f: &Fixture, owner: usize) -> GraphHandle {
    let handle = *f.model.verify_kgamma_graph.lock().get(&(owner, 5)).unwrap();
    assert_ne!(
        handle.0, 0,
        "real K5 must publish a nonnull mock capture handle"
    );
    handle
}

fn graph_map(f: &Fixture) -> std::collections::BTreeMap<(usize, usize), u64> {
    f.model
        .verify_kgamma_graph
        .lock()
        .iter()
        .map(|(key, handle)| (*key, handle.0))
        .collect()
}

fn capture_and_record(f: &mut Fixture, owner: usize, history: &flow::History) -> GraphHandle {
    assert!(!f.model.suppress_graphs.load(Ordering::Relaxed));
    f.gpu.clear();
    flow::head_verdict(f, owner, history, 4);
    flow::acknowledge(f, owner, 4, false);
    let handle = graph(f, owner);
    let events = f.gpu.trace();
    assert!(events.contains(&Event::BeginCapture(DEFAULT)));
    assert!(events.contains(&Event::EndCapture(DEFAULT)));
    assert!(events.contains(&Event::LaunchGraph(handle.0, DEFAULT)));
    handle
}

#[test]
fn repeated_old_free_does_not_destroy_replacement_actual_kgamma_graph() {
    if isolated(
        "repeated_old_free_does_not_destroy_replacement_actual_kgamma_graph",
        true,
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let (mut f, histories) = flow::prepare(rank, [owner, 1 - owner]);
            f.gpu.capture_handles.store(true, Ordering::Relaxed);
            let old_graph = capture_and_record(&mut f, owner, &histories[owner]);
            f.gpu.clear();
            f.model.free_sequence(&mut f.seqs[owner]).unwrap();
            assert!(f.gpu.trace().contains(&Event::DestroyGraph(old_graph.0)));
            assert!(!f.model.verify_kgamma_graph.lock().contains_key(&(owner, 5)));
            let new = f.model.alloc_sequence().unwrap();
            assert_eq!(new.slot_idx, owner);
            assert_eq!(new.ssm_slot_idx(), Some(owner));
            let mut old = std::mem::replace(&mut f.seqs[owner], new);
            assert!(old.ssm_slot_idx().is_none());
            assert!(old.block_table.is_empty());
            assert!(flow::private(&old).block_table.is_empty());

            let prompt = [1, 2, 3, 4];
            f.seqs[owner].prompt_len = prompt.len();
            f.model
                .prefill(&prompt, &mut f.seqs[owner], CALLER)
                .unwrap();
            f.model.decode(5, &mut f.seqs[owner], CALLER).unwrap();
            let base = f.seqs[owner].seq_len;
            let drafts = f
                .model
                .run_mtp_propose_inner(7, base, 4, &mut f.seqs[owner], None)
                .unwrap();
            let history = flow::History {
                base,
                issued: std::iter::once(7).chain(drafts).collect(),
                canonical: vec![], // Not consumed by capture_and_record.
                bonus: vec![],
            };
            let replacement_graph = capture_and_record(&mut f, owner, &history);
            assert_ne!(old_graph.0, replacement_graph.0);
            let captured_graphs = graph_map(&f);
            let target_blocks = f.seqs[owner].block_table.clone();
            let private_blocks = flow::private(&f.seqs[owner]).block_table.clone();
            let free_target = f.model.kv_cache.lock().num_free_blocks();
            let slab = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            f.gpu.clear();
            f.model.free_sequence(&mut old).unwrap();
            assert!(
                f.gpu.trace().is_empty(),
                "old no-lease free touched replacement resources: {:?}",
                f.gpu.trace()
            );
            assert_eq!(graph_map(&f), captured_graphs);
            assert_eq!(f.seqs[owner].block_table, target_blocks);
            assert_eq!(flow::private(&f.seqs[owner]).block_table, private_blocks);
            assert_eq!(f.model.kv_cache.lock().num_free_blocks(), free_target);
            assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), slab);
            assert!(!f.model.ssm_pool.slot_is_free(owner));

            // Exercise the real next proposal and K5 lookup. The mock graph
            // launch proves handle liveness/order only, not replay arithmetic.
            let next = f.seqs[owner].seq_len;
            let drafts = f
                .model
                .run_mtp_propose_inner(6, next, 4, &mut f.seqs[owner], None)
                .unwrap();
            let issued: Vec<_> = std::iter::once(6).chain(drafts).collect();
            f.gpu.clear();
            let result = f
                .model
                .decode_verify_graphed_kgamma(&issued, &mut f.seqs[owner], CALLER)
                .unwrap();
            assert_eq!(result.len(), 5);
            assert_eq!(graph(&f, owner).0, replacement_graph.0);
            assert!(
                f.gpu
                    .trace()
                    .contains(&Event::LaunchGraph(replacement_graph.0, DEFAULT))
            );
            assert!(!f.gpu.trace().iter().any(|event| matches!(
                event,
                Event::BeginCapture(_) | Event::EndCapture(_) | Event::DestroyGraph(_)
            )));
        }
    }
}
