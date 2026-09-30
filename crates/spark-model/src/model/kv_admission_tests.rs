// SPDX-License-Identifier: AGPL-3.0-only

//! `admit`: the all-rank verdict and the rollback that makes a refused
//! chunk retryable, on a real `PagedKvCache`.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};
use spark_runtime::prefix_cache::NoPrefixCaching;
use std::cell::Cell;

const BLOCKS: usize = 4;

fn cache(gpu: &MockGpuBackend) -> PagedKvCache {
    let config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    PagedKvCache::new(config, BLOCKS, gpu).unwrap()
}

/// Reserve through `last_block` with `vote` standing in for the ranks.
fn run(
    seq: &mut SequenceState,
    last_block: usize,
    kv: &mut PagedKvCache,
    gpu: &MockGpuBackend,
    vote: impl FnOnce(bool) -> Result<bool>,
) -> Result<()> {
    admit(seq, last_block, kv, &NoPrefixCaching, gpu, 0, false, vote)
}

fn refused_by_peer(r: Result<()>) -> Option<bool> {
    r.err()?
        .downcast_ref::<KvAdmissionRefused>()
        .map(|r| r.by_peer)
}

#[test]
fn unanimous_admission_keeps_the_reservation() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu);
    let mut seq = SequenceState::host_only(0);
    let voted = Cell::new(None);
    run(&mut seq, 1, &mut kv, &gpu, |ok| {
        voted.set(Some(ok));
        Ok(ok)
    })
    .unwrap();
    assert_eq!(voted.get(), Some(true));
    assert_eq!(seq.block_table.len(), 2);
    assert_eq!(kv.num_free_blocks(), BLOCKS - 2);
}

#[test]
fn local_exhaustion_is_refused_and_leaves_no_partial_blocks() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu);
    let held: Vec<u32> = (0..BLOCKS - 2).map(|_| kv.alloc_block().unwrap()).collect();
    let mut seq = SequenceState::host_only(0);
    seq.block_table.push(kv.alloc_block().unwrap()); // e.g. a matched prefix
    let voted = Cell::new(None);
    // Blocks 1 and 2 are needed and one is free: the helper allocates block
    // 1, fails on block 2, and must hand block 1 back.
    let r = run(&mut seq, 2, &mut kv, &gpu, |ok| {
        voted.set(Some(ok));
        Ok(ok)
    });
    assert_eq!(voted.get(), Some(false), "the failing rank votes no");
    assert_eq!(refused_by_peer(r), Some(false));
    assert_eq!(
        seq.block_table.len(),
        1,
        "only this call's blocks roll back"
    );
    assert_eq!(kv.num_free_blocks(), 1);

    // The preempted victim's blocks come back: the same request now fits
    // and ends exactly where an uninterrupted reservation would.
    kv.free_blocks(&held);
    run(&mut seq, 2, &mut kv, &gpu, Ok).unwrap();
    assert_eq!(seq.block_table.len(), 3);
    assert_eq!(kv.num_free_blocks(), BLOCKS - 3);
}

#[test]
fn a_peer_refusal_rolls_back_a_local_success() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu);
    let mut seq = SequenceState::host_only(0);
    let r = run(&mut seq, 1, &mut kv, &gpu, |ok| {
        assert!(ok, "this rank could admit");
        Ok(false)
    });
    assert_eq!(refused_by_peer(r), Some(true));
    assert!(seq.block_table.is_empty());
    assert_eq!(kv.num_free_blocks(), BLOCKS);
}

#[test]
fn a_failed_vote_surfaces_the_transport_error() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu);
    let mut seq = SequenceState::host_only(0);
    let e = run(&mut seq, 0, &mut kv, &gpu, |_| anyhow::bail!("nccl down")).unwrap_err();
    assert!(!is_kv_admission_refused(&e), "not retryable: {e:#}");
    assert!(format!("{e:#}").contains("nccl down"));
}

#[test]
fn only_the_typed_refusal_counts_even_through_context() {
    let refused: anyhow::Error = KvAdmissionRefused { by_peer: false }.into();
    let refused = refused.context("prefill chunk");
    assert!(is_kv_admission_refused(&refused));
    assert!(format!("{refused:#}").contains("KV cache exhausted"));
    // A same-worded error raised elsewhere (e.g. mid-forward) is NOT one.
    let plain = anyhow::anyhow!("KV cache exhausted: no free blocks");
    assert!(!is_kv_admission_refused(&plain));
}

#[test]
fn the_worker_survives_only_an_agreed_refusal() {
    let refused = Err(KvAdmissionRefused { by_peer: true }.into());
    assert!(worker_step_outcome(refused).unwrap());
    let fatal = worker_step_outcome(Err(anyhow::anyhow!("KV cache exhausted: no free blocks")));
    assert!(fatal.is_err());
    assert!(!worker_step_outcome(Ok(false)).unwrap());
}

#[test]
fn the_worker_rejects_a_chunk_away_from_its_progress() {
    let mut seq = SequenceState::host_only(3);
    seq.seq_len = 4096;
    check_worker_chunk_start(&seq, 4096).unwrap();
    let e = check_worker_chunk_start(&seq, 0).unwrap_err();
    assert!(format!("{e:#}").contains("diverged"), "{e:#}");
}
