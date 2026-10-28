// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn reclaim_seam_free_failure_retains_pointers() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let own = owner(3, 77);
    let mut dstate = live_state(&gpu, own);
    let kv_cache = parking_lot::Mutex::new(
        PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 8,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            8,
            &gpu,
        )
        .unwrap(),
    );
    for _ in 0..2 {
        let block = kv_cache.lock().try_alloc_block().expect("block available");
        dstate.block_table.push(block);
    }

    gpu.fail_next_free();
    gpu.fail_next_free();
    dstate.reclaim_on_owner_failure(&gpu, &kv_cache);

    // Both failed frees keep their handles (retryable)…
    assert_ne!(dstate.ctx_hidden_acc.0, 0);
    assert!(dstate.block_table_dev.is_some());
    // …blocks are still returned, and a plain second reclaim retries both
    // frees (injections are one-shot).
    assert!(dstate.block_table.is_empty());
    dstate.reclaim_on_owner_failure(&gpu, &kv_cache);
    assert_eq!(dstate.ctx_hidden_acc.0, 0);
    assert!(dstate.block_table_dev.is_none());
}
