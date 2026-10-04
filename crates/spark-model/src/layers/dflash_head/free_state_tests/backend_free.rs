// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn real_free_state_backend_free_failure_retains_pointer_for_retry() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);

    // The accumulator goes to the reuse pool (no backend free); the
    // injected failure lands on the block-table-dev free — its handle must
    // be RESTORED so a retry can release it.
    gpu.fail_next_free();
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("free_state succeeds despite the backend free failure");

    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(
        boxed.block_table_dev.is_some(),
        "handle restored, retryable"
    );
    // …and the retry DOES release it (flag is one-shot).
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("second free retries the block table free");
    assert!(boxed.block_table_dev.is_none());
}
