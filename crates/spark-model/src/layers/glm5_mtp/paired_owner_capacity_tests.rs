// SPDX-License-Identifier: AGPL-3.0-only
//! Actual cold head/cache/lease allocation, not Model or four-owner E6 authority.
use super::*;

#[test]
fn actual_four_owner_constructor_and_slot3_reuse_preserve_peers() {
    let gpu = TestGpu::new();
    let head = configured_owner_head(&gpu, true, true, Some(4)).unwrap();
    let slab = head.paired.as_ref().unwrap().lock().slab;
    assert_eq!(head.paired.as_ref().unwrap().lock().slots.len(), 4);
    assert_eq!(gpu.live_allocations()[&slab.0], 4 * SLOT_BYTES);
    {
        let cache = head.kv_cache.lock();
        assert_eq!(cache.num_blocks(), 512);
        assert_eq!(cache.num_free_blocks(), 512);
        for (ptr, stride) in [
            (
                cache.sparse_index_pool_ptr(0),
                cache.sparse_index_block_stride_bytes(0),
            ),
            (
                cache.sparse_index_tail_pool_ptr(0),
                cache.sparse_index_tail_block_stride_bytes(0),
            ),
        ] {
            assert!(!ptr.is_null());
            assert_eq!(gpu.live_allocations()[&ptr.0], 512 * stride);
        }
    }
    let mut states: Vec<_> = (0..4)
        .map(|_| head.alloc_state_inner(&gpu).unwrap())
        .collect();
    for (index, state) in states.iter().enumerate() {
        let lease = state.paired.as_ref().unwrap();
        assert_eq!(lease.slot, index);
        assert_eq!(lease.slab, slab);
        assert_eq!(state.block_table.len(), 128);
        head.validate_paired_live(state, &gpu).unwrap();
        for other in &states[..index] {
            assert!(
                state
                    .block_table
                    .iter()
                    .all(|block| !other.block_table.contains(block))
            );
        }
        gpu.memset(
            slab.offset(index * SLOT_BYTES),
            0x30 + index as u8,
            SLOT_BYTES,
        )
        .unwrap();
    }
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
    gpu.clear();
    assert!(head.alloc_state_inner(&gpu).is_err());
    assert!(
        gpu.trace().is_empty(),
        "fifth owner must refuse before allocation"
    );
    let peer_blocks: Vec<_> = states[..3].iter().map(|s| s.block_table.clone()).collect();
    let peers = gpu.read_span(slab, 3 * SLOT_BYTES).unwrap();
    let generation = states[3].paired.as_ref().unwrap().generation;
    let retired_blocks = states[3].block_table.clone();
    head.free_state(&gpu, &mut states[3]).unwrap();
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 128);
    assert!(head.validate_paired_live(&states[3], &gpu).is_err());
    let replacement = head.alloc_state_inner(&gpu).unwrap();
    let lease = replacement.paired.as_ref().unwrap();
    assert_eq!(lease.slot, 3);
    assert!(lease.generation > generation);
    assert!(
        replacement
            .block_table
            .iter()
            .all(|block| retired_blocks.contains(block))
    );
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
    assert_eq!(gpu.read_span(slab, 3 * SLOT_BYTES).unwrap(), peers);
    for (index, state) in states[..3].iter().enumerate() {
        assert_eq!(state.block_table, peer_blocks[index]);
        head.validate_paired_live(state, &gpu).unwrap();
    }
    // Retired object has no authority after actual same-slot generation reuse.
    let before_old_free = gpu.read_span(slab, 4 * SLOT_BYTES).unwrap();
    gpu.clear();
    head.free_state(&gpu, &mut states[3]).unwrap();
    assert!(gpu.trace().is_empty());
    assert_eq!(
        gpu.read_span(slab, 4 * SLOT_BYTES).unwrap(),
        before_old_free
    );
    head.validate_paired_live(&replacement, &gpu).unwrap();
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
}

#[test]
fn actual_four_owner_failed_slot3_retirement_stays_quarantined() {
    let gpu = TestGpu::new();
    let head = configured_owner_head(&gpu, true, false, Some(4)).unwrap();
    let mut states: Vec<_> = (0..4)
        .map(|_| head.alloc_state_inner(&gpu).unwrap())
        .collect();
    let views: Vec<_> = states.iter().map(|s| s.block_table.clone()).collect();
    gpu.clear();
    gpu.fail_sync_at.store(1, Ordering::Relaxed);
    assert!(head.free_state(&gpu, &mut states[3]).is_err());
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
    gpu.clear();
    gpu.synchronize(gpu.default_stream()).unwrap();
    gpu.clear();
    assert!(head.free_state(&gpu, &mut states[3]).is_err());
    assert!(head.alloc_state_inner(&gpu).is_err());
    assert!(gpu.trace().is_empty());
    for (index, state) in states.iter().enumerate() {
        assert_eq!(state.block_table, views[index]);
        assert_eq!(head.validate_paired_live(state, &gpu).is_ok(), index != 3);
    }
}

#[test]
fn actual_three_owner_constructor_and_invalid_capacity_bounds() {
    let gpu = TestGpu::new();
    let head = configured_owner_head(&gpu, true, false, Some(3)).unwrap();
    let slab = head.paired.as_ref().unwrap().lock().slab;
    assert_eq!(gpu.live_allocations()[&slab.0], 3 * SLOT_BYTES);
    assert_eq!(head.kv_cache.lock().num_blocks(), 384);
    let states: Vec<_> = (0..3)
        .map(|_| head.alloc_state_inner(&gpu).unwrap())
        .collect();
    for (index, state) in states.iter().enumerate() {
        assert_eq!(state.paired.as_ref().unwrap().slot, index);
        head.validate_paired_live(state, &gpu).unwrap();
    }
    gpu.clear();
    assert!(head.alloc_state_inner(&gpu).is_err());
    assert!(gpu.trace().is_empty());
    for invalid in [0, 1, 5, usize::MAX] {
        let gpu = TestGpu::new();
        let error = configured_owner_head(&gpu, true, false, Some(invalid))
            .err()
            .expect("invalid explicit capacity must fail");
        assert!(format!("{error:#}").contains("capacity must be2..4"));
        // Only the six fixture-supplied weights exist, no cache/slab allocation.
        assert_eq!(gpu.live_allocations().len(), 6);
        assert_eq!(gpu.trace().len(), 6);
    }
}

#[test]
fn actual_four_owner_slab_checks_its_full_extent_before_claiming_alias() {
    let gpu = TestGpu::new();
    let head = configured_owner_head(&gpu, true, true, Some(4)).unwrap();
    let cache = head.kv_cache.lock();
    let original = kv_rows::cache_spans(&cache)
        .unwrap()
        .into_iter()
        .map(|span| span.ptr.0)
        .min()
        .unwrap();
    // Only the final quarter of a four-owner slab intersects the first cache owner;
    // the old two-owner span would miss it. This is an injected allocator lie,
    // not a second allocation or permission to free the underlying bytes.
    let alias = original.checked_sub((3 * SLOT_BYTES) as u64).unwrap();
    let before = gpu.live_allocations();
    gpu.clear();
    gpu.next_alloc_alias.store(alias, Ordering::Relaxed);
    assert!(Pool::new(&gpu, 2044, &cache, 0, OwnerCapacity::new(4).unwrap()).is_err());
    assert_eq!(gpu.free_count(), 0);
    assert_eq!(gpu.live_allocations(), before);
}
