// SPDX-License-Identifier: AGPL-3.0-only

//! Window-bounded KV pools for sliding-window layers (`window.rs`).

use super::*;

const WINDOW: usize = 1024;

/// Four attention layers, the odd ones sliding-window, 16-token blocks.
fn windowed(blocks: usize, local: usize, pins: usize, gpu: &MockGpuBackend) -> PagedKvCache {
    windowed_over(WINDOW, blocks, local, pins, gpu)
}

/// [`windowed`] with a `window`-key window.
fn windowed_over(
    window: usize,
    blocks: usize,
    local: usize,
    pins: usize,
    gpu: &MockGpuBackend,
) -> PagedKvCache {
    let config = KvCacheConfig {
        num_layers: 4,
        dtype: KvCacheDtype::Bf16,
        ..test_config()
    };
    let plan = WindowPlan {
        layers: vec![1, 3],
        window,
        blocks: local,
        pins,
    };
    PagedKvCache::new_windowed(config, blocks, plan, gpu).unwrap()
}

fn local_of(cache: &PagedKvCache, gpu: &MockGpuBackend, block: u32) -> u32 {
    let map = gpu.read_alloc(cache.window_map_ptr()).unwrap();
    let at = block as usize * 4;
    u32::from_le_bytes(map[at..at + 4].try_into().unwrap())
}

#[test]
fn first_block_starts_at_the_key_tile_below_the_window() {
    let plan = WindowPlan {
        layers: vec![1],
        window: WINDOW,
        blocks: 2,
        pins: 0,
    };
    // the first row at p reads keys from p + 1 - W, loaded from the 32-key
    // tile that holds it (the prefill kernels' tile start)
    for (first_pos, block) in [(0, 0), (1023, 0), (1024, 0), (1056, 2), (5000, 248)] {
        assert_eq!(
            plan.first_block(first_pos, 16),
            block,
            "first row {first_pos}"
        );
    }
}

#[test]
fn plan_holds_every_live_span_the_batch_transient_and_the_pins() {
    // W + 31 tile keys + 16 verify rows over 16-token blocks, plus the
    // block a row run can straddle
    assert_eq!(WindowPlan::span_blocks(WINDOW, 16, 16), 68);
    assert_eq!(WindowPlan::span_blocks(WINDOW, 16, 0), 67);
    let blocks = WindowPlan::size_blocks(WINDOW, 16, 8, 16, 4104, 8);
    // null block + 8 spans + the batch's rows (and a block per sequence it
    // can straddle) + 8 pinned windows
    assert_eq!(blocks, 1 + 8 * 68 + 4104usize.div_ceil(16) + 8 + 8 * 67);
}

#[test]
fn window_layers_live_outside_the_primary_pools() {
    let gpu = MockGpuBackend::new();
    let cache = windowed(64, 8, 0, &gpu);
    for layer in [1, 3] {
        assert!(cache.is_window_layer(layer));
        assert!(!cache.k_pool_ptr(layer).is_null());
        assert_ne!(cache.k_pool_ptr(layer), cache.v_pool_ptr(layer));
    }
    assert!(!cache.is_window_layer(0) && !cache.is_window_layer(2));
    // zeroing a primary block touches the two primary layers' K and V only
    let before = gpu.memset_count();
    cache.zero_blocks(&[5], &gpu, 0).unwrap();
    assert_eq!(gpu.memset_count() - before, 2 * 2);
}

#[test]
fn hold_lends_a_zeroed_local_block_and_publishes_the_map() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    assert_eq!(local_of(&cache, &gpu, 5), NO_WINDOW);
    let junk = vec![0xABu8; 8 * cache.k_block_stride_bytes_for_layer(1)];
    gpu.copy_h2d(&junk, cache.k_pool_ptr(1)).unwrap();
    cache.window_hold(&[5, 6], &gpu, 0).unwrap();
    let (l5, l6) = (local_of(&cache, &gpu, 5), local_of(&cache, &gpu, 6));
    assert!(l5 != NO_WINDOW && l6 != NO_WINDOW && l5 != l6);
    // block 0 of the window pool is the null block, never lent
    assert!(l5 != 0 && l6 != 0);
    let stride = cache.k_block_stride_bytes_for_layer(1);
    let pool = gpu.read_alloc(cache.k_pool_ptr(1)).unwrap();
    for l in [l5, l6] {
        let at = l as usize * stride;
        assert!(pool[at..at + stride].iter().all(|&b| b == 0), "local {l}");
    }
    assert!(cache.window_mapped(5) && cache.window_mapped(6));
}

#[test]
fn holders_of_one_primary_block_share_its_window_rows() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    cache.window_hold(&[5], &gpu, 0).unwrap();
    let local = local_of(&cache, &gpu, 5);
    cache.window_hold(&[5], &gpu, 0).unwrap();
    assert_eq!(local_of(&cache, &gpu, 5), local);
    assert_eq!(cache.window_free_blocks(), 6);
    cache.window_release(&[5], &gpu, 0).unwrap();
    assert!(cache.window_mapped(5));
    cache.window_release(&[5], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(5));
    assert_eq!(cache.window_free_blocks(), 7);
}

#[test]
fn an_exhausted_window_pool_refuses_the_whole_hold() {
    let gpu = MockGpuBackend::new();
    // null block + two lendable blocks
    let mut cache = windowed(64, 3, 0, &gpu);
    cache.window_hold(&[1], &gpu, 0).unwrap();
    let err = cache.window_hold(&[2, 3], &gpu, 0).unwrap_err();
    // the scheduler reads this as an exhausted pool (preempt and retry)
    assert!(format!("{err:#}").contains("KV cache exhausted"), "{err:#}");
    // nothing of the refused hold stays lent
    assert!(!cache.window_mapped(2) && !cache.window_mapped(3));
    assert_eq!(cache.window_free_blocks(), 1);
}

#[test]
fn pins_keep_a_snapshot_window_until_the_pool_needs_it() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 5, 4, &gpu);
    let (a, b) = (cache.alloc_block().unwrap(), cache.alloc_block().unwrap());
    cache.window_hold(&[a, b], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[a, b], 0, usize::MAX, PinRank::Prompt));
    // the pin holds the primary blocks too
    assert_eq!((cache.ref_count(a), cache.ref_count(b)), (2, 2));
    // the sequence leaves; the pin keeps the window
    cache.window_release(&[a, b], &gpu, 0).unwrap();
    cache.free_blocks(&[a, b]);
    assert!(cache.window_mapped(a) && cache.window_mapped(b));
    // two blocks left free; a third evicts the pin
    cache.window_hold(&[10, 11], &gpu, 0).unwrap();
    assert!(cache.window_mapped(a));
    cache.window_hold(&[12], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(a) && !cache.window_mapped(b));
    assert_eq!((cache.ref_count(a), cache.ref_count(b)), (0, 0));
    assert_eq!(cache.window_pins(), 0);
}

#[test]
fn pins_past_their_count_evict_the_oldest() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 1, &gpu);
    let (a, b) = (cache.alloc_block().unwrap(), cache.alloc_block().unwrap());
    cache.window_hold(&[a, b], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[a], 0, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&[b], 0, usize::MAX, PinRank::Prompt));
    assert_eq!(cache.window_pins(), 1);
    assert_eq!((cache.ref_count(a), cache.ref_count(b)), (1, 2));
    cache.window_release(&[a, b], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(a) && cache.window_mapped(b));
}

#[test]
fn a_window_with_unmapped_rows_is_not_pinned() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 4, &gpu);
    cache.window_hold(&[1], &gpu, 0).unwrap();
    assert!(!cache.window_pin(&[1, 2], 0, usize::MAX, PinRank::Prompt));
    assert_eq!(cache.window_pins(), 0);
    assert_eq!(cache.ref_count(1), 0);
}

#[test]
fn a_freed_primary_block_drops_its_window_rows() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    let block = cache.alloc_block().unwrap();
    cache.window_hold(&[block], &gpu, 0).unwrap();
    // a holder without a primary ref is a bug; the free still reclaims the
    // rows rather than leave them mapped to a block another sequence gets
    cache.free_block(block);
    assert!(!cache.window_mapped(block));
    assert_eq!(cache.window_free_blocks(), 7);
}

#[test]
fn release_returns_the_window_pools() {
    use atlas_core::scope::ModelResource;
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    let live = gpu.alloc_count();
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
    assert!(gpu.alloc_count() < live);
}

#[test]
fn a_resumed_window_goes_to_the_back_of_the_pin_queue() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 2, &gpu);
    let blocks: Vec<u32> = (0..3).map(|_| cache.alloc_block().unwrap()).collect();
    cache.window_hold(&blocks, &gpu, 0).unwrap();
    assert!(cache.window_pin(&blocks[..1], 0, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&blocks[1..2], 0, usize::MAX, PinRank::Prompt));
    // a resume reads the oldest pin's window: it is now the newest
    cache.window_touch(&blocks[..1]);
    assert!(cache.window_pin(&blocks[2..], 0, usize::MAX, PinRank::Prompt));
    cache.window_release(&blocks, &gpu, 0).unwrap();
    assert!(cache.window_mapped(blocks[0]) && cache.window_mapped(blocks[2]));
    assert!(!cache.window_mapped(blocks[1]));
}

#[test]
fn a_deeper_pin_on_the_same_path_sends_the_shallower_one_out_first() {
    let gpu = MockGpuBackend::new();
    // a 32-key window: a pin spans 4 blocks
    let mut cache = windowed_over(32, 64, 24, 3, &gpu);
    // two sequences: `x` and `y`, each with its own blocks
    let x: Vec<u32> = (0..12).map(|_| cache.alloc_block().unwrap()).collect();
    let y: Vec<u32> = (0..2).map(|_| cache.alloc_block().unwrap()).collect();
    cache.window_hold(&x, &gpu, 0).unwrap();
    cache.window_hold(&y, &gpu, 0).unwrap();
    // y pins, x pins at blocks 0..2, then x pins far deeper along its own path
    assert!(cache.window_pin(&y, 0, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&x[..2], 0, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&x, 10, usize::MAX, PinRank::Prompt));
    // the fourth pin evicts x's shallow one, not y's older pin
    let z = cache.alloc_block().unwrap();
    cache.window_hold(&[z], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[z], 0, usize::MAX, PinRank::Prompt));
    cache.window_release(&x, &gpu, 0).unwrap();
    cache.window_release(&y, &gpu, 0).unwrap();
    assert!(!cache.window_mapped(x[0]) && !cache.window_mapped(x[1]));
    assert!(cache.window_mapped(x[10]) && cache.window_mapped(y[0]));
}

#[test]
fn a_pin_a_tile_deeper_on_the_same_path_merges_with_the_shallower_one() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed_over(32, 64, 24, 2, &gpu);
    let x: Vec<u32> = (0..7).map(|_| cache.alloc_block().unwrap()).collect();
    let y: Vec<u32> = (0..2).map(|_| cache.alloc_block().unwrap()).collect();
    cache.window_hold(&x, &gpu, 0).unwrap();
    cache.window_hold(&y, &gpu, 0).unwrap();
    // a prompt's checkpoint a block short of its tail, then the tail, whose
    // window starts a key tile later: one pin over both windows
    assert!(cache.window_pin(&x[..6], 2, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&x, 4, usize::MAX, PinRank::Prompt));
    assert_eq!(cache.window_pins(), 1);
    assert!(cache.window_pin(&y, 0, usize::MAX, PinRank::Prompt));
    cache.window_release(&x, &gpu, 0).unwrap();
    cache.window_release(&y, &gpu, 0).unwrap();
    // both resume points keep their rows, and the refs balance
    assert!((2..7).all(|i| cache.window_mapped(x[i])));
    assert!(!cache.window_mapped(x[0]) && !cache.window_mapped(x[1]));
    assert_eq!(cache.ref_count(x[2]), 2);
    // a resume at the shallower point refreshes the merged pin
    cache.window_touch(&x[2..6]);
    let z = cache.alloc_block().unwrap();
    cache.window_hold(&[z], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[z], 0, usize::MAX, PinRank::Prompt));
    cache.window_release(&[z], &gpu, 0).unwrap();
    assert!(cache.window_mapped(x[2]) && !cache.window_mapped(y[0]));
}

#[test]
fn pinning_a_window_already_pinned_refreshes_it() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 2, &gpu);
    let blocks: Vec<u32> = (0..3).map(|_| cache.alloc_block().unwrap()).collect();
    cache.window_hold(&blocks, &gpu, 0).unwrap();
    assert!(cache.window_pin(&blocks[..1], 0, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&blocks[1..2], 0, usize::MAX, PinRank::Prompt));
    // the same window again (a second sequence on the same prefix): one pin,
    // one ref, and it is now the newest
    assert!(cache.window_pin(&blocks[..1], 0, usize::MAX, PinRank::Prompt));
    assert_eq!(cache.window_pins(), 2);
    assert_eq!(cache.ref_count(blocks[0]), 2);
    assert!(cache.window_pin(&blocks[2..], 0, usize::MAX, PinRank::Prompt));
    cache.window_release(&blocks, &gpu, 0).unwrap();
    assert!(cache.window_mapped(blocks[0]) && !cache.window_mapped(blocks[1]));
}

#[test]
fn a_window_in_generated_text_goes_before_an_older_prompt_window() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 2, &gpu);
    let blocks: Vec<u32> = (0..3).map(|_| cache.alloc_block().unwrap()).collect();
    cache.window_hold(&blocks, &gpu, 0).unwrap();
    // a prompt's tail, then the answer's snapshot further along the same
    // blocks: the answer does not supersede the prompt, it goes first
    assert!(cache.window_pin(&blocks[..1], 0, usize::MAX, PinRank::Prompt));
    assert!(cache.window_pin(&blocks[..2], 1, usize::MAX, PinRank::Generated));
    assert!(cache.window_pin(&blocks[2..], 0, usize::MAX, PinRank::Prompt));
    cache.window_release(&blocks, &gpu, 0).unwrap();
    assert!(cache.window_mapped(blocks[0]) && cache.window_mapped(blocks[2]));
    assert!(!cache.window_mapped(blocks[1]));
}

#[test]
fn the_next_turn_supersedes_a_tail_whose_last_block_it_never_shares() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed_over(32, 64, 24, 2, &gpu);
    // y pins, then turn 1 pins its tail: blocks 2..6 of its path, the last
    // one partly written (5 whole blocks)
    let turn1: Vec<u32> = (0..6).map(|_| cache.alloc_block().unwrap()).collect();
    let y = cache.alloc_block().unwrap();
    cache.window_hold(&turn1, &gpu, 0).unwrap();
    cache.window_hold(&[y], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[y], 0, 1, PinRank::Prompt));
    assert!(cache.window_pin(&turn1, 2, 5, PinRank::Prompt));
    // turn 2 shares turn 1's whole blocks, then goes on in blocks of its own
    let mut turn2 = turn1[..5].to_vec();
    turn2.extend((0..9).map(|_| cache.alloc_block().unwrap()));
    cache.window_hold(&turn2[5..], &gpu, 0).unwrap();
    assert!(cache.window_pin(&turn2, 10, 14, PinRank::Prompt));
    // turn 1's window is superseded: it goes before y's older pin
    cache.window_release(&turn1, &gpu, 0).unwrap();
    cache.window_release(&turn2[5..], &gpu, 0).unwrap();
    cache.window_release(&[y], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(turn1[2]) && !cache.window_mapped(turn1[5]));
    assert!(cache.window_mapped(y) && cache.window_mapped(turn2[10]));
}

#[test]
fn a_merge_never_spans_rows_that_are_gone() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed_over(32, 64, 24, 4, &gpu);
    let b: Vec<u32> = (0..2).map(|_| cache.alloc_block().unwrap()).collect();
    let x: Vec<u32> = (0..4).map(|_| cache.alloc_block().unwrap()).collect();
    cache.window_hold(&b, &gpu, 0).unwrap();
    cache.window_hold(&x[2..], &gpu, 0).unwrap();
    // a pin with one whole block, then a near pin along it whose window
    // starts past two blocks nothing holds
    assert!(cache.window_pin(&b, 0, 1, PinRank::Prompt));
    let path = [b[0], x[0], x[1], x[2], x[3]];
    assert!(cache.window_pin(&path, 3, 5, PinRank::Prompt));
    // the first pin is superseded instead, and nothing was lent for x[0..2]
    assert_eq!(cache.window_pins(), 2);
    assert!(!cache.window_mapped(x[0]) && !cache.window_mapped(x[1]));
    assert_eq!(cache.ref_count(x[2]), 2);
}
