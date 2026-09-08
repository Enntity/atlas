// SPDX-License-Identifier: AGPL-3.0-only
//! Actual checked pool-reader tests; no CUDA numerics or arbitrary raw owner.
use super::*;
#[path = "hidden_trace_kv_test_gpu.rs"]
mod support;
use std::sync::atomic::Ordering;
use support::{Event, Gpu, cache, context};

fn reference(rows: usize, appended: bool) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(if appended {
        b"atlas/glm53/mtp-kv/appended/v1\0".as_slice()
    } else {
        b"atlas/glm53/mtp-kv/prefix/v1\0".as_slice()
    });
    hash.update((rows as u64).to_le_bytes());
    hash.update(512u32.to_le_bytes());
    hash.update(2u32.to_le_bytes());
    let range = if appended { rows..rows + 1 } else { 0..rows };
    for row in range {
        for side in 0..2 {
            let bytes: Vec<_> = (0..1024)
                .map(|i| (row.wrapping_mul(73) ^ i ^ (i >> 8) ^ (side * 31)) as u8)
                .collect();
            hash.update(bytes);
        }
    }
    hash.finalize().into()
}

#[test]
fn actual_reader_full_valid_rows_maps_and_exact_transfer_budgets() {
    for rows in [1, 15, 16, 17, 148, 1984, 2043] {
        let mut hashes = Vec::new();
        for reverse in [false, true] {
            let gpu = Gpu::new();
            context(&gpu, |ctx| {
                let (cache, blocks) = cache(&gpu, rows, reverse);
                let mut probe = Probe::before(&cache, &blocks, rows, ctx, 7).unwrap();
                probe.after(&cache, &blocks, rows, ctx, 7).unwrap();
                assert_eq!(probe.prefix, reference(rows, false));
                assert_eq!(probe.appended, Some(reference(rows, true)));
                let mut expected = Vec::new();
                for logical in (0..rows).step_by(16) {
                    let n = (rows - logical).min(16) * SIDE_ROW;
                    for p in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)] {
                        expected.push(Event::Read(
                            p.offset(blocks[logical / 16] as usize * SIDE_BLOCK),
                            n,
                            7,
                        ));
                    }
                }
                for p in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)] {
                    expected.push(Event::Read(
                        p.offset(blocks[rows / 16] as usize * SIDE_BLOCK + rows % 16 * SIDE_ROW),
                        SIDE_ROW,
                        7,
                    ));
                }
                assert_eq!(*gpu.events.lock(), expected);
                let bytes: usize = expected
                    .iter()
                    .map(|e| {
                        if let Event::Read(_, n, _) = e {
                            *n
                        } else {
                            panic!("only reads")
                        }
                    })
                    .sum();
                assert_eq!(bytes, (rows + 1) * 2048);
                assert_eq!(expected.len(), 2 * rows.div_ceil(16) + 2);
                hashes.push((probe.prefix, probe.block_map));
            });
        }
        assert_eq!(hashes[0].0, hashes[1].0);
        if rows >= 16 {
            assert_ne!(hashes[0].1, hashes[1].1);
        }
    }
}

#[test]
fn valid_boundary_changes_count_but_unwritten_tails_do_not() {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        let rows = 17;
        let (cache, blocks) = cache(&gpu, rows, true);
        let baseline = Probe::before(&cache, &blocks, rows, ctx, 7).unwrap().prefix;
        for side in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)] {
            for logical in [0, rows - 1] {
                for byte in [0, 1023] {
                    let p = side.offset(
                        blocks[logical / 16] as usize * SIDE_BLOCK + logical % 16 * SIDE_ROW + byte,
                    );
                    let mut old = [0];
                    gpu.inner.copy_d2h(p, &mut old).unwrap();
                    gpu.inner.copy_h2d(&[old[0] ^ 1], p).unwrap();
                    assert_ne!(
                        Probe::before(&cache, &blocks, rows, ctx, 7).unwrap().prefix,
                        baseline
                    );
                    gpu.inner.copy_h2d(&old, p).unwrap();
                }
            }
            let unused = side.offset(blocks[rows / 16] as usize * SIDE_BLOCK + 15 * SIDE_ROW);
            gpu.inner.copy_h2d(&[0x92; SIDE_ROW], unused).unwrap();
            assert_eq!(
                Probe::before(&cache, &blocks, rows, ctx, 7).unwrap().prefix,
                baseline
            );
        }
    });
}

#[test]
fn every_prefix_and_appended_copy_fault_stops_without_backend_side_effects() {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        let rows = 2043;
        let (cache, blocks) = cache(&gpu, rows, true);
        for fail in 1..=258 {
            gpu.events.lock().clear();
            gpu.fail.store(fail, Ordering::Relaxed);
            let result = Probe::before(&cache, &blocks, rows, ctx, 7)
                .and_then(|mut probe| probe.after(&cache, &blocks, rows, ctx, 7));
            assert!(result.unwrap_err().to_string().contains("injected KV copy"));
            assert_eq!(gpu.events.lock().len(), fail);
            assert!(
                gpu.events
                    .lock()
                    .iter()
                    .all(|e| matches!(e, Event::Read(..)))
            );
        }
    });
}

#[test]
fn actual_pool_geometry_and_malformed_allocator_reject_before_any_copy() {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        for fault in 0..12 {
            let mut config = KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            };
            match fault {
                0 => config.block_size = 32,
                1 => config.num_layers = 2,
                2 => config.head_dim = 256,
                3 => config.num_kv_heads = 2,
                4 => config.dtype = KvCacheDtype::Fp8,
                5 => config.layer_dtypes = vec![KvCacheDtype::Fp8],
                6 => config.layer_dims = vec![(1, 512)],
                7 => config.cache_blocks_per_seq = Some(8),
                8..=11 => gpu.bad_allocation.store(fault - 7, Ordering::Relaxed),
                _ => unreachable!(),
            }
            let mut wrong = PagedKvCache::new(config, 128, &gpu).unwrap();
            let blocks = [wrong.alloc_block().unwrap(), wrong.alloc_block().unwrap()];
            gpu.events.lock().clear();
            assert!(Probe::before(&wrong, &blocks, 17, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
            gpu.bad_allocation.store(0, Ordering::Relaxed);
        }
        let (cache, blocks) = cache(&gpu, 17, false);
        for fault in 0..4 {
            let mut config = ctx.config.clone();
            match fault {
                0 => config.index_topk = 17,
                1 => config.kv_lora_rank = 256,
                2 => config.qk_rope_head_dim = 64,
                _ => config.model_type = "not-glm".into(),
            }
            let bad = ForwardContext {
                config: &config,
                midchunk_capture: None,
                ..*ctx
            };
            gpu.events.lock().clear();
            assert!(Probe::before(&cache, &blocks, 17, &bad, 7).is_err());
            assert!(gpu.events.lock().is_empty());
        }
    });
}

#[test]
fn actual_owner_cap_map_and_capture_faults_refuse_before_reads() {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        for rows in [0, 2044, usize::MAX] {
            let (cache, blocks) = cache(&gpu, 17, false);
            assert!(Probe::before(&cache, &blocks, rows, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
        }
        for fault in 0..6 {
            let (mut cache, mut blocks) = cache(&gpu, 17, false);
            match fault {
                0 => {
                    blocks.pop();
                }
                1 => blocks[1] = blocks[0],
                2 => blocks[1] = 128,
                3 => cache.inc_ref(blocks[0]),
                4 => {
                    cache.free_block(blocks[0]);
                }
                5 => gpu.capturing.store(true, Ordering::Relaxed),
                _ => unreachable!(),
            }
            assert!(Probe::before(&cache, &blocks, 17, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
            gpu.capturing.store(false, Ordering::Relaxed);
        }
    });
}

#[test]
fn changed_post_body_pool_map_cursor_or_duplicate_cannot_be_read() {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        for fault in 0..5 {
            let (cache, mut blocks) = cache(&gpu, 17, false);
            let mut probe = Probe::before(&cache, &blocks, 17, ctx, 7).unwrap();
            let (other, _) = support::cache(&gpu, 17, false);
            if fault == 0 {
                blocks.reverse();
            }
            if fault == 4 {
                probe.after(&cache, &blocks, 17, ctx, 7).unwrap();
            }
            gpu.events.lock().clear();
            let result = probe.after(
                if fault == 1 { &other } else { &cache },
                &blocks,
                if fault == 2 { 18 } else { 17 },
                ctx,
                if fault == 3 { 8 } else { 7 },
            );
            assert!(result.is_err());
            assert!(gpu.events.lock().is_empty());
        }
    });
}

#[test]
fn future_blocks_capacity_capture_and_post_failure_are_bounded() {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        let (mut cache, mut blocks) = cache(&gpu, 17, false);
        blocks.push(cache.alloc_block().unwrap());
        let baseline = Probe::before(&cache, &blocks, 17, ctx, 7).unwrap().prefix;
        for pool in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)] {
            gpu.inner
                .copy_h2d(
                    &[0x81; SIDE_BLOCK],
                    pool.offset(blocks[2] as usize * SIDE_BLOCK),
                )
                .unwrap();
        }
        assert_eq!(
            Probe::before(&cache, &blocks, 17, ctx, 7).unwrap().prefix,
            baseline
        );
        let capturing = ForwardContext {
            graph_capture: true,
            midchunk_capture: None,
            ..*ctx
        };
        gpu.events.lock().clear();
        assert!(Probe::before(&cache, &blocks, 17, &capturing, 7).is_err());
        assert!(gpu.events.lock().is_empty());
        for capacity in [0, 1, 129] {
            let cfg = KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            };
            let mut wrong = PagedKvCache::new(cfg, capacity, &gpu).unwrap();
            let wrong_blocks: Vec<_> = (0..capacity.min(2))
                .map(|_| wrong.alloc_block().unwrap())
                .collect();
            gpu.events.lock().clear();
            assert!(Probe::before(&wrong, &wrong_blocks, 17, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
        }
        for fail in 1..=2 {
            let mut probe = Probe::before(&cache, &blocks, 17, ctx, 7).unwrap();
            gpu.events.lock().clear();
            gpu.fail.store(fail, Ordering::Relaxed);
            assert!(probe.after(&cache, &blocks, 17, ctx, 7).is_err());
            assert!(probe.appended.is_none());
            gpu.events.lock().clear();
            gpu.fail.store(usize::MAX, Ordering::Relaxed);
            assert!(probe.after(&cache, &blocks, 17, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
        }
    });
}
