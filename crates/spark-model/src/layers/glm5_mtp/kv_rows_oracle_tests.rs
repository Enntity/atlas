// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn fixture(run: impl FnOnce(&mut PagedKvCache, &MockGpuBackend)) {
    let gpu = MockGpuBackend::new();
    let config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut cache = PagedKvCache::new(config, 4, &gpu).unwrap();
    for block in 0..4 {
        assert_eq!(cache.alloc_block().unwrap(), block);
        cache
            .write_block(
                0,
                block,
                &vec![block as u8; 16384],
                &vec![block as u8 + 8; 16384],
                &gpu,
            )
            .unwrap();
    }
    run(&mut cache, &gpu);
}

fn write_rows(cache: &PagedKvCache, gpu: &dyn GpuBackend, slots: &[i64]) -> Result<()> {
    for (row, &slot) in slots.iter().enumerate() {
        let block = slot as u32 / 16;
        let offset = slot as usize % 16 * 1024;
        gpu.copy_h2d(
            &vec![0x20 + row as u8; 1024],
            cache.k_cache_ptr(0, block).offset(offset),
        )?;
        gpu.copy_h2d(
            &vec![0x30 + row as u8; 1024],
            cache.v_cache_ptr(0, block).offset(offset),
        )?;
    }
    Ok(())
}

#[test]
fn oracle_compares_full_rows_and_preserves_every_guard_without_gpu_allocations() {
    fixture(|cache, gpu| {
        for slots in [vec![47, 0, 1, 2], vec![0], vec![16, 17, 18], vec![62, 63]] {
            let layout = OracleLayout::new(cache, &slots, 1, true).unwrap();
            let before = snapshot(cache, gpu, &[0, 1, 2, 3]).unwrap();
            let allocs = gpu.alloc_count();
            verify(
                cache,
                gpu,
                0,
                &layout,
                |cache| write_rows(cache, gpu, &layout.reference),
                |cache| write_rows(cache, gpu, &slots),
            )
            .unwrap();
            assert_eq!(gpu.alloc_count(), allocs);
            let after = snapshot(cache, gpu, &[0, 1, 2, 3]).unwrap();
            check_guards(&before, &after, &slots).unwrap();
        }
    });
}

#[test]
fn oracle_restores_reference_errors_candidate_errors_and_numerical_or_guard_mismatch() {
    fixture(|cache, gpu| {
        let layout = OracleLayout::new(cache, &[47, 0, 1, 2], 1, true).unwrap();
        for failure in 0..6 {
            let before = snapshot(cache, gpu, &[0, 1, 2, 3]).unwrap();
            let result = verify(
                cache,
                gpu,
                0,
                &layout,
                |cache| {
                    write_rows(cache, gpu, &layout.reference)?;
                    if failure == 0 {
                        anyhow::bail!("reference failed after write");
                    }
                    if failure == 4 {
                        gpu.copy_h2d(&[0xff], cache.k_cache_ptr(0, 1).offset(8192))?;
                    }
                    if failure == 5 {
                        gpu.copy_h2d(&0x7fc0u16.to_le_bytes(), cache.k_cache_ptr(0, 1))?;
                    }
                    Ok(())
                },
                |cache| {
                    write_rows(cache, gpu, &layout.actual)?;
                    if failure == 1 {
                        anyhow::bail!("candidate failed after write");
                    }
                    if failure == 2 {
                        gpu.copy_h2d(&[0xff], cache.v_cache_ptr(0, 2).offset(15 * 1024))?;
                    }
                    if failure == 3 {
                        gpu.copy_h2d(&[0xff], cache.k_cache_ptr(0, 0).offset(8192))?;
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "failure mode {failure}");
            assert_eq!(snapshot(cache, gpu, &[0, 1, 2, 3]).unwrap(), before);
        }
    });
}

#[test]
fn oracle_rejects_unsupported_capability_and_malformed_spans_before_writes() {
    fixture(|cache, gpu| {
        let before = (
            gpu.alloc_count(),
            gpu.launch_count(),
            gpu.d2h_blocking_count(),
        );
        for (slots, block, capable) in [
            (vec![], 0, true),
            (vec![0; 5], 0, true),
            (vec![0], 4, true),
            (vec![0], 0, false),
            (vec![-1], 0, true),
            (vec![64], 0, true),
            (vec![0, 0], 0, true),
            (vec![0, 16, 32, 48], 0, true),
        ] {
            assert!(OracleLayout::new(cache, &slots, block, capable).is_err());
        }
        assert!(
            validate_pool_spans(
                DeviceSpan {
                    ptr: DevicePtr(100),
                    bytes: 20
                },
                DeviceSpan {
                    ptr: DevicePtr(110),
                    bytes: 20
                }
            )
            .is_err()
        );
        assert!(
            validate_pool_spans(
                DeviceSpan {
                    ptr: DevicePtr(u64::MAX - 1),
                    bytes: 20
                },
                DeviceSpan {
                    ptr: DevicePtr(100),
                    bytes: 20
                }
            )
            .is_err()
        );
        validate_pool_spans(
            DeviceSpan {
                ptr: DevicePtr(100),
                bytes: 20,
            },
            DeviceSpan {
                ptr: DevicePtr(120),
                bytes: 20,
            },
        )
        .unwrap();
        assert_eq!(
            (
                gpu.alloc_count(),
                gpu.launch_count(),
                gpu.d2h_blocking_count()
            ),
            before
        );
    });
}

#[test]
fn restoration_attempts_remaining_blocks_and_reports_unrecoverable_copy_errors() {
    fixture(|cache, gpu| {
        let layout = OracleLayout::new(cache, &[47, 0, 1, 2], 1, true).unwrap();
        let original = snapshot(cache, gpu, &layout.blocks).unwrap();
        let result = verify(
            cache,
            gpu,
            0,
            &layout,
            |cache| {
                for &block in &layout.blocks {
                    gpu.copy_h2d(&[0xab; 1024], cache.k_cache_ptr(0, block))?;
                }
                // Simulate an inaccessible V allocation after writes have begun.
                // Every K side must still be restored despite each V copy failing.
                gpu.free(cache.v_cache_ptr(0, 0))?;
                anyhow::bail!("injected device allocation failure")
            },
            |_| panic!("candidate cannot run after a reference failure"),
        );
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("injected device allocation failure"));
        assert!(error.contains("restoration failed"));
        assert!(error.contains("must not resume"));
        for block in original {
            let mut actual = vec![0; block.k.len()];
            gpu.copy_d2h(cache.k_cache_ptr(0, block.block), &mut actual)
                .unwrap();
            assert_eq!(actual, block.k);
        }
    });
}
