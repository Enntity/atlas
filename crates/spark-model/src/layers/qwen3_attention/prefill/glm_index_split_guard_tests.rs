// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the `glm_index_split` range guard: what it launches behind
//! each exchange, and when the clamped-id counter is read and fails a chunk.
//! The kernel itself is checked on a GPU by
//! `scripts/dev/glm_index_split_guard_bench.cu`.

use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::bail;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{KernelArg, KernelHandle};

use super::super::tests::{Pair, split, with_ctx};
use super::super::{IndexSplit, OwnerRows};
use super::*;

/// The owner under test: 258 rows of 16 bytes (four ids), so a quarter is 64
/// rows, continuing a sequence at token 8196.
const ROWS: usize = 258;
const RB: usize = 16;
const Q: usize = 64;
const END: usize = 8196 + ROWS;
const KERNEL: u64 = 0x51d;

/// One launch: `(kernel, grid, block, stream, arguments)`.
type Launch = (u64, [u32; 3], [u32; 3], u64, Vec<u64>);

/// The mock GPU, recording each launch's arguments and each synchronized
/// stream.
#[derive(Default)]
struct Capture {
    inner: MockGpuBackend,
    launches: Mutex<Vec<Launch>>,
    synced: Mutex<Vec<u64>>,
}

impl GpuBackend for Capture {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.synced.lock().unwrap().push(stream);
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        assert_eq!((module, symbol), ("glm_indexer", "glm_index_clamp_ids"));
        Ok(KernelHandle(KERNEL))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut c_void],
    ) -> Result<()> {
        bail!("expected the typed launch")
    }
    fn launch_typed(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let word = |a: &KernelArg<'_>| match a {
            KernelArg::Buffer(p) => p.0,
            KernelArg::Bytes(b) => b.iter().rev().fold(0, |w, &byte| w << 8 | byte as u64),
        };
        let args = args.iter().map(word).collect();
        let launch = (kernel.0, grid, block, stream, args);
        self.launches.lock().unwrap().push(launch);
        Ok(())
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}

/// Run `f` on `rank` with a fresh chunk's guard state, the owner's rows and
/// an exchange of them.
fn on_rank<R>(
    gpu: &Capture,
    rank: usize,
    end: usize,
    f: impl FnOnce(&ForwardContext, DevicePtr, &dyn Fn() -> Result<()>) -> R,
) -> R {
    PENDING.set(false);
    let pair = Pair::new(&gpu.inner, rank);
    with_ctx(&gpu.inner, Some(&pair), false, 2, |ctx| {
        let ctx = ForwardContext {
            gpu,
            midchunk_capture: None,
            ..*ctx
        };
        let selected = ctx.buffers.expert_down_out();
        let owner = OwnerRows {
            selected,
            row_bytes: RB,
            end,
            scratch: selected.offset(ROWS * RB),
            inputs: [(selected, 0); 2],
        };
        let split: IndexSplit = split(ROWS, rank, false);
        f(&ctx, selected, &|| split.exchange(&owner, 0, &ctx, 7))
    })
}

#[test]
fn each_received_quarter_is_clamped_once_behind_its_exchange() {
    for rank in 0..2 {
        let gpu = Capture::default();
        on_rank(&gpu, rank, END, |ctx, selected, exchange| {
            let memsets = gpu.inner.memset_count();
            exchange().unwrap();
            let at = |row: usize| selected.offset(row * RB).0;
            // Rank 0 receives Q1 and Q2, rank 1 Q0 and Q3: four ids a row.
            let received = if rank == 0 {
                [at(Q), at(2 * Q)]
            } else {
                [at(0), at(3 * Q)]
            };
            let ids = (Q * RB / 4) as u64;
            let count = counter(ctx.gpu).unwrap().0;
            let want = received.map(|rows| {
                let args = vec![rows, ids, END as u64, count];
                (KERNEL, [1, 1, 1], [256, 1, 1], 7, args)
            });
            assert_eq!(*gpu.launches.lock().unwrap(), want, "rank {rank}");
            // The count starts from zero once, and nothing waits on the device.
            assert_eq!(gpu.inner.memset_count() - memsets, 1);
            assert_eq!(gpu.inner.read_alloc(DevicePtr(count)), Some(vec![0; 4]));
            assert!(gpu.synced.lock().unwrap().is_empty());
        });
    }
}

#[test]
fn the_grid_covers_every_received_id() {
    assert_eq!(grid(1), 1);
    assert_eq!(grid(IDS_PER_CTA), 1);
    assert_eq!(grid(IDS_PER_CTA + 1), 2);
    // A quarter of a 4096-row owner of 2051-id rows.
    assert_eq!(grid(1024 * 2051), 1026);
}

#[test]
fn a_bound_past_the_kernels_range_fails_before_any_launch() {
    let gpu = Capture::default();
    on_rank(&gpu, 0, 1 << 31, |_, _, exchange| {
        assert!(exchange().is_err());
        assert!(gpu.launches.lock().unwrap().is_empty());
    });
}

#[test]
fn the_counter_is_read_once_a_chunk_and_only_after_a_split() {
    let gpu = Capture::default();
    PENDING.set(false);
    // No split in this chunk: no counter, no read, no wait.
    let allocs = gpu.inner.alloc_count();
    check_index_split_rows(&gpu, 7).unwrap();
    assert_eq!(gpu.inner.alloc_count(), allocs);
    assert_eq!(gpu.inner.d2h_blocking_count(), 0);
    assert!(gpu.synced.lock().unwrap().is_empty());
    on_rank(&gpu, 1, END, |ctx, _, exchange| {
        // Two owners (or layers) of one chunk share the count.
        let memsets = gpu.inner.memset_count();
        exchange().unwrap();
        exchange().unwrap();
        assert_eq!(gpu.inner.memset_count() - memsets, 1);
        assert_eq!(gpu.launches.lock().unwrap().len(), 4);
        let reads = gpu.inner.d2h_blocking_count();
        check_index_split_rows(ctx.gpu, 7).unwrap();
        check_index_split_rows(ctx.gpu, 7).unwrap();
        assert_eq!(gpu.inner.d2h_blocking_count() - reads, 1);
        assert_eq!(*gpu.synced.lock().unwrap(), [7]);
    });
}

#[test]
fn clamped_ids_fail_the_chunk_naming_the_peer_and_the_next_chunk_starts_clean() {
    let gpu = Capture::default();
    on_rank(&gpu, 0, END, |ctx, _, exchange| {
        exchange().unwrap();
        // As the kernel leaves the count after clamping three ids.
        let count = counter(ctx.gpu).unwrap();
        gpu.copy_h2d(&3u32.to_le_bytes(), count).unwrap();
        let msg = format!("{:#}", check_index_split_rows(ctx.gpu, 7).unwrap_err());
        assert!(
            msg.contains(
                "the peer's index-split rows held 3 out-of-range token ids (a desynchronized peer)"
            ),
            "{msg}"
        );
        assert!(msg.contains("nothing was read out of range"), "{msg}");
        // The failed chunk's count does not reach the next one.
        check_index_split_rows(ctx.gpu, 7).unwrap();
        exchange().unwrap();
        check_index_split_rows(ctx.gpu, 7).unwrap();
    });
}
