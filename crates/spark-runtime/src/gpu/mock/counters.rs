// SPDX-License-Identifier: AGPL-3.0-only
//! Test-assertion accessors of [`MockGpuBackend`]: operation counters,
//! queued failures and recorded-request snapshots.

use std::sync::atomic::Ordering;

use super::{MockGpuBackend, MockLaunch};
use crate::gpu::Bf16Gemm;

impl MockGpuBackend {
    /// `(streams, events)` created and not yet destroyed: the leak check of
    /// whatever owns them.
    pub fn live_streams_and_events(&self) -> (usize, usize) {
        (
            self.live_streams.load(Ordering::Relaxed),
            self.live_events.load(Ordering::Relaxed),
        )
    }

    /// Queue one `free` failure: the next `free` call returns an error and
    /// consumes one queued failure. Lets cleanup-path tests prove a failed
    /// free keeps its resource observable/retryable instead of silently
    /// leaking (queue N to fail N consecutive frees).
    pub fn fail_next_free(&self) {
        self.pending_free_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// `destroy_graph` calls so far — the tripwire proving a cleanup error
    /// path does NOT destroy pooled graphs.
    pub fn destroy_graph_count(&self) -> usize {
        self.destroyed_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `free` calls whose pointer was not the base of a live allocation: a
    /// double free, or an offset view into a block. The real backend rejects
    /// both with `CUDA_ERROR_INVALID_VALUE` and puts the pointer back on its
    /// cleanup ledger; the mock accepts them silently, so this counter is how
    /// a teardown test sees them.
    pub fn invalid_free_count(&self) -> usize {
        self.invalid_frees.load(Ordering::Relaxed)
    }

    /// Required kernel lookups in order, including failed resolution attempts.
    pub fn kernel_requests_snapshot(&self) -> Vec<(String, String)> {
        self.kernel_requests.lock().clone()
    }

    /// Fail the next required lookup, so pre-allocation closure checks are tested.
    pub fn fail_next_kernel(&self) {
        self.pending_kernel_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn fail_next_bf16_gemm(&self) {
        self.pending_gemm_failures.fetch_add(1, Ordering::Relaxed);
    }
    pub fn bf16_gemm_requests_snapshot(&self) -> Vec<Bf16Gemm> {
        self.bf16_gemms.lock().clone()
    }
    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }

    pub fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    /// `synchronize` calls so far — a proxy for "full stream drains", the cost
    /// a batched gather exists to amortize.
    pub fn sync_count(&self) -> usize {
        self.syncs.load(Ordering::Relaxed)
    }

    /// BLOCKING `copy_d2h` calls (each one drains the stream on the real
    /// backend). A bulk gather must have zero of these.
    pub fn d2h_blocking_count(&self) -> usize {
        self.d2h_blocking.load(Ordering::Relaxed)
    }

    /// `copy_d2h_async` calls (enqueue-only).
    pub fn d2h_async_count(&self) -> usize {
        self.d2h_async.load(Ordering::Relaxed)
    }

    /// `copy_d2d` + `copy_d2d_async` calls so far — one eager launch each on
    /// the real backend.
    pub fn d2d_count(&self) -> usize {
        self.d2d.load(Ordering::Relaxed)
    }

    /// `copy_d2d_2d_async` calls so far — one `cudaMemcpy2DAsync` each,
    /// whatever the row count.
    pub fn d2d_2d_count(&self) -> usize {
        self.d2d_2d.load(Ordering::Relaxed)
    }

    /// Pitched host↔device copies so far (either direction).
    pub fn host_pitched_count(&self) -> usize {
        self.host_pitched.load(Ordering::Relaxed)
    }

    /// `memset_async` calls so far.
    pub fn memset_count(&self) -> usize {
        self.memsets.load(Ordering::Relaxed)
    }

    /// `alloc_host_pinned` calls — the tripwire for a staging buffer that is
    /// re-allocated per event instead of reused.
    pub fn host_pinned_alloc_count(&self) -> usize {
        self.host_pinned_allocs.load(Ordering::Relaxed)
    }

    /// Every launch recorded so far, in dispatch order. Lets a test assert
    /// WHICH kernel shape ran (grid/block signature), not just how many —
    /// the mock's `kernel()` hands out one shared handle, so geometry is
    /// the only per-launch identity available.
    pub fn launches_snapshot(&self) -> Vec<MockLaunch> {
        self.launches.lock().clone()
    }
}
