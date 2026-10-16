// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_RDMA_KERNEL_FLAGS=1`: the RDMA pair exchange with the `ready`
//! publish and `arrived` wait done by kernels (system-scope release/acquire)
//! instead of stream memory operations.

use super::{
    ARRIVED, Job, READY, RdmaPair, cu, cuMemcpyAsync, cuStreamIsCapturing, flag_off, recv_off,
    segments, send_off,
};
use anyhow::Result;
use std::sync::atomic::Ordering;

impl RdmaPair {
    /// Whether `ATLAS_RDMA_KERNEL_FLAGS=1`: publish `ready` and wait for
    /// `arrived` with kernels (system-scope release/acquire) instead of
    /// stream memory operations.
    pub(in crate::nccl_backend) fn kernel_flags() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("ATLAS_RDMA_KERNEL_FLAGS").as_deref() == Ok("1"))
    }

    /// `exchange` with kernel-side flags for single-segment payloads; larger
    /// (segmented) payloads take the stream-memop `exchange`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::nccl_backend) fn exchange_kernel_flags(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        stream: u64,
        signal: impl Fn(u64, u64) -> Result<()>,
        land_wait: impl Fn(u64, u64, usize, u64, u64) -> Result<()>,
        land: impl Fn(u64, u64, usize) -> Result<()>,
    ) -> Result<bool> {
        if bytes > self.capacity || segments(bytes).len() != 1 {
            return self.exchange(src, dst, bytes, stream, land);
        }
        let mut capturing = 0i32;
        cu(
            unsafe { cuStreamIsCapturing(stream, &mut capturing) },
            "cuStreamIsCapturing",
        )?;
        if capturing != 0 {
            return Ok(false);
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let slot = (seq & 1) as usize;
        let flags = self.dev + flag_off(self.capacity) as u64;
        let send = self.dev + send_off(self.capacity, slot) as u64;
        let recv = self.dev + recv_off(self.capacity, slot) as u64;
        let first = self.segs.fetch_add(1, Ordering::Relaxed);
        cu(
            unsafe { cuMemcpyAsync(send, src, bytes, stream) },
            "cuMemcpyAsync(RDMA send slot)",
        )?;
        signal(flags + READY as u64, first + 1)?;
        self.jobs.lock().push_back(Job {
            seq,
            first,
            parts: vec![(0, bytes)],
        });
        land_wait(dst, recv, bytes, flags + ARRIVED as u64, first + 1)?;
        Ok(true)
    }
}
