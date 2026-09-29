// SPDX-License-Identifier: AGPL-3.0-only

//! Two-rank fast-path helpers on [`NcclBackend`]: the BF16 add launch, the
//! RDMA pair connect/dispatch, and the peer lifeline accessor.

#[cfg(atlas_rdma_verbs)]
use super::{ALL_REDUCE_DTYPE_BYTES, ensure_payload_fits, rdma_pair};
use super::{NcclBackend, cuLaunchKernel};
use crate::peer_lifeline::PeerLifeline;
use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::Ordering;

impl NcclBackend {
    /// `dst[i] += src[i]` over `count` BF16 values on `stream`.
    pub(super) fn launch_add(&self, dst: u64, src: u64, count: usize, stream: u64) -> Result<()> {
        let kernel = self.add_kernel.load(Ordering::Relaxed);
        if kernel == 0 {
            anyhow::bail!("bf16_add_inplace kernel not set — call set_add_kernel() first");
        }
        let threads: u32 = 256;
        let blocks: u32 = (count as u32).div_ceil(threads);
        let mut p_dst = dst;
        let mut p_src = src;
        let mut p_n = count as i32;
        let mut params: [*mut c_void; 3] = [
            &mut p_dst as *mut u64 as *mut c_void,
            &mut p_src as *mut u64 as *mut c_void,
            &mut p_n as *mut i32 as *mut c_void,
        ];
        let status = unsafe {
            cuLaunchKernel(
                kernel,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if status != 0 {
            anyhow::bail!("cuLaunchKernel (bf16_add_inplace) failed: status {status}");
        }
        Ok(())
    }

    /// 2-rank all-reduce over the RDMA pair on `stream` when it is up and the
    /// payload qualifies; `false` means the caller takes the NCCL path.
    pub(super) fn try_rdma_all_reduce(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma {
            ensure_payload_fits(bytes, self.recv_capacity, self.rank, self.world_size)?;
            if bytes == 0 {
                return Ok(true);
            }
            if rdma.exchange(ptr, ptr, bytes, stream, |dst, src, len| {
                self.launch_add(dst, src, len / ALL_REDUCE_DTYPE_BYTES, stream)
            })? {
                return Ok(true);
            }
        }
        let _ = (ptr, bytes, stream);
        Ok(false)
    }

    /// Reduce-scatter/all-gather step over the RDMA pair (see
    /// `CommBackend::exchange_async`); `false` when the pair is not up.
    pub(super) fn try_rdma_exchange(
        &self,
        send: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma
            && bytes > 0
            && (!add || self.add_kernel.load(Ordering::Relaxed) != 0)
        {
            return rdma.exchange(send, dst, bytes, stream, |dst, src, len| {
                if add {
                    self.launch_add(dst, src, len / ALL_REDUCE_DTYPE_BYTES, stream)
                } else {
                    rdma_pair::copy_async(dst, src, len, stream)
                }
            });
        }
        let _ = (send, dst, bytes, add, stream);
        Ok(false)
    }

    /// Connections to this rank's peers from the initial bootstrap; watch them
    /// to learn when a peer process exits.
    pub fn peer_lifeline(&self) -> &PeerLifeline {
        &self.peer_lifeline
    }

    /// RDMA pair payload capacity, when the pair is up.
    pub(super) fn rdma_capacity(&self) -> Option<usize> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma {
            return Some(rdma.capacity());
        }
        None
    }

    /// The RDMA pair for a 2-rank world when `ATLAS_RDMA_ALLREDUCE=1`.
    #[cfg(atlas_rdma_verbs)]
    pub(super) fn connect_rdma(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
        recv_capacity: usize,
    ) -> Result<Option<rdma_pair::RdmaPair>> {
        // Port +1 is the reconnect bootstrap; +2 carries the RDMA identities.
        let rdma = if world_size == 2 && rdma_pair::RdmaPair::requested() {
            Some(rdma_pair::RdmaPair::connect(
                rank,
                master_addr,
                master_port.wrapping_add(2),
                recv_capacity.next_multiple_of(64),
            )?)
        } else {
            None
        };
        Ok(rdma)
    }
}
