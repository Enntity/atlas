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

/// The capturable collectives' kernel for `world_size` ranks (module
/// `rdma_oneshot`): `rdma_oneshot_bf16` for two, `rdma_chain_bf16` beyond.
pub(super) fn oneshot_kernel_name(world_size: usize) -> &'static str {
    if world_size > 2 {
        "rdma_chain_bf16"
    } else {
        "rdma_oneshot_bf16"
    }
}

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

    /// The one-shot channel (`rdma_pair::oneshot`) of the RDMA pair or chain,
    /// when `ATLAS_RDMA_ONESHOT=1`.
    #[cfg(atlas_rdma_verbs)]
    fn oneshot(&self) -> Option<&rdma_pair::OneShot> {
        let chain = self.chain.as_ref().map(|c| c.oneshot());
        chain.or_else(|| self.rdma.as_ref().and_then(|r| r.oneshot()))
    }

    /// One-shot exchange over the RDMA pair or all-reduce over the chain
    /// (`rdma_pair::oneshot`): on `stream`, graph-capturable, sharing one
    /// device sequence between eager calls and replays. `false` (nothing
    /// enqueued) when `ATLAS_RDMA_ONESHOT` is off, the kernel is missing or
    /// the size (or, on a chain, the call) is ineligible.
    pub(super) fn try_oneshot(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.oneshot() {
            return oneshot.enqueue(src, dst, bytes, add, stream);
        }
        let _ = (src, dst, bytes, add, stream);
        Ok(false)
    }

    /// A chain all-reduce of any size on `stream`: consecutive one-shot ops of
    /// at most the one-shot maximum (each capturable, like one). `false` off
    /// the chain or while its kernel is missing.
    pub(super) fn try_chain_all_reduce(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.all_reduce(ptr, bytes, stream);
        }
        let _ = (ptr, bytes, stream);
        Ok(false)
    }

    /// A chain all-gather of every rank's part (`CommBackend::gather_async`)
    /// on `stream`; `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_gather(
        &self,
        base: u64,
        stride: i64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.gather(base, stride, bytes, stream);
        }
        let _ = (base, stride, bytes, stream);
        Ok(false)
    }

    /// A chain broadcast of any size (`CommBackend::broadcast_async`) on
    /// `stream`; `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_broadcast_parts(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.broadcast(ptr, bytes, root, stream);
        }
        let _ = (ptr, bytes, root, stream);
        Ok(false)
    }

    /// A four-rank chain reduce-scatter (`CommBackend::reduce_scatter_async`,
    /// or with `fp8` `reduce_scatter_fp8_async`) on `stream`; `false` off the
    /// chain or when it does not apply.
    pub(super) fn try_chain_reduce_scatter(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        fp8: bool,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.reduce_scatter(seg0, stride, bytes, fp8, stream);
        }
        let _ = (seg0, stride, bytes, fp8, stream);
        Ok(false)
    }

    /// The chain's fused reduce-scatter (`CommBackend::rs_fused_begin` and
    /// the rest); `None` / `false` off the chain.
    pub(super) fn try_chain_rs_fused_begin(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
    ) -> Result<Option<crate::RsFused>> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.rs_fused_begin(seg0, stride, bytes);
        }
        let _ = (seg0, stride, bytes);
        Ok(None)
    }

    pub(super) fn try_chain_reduce_scatter_fused(
        &self,
        fused: &crate::RsFused,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.reduce_scatter_fused(fused, stream);
        }
        let _ = (fused, stream);
        Ok(false)
    }

    pub(super) fn try_chain_rs_fused_wait(
        &self,
        fused: &crate::RsFused,
        piece: usize,
        stream: u64,
    ) -> Result<()> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.rs_fused_wait(fused, piece, stream);
        }
        let _ = (fused, piece, stream);
        anyhow::bail!("fused reduce-scatter: no RDMA chain")
    }

    /// A chain all-to-all (`CommBackend::all_to_all_async`) on `stream`;
    /// `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_all_to_all(
        &self,
        planes: &[crate::AllToAllPlane],
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.all_to_all(planes, stream);
        }
        let _ = (planes, stream);
        Ok(false)
    }

    /// Whether the chain's reduce-scatter can run (its kernel is set and a
    /// slice fits the one-shot maximum).
    pub(super) fn chain_rs_ready(&self) -> bool {
        #[cfg(atlas_rdma_verbs)]
        if self.chain.is_some() {
            return rdma_pair::chain::RdmaChain::rs_slice(self.chain_oneshot_max_bytes()) > 0;
        }
        false
    }

    /// The chain's one-shot maximum (0 off the chain or while its kernel is
    /// missing).
    pub(super) fn chain_oneshot_max_bytes(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.oneshot().max_bytes();
        }
        0
    }

    /// A chain broadcast of rank `root`'s `ptr[..bytes]` on `stream` over the
    /// one-shot channel; `false` off the chain or when ineligible.
    pub(super) fn try_chain_broadcast(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.oneshot().broadcast(ptr, bytes, root, stream);
        }
        let _ = (ptr, bytes, root, stream);
        Ok(false)
    }

    /// Largest one-shot payload (0 when the channel is unavailable).
    pub(super) fn oneshot_max_bytes(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.oneshot() {
            return oneshot.max_bytes();
        }
        0
    }

    /// Provide the one-shot kernel ([`oneshot_kernel_name`]) to the channel,
    /// if it is up.
    pub(super) fn set_oneshot_kernel_handle(&self, handle: u64) {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            for oneshot in chain.channels() {
                oneshot.set_kernel(handle);
            }
            tracing::info!(
                "RDMA chain one-shot kernel set on {} channel(s)",
                chain.channels().len()
            );
        } else if let Some(oneshot) = self.oneshot() {
            oneshot.set_kernel(handle);
            tracing::info!("RDMA one-shot kernel set (graph-capturable collectives enabled)");
        }
        let _ = handle;
    }

    /// One-shot channels the RDMA chain opened (1 off a chain).
    pub(super) fn oneshot_channels(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.channels().len();
        }
        1
    }

    /// Why the one-shot channel stopped (a peer never arrived, the ranks
    /// diverged, or the host could no longer serve it), if it did.
    pub fn oneshot_poisoned(&self) -> Option<String> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.channels().iter().find_map(|o| o.poisoned());
        }
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.oneshot() {
            return oneshot.poisoned();
        }
        None
    }

    /// The host command ring of the RDMA pair or chain, when
    /// `ATLAS_GLM_CMD_RDMA=1` (see `rdma_pair::cmd_ring`): the words a
    /// message may carry, else 0.
    pub(super) fn cmd_ring_max_words(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if self.chain.as_ref().and_then(|c| c.cmd()).is_some()
            || self.rdma.as_ref().and_then(|r| r.cmd()).is_some()
        {
            return rdma_pair::cmd_ring::MAX_WORDS;
        }
        0
    }

    /// Send `words` to the peer (on a chain: down the chain) over the
    /// command ring.
    pub(super) fn cmd_ring_send(&self, words: &[u32]) -> Result<()> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.send_command_words(words);
        }
        #[cfg(atlas_rdma_verbs)]
        if let Some(ring) = self.rdma.as_ref().and_then(|r| r.cmd()) {
            return ring.send(words);
        }
        anyhow::bail!(
            "no command ring for {} command words (ATLAS_GLM_CMD_RDMA)",
            words.len()
        )
    }

    /// Receive the peer's next `words.len()` words from the command ring (on
    /// a chain also passing them on to the rank above).
    pub(super) fn cmd_ring_recv(&self, words: &mut [u32]) -> Result<()> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.recv_command_words(words);
        }
        #[cfg(atlas_rdma_verbs)]
        if let Some(ring) = self.rdma.as_ref().and_then(|r| r.cmd()) {
            return ring.recv(words);
        }
        anyhow::bail!(
            "no command ring for {} command words (ATLAS_GLM_CMD_RDMA)",
            words.len()
        )
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

    /// The RDMA chain for three or four ranks when `ATLAS_RDMA_ALLREDUCE=1`,
    /// bootstrapping through rank 0 on the NCCL bootstrap connections
    /// (`peers`) at the pair's port.
    #[cfg(atlas_rdma_verbs)]
    pub(super) fn connect_chain(
        rank: usize,
        world_size: usize,
        peers: &[std::net::TcpStream],
        master_port: u16,
    ) -> Result<Option<rdma_pair::chain::RdmaChain>> {
        if world_size <= 2 || !rdma_pair::RdmaPair::requested() {
            return Ok(None);
        }
        let socket = std::env::var("NCCL_NET").is_ok_and(|v| v.eq_ignore_ascii_case("socket"))
            || std::env::var("NCCL_IB_DISABLE").as_deref() == Ok("1");
        if !socket {
            tracing::warn!(
                "RDMA chain: NCCL still carries startup and rare collectives between ranks that \
                 share no RDMA link; set NCCL_IB_DISABLE=1 (or NCCL_NET=Socket) and \
                 NCCL_SOCKET_IFNAME to the routed IP interface on every rank"
            );
        }
        let at =
            rdma_pair::bootstrap::chain::Rendezvous::of(rank, peers, master_port.wrapping_add(2))?;
        rdma_pair::chain::RdmaChain::connect(rank, world_size, &at).map(Some)
    }

    /// The RDMA pair for a 2-rank world when `ATLAS_RDMA_ALLREDUCE=1`. Its
    /// bootstrap stays on the link of the NCCL bootstrap connection
    /// (`peers`; see `bootstrap::Link::of`).
    #[cfg(atlas_rdma_verbs)]
    pub(super) fn connect_rdma(
        rank: usize,
        world_size: usize,
        peers: &[std::net::TcpStream],
        master_addr: &str,
        master_port: u16,
        recv_capacity: usize,
    ) -> Result<Option<rdma_pair::RdmaPair>> {
        // Port +1 is the reconnect bootstrap; +2 carries the RDMA identities.
        let rdma = if world_size == 2 && rdma_pair::RdmaPair::requested() {
            let port = master_port.wrapping_add(2);
            let link = rdma_pair::bootstrap::Link::of(rank, &peers[0], port, master_addr)?;
            Some(rdma_pair::RdmaPair::connect(
                rank,
                &link,
                recv_capacity.next_multiple_of(64),
            )?)
        } else {
            if world_size == 2 && rdma_pair::cmd_ring::requested() {
                tracing::warn!(
                    "ATLAS_GLM_CMD_RDMA=1 needs the RDMA pair (two ranks, ATLAS_RDMA_ALLREDUCE=1): \
                     command words stay on NCCL broadcasts"
                );
            }
            None
        };
        Ok(rdma)
    }
}
