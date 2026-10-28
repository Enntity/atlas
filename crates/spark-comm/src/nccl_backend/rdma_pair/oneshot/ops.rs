// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl OneShot {
    pub(super) fn new(cfg: Config, shape: Shape, host: usize, dev: u64) -> Result<Self> {
        shape.check()?;
        let (mut state, zero) = (0u64, [0u8; STATE_BYTES]);
        cu(
            unsafe { cuMemAlloc_v2(&mut state, STATE_BYTES) },
            "cuMemAlloc(one-shot state)",
        )?;
        cu(
            unsafe { cuMemcpyHtoD_v2(state, zero.as_ptr().cast(), STATE_BYTES) },
            "cuMemcpyHtoD(one-shot state)",
        )?;
        Ok(Self {
            cfg,
            shape,
            host,
            dev,
            state,
            kernel: AtomicU64::new(0),
            fused: None,
            epoch: AtomicU32::new(0),
            fused_last: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Give channel 0 of a four-rank chain the fused region at device
    /// address `dev` (`fused::entries` slots of `cap` bytes).
    pub(super) fn with_fused(mut self, dev: u64, cap: usize) -> Result<Self> {
        let mut done = 0u64;
        let bytes = 4 * MAX_PIECES;
        cu(
            unsafe { cuMemAlloc_v2(&mut done, bytes) },
            "cuMemAlloc(fused done)",
        )?;
        let zero = [0u8; 4 * MAX_PIECES];
        cu(
            unsafe { cuMemcpyHtoD_v2(done, zero.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD(fused done)",
        )?;
        self.fused = Some(FusedRegion { dev, cap, done });
        Ok(self)
    }

    /// Provide the kernel (`oneshot_kernel_name`); the channel is
    /// unavailable until then.
    pub(in crate::nccl_backend) fn set_kernel(&self, handle: u64) {
        self.kernel.store(handle, Ordering::Relaxed);
    }

    /// Ranks of the channel.
    pub(in crate::nccl_backend) fn ranks(&self) -> usize {
        self.shape.ranks
    }

    /// Largest payload [`Self::enqueue`] accepts (0 while unavailable).
    pub(in crate::nccl_backend) fn max_bytes(&self) -> usize {
        if self.kernel.load(Ordering::Relaxed) == 0 {
            0
        } else {
            self.cfg.max
        }
    }

    /// Why the channel stopped (the kernel trapped or will), if it did.
    pub(in crate::nccl_backend) fn poisoned(&self) -> Option<String> {
        poison_at(self.ctrl())
    }

    /// Host address of the control page.
    fn ctrl(&self) -> usize {
        self.host + ctrl_off(self.cfg.max, self.shape.origins())
    }

    /// Send `src[..bytes]` to the peer and land its payload in `dst` (added
    /// in place when `add`, else copied), all on `stream` and capturable.
    /// A chain only sums in place (`add`, `src == dst`): every rank's
    /// payload, the same on all. `false` (nothing enqueued) when the size or
    /// the call is ineligible or the kernel is missing; every rank sees the
    /// same answer.
    pub(in crate::nccl_backend) fn enqueue(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks > 2 && (!add || src != dst) {
            return Ok(false);
        }
        self.launch(src, dst, bytes, add, Land::Sum, self.cfg.timeout_ns, stream)
    }

    /// A chain's all-gather of every rank's `base + rank * stride`
    /// `[..bytes]` into the same place on every rank, on `stream` and
    /// capturable; `false` as for [`Self::enqueue`], and on a pair.
    pub(in crate::nccl_backend) fn gather(
        &self,
        base: u64,
        stride: i64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks == 2 {
            return Ok(false);
        }
        let own = base.wrapping_add_signed(stride * self.shape.rank as i64);
        self.launch(
            own,
            base,
            bytes,
            false,
            Land::Gather(stride),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// The four-rank chain's reduce-scatter: segment `d` of every rank's
    /// partial at `seg0 + d * stride` (`bytes / 4` each), summed in the
    /// all-reduce's pairwise tree, lands in rank `d`'s own segment, on
    /// `stream` and capturable. `false` as for [`Self::enqueue`], and unless
    /// four ranks, 16-byte alignment and a payload of whole 64-byte pieces
    /// below [`RS_BIT`]. `fp8`: the pieces travel as E4M3 and f32 scales,
    /// summed in f32 (lossy, `oneshot_rs4_fp8`), and must be whole KiB.
    pub(in crate::nccl_backend) fn reduce_scatter(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        fp8: bool,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks != MAX_RANKS
            || !bytes.is_multiple_of(RS_PIECES * 64)
            || bytes >= RS_BIT
            || !seg0.is_multiple_of(16)
            || stride % 16 != 0
            || fp8 && !bytes.is_multiple_of(MAX_RANKS * rs_per_segment(bytes) * 1024)
        {
            return Ok(false);
        }
        // The kernel takes `add` as the reduce-scatter's FP8 switch.
        self.launch(
            seg0,
            seg0,
            bytes,
            fp8,
            Land::ReduceScatter(stride),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// The four-rank chain's all-to-all of `a2a` (`CommBackend::
    /// all_to_all_async`, one op: its four segments within the one-shot
    /// maximum) on `stream`, capturable; `false` as for [`Self::enqueue`],
    /// and unless four ranks and a shape [`A2a::valid`] admits.
    pub(in crate::nccl_backend) fn all_to_all(&self, a2a: A2a, stream: u64) -> Result<bool> {
        let bytes = MAX_RANKS * a2a.segment();
        if self.shape.ranks != MAX_RANKS || !a2a.valid() || bytes >= RS_BIT {
            return Ok(false);
        }
        self.launch(
            0,
            0,
            bytes,
            false,
            Land::AllToAll(a2a),
            self.cfg.timeout_ns,
            stream,
        )
    }
}
