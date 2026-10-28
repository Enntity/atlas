// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl OneShot {
    /// Begin a fused reduce-scatter of `seg`-byte segments at `seg0 + d *
    /// stride` (`CommBackend::rs_fused_begin`): its producer plan, its
    /// pieces in the order the producer finishes them (at the ends the raw
    /// segments as `relay::rs_role` sends them, farthest first; in the
    /// middle the segments it sums on, then its raw one; its own last). `None` off channel 0 of a four-rank
    /// chain with a fused region, for a segment beyond its capacity or of
    /// misaligned pieces, and right after another fused op.
    pub(in crate::nccl_backend) fn rs_fused_begin(
        &self,
        seg0: u64,
        stride: i64,
        seg: usize,
    ) -> Result<Option<crate::RsFused>> {
        let Some(f) = self.fused else {
            return Ok(None);
        };
        let bytes = MAX_RANKS * seg;
        let k = rs_per_segment(bytes);
        if self.kernel.load(Ordering::Relaxed) == 0
            || seg == 0
            || seg > f.cap
            || !bytes.is_multiple_of(RS_PIECES * 64)
            || !(seg / k).is_multiple_of(16)
            || !seg0.is_multiple_of(16)
            || stride % 16 != 0
            || (stride.unsigned_abs() as usize) < seg
            || self.fused_last.load(Ordering::Relaxed)
        {
            return Ok(None);
        }
        let piece = seg / k;
        let rank = self.shape.rank;
        let role = relay::rs_role(rank);
        let ctrl = self.dev + ctrl_off(self.cfg.max, self.shape.origins()) as u64;
        let mut plan = crate::RsFusedPlan {
            ready: ctrl + READY as u64,
            state: self.state,
            stage: ctrl + STAGE as u64,
            poison: ctrl + POISON as u64,
            timeout_ns: self.cfg.timeout_ns,
            code: fused::code(bytes) as u32,
            npieces: (MAX_RANKS * k) as u32,
            piece_bytes: piece as u32,
            mode: self.cfg.sm_stage.max(1),
            pad: 0,
            piece: [crate::RsFusedPiece::default(); crate::RS_FUSED_MAX_PIECES],
        };
        // A middle rank produces the segments it sums on first: their raw
        // inputs arrive first (the ends send those first), and their pairs
        // feed the far side's sums, so its links fill from the start; its
        // raw segment (one hop, for an end's own sum) follows.
        let computed = role.computed.map_or(&[][..], |(_, segs)| segs);
        let order = if computed.is_empty() {
            [(role.raw.1, true), (computed, false), (&[rank][..], false)]
        } else {
            [(computed, false), (role.raw.1, true), (&[rank][..], false)]
        };
        let mut at = plan.piece.iter_mut();
        for (segs, raw) in order {
            for &g in segs {
                for j in 0..k {
                    let slot = at.next().context("fused reduce-scatter: too many pieces")?;
                    *slot = crate::RsFusedPiece {
                        out: seg0.wrapping_add_signed(g as i64 * stride) + (j * piece) as u64,
                        s: if raw {
                            f.dev + (fused::raw_index(rank, g) * f.cap + j * piece) as u64
                        } else {
                            0
                        },
                        p: (g * k + j) as u32,
                        pad: 0,
                    };
                }
            }
        }
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(Some(crate::RsFused {
            plan,
            done: f.done,
            epoch,
            piece_bytes: piece,
            pieces: k,
            seg0,
            stride,
            bytes: seg,
        }))
    }

    /// The chain kernel of the fused op `fused` on `stream`, capturable;
    /// `false` as for [`Self::enqueue`].
    pub(in crate::nccl_backend) fn reduce_scatter_fused(
        &self,
        fused: &crate::RsFused,
        stream: u64,
    ) -> Result<bool> {
        self.launch(
            fused.seg0,
            fused.seg0,
            MAX_RANKS * fused.bytes,
            false,
            Land::Fused(fused.stride, fused.epoch),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// `stream` waits until piece `piece` of this rank's segment of `fused`
    /// is final.
    pub(in crate::nccl_backend) fn rs_fused_wait(
        &self,
        fused: &crate::RsFused,
        piece: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            piece < fused.pieces,
            "fused reduce-scatter: piece {piece} of {}",
            fused.pieces
        );
        cu(
            unsafe {
                cuStreamWaitValue32_v2(
                    stream,
                    fused.done + 4 * piece as u64,
                    fused.epoch,
                    CU_STREAM_WAIT_VALUE_GEQ,
                )
            },
            "cuStreamWaitValue32(fused done)",
        )
    }

    /// A chain's broadcast of rank `root`'s `ptr[..bytes]` into every rank's
    /// `ptr`, on `stream` and capturable; `false` as for [`Self::enqueue`],
    /// and on a pair (its broadcasts stay on NCCL). It waits for the root
    /// without a limit, as ncclBroadcast does: a worker idles here until the
    /// head's next command, and the host times a slow broadcast afterwards.
    pub(in crate::nccl_backend) fn broadcast(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks == 2 || root >= self.shape.ranks {
            return Ok(false);
        }
        self.launch(ptr, ptr, bytes, false, Land::Root(root), 0, stream)
    }

    /// One op: stage `src`, then land what `land` says in `dst`, waiting up
    /// to `timeout_ns` (0: no limit).
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        land: Land,
        timeout_ns: u64,
        stream: u64,
    ) -> Result<bool> {
        let ranks = self.shape.ranks;
        // The kernel's `root`: < ranks a broadcast, == ranks the sum (or
        // the pair's exchange), ranks + 1 a gather `stride` bytes apart.
        let (root, stride) = match land {
            Land::Sum => (ranks, 0),
            Land::Root(root) => (root, 0),
            Land::Gather(stride) => (ranks + 1, stride),
            Land::ReduceScatter(stride) => (ranks + 2, stride),
            Land::AllToAll(_) => (ranks + 3, 0),
            Land::Fused(stride, _) => (ranks + 4, stride),
        };
        let mut a2a = match land {
            Land::AllToAll(a2a) => a2a,
            _ => A2a::default(),
        };
        let (mut fused, region) = match (land, self.fused) {
            (Land::Fused(_, epoch), Some(f)) => (
                FusedArg {
                    done: f.done,
                    epoch,
                    pad: 0,
                },
                Some(f),
            ),
            (Land::Fused(..), None) => return Ok(false),
            _ => (FusedArg::default(), None),
        };
        // The reduce-scatter's and the all-to-all's kernels always stage
        // (pipelined, piece by piece), and run every block.
        let rs = matches!(
            land,
            Land::ReduceScatter(_) | Land::AllToAll(_) | Land::Fused(..)
        );
        let kernel = self.kernel.load(Ordering::Relaxed);
        let fits = match region {
            Some(f) => bytes <= MAX_RANKS * f.cap,
            None => self.cfg.eligible(bytes),
        };
        if kernel == 0 || !fits {
            return Ok(false);
        }
        // Whether a fused op may follow (`fused` module doc): not right
        // after one, until an op every rank hears from its neighbours.
        match land {
            Land::Fused(..) => self.fused_last.store(true, Ordering::Relaxed),
            Land::Root(_) => {}
            _ => self.fused_last.store(false, Ordering::Relaxed),
        }
        let ctrl = self.dev + ctrl_off(self.cfg.max, self.shape.origins()) as u64;
        let stage = ctrl + STAGE as u64;
        let sm = if rs {
            self.cfg.sm_stage.max(1)
        } else {
            self.cfg.sm_stage
        };
        let fence = self.cfg.stage_fence && sm == 0;
        if sm == 0 {
            cu(
                unsafe { cuStreamWaitValue32_v2(stream, stage, 0, CU_STREAM_WAIT_VALUE_EQ) },
                "cuStreamWaitValue32(one-shot stage)",
            )?;
            cu(
                unsafe { cuMemcpyAsync(self.dev, src, bytes, stream) },
                "cuMemcpyAsync(one-shot stage)",
            )?;
        }
        if fence {
            // Default flags fence the copy before the word lands.
            cu(
                unsafe { cuStreamWriteValue32_v2(stream, stage, bytes as u32, 0) },
                "cuStreamWriteValue32(one-shot stage)",
            )?;
        }
        let blocks = if rs {
            rs_blocks()
        } else {
            (bytes / 16).div_ceil(THREADS).clamp(1, MAX_BLOCKS)
        } as u32;
        // The kernel's pieces: a reduce-scatter's (`rs_pieces`), else the
        // flood's or the pair's stripes.
        let rails = if rs {
            (MAX_RANKS * rs_per_segment(bytes)) as u32
        } else {
            self.shape.stripe_count(bytes, self.cfg.stripe_min)
        };
        // A fused op lands in its region, slots `cap` bytes apart.
        let (recv, slot_stride) = match region {
            Some(f) => (f.dev, f.cap as u64),
            None => (
                self.dev + recv_off(self.cfg.max, 0, 0) as u64,
                self.cfg.max as u64,
            ),
        };
        let mut args = (
            dst,
            recv,
            slot_stride,
            bytes as u32,
            add as u32,
            ctrl + FLAGS as u64,
            rails,
            self.state,
            if fence { 0 } else { stage },
            ctrl + POISON as u64,
            timeout_ns,
            src,
            self.dev,
            sm,
            self.shape.ranks as u32,
            self.shape.rank as u32,
            root as u32,
            stride,
        );
        // The driver reads as many as the kernel takes: the pair's first 14.
        let mut params: [*mut c_void; 20] = [
            (&raw mut args.0).cast(),
            (&raw mut args.1).cast(),
            (&raw mut args.2).cast(),
            (&raw mut args.3).cast(),
            (&raw mut args.4).cast(),
            (&raw mut args.5).cast(),
            (&raw mut args.6).cast(),
            (&raw mut args.7).cast(),
            (&raw mut args.8).cast(),
            (&raw mut args.9).cast(),
            (&raw mut args.10).cast(),
            (&raw mut args.11).cast(),
            (&raw mut args.12).cast(),
            (&raw mut args.13).cast(),
            (&raw mut args.14).cast(),
            (&raw mut args.15).cast(),
            (&raw mut args.16).cast(),
            (&raw mut args.17).cast(),
            (&raw mut a2a).cast(),
            (&raw mut fused).cast(),
        ];
        // A forced kernel stage (the reduce-scatter under a copy-engine
        // configuration) launches as the copy-engine path does: no PDL.
        let status = if sm != 0 && self.cfg.sm_stage != 0 && self.cfg.pdl {
            // The kernel enters with `atlas_pdl_enter()` before it reads
            // anything its predecessor wrote.
            let mut value = [0u8; 64];
            value[..4].copy_from_slice(&1i32.to_ne_bytes());
            let mut attr = CuLaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value,
            };
            let config = CuLaunchConfig {
                grid: [blocks, 1, 1],
                block: [THREADS as u32, 1, 1],
                shared_mem_bytes: 0,
                stream,
                attrs: &mut attr,
                num_attrs: 1,
            };
            unsafe { cuLaunchKernelEx(&config, kernel, params.as_mut_ptr(), std::ptr::null_mut()) }
        } else {
            unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    THREADS as u32,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        };
        if status != 0 && fence {
            // `stage` is already on the stream: the proxy will serve an op no
            // kernel consumes, and its count would stay ahead of the device
            // sequence (stale receive slots). Stop the channel instead.
            poison_host(self.ctrl(), POISON_LAUNCH);
        }
        cu(status, oneshot_kernel_name(self.shape.ranks))?;
        Ok(true)
    }
}
