// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split at run time (`rank_split`): a head's half buffers, its
//! swaps over the pair exchange, the pieces both ranks run, and the worker's
//! walk.

use anyhow::{Context, Result, bail, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::{BlockDiffusionDraftHead, DflashScratch};
use super::{Geometry, Graphs, Parts, Piece, SplitOps, Swap, half, walk};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

/// A head's split state: the plan, the half buffers and how far the current
/// propose got.
pub struct RankSplit {
    pub(crate) geometry: Geometry,
    /// This rank's half of gate (then of the gated activation) and of up.
    pub(super) gate: DevicePtr,
    up: DevicePtr,
    /// The peer's half of the gated activation.
    activation_peer: DevicePtr,
    /// This rank's and the peer's half of the down projection.
    pub(super) down: DevicePtr,
    down_peer: DevicePtr,
    /// This rank's and the peer's half of the logits.
    pub(super) logits: DevicePtr,
    logits_peer: DevicePtr,
    /// Where a one-way swap's unused direction lands and what it sends.
    sink: DevicePtr,
    /// Swaps issued by the current propose.
    issued: AtomicUsize,
}

impl RankSplit {
    /// Allocate the half buffers for `geometry`.
    pub(crate) fn new(geometry: Geometry, gpu: &dyn GpuBackend) -> Result<Self> {
        geometry.validate()?;
        let alloc = |swap: Swap, on: bool| -> Result<DevicePtr> {
            if on {
                gpu.alloc(geometry.bytes(swap))
            } else {
                Ok(DevicePtr::NULL)
            }
        };
        let (mlp, head) = (geometry.parts.mlp, geometry.parts.head);
        let sink_bytes = geometry
            .swaps()
            .iter()
            .map(|&s| geometry.bytes(s))
            .max()
            .context("a split has swaps")?;
        Ok(Self {
            geometry,
            gate: alloc(Swap::Activation(0), mlp)?,
            up: alloc(Swap::Activation(0), mlp)?,
            activation_peer: alloc(Swap::Activation(0), mlp)?,
            down: alloc(Swap::Output(0), mlp)?,
            down_peer: alloc(Swap::Output(0), mlp)?,
            logits: alloc(Swap::Logits, head)?,
            logits_peer: alloc(Swap::Logits, head)?,
            sink: gpu.alloc(sink_bytes)?,
            issued: AtomicUsize::new(0),
        })
    }

    /// The largest swap, which the pair exchange must be able to carry.
    pub(crate) fn max_bytes(&self) -> usize {
        let g = &self.geometry;
        g.swaps().iter().map(|&s| g.bytes(s)).max().unwrap_or(0)
    }

    /// A propose begins: no swap issued yet.
    pub(crate) fn begin(&self) {
        self.issued.store(0, Ordering::Relaxed);
    }

    /// What `rank` sends and where the peer's payload lands.
    pub(super) fn ends(
        &self,
        swap: Swap,
        rank: usize,
        scratch: &DflashScratch,
    ) -> (DevicePtr, DevicePtr) {
        match swap {
            // One way, head to worker. The rows are in `norm_buf` on both.
            Swap::Input(_) | Swap::Hidden if rank == 0 => (scratch.norm_buf, self.sink),
            Swap::Input(_) | Swap::Hidden => (self.sink, scratch.norm_buf),
            Swap::Activation(_) => (self.gate, self.activation_peer),
            Swap::Output(_) => (self.down, self.down_peer),
            Swap::Logits => (self.logits, self.logits_peer),
        }
    }

    /// Issue `swap` on `stream`, then place the halves it completed where
    /// the unsplit launch would have written the whole rows.
    pub(crate) fn swap(
        &self,
        swap: Swap,
        rank: usize,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<()> {
        let (send, dst) = self.ends(swap, rank, scratch);
        self.exchange(swap, send, dst, comm, stream)?;
        let g = &self.geometry;
        match swap {
            Swap::Activation(_) => self.join(
                scratch.mlp_intermediate,
                g.inter,
                (self.gate, self.activation_peer),
                rank,
                gpu,
                stream,
            ),
            Swap::Output(_) if rank == 0 => self.join(
                scratch.stream_acc,
                g.hidden,
                (self.down, self.down_peer),
                rank,
                gpu,
                stream,
            ),
            Swap::Logits if rank == 0 => self.join(
                scratch.logits,
                g.vocab,
                (self.logits, self.logits_peer),
                rank,
                gpu,
                stream,
            ),
            _ => Ok(()),
        }
    }

    fn exchange(
        &self,
        swap: Swap,
        send: DevicePtr,
        dst: DevicePtr,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<()> {
        let bytes = self.geometry.bytes(swap);
        ensure!(
            comm.exchange_async(send.0, dst.0, bytes, false, stream)?,
            "rank-split propose: pair exchange refused {swap:?} ({bytes} bytes)"
        );
        self.issued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Write `[gamma, n]` rows at `dst` from the two ranks' shares
    /// `(own, peer)`, each `[gamma, its rows]`: two pitched copies.
    fn join(
        &self,
        dst: DevicePtr,
        n: usize,
        (own, peer): (DevicePtr, DevicePtr),
        rank: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for (src, r) in [(own, rank), (peer, 1 - rank)] {
            let (first, rows) = half(n, r);
            gpu.copy_d2d_2d_async(
                src,
                rows * 2,
                dst.offset(first * 2),
                n * 2,
                rows * 2,
                self.geometry.gamma,
                stream,
            )?;
        }
        Ok(())
    }

    /// A propose ends, however it ended: issue every swap it did not reach,
    /// so the peer, which walks them all, is never left waiting in one.
    pub(crate) fn finish(&self, comm: &dyn CommBackend, stream: u64) -> Result<()> {
        let swaps = self.geometry.swaps();
        let issued = self.issued.load(Ordering::Relaxed);
        for &swap in swaps.iter().skip(issued) {
            self.exchange(swap, self.sink, self.sink, comm, stream)?;
        }
        if issued < swaps.len() {
            tracing::warn!(
                "rank-split propose stopped after {issued} of {} swaps; the rest were drained",
                swaps.len()
            );
        }
        Ok(())
    }
}

/// Rows `[first, first + rows)` of an NVFP4 `[n, k]` twin: packed bytes and
/// group scales are row-major, the tensor scale is the twin's.
pub(super) fn row_range(w: &QuantizedWeight, first: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: w.weight.offset(first * k / 2),
        weight_scale: w.weight_scale.offset(first * k / 16),
        ..*w
    }
}

impl BlockDiffusionDraftHead {
    /// Turn the rank split on for this head (either rank). Refuses a head
    /// the split cannot serve: it must not run unsplit on one rank only.
    pub fn enable_rank_split(&mut self, parts: Parts, gpu: &dyn GpuBackend) -> Result<()> {
        let tier = crate::layers::w4a16_gemv_tiers::tc_kernel(self.gamma as u32);
        let layers_ok = self.twins.nvfp4_tc
            && self.layers.iter().all(|l| {
                l.gate_proj_nvfp4.is_some()
                    && l.up_proj_nvfp4.is_some()
                    && l.down_proj_nvfp4.is_some()
            });
        ensure!(
            tier.0 != 0
                && (!parts.mlp || layers_ok)
                && (!parts.head || self.twins.lm_head_q4.is_some()),
            "ATLAS_GLM_DRAFT_TP needs the drafter's NVFP4 twins on the tensor-core GEMV \
             (ATLAS_W4A16_TC=1, ATLAS_DFLASH_NVFP4_TC=1, ATLAS_DFLASH_NVFP4_HEAD=1)"
        );
        ensure!(
            self.startup.option_b_enabled
                && self.lane_count() == 1
                && !self.startup.debug_dump
                && !self.startup.graph_ineligible_diags,
            "ATLAS_GLM_DRAFT_TP needs the paged single-lane propose without diagnostics"
        );
        let geometry = Geometry {
            gamma: self.gamma,
            hidden: self.hidden_size,
            inter: self.intermediate_size,
            vocab: self.vocab_size,
            layers: self.layers.len(),
            parts,
        };
        tracing::info!(
            "DFlash rank split: {parts:?}, {} swaps per propose, largest {} bytes",
            geometry.swaps().len(),
            geometry
                .swaps()
                .iter()
                .map(|&s| geometry.bytes(s))
                .max()
                .unwrap_or(0)
        );
        self.rank_split = Some(RankSplit::new(geometry, gpu)?);
        Ok(())
    }

    /// The split and communicator a single-sequence propose on the head rank
    /// swaps over, if it splits. The one predicate the model announces by and
    /// the forward splits by.
    pub(crate) fn rank_split_with<'a>(
        &'a self,
        comm: Option<&'a dyn CommBackend>,
        grammar: bool,
    ) -> Option<(&'a RankSplit, &'a dyn CommBackend)> {
        let (split, comm) = (self.rank_split.as_ref()?, comm?);
        (!grammar
            && comm.rank() == 0
            && comm.world_size() == 2
            && comm.supports_exchange_async(split.max_bytes()))
        .then_some((split, comm))
    }

    /// `out[gamma, rows] = input[gamma, k] · W[first.., ..]ᵀ` for this rank's
    /// half of an `[n, k]` twin.
    #[allow(clippy::too_many_arguments)]
    fn split_gemv(
        &self,
        gpu: &dyn GpuBackend,
        w: Option<&QuantizedWeight>,
        input: DevicePtr,
        out: DevicePtr,
        n: usize,
        k: usize,
        rank: usize,
        stream: u64,
    ) -> Result<()> {
        let w = w.context("rank-split propose: NVFP4 twin missing")?;
        let (first, rows) = half(n, rank);
        ensure!(
            self.nvfp4_tc_rows(
                gpu,
                &row_range(w, first, k),
                input,
                out,
                self.gamma as u32,
                rows as u32,
                k as u32,
                stream,
            )?,
            "rank-split propose: tensor-core GEMV tier missing"
        );
        Ok(())
    }

    /// A piece both ranks run, over `rank`'s half.
    pub(crate) fn split_piece(
        &self,
        split: &RankSplit,
        piece: Piece,
        rank: usize,
        gpu: &dyn GpuBackend,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<()> {
        let g = &split.geometry;
        match piece {
            Piece::GateUp(l) => {
                let layer = &self.layers[l];
                for (w, out) in [
                    (&layer.gate_proj_nvfp4, split.gate),
                    (&layer.up_proj_nvfp4, split.up),
                ] {
                    self.split_gemv(
                        gpu,
                        w.as_ref(),
                        scratch.norm_buf,
                        out,
                        g.inter,
                        g.hidden,
                        rank,
                        stream,
                    )?;
                }
                ops::silu_mul(
                    gpu,
                    self.kernels.silu_mul,
                    split.gate,
                    split.up,
                    split.gate,
                    (g.gamma * half(g.inter, rank).1) as u32,
                    stream,
                )
            }
            Piece::Down(l) => self.split_gemv(
                gpu,
                self.layers[l].down_proj_nvfp4.as_ref(),
                scratch.mlp_intermediate,
                split.down,
                g.hidden,
                g.inter,
                rank,
                stream,
            ),
            Piece::Vocab => self.split_gemv(
                gpu,
                self.twins.lm_head_q4.as_ref(),
                scratch.norm_buf,
                split.logits,
                g.vocab,
                g.hidden,
                rank,
                stream,
            ),
            other => bail!("rank-split propose: {other:?} is not a shared piece"),
        }
    }

    /// Worker side of a split propose: enqueue this rank's whole walk on
    /// `stream`. No host sync: the swaps order it against the head.
    pub fn rank_split_serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<()> {
        let split = self
            .rank_split
            .as_ref()
            .context("rank-split propose announced to a head without the split")?;
        ensure!(
            comm.rank() == 1 && comm.world_size() == 2,
            "rank-split propose serves on rank 1 of 2"
        );
        let worker = Worker {
            head: self,
            split,
            gpu,
            comm,
            stream,
        };
        split.begin();
        let walked = walk(
            &split.geometry.steps(1),
            Graphs::Eager,
            gpu,
            stream,
            &worker,
        );
        // The head is already walking these swaps: issue them all.
        let drained = split.finish(comm, stream);
        walked.and(drained)
    }
}

struct Worker<'a> {
    head: &'a BlockDiffusionDraftHead,
    split: &'a RankSplit,
    gpu: &'a dyn GpuBackend,
    comm: &'a dyn CommBackend,
    stream: u64,
}

impl SplitOps for Worker<'_> {
    fn piece(&self, piece: Piece) -> Result<()> {
        let scratch = &self.head.scratch;
        self.head
            .split_piece(self.split, piece, 1, self.gpu, scratch, self.stream)
    }
    fn swap(&self, swap: Swap) -> Result<()> {
        let scratch = &self.head.scratch;
        self.split
            .swap(swap, 1, self.gpu, self.comm, scratch, self.stream)
    }
}
