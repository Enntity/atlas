// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 extensions to [`BlockDiffusionDraftHead`].
//!
//! DFlash2 keeps the DFlash backbone and adds two pieces, both optional on
//! the head (`None` for DFlash v1 / DSpark checkpoints):
//!
//! * **Grouped dynamic conv** around each sublayer. `prepare` projects the
//!   normed block rows to per-row, per-group tap deltas and convolves the
//!   rows in-block (side 0) before attention / MLP; `finish` convolves the
//!   sublayer output with the other half of those deltas (side 1) before the
//!   residual add. Context K/V never pass through the conv.
//! * **Candidate selector** in place of the per-row argmax: the lm_head top-K
//!   of each mask row is re-scored with a low-rank transition
//!   `pred[prev] ⊙ hidden_projection(h) · succ[cand]` and walked greedily
//!   from the block anchor (temperature 0).
//!
//! Every launch reads stable scratch pointers, so the conv calls sit inside
//! the captured pre/post-attention subgraphs and the selector inside the
//! captured tail.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{BlockDiffusionDraftHead, DflashScratch};
use crate::layers::ops;
use crate::weight_loader::dflash_loader::{Dflash2ConvWeights, Dflash2Weights};
use crate::weight_map::DenseWeight;

/// Which grouped conv of a DFlash2 layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConvSite {
    Attention,
    Mlp,
}

pub struct Dflash2Kernels {
    pub grouped_conv: KernelHandle,
    pub topk: KernelHandle,
    pub selector_walk: KernelHandle,
}

/// Loaded DFlash2 weights plus their kernels.
pub struct Dflash2Head {
    /// Per drafter layer: `(attention_conv, mlp_conv)`.
    pub layers: Vec<(Dflash2ConvWeights, Dflash2ConvWeights)>,
    pub hidden_projection: DenseWeight,
    pub predecessor_codebook: DenseWeight,
    pub successor_codebook: DenseWeight,
    pub conv_group_size: usize,
    pub conv_taps: usize,
    pub selector_rank: usize,
    pub kernels: Dflash2Kernels,
}

/// Per-lane DFlash2 scratch (captured graphs bake these pointers).
pub struct Dflash2Scratch {
    /// `[γ, 2 * taps * groups]` BF16 conv deltas; `prepare` fills it, the
    /// matching `finish` consumes side 1 before the next `prepare`.
    pub conv_delta: DevicePtr,
    /// `[γ, DFLASH2_TOPK]` u32 lm_head candidates per block row.
    pub cand_ids: DevicePtr,
    /// `[γ, DFLASH2_TOPK]` f32 unary logits of `cand_ids`.
    pub cand_vals: DevicePtr,
    /// `[γ, rank]` BF16 selector projection of the final-normed rows.
    pub selector_hidden: DevicePtr,
}

impl Dflash2Head {
    pub fn new(weights: Dflash2Weights, num_layers: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        anyhow::ensure!(
            weights.layers.len() == num_layers,
            "DFlash2: {} conv layer pairs for {num_layers} drafter layers",
            weights.layers.len()
        );
        anyhow::ensure!(
            weights.selector_top_k == ops::DFLASH2_TOPK,
            "DFlash2: selector_top_k={} unsupported (kernel is built for {})",
            weights.selector_top_k,
            ops::DFLASH2_TOPK
        );
        anyhow::ensure!(
            (1..=ops::DFLASH2_MAX_TAPS).contains(&weights.conv_taps),
            "DFlash2: conv_kernel_size={} outside 1..={}",
            weights.conv_taps,
            ops::DFLASH2_MAX_TAPS
        );
        anyhow::ensure!(
            (1..=ops::DFLASH2_MAX_RANK).contains(&weights.selector_rank),
            "DFlash2: selector_rank={} outside 1..={}",
            weights.selector_rank,
            ops::DFLASH2_MAX_RANK
        );
        Ok(Self {
            kernels: Dflash2Kernels {
                grouped_conv: gpu.kernel("dflash2", "dflash2_grouped_conv_bf16")?,
                topk: gpu.kernel("dflash2", "dflash2_topk_bf16")?,
                selector_walk: gpu.kernel("dflash2", "dflash2_selector_walk")?,
            },
            layers: weights.layers,
            hidden_projection: weights.hidden_projection,
            predecessor_codebook: weights.predecessor_codebook,
            successor_codebook: weights.successor_codebook,
            conv_group_size: weights.conv_group_size,
            conv_taps: weights.conv_taps,
            selector_rank: weights.selector_rank,
        })
    }

    fn conv(&self, layer_idx: usize, site: ConvSite) -> Result<&Dflash2ConvWeights> {
        let (attention, mlp) = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("DFlash2 conv layer {layer_idx} is missing"))?;
        Ok(match site {
            ConvSite::Attention => attention,
            ConvSite::Mlp => mlp,
        })
    }

    /// Row width of the conv delta buffer: `2 sides × taps × groups`.
    fn delta_width(&self, hidden: usize) -> usize {
        2 * self.conv_taps * (hidden / self.conv_group_size)
    }

    /// Zeroed scratch for `rows` (= γ) block rows.
    pub fn alloc_scratch(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        hidden: usize,
    ) -> Result<Dflash2Scratch> {
        let sizes = [
            rows * self.delta_width(hidden) * 2,
            rows * ops::DFLASH2_TOPK * 4,
            rows * ops::DFLASH2_TOPK * 4,
            rows * self.selector_rank * 2,
        ];
        let mut ptrs = [DevicePtr::NULL; 4];
        for (ptr, bytes) in ptrs.iter_mut().zip(sizes) {
            *ptr = gpu.alloc(bytes)?;
            gpu.memset(*ptr, 0, bytes)?;
        }
        let [conv_delta, cand_ids, cand_vals, selector_hidden] = ptrs;
        Ok(Dflash2Scratch {
            conv_delta,
            cand_ids,
            cand_vals,
            selector_hidden,
        })
    }
}

impl BlockDiffusionDraftHead {
    /// DFlash2 head + this lane's scratch, or `None` for DFlash v1.
    fn dflash2_parts<'a>(
        &'a self,
        scratch: &'a DflashScratch,
    ) -> Result<Option<(&'a Dflash2Head, &'a Dflash2Scratch)>> {
        match (self.dflash2.as_ref(), scratch.dflash2.as_ref()) {
            (None, _) => Ok(None),
            (Some(head), Some(lane)) => Ok(Some((head, lane))),
            (Some(_), None) => {
                anyhow::bail!("DFlash2 head bound to a lane without DFlash2 scratch")
            }
        }
    }

    /// Small-M BF16 projection `[m, k] × [n, k]ᵀ → [m, n]` (m = γ rows).
    fn dflash2_project(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let m = self.gamma as u32;
        let k = self.hidden_size as u32;
        if m <= ops::DENSE_GEMV_BATCHM_MAX_M {
            ops::dense_gemv_batchm(
                gpu,
                self.kernels.dense_gemv_batchm,
                input,
                weight,
                output,
                m,
                n,
                k,
                n,
                stream,
            )
        } else {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// One conv side (0 = prepare, 1 = finish) in place on the γ rows at `x`.
    #[allow(clippy::too_many_arguments)]
    fn dflash2_conv_side(
        &self,
        gpu: &dyn GpuBackend,
        head: &Dflash2Head,
        lane: &Dflash2Scratch,
        conv: &Dflash2ConvWeights,
        side: usize,
        x: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let taps = head.conv_taps;
        let groups = self.hidden_size / head.conv_group_size;
        ops::dflash2_grouped_conv(
            gpu,
            head.kernels.grouped_conv,
            x,
            lane.conv_delta,
            conv.base_kernel
                .weight
                .offset(side * taps * self.hidden_size * 2),
            self.gamma as u32,
            self.hidden_size as u32,
            head.conv_group_size as u32,
            taps as u32,
            self.gamma as u32,
            head.delta_width(self.hidden_size) as u32,
            (side * taps * groups) as u32,
            stream,
        )
    }

    /// `conv.prepare` on the γ block rows at `x` (in place): project the
    /// per-row deltas from `x`, then apply the side-0 conv. No-op for v1.
    pub(super) fn dflash2_conv_prepare(
        &self,
        gpu: &dyn GpuBackend,
        scratch: &DflashScratch,
        layer_idx: usize,
        site: ConvSite,
        x: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some((head, lane)) = self.dflash2_parts(scratch)? else {
            return Ok(());
        };
        let conv = head.conv(layer_idx, site)?;
        let width = head.delta_width(self.hidden_size) as u32;
        self.dflash2_project(
            gpu,
            x,
            &conv.kernel_projection,
            lane.conv_delta,
            width,
            stream,
        )?;
        self.dflash2_conv_side(gpu, head, lane, conv, 0, x, stream)
    }

    /// `conv.finish` on the sublayer output at `y` (in place): side-1 conv
    /// with the deltas of the matching `prepare`. No-op for v1.
    pub(super) fn dflash2_conv_finish(
        &self,
        gpu: &dyn GpuBackend,
        scratch: &DflashScratch,
        layer_idx: usize,
        site: ConvSite,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some((head, lane)) = self.dflash2_parts(scratch)? else {
            return Ok(());
        };
        let conv = head.conv(layer_idx, site)?;
        self.dflash2_conv_side(gpu, head, lane, conv, 1, y, stream)
    }

    /// Candidate-selector tail. `normed` holds the γ final-normed block rows
    /// and `scratch.logits` their lm_head logits; writes all γ entries of
    /// `scratch.draft_tokens_dev` (row 0 = anchor-row argmax, rows 1.. =
    /// greedy walk from the anchor in `scratch.markov_prev_dev`).
    pub(super) fn dflash2_select_drafts(
        &self,
        gpu: &dyn GpuBackend,
        scratch: &DflashScratch,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (head, lane) = self
            .dflash2_parts(scratch)?
            .ok_or_else(|| anyhow::anyhow!("DFlash2 selector requested on a DFlash v1 head"))?;
        let gamma = self.gamma as u32;
        let vocab = self.vocab_size as u32;
        let rank = head.selector_rank as u32;
        ops::dflash2_topk(
            gpu,
            head.kernels.topk,
            scratch.logits,
            lane.cand_ids,
            lane.cand_vals,
            gamma,
            vocab,
            stream,
        )?;
        self.dflash2_project(
            gpu,
            normed,
            &head.hidden_projection,
            lane.selector_hidden,
            rank,
            stream,
        )?;
        ops::dflash2_selector_walk(
            gpu,
            head.kernels.selector_walk,
            lane.cand_ids,
            lane.cand_vals,
            lane.selector_hidden,
            head.predecessor_codebook.weight,
            head.successor_codebook.weight,
            scratch.markov_prev_dev,
            scratch.draft_tokens_dev,
            1,
            gamma,
            rank,
            vocab,
            stream,
        )
    }
}
