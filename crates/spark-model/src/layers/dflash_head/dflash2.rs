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
            self.kernels.linear(
                gpu,
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

/// Batched (`[n, γ]`) DFlash2 proposal forward. Mirrors the single-sequence
/// Option-B layer (`forward_block_layer_pre_attn` / `_attention` /
/// `_post_attn`) and the selector tail with every projection run once over all
/// `n·γ` rows, so the drafter weights are read once per step instead of once
/// per sequence. Convs are block-local (`block_size = γ`), attention and the
/// selector walk run per sequence.
///
/// Inputs are the staged batch buffers of `propose_batch`: query embeddings in
/// `batch_query_embed`, packed positions, slot mapping, per-sequence
/// `[kv_len, q_offset, q_rope_pos]` in `batch_attention_args` and anchors in
/// `batch_markov_prev`. Writes `n·γ` row tokens to `batch_tokens` (row 0 of
/// each sequence = anchor argmax, rows 1.. = the walk), the same row order as
/// the single-sequence `draft_tokens_dev`.
impl BlockDiffusionDraftHead {
    /// Whether this head batches DFlash2 proposals natively (BF16 weights,
    /// one proposal lane). `ATLAS_DFLASH2_BATCH=0` keeps the serial fallback.
    pub(super) fn dflash2_batch_enabled(&self) -> bool {
        self.dflash2.is_some()
            && self.lane_count() == 1
            && self.lm_head_nvfp4.is_none()
            && !matches!(self.quant, super::DflashQuantization::Fp8Weights)
            && std::env::var("ATLAS_DFLASH2_BATCH").as_deref() != Ok("0")
    }

    /// `conv.prepare` (`side == 0`, projects `deltas` from `x` first) or
    /// `conv.finish` (`side == 1`) over `rows` block rows at `x`, in place.
    #[allow(clippy::too_many_arguments)]
    fn dflash2_batch_conv(
        &self,
        gpu: &dyn GpuBackend,
        head: &Dflash2Head,
        layer_idx: usize,
        site: ConvSite,
        side: usize,
        x: DevicePtr,
        deltas: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let conv = head.conv(layer_idx, site)?;
        let hidden = self.hidden_size as u32;
        let width = head.delta_width(self.hidden_size) as u32;
        if side == 0 {
            self.kernels
                .linear(gpu, x, &conv.kernel_projection, deltas, rows, width, hidden, stream)?;
        }
        let taps = head.conv_taps;
        let groups = self.hidden_size / head.conv_group_size;
        ops::dflash2_grouped_conv(
            gpu,
            head.kernels.grouped_conv,
            x,
            deltas,
            conv.base_kernel
                .weight
                .offset(side * taps * self.hidden_size * 2),
            rows,
            hidden,
            head.conv_group_size as u32,
            taps as u32,
            self.gamma as u32,
            width,
            (side * taps * groups) as u32,
            stream,
        )
    }

    pub(super) fn run_batched_dflash2(
        &self,
        n: usize,
        block_tables: &[u64],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let head = self
            .dflash2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batched DFlash2 on a DFlash v1 head"))?;
        let gpu = ctx.gpu;
        let g = self.gamma;
        let rows = u32::try_from(n * g)?;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let vocab = self.vocab_size as u32;
        let rank = head.selector_rank as u32;
        let width = head.delta_width(self.hidden_size);
        anyhow::ensure!(
            block_tables.len() == n && width <= self.hidden_size && width <= self.intermediate_size,
            "batched DFlash2 needs {n} block tables and conv deltas within the borrowed buffers"
        );
        // Conv deltas live in buffers idle during their sublayer: the MLP up
        // rows around attention, the attention projection rows around the MLP.
        let (attn_deltas, mlp_deltas) = (self.batch_mlp_up, self.batch_attn_proj);
        let q_seq_bytes = g * q_dim as usize * 2;
        let inv_sqrt_d = 1.0 / (self.head_dim as f32).sqrt();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.batch_query_embed,
                &layer.input_layernorm,
                self.batch_norm,
                rows,
                h,
                self.rms_norm_eps,
                stream,
            )?;
            self.dflash2_batch_conv(
                gpu, head, layer_idx, ConvSite::Attention, 0, self.batch_norm, attn_deltas, rows,
                stream,
            )?;
            let lin = |x, w: &DenseWeight, mx: Option<&super::Mxfp8Weight>, y, n_out, k_in| {
                self.kernels
                    .project(gpu, x, w, mx, y, rows, n_out, k_in, stream)
            };
            lin(self.batch_norm, &layer.q_proj, layer.q_proj_mx.as_ref(), self.batch_q, q_dim, h)?;
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.batch_q,
                &layer.q_norm,
                self.batch_q,
                rows * self.num_q_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;
            lin(self.batch_norm, &layer.k_proj, None, self.batch_k, kv_dim, h)?;
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.batch_k,
                &layer.k_norm,
                self.batch_k,
                rows * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;
            lin(self.batch_norm, &layer.v_proj, None, self.batch_v, kv_dim, h)?;
            ops::rope_yarn(
                gpu,
                self.kernels.rope_qwen3,
                self.batch_q,
                self.batch_k,
                self.batch_position_ids,
                rows,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rotary_dim as u32,
                self.yarn_inv_freq,
                self.rope_theta,
                stream,
            )?;
            let (k_pool, v_pool) = {
                let cache = self.kv_cache.lock();
                (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
            };
            ops::reshape_and_cache(
                gpu,
                self.kernels.reshape_cache_bf16,
                self.batch_k,
                self.batch_v,
                k_pool,
                v_pool,
                self.batch_slot_mapping,
                rows,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                16,
                kv_dim,
                kv_dim,
                0,
                stream,
            )?;
            let sinks = layer
                .attention_sink_bias
                .as_ref()
                .map_or(DevicePtr::NULL, |w| w.weight);
            for (sequence, &table) in block_tables.iter().enumerate() {
                ops::prefill_attention_paged_dflash_bf16_indirect(
                    gpu,
                    self.kernels.prefill_attn_dflash_bf16_indirect,
                    self.batch_q.offset(sequence * q_seq_bytes),
                    k_pool,
                    v_pool,
                    self.batch_attn_out.offset(sequence * q_seq_bytes),
                    DevicePtr(table),
                    g as u32,
                    self.batch_attention_args.offset(sequence * 12),
                    self.num_q_heads as u32,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16,
                    self.attn_sliding_window(),
                    self.attn_causal(),
                    inv_sqrt_d,
                    sinks,
                    stream,
                )?;
            }
            lin(self.batch_attn_out, &layer.o_proj, layer.o_proj_mx.as_ref(), self.batch_attn_proj, h, q_dim)?;
            self.dflash2_batch_conv(
                gpu, head, layer_idx, ConvSite::Attention, 1, self.batch_attn_proj, attn_deltas,
                rows, stream,
            )?;
            ops::residual_add(
                gpu,
                self.kernels.residual_add,
                self.batch_query_embed,
                self.batch_attn_proj,
                rows * h,
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.batch_query_embed,
                &layer.post_attention_layernorm,
                self.batch_norm,
                rows,
                h,
                self.rms_norm_eps,
                stream,
            )?;
            self.dflash2_batch_conv(
                gpu, head, layer_idx, ConvSite::Mlp, 0, self.batch_norm, mlp_deltas, rows, stream,
            )?;
            lin(self.batch_norm, &layer.gate_proj, layer.gate_proj_mx.as_ref(), self.batch_mlp_gate, inter, h)?;
            lin(self.batch_norm, &layer.up_proj, layer.up_proj_mx.as_ref(), self.batch_mlp_up, inter, h)?;
            ops::silu_mul(
                gpu,
                self.kernels.silu_mul,
                self.batch_mlp_gate,
                self.batch_mlp_up,
                self.batch_mlp_gate,
                rows * inter,
                stream,
            )?;
            lin(self.batch_mlp_gate, &layer.down_proj, layer.down_proj_mx.as_ref(), self.batch_mlp_down, h, inter)?;
            self.dflash2_batch_conv(
                gpu, head, layer_idx, ConvSite::Mlp, 1, self.batch_mlp_down, mlp_deltas, rows,
                stream,
            )?;
            ops::residual_add(
                gpu,
                self.kernels.residual_add,
                self.batch_query_embed,
                self.batch_mlp_down,
                rows * h,
                stream,
            )?;
        }

        // Selector tail. Candidates and selector rows borrow the (now idle)
        // MLP gate rows.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.batch_query_embed,
            &self.norm,
            self.batch_norm,
            rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        self.kernels.linear(
            gpu,
            self.batch_norm,
            &DenseWeight {
                weight: self.lm_head_shared,
            },
            self.batch_logits,
            rows,
            vocab,
            h,
            stream,
        )?;
        let cand_bytes = n * g * ops::DFLASH2_TOPK * 4;
        let (cand_ids, cand_vals) = (self.batch_mlp_gate, self.batch_mlp_gate.offset(cand_bytes));
        let selector_hidden = self.batch_mlp_gate.offset(2 * cand_bytes);
        anyhow::ensure!(
            2 * cand_bytes + n * g * rank as usize * 2 <= n * g * self.intermediate_size * 2,
            "batched DFlash2 selector scratch exceeds the MLP rows"
        );
        ops::dflash2_topk(
            gpu,
            head.kernels.topk,
            self.batch_logits,
            cand_ids,
            cand_vals,
            rows,
            vocab,
            stream,
        )?;
        self.kernels.linear(
            gpu,
            self.batch_norm,
            &head.hidden_projection,
            selector_hidden,
            rows,
            rank,
            h,
            stream,
        )?;
        ops::dflash2_selector_walk(
            gpu,
            head.kernels.selector_walk,
            cand_ids,
            cand_vals,
            selector_hidden,
            head.predecessor_codebook.weight,
            head.successor_codebook.weight,
            self.batch_markov_prev,
            self.batch_tokens,
            u32::try_from(n)?,
            g as u32,
            rank,
            vocab,
            stream,
        )
    }
}
