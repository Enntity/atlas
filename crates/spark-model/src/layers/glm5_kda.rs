// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 KDA recurrent block for the conservative GB10 bring-up path.
mod forward_attention;
mod forward_ffn;
mod forward_recurrent;
mod hc;
mod indexed_core;
mod multi_seq;
mod paired_verify;
mod profile;
mod projection;
mod recurrent;
mod shared_cache;

use std::mem::size_of;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::layers::qwen3_attention::HcWeights;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::DenseWeight;

pub use projection::{Glm5KdaWeights, Glm5Projection};

fn verify_batched_ffn_enabled() -> bool {
    std::env::var("ATLAS_GLM_KDA_BATCHED_FFN").ok().as_deref() == Some("1")
}

fn verify_batched_conv_snapshot_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_BATCHED_CONV_SNAPSHOT")
        .ok()
        .as_deref()
        == Some("1")
}

fn verify_batched_recurrent_snapshot_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_BATCHED_RECURRENT_SNAPSHOT")
        .ok()
        .as_deref()
        == Some("1")
}

fn verify_fused_qkv_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_QKV").ok().as_deref() == Some("1")
}

fn verify_fused_dense_pairs_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_DENSE_PAIRS")
        .ok()
        .as_deref()
        == Some("1")
}

fn verify_fused_dense_triple_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_DENSE_TRIPLE")
        .ok()
        .as_deref()
        == Some("1")
}

/// Exact K=5-only seam: exchange the remote row-parallel KDA O projection and
/// combine it inside mHC post-mixing. The launcher retains a fallback switch.
fn verify_fused_tp_hc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_K5_FUSED_TP_HC").ok().as_deref() == Some("1"))
}

fn verify_fused_moe_hc_check_once() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_GLM_K5_FUSED_MOE_HC_CHECK")
            .ok()
            .as_deref()
            == Some("1")
    }) && !CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

pub struct Glm5KdaLayer {
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    weights: Glm5KdaWeights,
    ffn: FfnComponent,
    hc: HcWeights,
    layer_idx: usize,
    ssm_ordinal: usize,
    indexed_trace_logged: std::sync::atomic::AtomicBool,
    hidden_size: usize,
    heads: usize,
    dim: usize,
    conv_width: usize,
    lower_bound: f32,
    h_state_bytes: usize,
    conv_state_bytes: usize,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    dense_gemv_batchm_k: KernelHandle,
    dense_gemv_batch5_k: KernelHandle,
    dense_gemv_batch5_dual_k: KernelHandle,
    dense_gemv_batch5_triple_n_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_sw_k: KernelHandle,
    w4a16_gemv_batch2_k: KernelHandle,
    w4a16_gemv_batch3_k: KernelHandle,
    w4a16_gemv_batch5_qkv_k: KernelHandle,
    w4a16_gemv_batchm: W4a16BatchmTiers,
    w4a16_gemm_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    dense_gemm_pipelined_k: KernelHandle,
    conv_prefill_k: KernelHandle,
    conv_prefill_tp_k: KernelHandle,
    conv_prefill_tp_snap_k: KernelHandle,
    conv_indexed_k: KernelHandle,
    pack_k: KernelHandle,
    recurrent_k: KernelHandle,
    recurrent_indexed_k: KernelHandle,
    recurrent_verify_snap_k: KernelHandle,
    preprocess_regresident_k: KernelHandle,
    recurrent_regresident_k: KernelHandle,
    register_resident_prefill: bool,
    gated_norm_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_pre_k: KernelHandle,
    hc_pre_from_raw_mix_k: KernelHandle,
    hc_post_k: KernelHandle,
    hc_post_bf16_add_k: KernelHandle,
    hc_post_moe_blend_k: KernelHandle,
    hc_contract_k: KernelHandle,
}

/// Non-owning continuation for an immediate FFN call on the same layer/context/stream.
///
/// This carries the original profiling timer, not a new phase timer. It is not an
/// owner-safe snapshot: `normed` aliases singleton `norm_output`, and FFN/post-mHC
/// still reads `hc_streams`, `hc_post`, and `hc_comb` from `ctx.buffers`.
/// Another owner's attention phase may overwrite all four. Pair traversal must
/// provide disjoint checked views or preserve/restore their contents first.
struct FfnPhase {
    hidden: DevicePtr,
    normed: DevicePtr,
    tokens: usize,
    decode: bool,
    capture_verify_intermediates: bool,
    profile_timer: Option<std::time::Instant>,
}

impl Glm5KdaLayer {
    pub fn new(
        input_norm: DenseWeight,
        post_attn_norm: DenseWeight,
        weights: Glm5KdaWeights,
        ffn: FfnComponent,
        hc: HcWeights,
        layer_idx: usize,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.linear_key_head_dim == 128,
            "GLM-5 KDA requires head_dim=128"
        );
        ensure!(
            config.linear_num_key_heads == config.linear_num_value_heads,
            "GLM-5 KDA requires equal Q/K/V head counts"
        );
        ensure!(config.hc_mult == 4, "GLM-5 KDA requires hc_mult=4");
        let heads = config.linear_num_key_heads;
        let dim = config.linear_key_head_dim;
        let register_resident_prefill = recurrent::parse_register_resident_prefill(
            std::env::var("ATLAS_KDA_REGRESIDENT_PREFILL")
                .ok()
                .as_deref(),
        )?;
        let recurrent_regresident_k = if register_resident_prefill {
            gpu.kernel("kda", "kda_recurrent_bf16_regresident")?
        } else {
            KernelHandle(0)
        };
        let preprocess_regresident_k = if register_resident_prefill {
            ensure!(
                recurrent::scratch_is_sufficient(
                    heads,
                    dim,
                    config.num_experts_per_tok,
                    config.moe_intermediate_size,
                    config.hidden_size,
                ),
                "GLM-5 KDA register-resident prefill scratch does not fit expert buffers"
            );
            gpu.kernel("kda", "kda_preprocess_regresident")?
        } else {
            KernelHandle(0)
        };
        let layer = Self {
            input_norm,
            post_attn_norm,
            weights,
            ffn,
            hc,
            layer_idx,
            ssm_ordinal: indexed_core::ordinal(config, layer_idx)?,
            indexed_trace_logged: std::sync::atomic::AtomicBool::new(false),
            hidden_size: config.hidden_size,
            heads,
            dim,
            conv_width: config.linear_conv_kernel_dim,
            lower_bound: config.kda_gate_lower_bound,
            h_state_bytes: heads * dim * dim * 4,
            conv_state_bytes: 3 * heads * dim * config.linear_conv_kernel_dim * 4,
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batchm_k: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
            dense_gemv_batch5_k: super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batch5",
            ),
            dense_gemv_batch5_dual_k: super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batch5_dual",
            ),
            dense_gemv_batch5_triple_n_k: super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batch5_triple_n",
            ),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemv_batch5_qkv_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch5_qkv"),
            w4a16_gemv_batchm: W4a16BatchmTiers::resolve(gpu),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined"),
            conv_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv_prefill_tp_k: super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp",
            ),
            conv_prefill_tp_snap_k: super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp_snap",
            ),
            pack_k: gpu.kernel("kda", "kda_pack_qkv")?,
            conv_indexed_k: super::try_kernel(gpu, "causal_conv1d", "glm_kda_conv_indexed"),
            recurrent_k: gpu.kernel("kda", "kda_recurrent_bf16")?,
            recurrent_indexed_k: super::try_kernel(gpu, "kda", "glm_kda_recurrent_indexed"),
            recurrent_verify_snap_k: super::try_kernel(
                gpu,
                "kda",
                "kda_recurrent_bf16_verify_snap",
            ),
            preprocess_regresident_k,
            recurrent_regresident_k,
            register_resident_prefill,
            gated_norm_k: gpu.kernel("kda", "kda_sigmoid_gated_rms_norm")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_pre_k: gpu.kernel("hyper_connection", "hc_pre")?,
            hc_pre_from_raw_mix_k: gpu.kernel("hyper_connection", "hc_pre_from_raw_mix")?,
            hc_post_k: gpu.kernel("hyper_connection", "hc_post")?,
            hc_post_bf16_add_k: super::try_kernel(gpu, "hyper_connection", "hc_post_bf16_add"),
            hc_post_moe_blend_k: super::try_kernel(gpu, "hyper_connection", "hc_post_moe_blend"),
            hc_contract_k: gpu.kernel("hyper_connection", "hc_contract")?,
        };
        if crate::model::glm_independent::enabled(&config.model_type)? {
            let handles = std::array::from_fn(|i| match i + 2 {
                2 => layer.w4a16_gemv_batch2_k.0,
                3 => layer.w4a16_gemv_batch3_k.0,
                n => layer.w4a16_gemv_batchm.kernel(n as u32).0,
            });
            crate::model::glm_independent::validate_projection_handles(
                handles,
                layer.dense_gemv_batchm_k.0,
            )?;
            ensure!(
                layer.conv_indexed_k.0 != 0 && layer.recurrent_indexed_k.0 != 0,
                "independent KDA requires both indexed kernels"
            );
        }
        Ok(layer)
    }

    fn forward_inner(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let phase = self.forward_attention(
            hidden,
            state,
            tokens,
            decode,
            capture_verify_intermediates,
            ctx,
            stream,
        )?;
        self.forward_ffn(phase, ctx, stream)
    }
}

impl TransformerLayer for Glm5KdaLayer {
    fn validate_glm_owner_verify(
        &self,
        ctx: &ForwardContext,
        shape: crate::layer::glm_owner_verify::GlmOwnerBatchShape,
        stream: u64,
    ) -> Result<()> {
        let workspace = crate::layer::glm_owner_verify::GlmOwnerBatchWorkspace::new(ctx, shape)?;
        self.validate_temporal(
            &workspace.scratch,
            &[ctx; 8][..shape.owners()],
            crate::layer::glm_verify_ffn::GlmVerifyFfn::Owners(shape),
            stream,
        )
    }

    fn decode_glm_owner_verify(
        &self,
        owners: &mut [crate::layer::glm_pair_verify::GlmPairLayerInput<'_>],
        _cache: &mut PagedKvCache,
        workspace: &mut crate::layer::glm_owner_verify::GlmOwnerBatchWorkspace,
        ctx: &[&ForwardContext],
        stream: u64,
    ) -> Result<()> {
        let mode = crate::layer::glm_verify_ffn::GlmVerifyFfn::Owners(workspace.shape());
        self.decode_temporal(owners, &mut workspace.scratch, ctx, mode, stream)
    }

    fn supports_glm_pair_verify(&self) -> bool {
        self.pair_supported()
    }

    fn validate_glm_pair_verify(
        &self,
        ctx: &ForwardContext,
        mode: crate::layer::glm_pair_verify::GlmPairFfn,
        stream: u64,
    ) -> Result<()> {
        let workspace = crate::layer::glm_pair_verify::GlmPairWorkspace::new(ctx, mode)?;
        self.validate_pair(&workspace, [ctx, ctx], stream)
    }

    fn decode_glm_pair_verify(
        &self,
        owners: [crate::layer::glm_pair_verify::GlmPairLayerInput<'_>; 2],
        _cache: &mut PagedKvCache,
        workspace: &mut crate::layer::glm_pair_verify::GlmPairWorkspace,
        ctx: [&ForwardContext; 2],
        stream: u64,
    ) -> Result<()> {
        self.decode_pair(owners, workspace, ctx, stream)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_inner(hidden, state, 1, true, false, ctx, stream)
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_inner(hidden, state, num_tokens, false, false, ctx, stream)
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Speculative verify rows are one temporal sequence, not independent
        // requests. Batch the stateless projections/mHC work, but advance the
        // KDA recurrent state row-by-row and capture rollback intermediates.
        self.forward_inner(hidden, state, num_tokens, false, true, ctx, stream)
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let independent = crate::model::glm_independent::selected(ctx, num_seqs)?;
        if independent
            || crate::model::glm_c4::batched_kda_rows(
                num_seqs,
                multi_seq::enabled(),
                crate::model::glm_c4::enabled(&ctx.config.model_type),
            )?
        {
            if independent {
                crate::model::glm_independent::validate_positions(
                    seq_lens.iter().copied(),
                    num_seqs,
                    ctx.levers.max_decode_seqs as usize,
                )?;
            } else if num_seqs == 4 {
                crate::model::glm_c4::validate_positions(seq_lens.iter().copied(), 4)?;
            }
            self.decode_multi_seq_inner(hidden, num_seqs, states, ctx, stream)
        } else {
            let h = ctx.config.hidden_size;
            for i in 0..num_seqs {
                let mut bt = block_tables[i].clone();
                let mut disk = Vec::new();
                let mut last_offloaded = Vec::new();
                self.decode(
                    hidden.offset(i * h * 2),
                    residual.offset(i * h * 2),
                    states[i],
                    kv_cache,
                    seq_lens[i],
                    &mut bt,
                    &mut disk,
                    &mut last_offloaded,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16: false,
            h_prefill_stage: None,
        }))
    }
}
