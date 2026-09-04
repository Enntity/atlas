// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 KDA recurrent block for the conservative GB10 bring-up path.
mod hc;
mod multi_seq;
mod profile;
mod projection;
mod recurrent;

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

pub struct Glm5KdaLayer {
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    weights: Glm5KdaWeights,
    ffn: FfnComponent,
    hc: HcWeights,
    layer_idx: usize,
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
    pack_k: KernelHandle,
    recurrent_k: KernelHandle,
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
    hc_contract_k: KernelHandle,
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
        Ok(Self {
            input_norm,
            post_attn_norm,
            weights,
            ffn,
            hc,
            layer_idx,
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
            recurrent_k: gpu.kernel("kda", "kda_recurrent_bf16")?,
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
            hc_contract_k: gpu.kernel("hyper_connection", "hc_contract")?,
        })
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
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA expected SsmLayerState"))?;
        ensure!(!state.h_is_f16, "GLM-5 KDA requires FP32 recurrent state");
        let m = tokens as u32;
        let h = self.hidden_size as u32;
        let p = self.heads * self.dim;
        let bf16 = 2usize;
        let mut profile_timer = profile::start(ctx, stream)?;

        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                ctx.buffers.hc_streams(),
                m,
                h,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }
        self.hc_pre(&self.hc.attn, hidden, m, ctx, stream)?;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_norm,
            normed,
            m,
            h,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "hc_attn_norm")?;

        let projected = ctx.buffers.qkv_output();
        let plane_bytes = tokens * p * bf16;
        let fused_qkv = capture_verify_intermediates
            && m == 5
            && self.w4a16_gemv_batch5_qkv_k.0 != 0
            && verify_fused_qkv_enabled();
        if fused_qkv {
            ops::w4a16_gemv_batch5_qkv(
                ctx.gpu,
                self.w4a16_gemv_batch5_qkv_k,
                normed,
                &self.weights.q_proj.nvfp4,
                &self.weights.k_proj.nvfp4,
                &self.weights.v_proj.nvfp4,
                projected,
                m,
                p as u32,
                h,
                stream,
            )?;
        } else if capture_verify_intermediates {
            self.project_hot_verify(
                normed,
                &self.weights.q_proj,
                projected,
                m,
                p as u32,
                h,
                ctx,
                stream,
            )?;
        } else {
            self.project_hot(
                normed,
                &self.weights.q_proj,
                projected,
                m,
                p as u32,
                h,
                decode,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, &mut profile_timer, "q_proj")?;
        if !fused_qkv {
            if capture_verify_intermediates {
                self.project_hot_verify(
                    normed,
                    &self.weights.k_proj,
                    projected.offset(plane_bytes),
                    m,
                    p as u32,
                    h,
                    ctx,
                    stream,
                )?;
                self.project_hot_verify(
                    normed,
                    &self.weights.v_proj,
                    projected.offset(2 * plane_bytes),
                    m,
                    p as u32,
                    h,
                    ctx,
                    stream,
                )?;
            } else {
                self.project_hot(
                    normed,
                    &self.weights.k_proj,
                    projected.offset(plane_bytes),
                    m,
                    p as u32,
                    h,
                    decode,
                    ctx,
                    stream,
                )?;
                self.project_hot(
                    normed,
                    &self.weights.v_proj,
                    projected.offset(2 * plane_bytes),
                    m,
                    p as u32,
                    h,
                    decode,
                    ctx,
                    stream,
                )?;
            }
        }
        let beta = projected.offset(3 * plane_bytes);
        let fa = beta.offset(tokens * self.heads * bf16);
        let ga = fa.offset(tokens * self.dim * bf16);
        let fused_dense_pairs = capture_verify_intermediates
            && m == 5
            && self.dense_gemv_batch5_dual_k.0 != 0
            && verify_fused_dense_pairs_enabled();
        let fused_dense_triple = capture_verify_intermediates
            && m == 5
            && self.dense_gemv_batch5_triple_n_k.0 != 0
            && verify_fused_dense_triple_enabled();
        if fused_dense_triple {
            ops::dense_gemv_batch5_triple_n(
                ctx.gpu,
                self.dense_gemv_batch5_triple_n_k,
                normed,
                &self.weights.b_proj,
                &self.weights.f_a_proj,
                &self.weights.g_a_proj,
                beta,
                fa,
                ga,
                self.heads as u32,
                self.dim as u32,
                h,
                stream,
            )?;
            profile::step(ctx, stream, &mut profile_timer, "beta_f_a_g_a")?;
        } else {
            if capture_verify_intermediates {
                self.project_dense_verify(
                    normed,
                    &self.weights.b_proj,
                    beta,
                    m,
                    self.heads as u32,
                    h,
                    ctx,
                    stream,
                )?;
            } else {
                self.project_dense(
                    normed,
                    &self.weights.b_proj,
                    beta,
                    m,
                    self.heads as u32,
                    h,
                    ctx,
                    stream,
                )?;
            }
            profile::step(ctx, stream, &mut profile_timer, "kv_beta_proj")?;
            if fused_dense_pairs {
                ops::dense_gemv_batch5_dual(
                    ctx.gpu,
                    self.dense_gemv_batch5_dual_k,
                    normed,
                    normed,
                    &self.weights.f_a_proj,
                    &self.weights.g_a_proj,
                    fa,
                    ga,
                    self.dim as u32,
                    h,
                    stream,
                )?;
            } else {
                if capture_verify_intermediates {
                    self.project_dense_verify(
                        normed,
                        &self.weights.f_a_proj,
                        fa,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.project_dense(
                        normed,
                        &self.weights.f_a_proj,
                        fa,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                }
                profile::step(ctx, stream, &mut profile_timer, "f_a_proj")?;
                if capture_verify_intermediates {
                    self.project_dense_verify(
                        normed,
                        &self.weights.g_a_proj,
                        ga,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.project_dense(
                        normed,
                        &self.weights.g_a_proj,
                        ga,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                }
            }
            if fused_dense_pairs {
                profile::step(ctx, stream, &mut profile_timer, "f_a_g_a")?;
            }
        }

        let g1 = ctx.buffers.ssm_deinterleaved();
        let g2 = g1.offset(plane_bytes);
        if fused_dense_pairs {
            ops::dense_gemv_batch5_dual(
                ctx.gpu,
                self.dense_gemv_batch5_dual_k,
                fa,
                ga,
                &self.weights.f_b_proj,
                &self.weights.g_b_proj,
                g1,
                g2,
                p as u32,
                self.dim as u32,
                stream,
            )?;
        } else {
            if capture_verify_intermediates {
                self.project_dense_verify(
                    fa,
                    &self.weights.f_b_proj,
                    g1,
                    m,
                    p as u32,
                    self.dim as u32,
                    ctx,
                    stream,
                )?;
                self.project_dense_verify(
                    ga,
                    &self.weights.g_b_proj,
                    g2,
                    m,
                    p as u32,
                    self.dim as u32,
                    ctx,
                    stream,
                )?;
            } else {
                self.project_dense(
                    fa,
                    &self.weights.f_b_proj,
                    g1,
                    m,
                    p as u32,
                    self.dim as u32,
                    ctx,
                    stream,
                )?;
                self.project_dense(
                    ga,
                    &self.weights.g_b_proj,
                    g2,
                    m,
                    p as u32,
                    self.dim as u32,
                    ctx,
                    stream,
                )?;
            }
        }
        profile::step(ctx, stream, &mut profile_timer, "g_a_f_b_g_b")?;

        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(ctx.gpu, self.pack_k, projected, packed, m, p as u32, stream)?;
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        if capture_verify_intermediates {
            ensure!(
                state.h_state_intermediates.len() + 1 >= tokens
                    && state.conv_state_intermediates.len() >= tokens,
                "GLM-5 KDA verify needs K-1 h and K conv intermediates (h={}, conv={}, K={tokens})",
                state.h_state_intermediates.len(),
                state.conv_state_intermediates.len(),
            );
            let packed_row_bytes = 3 * p * bf16;
            let gate_row_bytes = p * bf16;
            let beta_row_bytes = self.heads * bf16;
            let conv_inter_base = state.conv_state_intermediates[0];
            let h_inter_base = state.h_state_intermediates[0];
            let fused_conv_snapshots = tokens == 5
                && self.conv_prefill_tp_snap_k.0 != 0
                && verify_batched_conv_snapshot_enabled()
                && !conv_inter_base.is_null()
                && state.conv_state_intermediates[..tokens - 1]
                    .iter()
                    .enumerate()
                    .all(|(t, ptr)| {
                        ptr.0 == conv_inter_base.0 + (t * self.conv_state_bytes) as u64
                    });
            let fused_recurrent_snapshots = tokens == 5
                && self.recurrent_verify_snap_k.0 != 0
                && verify_batched_recurrent_snapshot_enabled()
                && !h_inter_base.is_null()
                && state.h_state_intermediates[..tokens - 1]
                    .iter()
                    .enumerate()
                    .all(|(t, ptr)| ptr.0 == h_inter_base.0 + (t * self.h_state_bytes) as u64);

            if fused_conv_snapshots || fused_recurrent_snapshots {
                if fused_conv_snapshots {
                    ops::conv1d_update_prefill_tp_snap(
                        ctx.gpu,
                        self.conv_prefill_tp_snap_k,
                        state.conv_state,
                        packed,
                        &self.weights.conv,
                        DevicePtr::NULL,
                        convolved,
                        conv_inter_base,
                        self.conv_state_bytes / size_of::<f32>(),
                        (3 * p) as u32,
                        self.conv_width as u32,
                        m,
                        (3 * p) as u32,
                        (3 * p) as u32,
                        stream,
                    )?;
                } else {
                    for t in 0..tokens {
                        ops::conv1d_update_prefill(
                            ctx.gpu,
                            self.conv_prefill_k,
                            self.conv_prefill_tp_k,
                            state.conv_state,
                            packed.offset(t * packed_row_bytes),
                            &self.weights.conv,
                            DevicePtr::NULL,
                            convolved.offset(t * packed_row_bytes),
                            (3 * p) as u32,
                            self.conv_width as u32,
                            1,
                            (3 * p) as u32,
                            (3 * p) as u32,
                            stream,
                        )?;
                        if t + 1 < tokens {
                            ctx.gpu.copy_d2d_async(
                                state.conv_state,
                                state.conv_state_intermediates[t],
                                self.conv_state_bytes,
                                stream,
                            )?;
                        }
                    }
                }

                if fused_recurrent_snapshots {
                    ops::kda_recurrent_verify_snap(
                        ctx.gpu,
                        self.recurrent_verify_snap_k,
                        convolved,
                        g1,
                        beta,
                        self.weights.a_log.weight,
                        self.weights.dt_bias.weight,
                        state.h_state,
                        core_out,
                        h_inter_base,
                        self.h_state_bytes / size_of::<f32>(),
                        m,
                        self.heads as u32,
                        self.dim as u32,
                        self.lower_bound,
                        stream,
                    )?;
                } else {
                    for t in 0..tokens {
                        self.run_recurrent(
                            convolved.offset(t * packed_row_bytes),
                            g1.offset(t * gate_row_bytes),
                            beta.offset(t * beta_row_bytes),
                            state.h_state,
                            core_out.offset(t * gate_row_bytes),
                            1,
                            true,
                            ctx,
                            stream,
                        )?;
                        if t + 1 < tokens {
                            ctx.gpu.copy_d2d_async(
                                state.h_state,
                                state.h_state_intermediates[t],
                                self.h_state_bytes,
                                stream,
                            )?;
                        }
                    }
                }
            } else {
                // Preserve the original interleaved path as the exact fallback.
                for t in 0..tokens {
                    ops::conv1d_update_prefill(
                        ctx.gpu,
                        self.conv_prefill_k,
                        self.conv_prefill_tp_k,
                        state.conv_state,
                        packed.offset(t * packed_row_bytes),
                        &self.weights.conv,
                        DevicePtr::NULL,
                        convolved.offset(t * packed_row_bytes),
                        (3 * p) as u32,
                        self.conv_width as u32,
                        1,
                        (3 * p) as u32,
                        (3 * p) as u32,
                        stream,
                    )?;
                    self.run_recurrent(
                        convolved.offset(t * packed_row_bytes),
                        g1.offset(t * gate_row_bytes),
                        beta.offset(t * beta_row_bytes),
                        state.h_state,
                        core_out.offset(t * gate_row_bytes),
                        1,
                        true,
                        ctx,
                        stream,
                    )?;
                    // A partial accept can select states after rows 0..K-2;
                    // the post-row K-1 state is already canonical on full accept.
                    if t + 1 < tokens {
                        ctx.gpu.copy_d2d_async(
                            state.h_state,
                            state.h_state_intermediates[t],
                            self.h_state_bytes,
                            stream,
                        )?;
                        ctx.gpu.copy_d2d_async(
                            state.conv_state,
                            state.conv_state_intermediates[t],
                            self.conv_state_bytes,
                            stream,
                        )?;
                    }
                }
            }
        } else {
            ops::conv1d_update_prefill(
                ctx.gpu,
                self.conv_prefill_k,
                self.conv_prefill_tp_k,
                state.conv_state,
                packed,
                &self.weights.conv,
                DevicePtr::NULL,
                convolved,
                (3 * p) as u32,
                self.conv_width as u32,
                m,
                (3 * p) as u32,
                (3 * p) as u32,
                stream,
            )?;
            self.run_recurrent(
                convolved,
                g1,
                beta,
                state.h_state,
                core_out,
                m,
                decode,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, &mut profile_timer, "pack_conv")?;
        profile::step(ctx, stream, &mut profile_timer, "recurrent")?;
        let gated = projected;
        ops::kda_sigmoid_gated_norm(
            ctx.gpu,
            self.gated_norm_k,
            core_out,
            g2,
            self.weights.o_norm.weight,
            gated,
            m,
            self.heads as u32,
            self.dim as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "gated_norm")?;
        if capture_verify_intermediates {
            self.project_hot_verify(
                gated,
                &self.weights.o_proj,
                normed,
                m,
                h,
                p as u32,
                ctx,
                stream,
            )?;
        } else {
            self.project_hot(
                gated,
                &self.weights.o_proj,
                normed,
                m,
                h,
                p as u32,
                decode,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, &mut profile_timer, "o_proj")?;
        let fused_tp_hc = capture_verify_intermediates
            && tokens == 5
            && !ctx.graph_capture
            && verify_fused_tp_hc_enabled()
            && self.hc_post_bf16_add_k.0 != 0
            && ctx.config.tp_world_size == 2
            && ctx
                .comm
                .is_some_and(|comm| comm.world_size() == 2 && comm.supports_peer_exchange_async());
        if fused_tp_hc {
            // `moe_output` is a registered, caller-owned [M,H] BF16 arena.
            // The preceding KDA work is done with it, and FFN overwrites it
            // only after this immediate mHC post consumes the peer payload.
            ctx.comm.unwrap().peer_exchange_async(
                normed.0,
                ctx.buffers.moe_output().0,
                tokens * self.hidden_size * 2,
                stream,
            )?;
        } else if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(normed.0, tokens * self.hidden_size * 2, stream)?;
        }
        profile::step(ctx, stream, &mut profile_timer, "tp_reduce")?;
        if fused_tp_hc {
            ops::hc_post_bf16_add(
                ctx.gpu,
                self.hc_post_bf16_add_k,
                normed,
                ctx.buffers.moe_output(),
                ctx.buffers.hc_streams(),
                ctx.buffers.hc_post(),
                ctx.buffers.hc_comb(),
                ctx.buffers.hc_streams(),
                m,
                h,
                self.hc.hc_mult as u32,
                stream,
            )?;
        } else {
            self.hc_post(normed, m, ctx, stream)?;
        }
        profile::step(ctx, stream, &mut profile_timer, "hc_attn_post")?;

        self.hc_pre(&self.hc.ffn, hidden, m, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.post_attn_norm,
            normed,
            m,
            h,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "hc_ffn_norm")?;
        let ffn_out = if capture_verify_intermediates && tokens == 2 {
            self.ffn.forward_k2(normed, ctx, stream)?;
            ctx.buffers.moe_output()
        } else if capture_verify_intermediates && tokens == 3 {
            self.ffn.forward_k3(normed, ctx, stream)?;
            ctx.buffers.moe_output()
        } else if capture_verify_intermediates && tokens == 4 && verify_batched_ffn_enabled() {
            // Dense layers use one batch4 GEMV; GLM MoE layers preserve the
            // parallel K2 routed path while evaluating the shared expert once.
            self.ffn.forward_k4(normed, ctx, stream)?
        } else if capture_verify_intermediates && tokens == 5 && verify_batched_ffn_enabled() {
            self.ffn.forward_k5(normed, ctx, stream)?
        } else if capture_verify_intermediates {
            self.ffn.forward_batched(normed, tokens, ctx, stream)?;
            ctx.buffers.moe_output()
        } else if decode {
            self.ffn.forward(normed, ctx, stream)?
        } else {
            self.ffn.forward_prefill(normed, tokens, ctx, stream)?;
            ctx.buffers.moe_output()
        };
        profile::step(ctx, stream, &mut profile_timer, "ffn")?;
        self.hc_post(ffn_out, m, ctx, stream)?;
        profile::step(ctx, stream, &mut profile_timer, "hc_ffn_post")?;

        if self.layer_idx + 1 == ctx.config.num_hidden_layers {
            ops::hc_contract(
                ctx.gpu,
                self.hc_contract_k,
                ctx.buffers.hc_streams(),
                hidden,
                m,
                h,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }
        Ok(())
    }
}

impl TransformerLayer for Glm5KdaLayer {
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
        if multi_seq::enabled() && (2..=3).contains(&num_seqs) {
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
