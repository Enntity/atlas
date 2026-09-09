// SPDX-License-Identifier: AGPL-3.0-only

//! Attention through the FFN input normalization; the caller resumes FFN immediately.

use super::*;

impl Glm5KdaLayer {
    pub(super) fn forward_attention(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<FfnPhase> {
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

        let core_out = self.forward_recurrent(
            projected,
            g1,
            beta,
            state,
            tokens,
            decode,
            capture_verify_intermediates,
            ctx,
            stream,
        )?;
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
        Ok(FfnPhase {
            hidden,
            normed,
            tokens,
            decode,
            capture_verify_intermediates,
            profile_timer,
        })
    }
}
