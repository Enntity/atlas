// SPDX-License-Identifier: AGPL-3.0-only

//! Attention through the FFN input normalization; the caller resumes FFN immediately.

use super::*;

impl Glm5KdaLayer {
    /// The attention body over `tokens` rows. Everything here is row-local
    /// except `recurrent`, which advances whichever recurrent state(s) own the
    /// rows and returns the core output `[tokens, heads*dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_attention_rows(
        &self,
        hidden: DevicePtr,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
        recurrent: &mut dyn FnMut(DevicePtr, DevicePtr, DevicePtr) -> Result<DevicePtr>,
    ) -> Result<FfnPhase> {
        self.validate_exl3_forward(tokens, ctx, stream)?;
        let upstream = self
            .upstream_trace
            .as_ref()
            .map(|trace| trace.begin())
            .transpose()?
            .flatten();
        use spark_runtime::weights::exl3::ffn_io::DiagnosticDtype::Bf16;
        let m = tokens as u32;
        let h = self.hidden_size as u32;
        let p = self.heads * self.dim;
        // ATLAS_GLM_CANONICAL_VERIFY: one kernel family per projection for
        // every row count, K=5 seams off; a plain one-row decode takes the
        // verify projections too, so its row matches the same row verified.
        let canonical = crate::layers::canonical_verify::enabled();
        let verify_proj = capture_verify_intermediates || (decode && canonical);
        let bf16 = 2usize;
        let mut profile_timer = profile::start(ctx, stream)?;
        // Sequence-parallel prefill: the highway and `hidden` hold this rank's
        // rows compacted at row 0 (`layers::glm_sp`); `m_hc` is their count.
        let sp = crate::layers::glm_sp::current()
            .filter(|sp| !decode && !capture_verify_intermediates && tokens == sp.total());
        ensure!(
            upstream.is_none() || sp.is_none(),
            "EXL3 upstream capture requires full-row pure TP; sequence-parallel captures are unsupported"
        );
        let m_hc = sp.map_or(m, |sp| sp.rows as u32);
        let local = |x: DevicePtr| sp.map_or(x, |sp| sp.local(x, self.hidden_size));
        self.capture_upstream(
            upstream,
            "incoming_hidden",
            &[(
                "hidden",
                Bf16,
                local(hidden),
                &[m_hc as usize, self.hidden_size],
            )],
            ctx,
            stream,
        )?;

        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                local(hidden),
                ctx.buffers.hc_streams(),
                m_hc,
                h,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }
        self.capture_upstream(
            upstream,
            "incoming_highway",
            &[(
                "highway",
                Bf16,
                ctx.buffers.hc_streams(),
                &[m_hc as usize, self.hc.hc_mult, self.hidden_size],
            )],
            ctx,
            stream,
        )?;
        // ATLAS_GLM_DET_TRACE stages; `det_rows` are the seam (local) rows.
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        let (det_rows, row) = (
            (sp.map_or(0, |sp| sp.row0), m_hc as usize),
            h as usize * bf16,
        );
        let normed = ctx.buffers.norm_output();
        if !self.exl3_hc_norm(
            &self.hc.attn,
            self.input_norm.weight,
            local(normed),
            m_hc,
            ctx,
            stream,
        )? {
            self.hc_pre(&self.hc.attn, hidden, m_hc, ctx, stream)?;
            det.tap("in", hidden, det_rows, row);
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hidden,
                &self.input_norm,
                local(normed),
                m_hc,
                h,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
        }
        if let Some(sp) = sp {
            sp.all_gather(normed, self.hidden_size, ctx, stream)?;
        }
        self.capture_upstream_hc(upstream, "attn_mhc_norm", normed, tokens, ctx, stream)?;
        profile::step(ctx, stream, &mut profile_timer, "hc_attn_norm")?;

        let projected = ctx.buffers.qkv_output();
        let plane_bytes = tokens * p * bf16;
        self.forward_attention_qkv(
            normed,
            projected,
            plane_bytes,
            m,
            p,
            h,
            decode,
            verify_proj,
            &mut profile_timer,
            ctx,
            stream,
        )?;
        det.tap("x_qkv", projected, (0, tokens), 3 * p * bf16);
        let beta = projected.offset(3 * plane_bytes);
        let fa = beta.offset(tokens * self.heads * bf16);
        let ga = fa.offset(tokens * self.dim * bf16);
        let (g1, g2) = self.project_attention_aux(
            normed,
            projected,
            [beta, fa, ga],
            tokens,
            capture_verify_intermediates,
            verify_proj,
            canonical,
            upstream,
            &mut profile_timer,
            ctx,
            stream,
        )?;
        det.tap("x_g", g1, (0, tokens), 2 * p * bf16);
        self.capture_upstream(
            upstream,
            "recurrent_inputs",
            &[
                ("g1", Bf16, g1, &[tokens, p]),
                ("g2", Bf16, g2, &[tokens, p]),
            ],
            ctx,
            stream,
        )?;

        let core_out = recurrent(projected, g1, beta)?;
        self.capture_upstream(
            upstream,
            "recurrent_output",
            &[("core", Bf16, core_out, &[tokens, self.heads, self.dim])],
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
        self.capture_upstream(
            upstream,
            "gated_norm",
            &[("gated", Bf16, gated, &[tokens, p])],
            ctx,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "gated_norm")?;
        det.tap("x_gated", gated, (0, tokens), p * bf16);
        if verify_proj {
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
        self.capture_upstream(
            upstream,
            "attention_local_output",
            &[("output", Bf16, normed, &[tokens, self.hidden_size])],
            ctx,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "o_proj")?;
        det.tap("attn", normed, (0, tokens), row);
        let fused_tp_hc = self.native_mhc.is_none()
            && capture_verify_intermediates
            && !canonical
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
        } else if let Some(sp) = sp {
            sp.reduce_scatter(normed, self.hidden_size, ctx, stream)?;
        } else if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(normed.0, tokens * self.hidden_size * 2, stream)?;
        }
        self.capture_upstream(
            upstream,
            "attention_reduced_output",
            &[("output", Bf16, normed, &[tokens, self.hidden_size])],
            ctx,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "tp_reduce")?;
        if !fused_tp_hc {
            det.tap("attn_red", local(normed), det_rows, row);
        }
        let mut seam = false;
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
            seam = self.native_mhc.is_none()
                && crate::layers::qwen3_attention::hc_post_pre_prefill_fused(
                    &self.hc.ffn,
                    Some(local(normed)),
                    hidden,
                    m_hc,
                    self.hc.hc_mult as u32,
                    self.hc.sinkhorn_iters as u32,
                    self.hc.hc_eps,
                    ctx,
                    stream,
                )?;
            if !seam {
                self.hc_post(local(normed), m_hc, ctx, stream)?;
            }
        }
        self.capture_upstream(
            upstream,
            "post_attention_highway",
            &[(
                "highway",
                Bf16,
                ctx.buffers.hc_streams(),
                &[m_hc as usize, self.hc.hc_mult, self.hidden_size],
            )],
            ctx,
            stream,
        )?;
        profile::step(ctx, stream, &mut profile_timer, "hc_attn_post")?;

        if !self.exl3_hc_norm(
            &self.hc.ffn,
            self.post_attn_norm.weight,
            local(normed),
            m_hc,
            ctx,
            stream,
        )? {
            if !seam {
                self.hc_pre(&self.hc.ffn, hidden, m_hc, ctx, stream)?;
            }
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hidden,
                &self.post_attn_norm,
                local(normed),
                m_hc,
                h,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
        }
        // The routed MoE needs every row; a dense FFN runs only the local ones.
        if let Some(sp) = sp.filter(|_| matches!(self.ffn, crate::layers::FfnComponent::Moe(_))) {
            sp.all_gather(normed, self.hidden_size, ctx, stream)?;
        }
        self.capture_upstream_hc(upstream, "ffn_mhc_norm", normed, tokens, ctx, stream)?;
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
