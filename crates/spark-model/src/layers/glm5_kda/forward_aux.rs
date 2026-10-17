// SPDX-License-Identifier: AGPL-3.0-only

//! The KDA side projections (beta / f_a / g_a, then f_b / g_b) of the
//! attention rows. Split out of `forward_attention.rs` (500-LoC cap).

use super::*;

impl Glm5KdaLayer {
    /// Project `normed` into `beta`, `fa` and `ga`, then `fa` / `ga` into the
    /// returned `(g1, g2)` gate planes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_attention_aux(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        [beta, fa, ga]: [DevicePtr; 3],
        tokens: usize,
        capture_verify_intermediates: bool,
        verify_proj: bool,
        canonical: bool,
        upstream: Option<usize>,
        profile_timer: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        use spark_runtime::weights::exl3::ffn_io::DiagnosticDtype::Bf16;
        let m = tokens as u32;
        let h = self.hidden_size as u32;
        let p = self.heads * self.dim;
        let plane_bytes = tokens * p * 2;
        let fused_dense_pairs = capture_verify_intermediates
            && !canonical
            && m == 5
            && self.dense_gemv_batch5_dual_k.0 != 0
            && verify_fused_dense_pairs_enabled();
        let fused_dense_triple = capture_verify_intermediates
            && !canonical
            && m == 5
            && self.dense_gemv_batch5_triple_n_k.0 != 0
            && verify_fused_dense_triple_enabled();
        // Verify rows 2..=8 otherwise: the same batchm body per plane, one grid.
        let batchm_fused = capture_verify_intermediates
            && !canonical
            && (2..=ops::DENSE_GEMV_BATCHM_MAX_M).contains(&m)
            && self.dense_gemv_batchm_triple_n_k.0 != 0
            && self.dense_gemv_batchm_dual_k.0 != 0;
        // EXL3 takes the merged reference-ordered GEMM for every row count,
        // ahead of the verify GEMV fusions; FB/GB dispatch below is unchanged.
        if let Some(aux) = &self.native_aux {
            aux.forward(ctx.gpu, normed, [beta, fa, ga], tokens, stream)?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if canonical && verify_proj {
            // One canonical launch for the three side projections.
            let (heads, dim) = (self.heads as u32, self.dim as u32);
            crate::layers::canonical_verify::dense_planes(
                ctx.gpu,
                &[
                    (normed, &self.weights.b_proj, beta, heads, h),
                    (normed, &self.weights.f_a_proj, fa, dim, h),
                    (normed, &self.weights.g_a_proj, ga, dim, h),
                ],
                m,
                stream,
            )?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if batchm_fused && !fused_dense_triple {
            ops::dense_gemv_batchm_triple_n(
                ctx.gpu,
                self.dense_gemv_batchm_triple_n_k,
                normed,
                [
                    &self.weights.b_proj,
                    &self.weights.f_a_proj,
                    &self.weights.g_a_proj,
                ],
                [beta, fa, ga],
                m,
                [self.heads as u32, self.dim as u32],
                h,
                stream,
            )?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if fused_dense_triple {
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
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if !verify_proj
            && m > 1
            && self.dense_gemm_pipelined_k.0 != 0
            && self.dense_gemm_pipelined_triple_n_k.0 != 0
        {
            // The three pipelined launches `project_dense` would make, in one grid.
            ops::dense_gemm_pipelined_triple_n(
                ctx.gpu,
                self.dense_gemm_pipelined_triple_n_k,
                normed,
                [
                    &self.weights.b_proj,
                    &self.weights.f_a_proj,
                    &self.weights.g_a_proj,
                ],
                [beta, fa, ga],
                m,
                [self.heads as u32, self.dim as u32],
                h,
                stream,
            )?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else {
            if verify_proj {
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
            profile::step(ctx, stream, profile_timer, "kv_beta_proj")?;
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
                if verify_proj {
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
                profile::step(ctx, stream, profile_timer, "f_a_proj")?;
                if verify_proj {
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
                profile::step(ctx, stream, profile_timer, "f_a_g_a")?;
            }
        }

        self.capture_upstream(
            upstream,
            "attention_projections",
            &[
                ("qkv", Bf16, projected, &[3, tokens, p]),
                ("beta", Bf16, beta, &[tokens, self.heads]),
                ("fa", Bf16, fa, &[tokens, self.dim]),
                ("ga", Bf16, ga, &[tokens, self.dim]),
            ],
            ctx,
            stream,
        )?;
        let g1 = ctx.buffers.ssm_deinterleaved();
        let g2 = g1.offset(plane_bytes);
        if canonical && verify_proj {
            let dim = self.dim as u32;
            crate::layers::canonical_verify::dense_planes(
                ctx.gpu,
                &[
                    (fa, &self.weights.f_b_proj, g1, p as u32, dim),
                    (ga, &self.weights.g_b_proj, g2, p as u32, dim),
                ],
                m,
                stream,
            )?;
        } else if batchm_fused && !fused_dense_pairs {
            ops::dense_gemv_batchm_dual(
                ctx.gpu,
                self.dense_gemv_batchm_dual_k,
                [fa, ga],
                [&self.weights.f_b_proj, &self.weights.g_b_proj],
                [g1, g2],
                m,
                p as u32,
                self.dim as u32,
                stream,
            )?;
        } else if fused_dense_pairs {
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
        } else if verify_proj {
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
        profile::step(ctx, stream, profile_timer, "g_a_f_b_g_b")?;
        Ok((g1, g2))
    }
}
