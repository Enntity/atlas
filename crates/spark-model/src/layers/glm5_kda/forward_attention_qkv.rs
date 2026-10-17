// SPDX-License-Identifier: AGPL-3.0-only

//! Q/K/V projections of the KDA attention rows (fused K=5 verify GEMV,
//! verify projections, or the hot cast route shared across Q/K/V).

use super::*;

impl Glm5KdaLayer {
    /// Project `normed` into the Q, K and V planes of `projected`
    /// (`plane_bytes` apart).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_attention_qkv(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        plane_bytes: usize,
        m: u32,
        p: usize,
        h: u32,
        decode: bool,
        capture_verify_intermediates: bool,
        profile_timer: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let fused_qkv = !self.weights.q_proj.is_exl3()
            && capture_verify_intermediates
            && m == 5
            && self.w4a16_gemv_batch5_qkv_k.0 != 0
            && verify_fused_qkv_enabled()
            && !crate::layers::canonical_verify::enabled();
        if self.project_exl3_fused_qkv(normed, projected, m as usize, plane_bytes, ctx, stream)? {
            profile::step(ctx, stream, profile_timer, "q_proj")?;
            return Ok(());
        }
        // Whether Q took the Lt FP8 route, leaving `normed` cast in scratch.
        let mut q_fp8 = false;
        if fused_qkv {
            ops::w4a16_gemv_batch5_qkv(
                ctx.gpu,
                self.w4a16_gemv_batch5_qkv_k,
                ops::gemv_touch(ops::Twin::W4Batch5Qkv),
                normed,
                self.weights.q_proj.as_nvfp4()?.0,
                self.weights.k_proj.as_nvfp4()?.0,
                self.weights.v_proj.as_nvfp4()?.0,
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
            q_fp8 = self.project_hot_cast(
                normed,
                &self.weights.q_proj,
                projected,
                m,
                p as u32,
                h,
                decode,
                true,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, profile_timer, "q_proj")?;
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
                // K and V reuse Q's E4M3 cast of `normed` on the Lt FP8 route.
                let k_fp8 = self.project_hot_cast(
                    normed,
                    &self.weights.k_proj,
                    projected.offset(plane_bytes),
                    m,
                    p as u32,
                    h,
                    decode,
                    !q_fp8,
                    ctx,
                    stream,
                )?;
                self.project_hot_cast(
                    normed,
                    &self.weights.v_proj,
                    projected.offset(2 * plane_bytes),
                    m,
                    p as u32,
                    h,
                    decode,
                    !k_fp8,
                    ctx,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

impl Glm5KdaLayer {
    /// `ATLAS_EXL3_COOP_FUSED_QKV=1`: q|k|v as one cooperative launch over
    /// the fused reference matrix, writing the same three BF16 planes as the
    /// separate EXL3 launches. `Ok(false)` leaves them to the caller.
    fn project_exl3_fused_qkv(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        tokens: usize,
        plane_bytes: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (Glm5Projection::Exl3(q), Glm5Projection::Exl3(k), Glm5Projection::Exl3(v)) = (
            &self.weights.q_proj,
            &self.weights.k_proj,
            &self.weights.v_proj,
        ) else {
            return Ok(false);
        };
        let width = self.heads * self.dim;
        for weight in [q, k, v] {
            weight.validate_local_forward(
                tokens,
                self.hidden_size,
                width,
                stream,
                ctx.config.tp_rank,
                ctx.config.tp_world_size,
            )?;
        }
        q.launch_fused_local_bf16(
            ctx.gpu,
            k,
            v,
            normed,
            [
                projected,
                projected.offset(plane_bytes),
                projected.offset(2 * plane_bytes),
            ],
            tokens,
        )
    }
}
