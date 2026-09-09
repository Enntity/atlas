// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit shared width within the joint routed FFN; legacy K5 is separate.
use super::*;
use crate::layer::glm_pair_verify::GlmPairShared;

impl MoeLayer {
    pub(super) fn run_pair_shared(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        shared: GlmPairShared,
    ) -> Result<()> {
        let rows = match shared {
            GlmPairShared::TwoM5 => 5,
            GlmPairShared::M10 => 10,
        };
        self.run_verify_shared_rows(input, ctx, stream, 10, rows)
    }

    /// Same native-T arithmetic for explicitly checked temporal row groups.
    pub(super) fn run_verify_shared_rows(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        total_rows: usize,
        rows: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            matches!((total_rows, rows), (10, 5 | 10))
                || (total_rows == rows && matches!(rows, 15 | 20 | 25 | 30 | 35 | 40)),
            "bounded temporal shared width"
        );
        let h = ctx.config.hidden_size;
        let inter = ctx.config.shared_expert_intermediate_size;
        // Checked before any attention writer by the row/resource validator. Resolve
        // all three references before launching, preserving stop-first-error.
        let gate = self
            .shared_gate_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paired shared gate-T missing"))?;
        let up = self
            .shared_up_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paired shared up-T missing"))?;
        let down = self
            .shared_down_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paired shared down-T missing"))?;
        // All widths use the same generic-T kernel and elementwise activation;
        // wider row groups only share their existing weight scans.
        for owner in 0..(total_rows / rows) {
            let row_input = input.offset(owner * rows * h * 2);
            let gate_out = ctx
                .buffers
                .ssm_deinterleaved()
                .offset(owner * rows * inter * 2);
            let up_out = ctx.buffers.ssm_qkvz().offset(owner * rows * inter * 2);
            let down_out = ctx.buffers.attn_output().offset(owner * rows * h * 2);
            // The existing policy is preflight-validated OFF. Thus these are
            // precisely the control's nine-argument native-T GEMM launches,
            // with explicit per-owner destinations, not exact-K5 GEMVs.
            for (projection, weight, out) in [
                (shared_m16::SharedProjection::Gate, gate, gate_out),
                (shared_m16::SharedProjection::Up, up, up_out),
            ] {
                self.run_shared_m16(
                    projection,
                    row_input,
                    weight,
                    out,
                    rows as u32,
                    inter as u32,
                    h as u32,
                    ctx,
                    stream,
                    false,
                )?;
            }
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                gate_out,
                up_out,
                gate_out,
                (rows * inter) as u32,
                stream,
            )?;
            self.run_shared_m16(
                shared_m16::SharedProjection::Down,
                gate_out,
                down,
                down_out,
                rows as u32,
                h as u32,
                inter as u32,
                ctx,
                stream,
                false,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_exact_k5_shared(
        &self,
        input: DevicePtr,
        shared_gate_out: DevicePtr,
        shared_up_out: DevicePtr,
        shared_down_out: DevicePtr,
        h: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        aux: u64,
    ) -> Result<()> {
        let batch5 = self.w4a16_batchm.kernel(5);
        let fused_gate_up = self.w4a16_batch5_dual_k.0 != 0
            && std::env::var("ATLAS_GLM_K5_FUSED_SHARED_GATE_UP").as_deref() == Ok("1");
        if fused_gate_up {
            ops::w4a16_gemv_batch5_dual(
                ctx.gpu,
                self.w4a16_batch5_dual_k,
                input,
                &self.weights.shared_expert.gate_proj,
                &self.weights.shared_expert.up_proj,
                shared_gate_out,
                shared_up_out,
                5,
                shared_inter,
                h,
                aux,
            )?;
        } else {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                batch5,
                input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                5,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                batch5,
                input,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                5,
                shared_inter,
                h,
                aux,
            )?;
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            5 * shared_inter,
            aux,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            shared_gate_out,
            &self.weights.shared_expert.down_proj,
            shared_down_out,
            5,
            h,
            shared_inter,
            aux,
        )?;
        Ok(())
    }
}
