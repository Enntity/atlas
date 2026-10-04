// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit shared width within the joint routed FFN; legacy K5 is separate.
use super::*;

/// `ATLAS_GLM_SHARED_TP_SPLIT=1`. Read once. Both ranks must run the same
/// value (`model::startup_parity`): a rank splitting alone adds its half to
/// the peer's whole shared output.
pub(crate) fn shared_tp_split_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_SHARED_TP_SPLIT").as_deref() == Ok("1"))
}

/// `ATLAS_MOE_SHARED_REDUCE_OVERLAP=1`: run the shared expert on the prefill
/// stream while the EP all-reduce is in flight. A chunk that overlaps does
/// not take the TP split, so this must match across the ranks too.
pub(crate) fn shared_reduce_overlap_requested() -> bool {
    std::env::var("ATLAS_MOE_SHARED_REDUCE_OVERLAP").as_deref() == Ok("1")
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    /// Shared expert for an owner-batched verify of `rows` (9..=32) rows:
    /// the native NVFP4 projections through one tensor-core weight pass each
    /// (`kernel` from `w4a16_gemv_tiers::tc_kernel`), the arithmetic the
    /// <=8-row verify already uses.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_tc(
        &self,
        kernel: KernelHandle,
        input: DevicePtr,
        shared_gate_out: DevicePtr,
        shared_up_out: DevicePtr,
        shared_down_out: DevicePtr,
        rows: u32,
        h: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        aux: u64,
    ) -> Result<()> {
        let shared = &self.weights.shared_expert;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            input,
            &shared.gate_proj,
            shared_gate_out,
            rows,
            shared_inter,
            h,
            aux,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            input,
            &shared.up_proj,
            shared_up_out,
            rows,
            shared_inter,
            h,
            aux,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            rows * shared_inter,
            aux,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            shared_gate_out,
            &shared.down_proj,
            shared_down_out,
            rows,
            h,
            shared_inter,
            aux,
        )
    }

    /// Whether the shared expert can run TP-split for `rows` replicated rows
    /// (`ATLAS_GLM_SHARED_TP_SPLIT=1`, EP2, NVFP4 shared weights with scalar
    /// scale2, strided tensor-core tiers for `rows`).
    pub(super) fn shared_split_ready(&self, ctx: &ForwardContext, rows: u32) -> bool {
        let shared = &self.weights.shared_expert;
        let inter = ctx.config.shared_expert_intermediate_size;
        shared_tp_split_requested()
            && ctx.config.ep_world_size == 2
            && ctx.comm.is_some_and(|c| c.world_size() == 2)
            && inter.is_multiple_of(32)
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && [&shared.gate_proj, &shared.up_proj, &shared.down_proj]
                .iter()
                .all(|w| !w.is_null() && !w.has_per_row_scale2())
            && crate::layers::w4a16_gemv_tiers::tc_kernel(rows).0 != 0
            && crate::layers::w4a16_gemv_tiers::tc_ld_kernel(rows).0 != 0
    }

    /// This rank's half of the shared expert, intermediate columns
    /// `[rank * inter/2, +inter/2)`: gate and up rows (a pointer offset) and
    /// the matching K-slice of down (read through the strided tier).
    fn shared_split_weights(
        &self,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
    ) -> [QuantizedWeight; 3] {
        let shared = &self.weights.shared_expert;
        let col0 = (ctx.config.ep_rank as u32 * (inter / 2)) as usize;
        let gate_up_rows = |w: &QuantizedWeight| QuantizedWeight {
            weight: w.weight.offset(col0 * h as usize / 2),
            weight_scale: w.weight_scale.offset(col0 * h as usize / 16),
            ..*w
        };
        let down_cols = QuantizedWeight {
            weight: shared.down_proj.weight.offset(col0 / 2),
            weight_scale: shared.down_proj.weight_scale.offset(col0 / 16),
            ..shared.down_proj
        };
        [
            gate_up_rows(&shared.gate_proj),
            gate_up_rows(&shared.up_proj),
            down_cols,
        ]
    }

    /// `ATLAS_GLM_L2_AHEAD`: what the FFN of a verify of `rows` rows reads
    /// first: the NVFP4 shared expert's gate, up and down (this rank's half
    /// when it runs TP-split), then the BF16 router. A K=5 pass that defers
    /// the shared expert behind the routed experts (`k5_hc`, the caller's
    /// `forward_k5_for_hc` argument) asks for the router only. The paths that
    /// read the whole shared expert where this names the half (fixed K=2..5
    /// passes) only find less of it in L2.
    pub(crate) fn l2_ahead_lead(
        &self,
        rows: u32,
        k5_hc: bool,
        ctx: &ForwardContext,
    ) -> Vec<ops::L2Region> {
        let (h, inter) = (
            ctx.config.hidden_size as u32,
            ctx.config.shared_expert_intermediate_size as u32,
        );
        let shared = &self.weights.shared_expert;
        let mut lead = Vec::new();
        if inter > 0
            && !(rows == 5 && self.k5_defers_shared(k5_hc, ctx))
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && [&shared.gate_proj, &shared.up_proj, &shared.down_proj]
                .iter()
                .all(|w| !w.is_null())
        {
            let ([gate, up, down], n) = if self.shared_split_ready(ctx, rows) {
                (self.shared_split_weights(h, inter, ctx), inter / 2)
            } else {
                ([shared.gate_proj, shared.up_proj, shared.down_proj], inter)
            };
            for w in [&gate, &up] {
                lead.extend(ops::L2Region::nvfp4(w, n, h, h / 2, h / 16));
            }
            lead.extend(ops::L2Region::nvfp4(&down, h, n, inter / 2, inter / 16));
        }
        if self.gate_fp8.is_none()
            && self.gate_nvfp4.is_none()
            && !self.weights.gate.weight.is_null()
        {
            let router = self.router_logits_n as usize * h as usize * 2;
            lead.push(ops::L2Region::whole(self.weights.gate.weight, router));
        }
        lead
    }

    /// This rank's half of the shared expert ([`Self::shared_split_weights`]).
    /// `attn_output` then holds this rank's partial of the shared output; the
    /// EP all-reduce sums the two.
    pub(super) fn run_shared_split(
        &self,
        input: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let half = inter / 2;
        let [gate, up, down_cols] = self.shared_split_weights(h, inter, ctx);
        let (gate_out, up_out, down_out) = (
            ctx.buffers.ssm_deinterleaved(),
            ctx.buffers.ssm_qkvz(),
            ctx.buffers.attn_output(),
        );
        let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(rows);
        let gate_up = [(&gate, gate_out), (&up, up_out)];
        // ATLAS_GLM_DECODE_GEMV_BATCH: gate and up in one launch of the tc
        // body, both touched during the PDL wait (`ops::gemv_touch`).
        match ops::w4a16_pair_touch(ctx.gpu, tc, input, gate_up, rows, half, h, stream) {
            Some(done) => done?,
            None => {
                for (weight, out) in gate_up {
                    ops::w4a16_gemv_batchm(ctx.gpu, tc, input, weight, out, rows, half, h, stream)?;
                }
            }
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            rows * half,
            stream,
        )?;
        let (ld_half, ld_groups) = (inter / 2, inter / 16);
        let touch = crate::layers::w4a16_gemv_tiers::tc_rows(tc)
            .and_then(ops::w4a16_tc_twin)
            .and_then(ops::gemv_touch);
        if let Some(touch) = touch {
            return touch.w4a16_tc(
                ctx.gpu, gate_out, &down_cols, down_out, rows, h, half, ld_half, ld_groups, stream,
            );
        }
        ops::w4a16_gemv_tc_ld(
            ctx.gpu,
            crate::layers::w4a16_gemv_tiers::tc_ld_kernel(rows),
            gate_out,
            &down_cols,
            down_out,
            rows,
            h,
            half,
            ld_half,
            ld_groups,
            stream,
        )
    }

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
