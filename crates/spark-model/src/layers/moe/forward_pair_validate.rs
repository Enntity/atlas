// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed physical workspace checks; not sequence, generation or Model authority.
use super::*;

pub(super) fn validate_common(input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
    anyhow::ensure!(
        super::prequant_fp4::glm_grouped_shape(ctx.config)
            && ctx.config.tp_rank == ctx.config.ep_rank
            && ctx
                .comm
                .is_some_and(|c| c.world_size() == 2 && c.rank() == ctx.config.ep_rank)
            && ctx.ssm_batch.is_none()
            && !ctx.graph_capture
            && !ctx.profile
            && !ctx.gpu.stream_is_capturing(stream)
            && ctx.routed_lora_layers.is_none()
            && matches!(ctx.moe_lora_route, crate::layer::MoeLoraRoute::Skip)
            && input == ctx.buffers.norm_output(),
        "paired FFN requires eager native GLM TP2/EP2 ten normalized rows without adapters"
    );
    anyhow::ensure!(
        !super::dump::enabled(),
        "paired FFN forbids diagnostic readback"
    );
    let s = ctx.buffers.sizes();
    let a = ctx.buffers;
    // Requirements include the exact packed-A+scale and post-SiLU staging
    // lifetimes, not merely the final BF16 row extents. No allocation here.
    let spans = [
        (a.norm_output(), s.norm_output, 81920),
        (a.moe_output(), s.moe_output, 81920),
        (
            a.gate_logits(),
            s.gate_logits,
            5760, // Router logits exceed the 2116-byte routing metadata extent.
        ),
        (a.moe_router_in_f32(), s.moe_router_in_f32, 10256),
        (a.expert_gate_out(), s.expert_gate_out, 80 * 2048 * 2),
        (a.expert_up_out(), s.expert_up_out, 80 * 2048 * 2),
        (a.expert_down_out(), s.expert_down_out, 80 * 4096 * 2),
        (a.ssm_deinterleaved(), s.ssm_deinterleaved, 10 * 2048 * 2),
        (a.ssm_qkvz(), s.ssm_qkvz, 10 * 2048 * 2),
        (a.attn_output(), s.attn_output, 81920),
        (a.scratch(), s.scratch, 80 * 8),
    ];
    for (index, &(ptr, capacity, required)) in spans.iter().enumerate() {
        anyhow::ensure!(
            !ptr.is_null() && ptr.0 % 16 == 0 && capacity >= required,
            "paired FFN arena {index} capacity/alignment"
        );
        let end = ptr
            .0
            .checked_add(capacity as u64)
            .ok_or_else(|| anyhow::anyhow!("paired FFN arena extent overflow"))?;
        for &(other, other_capacity, _) in &spans[..index] {
            let other_end = other
                .0
                .checked_add(other_capacity as u64)
                .ok_or_else(|| anyhow::anyhow!("paired FFN arena extent overflow"))?;
            anyhow::ensure!(
                end <= other.0 || other_end <= ptr.0,
                "paired FFN concurrent arena alias"
            );
        }
        // Shared mHC survives FFN. Pair workspace ownership beyond these
        // original arena spans is checked by the actual layer/pair caller.
        for (other, bytes) in [
            (a.hc_streams(), s.hc_streams),
            (a.hc_post(), s.hc_post),
            (a.hc_comb(), s.hc_comb),
        ] {
            if bytes != 0 {
                let other_end = other
                    .0
                    .checked_add(bytes as u64)
                    .ok_or_else(|| anyhow::anyhow!("paired mHC extent overflow"))?;
                anyhow::ensure!(
                    end <= other.0 || other_end <= ptr.0,
                    "paired FFN aliases live mHC"
                );
            }
        }
    }
    Ok(())
}

impl MoeLayer {
    /// Additional read-only checks for the unchanged TwoK5 control. Both modes
    /// retain generic shared-T arithmetic after native exact-K5 parity failed.
    pub(super) fn validate_pair_k5_control(&self) -> Result<()> {
        for (key, expected) in [
            ("ATLAS_GLM_K5_GROUPED_MOE", "1"),
            ("ATLAS_GLM_K5_BATCHED_SHARED", "0"),
        ] {
            anyhow::ensure!(
                std::env::var(key).as_deref() == Ok(expected),
                "paired TwoK5 control requires {key}={expected}"
            );
        }
        anyhow::ensure!(
            !self.m5_projections.router.enabled() && !self.m16_gate_up.enabled(),
            "paired TwoK5 control excludes alternative BN4/M16 projection policies"
        );
        self.validate_pair_shared()?;
        // Router M5 may be selected by its already-latched production flag.
        // Require both actual router exports without duplicating the OnceLock.
        anyhow::ensure!(
            self.dense_gemm_router_m5.0 != 0,
            "paired TwoK5 control requires actual M5 router handle"
        );
        Ok(())
    }

    fn validate_pair_shared(&self) -> Result<()> {
        anyhow::ensure!(
            std::env::var("ATLAS_GLM_K5_BATCHED_SHARED").as_deref() == Ok("0"),
            "paired FFN preserves BATCHED_SHARED=0 generic-T control"
        );
        anyhow::ensure!(
            !self.m5_projections.shared.enabled() && !self.m16_gate_up.enabled(),
            "paired FFN excludes alternative shared/routed M16 policies"
        );
        anyhow::ensure!(
            self.shared_gate_fp8.is_none()
                && self.shared_up_fp8.is_none()
                && self.shared_down_fp8.is_none()
                && self.bf16_shared_expert.is_none(),
            "paired FFN requires generic shared native-T weights"
        );
        for weight in [&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t] {
            anyhow::ensure!(
                weight.as_ref().is_some_and(|w| !w.is_null()
                    && !w.weight_scale.is_null()
                    && w.weight.0.is_multiple_of(16)
                    && w.weight_scale.0.is_multiple_of(16)
                    && w.weight_scale_2.is_finite()
                    && !w.has_per_row_scale2()),
                "paired FFN shared-T projection missing or incompatible"
            );
        }
        anyhow::ensure!(
            self.w4a16_gemm_t.0 != 0,
            "paired FFN requires actual generic shared-T handle"
        );
        Ok(())
    }

    pub(super) fn validate_pair_verify(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        validate_common(input, ctx, stream)?;
        self.validate_pair_shared()?;
        super::forward_independent::validate_independent_environment()?;
        anyhow::ensure!(
            self.glm_grouped_resources(ctx)
                && self.use_t_layout_for_prefill()
                && matches!(self.btile_storage, gate_up_repack::Storage::Legacy)
                && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
                && !self.is_dflash_capture_layer,
            "paired FFN requires resident legacy-T native FP4 resources"
        );
        let fused = if self.nvfp4_vecscale {
            self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up
        } else {
            self.moe_w4a4_prequant_t_k64_compact_gate_up
        };
        let down = if self.nvfp4_vecscale {
            self.moe_w4a4_prequant_t_k64_vecscale
        } else {
            self.moe_w4a4_prequant_t_k64
        };
        for handle in [
            self.w4a16_gemm_t,
            self.moe_act_mul,
            self.dense_gemm_router,
            self.moe_topk_sigmoid_batched_k,
            self.moe_sort_by_expert,
            self.moe_build_tile_worklist_k,
            self.quantize_nvfp4_k,
            self.silu_mul_quant_nvfp4_k,
            fused,
            down,
            self.moe_unpermute_reduce_ep,
            self.moe_batched_blend,
        ] {
            anyhow::ensure!(handle.0 != 0, "paired FFN selected handle missing");
        }
        Ok(())
    }
}
