// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::layers::qwen3_attention::{glm_k3_mla_o, types::MlaWeights};
use anyhow::ensure;
impl Qwen3AttentionLayer {
    pub(super) fn glm_k3_o_stage(
        &self,
        c: &MultiSeqCtx<'_>,
        mla: &MlaWeights,
        long_verify: bool,
        output: DevicePtr,
    ) -> Result<Option<glm_k3_mla_o::StagePlan>> {
        let compare = glm_k3_mla_o::compare::enabled(&c.fwd.config.model_type)?;
        // Wider (DFlash) verifies keep the scalar per-row O projection.
        if !glm_k3_mla_o::enabled(&c.fwd.config.model_type)? || !long_verify || c.n != 3 {
            return Ok(None);
        }
        // validate_glm_long_verify already establishes causal consecutive rows,
        // repaired eager K3, TP2/EP2, BF16 caches and index head_dim128.
        ensure!(
            c.n == 3
                && c.h == 4096
                && c.bf16 == 2
                && c.nq == 32
                && mla.v_dim == 256
                && mla.rope == 0
                && mla.o_lora_rank == 0
                && mla.wo_nvfp4.is_none()
                && mla.wo.weight.0 != 0
                && mla.wo.weight.0 % 16 == 0
                && self.dense_gemv_batchm_k.0 != 0,
            "ATLAS_GLM_K3_MLA_O_BATCHM: unqualified K3 O geometry/weight/kernel"
        );
        let b = c.fwd.buffers;
        let s = b.sizes();
        // BufferArena allocates these separately. Check their actual spans too:
        // no staged row may overlap a later row's Q/K/index/attention scratch.
        // glm_index_decode_update's ONLY ssm_qkvz writes are BF16 key[128]
        // and gate[128] in bytes0..512. Selection uses other arenas below.
        let live = [
            (c.normed, 3 * 4096 * 2),
            (b.ssm_ba(), s.ssm_ba),
            (b.ssm_deinterleaved(), s.ssm_deinterleaved),
            (b.ssm_conv_out_f32(), s.ssm_conv_out_f32),
            (b.expert_up_out(), s.expert_up_out),
            (b.expert_gate_out(), s.expert_gate_out),
            (b.qkv_output(), s.qkv_output),
            (b.expert_down_out(), s.expert_down_out),
            (b.ssm_gates(), s.ssm_gates),
            // Last entry becomes candidate output only after every causal row ends.
            (b.attn_output(), s.attn_output),
        ];
        let mut plan =
            glm_k3_mla_o::StagePlan::new(b.ssm_qkvz(), s.ssm_qkvz, output, s.moe_output, &live)?;
        if compare {
            ensure!(
                self.dense_gemv_k.0 != 0,
                "ATLAS_GLM_K3_MLA_O_COMPARE: scalar kernel unavailable before cache writes"
            );
            let (candidate, other_live) = live.split_last().expect("attention scratch entry");
            // Only diagnostic mode allocates this small host list. Include the resident
            // matrix to prove that the alternate output cannot overwrite the operand.
            let mut comparison_live = other_live.to_vec();
            comparison_live.push((mla.wo.weight, 4096 * 8192 * 2));
            plan = plan.with_compare(candidate.0, candidate.1, &comparison_live)?;
        }
        Ok(Some(plan))
    }
}
