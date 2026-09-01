// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::GpuBackend;
use std::sync::Arc;

use super::types::{
    Glm5Layer, GlmAttentionWeights, GlmFfn, GlmHcWeights, GlmKernels, GlmSharedExpertSchedule,
};
use crate::layers::ops::{
    GLM53_DSA_CAUSAL_MLA_PREFILL_ENTRY, GLM53_DSA_MODULE, GLM53_DSA_POOL_ENTRY,
    GLM53_DSA_PREFILL_FAST_MODULE, GLM53_DSA_PREFILL_MODULE, GLM53_DSA_PREFILL_TC_MODULE,
    GLM53_DSA_PREFILL_TC_SCORES_ENTRY, GLM53_DSA_PREFILL_TC_VALUES_ENTRY, GLM53_DSA_SCORE_ENTRY,
    GLM53_DSA_SCORE_PREFILL_ENTRY, GLM53_DSA_SPARSE_MLA_ENTRY, GLM53_DSA_SPARSE_MLA_PREFILL_ENTRY,
    GLM53_DSA_SPARSE_MLA_PREFILL_WARP_ENTRY, GLM53_DSA_TOPK_ENTRY, GLM53_DSA_TOPK_PREFILL_ENTRY,
    GLM53_DSA_VERIFY_MULTI_LATENT_ENTRY, GLM53_DSA_VERIFY_MULTI_MODULE,
    GLM53_DSA_VERIFY_MULTI_POOL_ENTRY, GLM53_DSA_VERIFY_MULTI_SCORE_ENTRY,
    GLM53_DSA_VERIFY_MULTI_SPARSE_ENTRY, GLM53_KDA_FUSED_ENTRY, GLM53_KDA_MODULE,
    GLM53_KDA_VERIFY_NORM_ENTRY, GLM53_KDA_VERIFY_PREPARE_ENTRY, GLM53_KDA_VERIFY_RECURRENT_ENTRY,
    GLM53_KDA_VERIFY_TILED_MODULE,
};
use crate::weight_map::DenseWeight;

impl GlmKernels {
    fn resolve(gpu: &dyn GpuBackend) -> anyhow::Result<Self> {
        Ok(Self {
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm_f32out: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            hc_expand: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_pre: gpu.kernel("hyper_connection", "hc_pre")?,
            hc_pre_sqsum: gpu.kernel("hyper_connection", "glm53_hc_pre_sqsum")?,
            hc_pre_finish_norm: gpu.kernel("hyper_connection", "glm53_hc_pre_finish_norm")?,
            hc_post: gpu.kernel("hyper_connection", "hc_post")?,
            hc_mean: gpu.kernel("glm53_residual", "glm53_hc_mean")?,
            kda_decode: gpu.kernel(GLM53_KDA_MODULE, GLM53_KDA_FUSED_ENTRY)?,
            kda_verify_prepare: gpu.kernel(
                GLM53_KDA_VERIFY_TILED_MODULE,
                GLM53_KDA_VERIFY_PREPARE_ENTRY,
            )?,
            kda_verify_recurrent_tiled: gpu.kernel(
                GLM53_KDA_VERIFY_TILED_MODULE,
                GLM53_KDA_VERIFY_RECURRENT_ENTRY,
            )?,
            kda_verify_norm: gpu
                .kernel(GLM53_KDA_VERIFY_TILED_MODULE, GLM53_KDA_VERIFY_NORM_ENTRY)?,
            kda_conv: gpu.kernel("glm53_kda_prefill", "glm53_kda_conv_silu_chunk")?,
            kda_beta_transpose: gpu.kernel("glm53_kda_prefill", "glm53_kda_beta_transpose")?,
            kda_gated_norm: gpu.kernel("glm53_kda_prefill", "glm53_kda_gated_norm_chunk")?,
            kda_split_merged: gpu.kernel("glm53_kda_projection", "glm53_kda_split_merged_bf16")?,
            dsa_pool: gpu.kernel(GLM53_DSA_MODULE, GLM53_DSA_POOL_ENTRY)?,
            dsa_score: gpu.kernel(GLM53_DSA_MODULE, GLM53_DSA_SCORE_ENTRY)?,
            dsa_topk: gpu.kernel(GLM53_DSA_MODULE, GLM53_DSA_TOPK_ENTRY)?,
            dsa_sparse_mla: gpu.kernel(GLM53_DSA_MODULE, GLM53_DSA_SPARSE_MLA_ENTRY)?,
            dsa_score_prefill: gpu
                .kernel(GLM53_DSA_PREFILL_MODULE, GLM53_DSA_SCORE_PREFILL_ENTRY)?,
            dsa_topk_prefill: gpu.kernel(GLM53_DSA_PREFILL_MODULE, GLM53_DSA_TOPK_PREFILL_ENTRY)?,
            dsa_sparse_mla_prefill: gpu
                .kernel(GLM53_DSA_PREFILL_MODULE, GLM53_DSA_SPARSE_MLA_PREFILL_ENTRY)?,
            dsa_causal_mla_prefill: gpu.kernel(
                GLM53_DSA_PREFILL_FAST_MODULE,
                GLM53_DSA_CAUSAL_MLA_PREFILL_ENTRY,
            )?,
            dsa_sparse_mla_prefill_warp: gpu.kernel(
                GLM53_DSA_PREFILL_FAST_MODULE,
                GLM53_DSA_SPARSE_MLA_PREFILL_WARP_ENTRY,
            )?,
            dsa_prefill_tc_scores: gpu.kernel(
                GLM53_DSA_PREFILL_TC_MODULE,
                GLM53_DSA_PREFILL_TC_SCORES_ENTRY,
            )?,
            dsa_prefill_tc_values: gpu.kernel(
                GLM53_DSA_PREFILL_TC_MODULE,
                GLM53_DSA_PREFILL_TC_VALUES_ENTRY,
            )?,
            dsa_latent_append: gpu.kernel("glm53_dsa_projection", "glm53_dsa_latent_append")?,
            dsa_verify_multi_latent_append: gpu.kernel(
                GLM53_DSA_VERIFY_MULTI_MODULE,
                GLM53_DSA_VERIFY_MULTI_LATENT_ENTRY,
            )?,
            dsa_verify_multi_pool: gpu.kernel(
                GLM53_DSA_VERIFY_MULTI_MODULE,
                GLM53_DSA_VERIFY_MULTI_POOL_ENTRY,
            )?,
            dsa_verify_multi_score: gpu.kernel(
                GLM53_DSA_VERIFY_MULTI_MODULE,
                GLM53_DSA_VERIFY_MULTI_SCORE_ENTRY,
            )?,
            dsa_verify_multi_sparse_mla: gpu.kernel(
                GLM53_DSA_VERIFY_MULTI_MODULE,
                GLM53_DSA_VERIFY_MULTI_SPARSE_ENTRY,
            )?,
            dsa_index_norm: gpu.kernel("glm53_dsa_projection", "glm53_dsa_index_layernorm")?,
            dsa_absorb_query: gpu.kernel("glm53_dsa_projection", "glm53_dsa_absorb_query_bf16")?,
            dsa_expand_value: gpu.kernel("glm53_dsa_projection", "glm53_dsa_expand_value_bf16")?,
            silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            moe_topk_sigmoid_batched_f32: gpu
                .kernel("glm53_router", "glm53_moe_topk_sigmoid_batched_f32")?,
            exl3_bf16_to_fp16: gpu.kernel("glm53_exl3_moe", "glm53_exl3_bf16_to_fp16")?,
            exl3_prepare_routes: gpu.kernel("glm53_exl3_moe", "glm53_exl3_prepare_routes")?,
            exl3_moe: gpu.kernel("glm53_exl3_moe", "glm53_exl3_moe")?,
            exl3_fat_gather: gpu.kernel("glm53_exl3_fat", "glm53_exl3_fat_gather")?,
            exl3_fat_gate_up: gpu.kernel("glm53_exl3_fat", "glm53_exl3_fat_gemm_gate_up")?,
            exl3_fat_activate: gpu.kernel("glm53_exl3_fat", "glm53_exl3_fat_activate_down_had")?,
            exl3_fat_down: gpu.kernel("glm53_exl3_fat", "glm53_exl3_fat_gemm_down_scatter")?,
            exl3_fp32_to_bf16: gpu.kernel("glm53_exl3_moe", "glm53_exl3_fp32_to_bf16")?,
        })
    }
}

/// Resolve every kernel required by the native GLM-5.3 execution path.
///
/// Startup calls this before loading checkpoint tensors so a stale or
/// incomplete kernel target fails in seconds instead of after a 90 GiB load.
/// Layer construction uses the same resolver, keeping the preflight contract
/// identical to the handles the model will actually retain.
pub fn validate_kernel_contract(gpu: &dyn GpuBackend) -> anyhow::Result<()> {
    GlmKernels::resolve(gpu)?;
    Ok(())
}

impl Glm5Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_parts(
        layer_idx: usize,
        input_norm: DenseWeight,
        post_attention_norm: DenseWeight,
        attention: GlmAttentionWeights,
        ffn: GlmFfn,
        hc_attention: GlmHcWeights,
        hc_ffn: GlmHcWeights,
        shared_expert_schedule: Arc<GlmSharedExpertSchedule>,
        gpu: &dyn GpuBackend,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            layer_idx,
            input_norm,
            post_attention_norm,
            attention,
            ffn,
            hc_attention,
            hc_ffn,
            shared_expert_schedule,
            kernels: GlmKernels::resolve(gpu)?,
        })
    }
}
