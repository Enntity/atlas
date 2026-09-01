// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::{GpuBackend, KernelHandle};
use std::sync::Arc;

use crate::layers::ops::Glm53Exl3PointerTables;
use crate::weight_map::{DenseWeight, QuantWeight};

#[derive(Debug, Clone, Copy)]
pub struct GlmHcWeights {
    pub function: DenseWeight,
    pub function_tf32: DenseWeight,
    pub base: DenseWeight,
    pub scale: DenseWeight,
    pub use_tensor_core: bool,
}

pub struct KdaWeights {
    /// Load-time row concatenation of q/k/v/beta/forget-a/gate-a.
    ///
    /// The fixed TP2 appliance projects this once, then a tiny CUDA split
    /// kernel restores the layouts consumed by the recurrent kernels.  This
    /// removes five weight launches from every KDA layer invocation.
    pub input_merged: QuantWeight,
    pub query_conv: DenseWeight,
    pub key_conv: DenseWeight,
    pub value_conv: DenseWeight,
    pub forget_b: QuantWeight,
    pub dt_bias: DenseWeight,
    pub a_log: DenseWeight,
    pub gate_b: QuantWeight,
    pub output_norm: DenseWeight,
    pub output: QuantWeight,
}

pub struct DsaWeights {
    pub query_a: QuantWeight,
    pub query_a_norm: DenseWeight,
    pub query_b: QuantWeight,
    pub kv_a: QuantWeight,
    pub kv_a_norm: DenseWeight,
    pub kv_b: DenseWeight,
    pub output: QuantWeight,
    pub index_query: QuantWeight,
    pub index_key: QuantWeight,
    pub index_key_norm: DenseWeight,
    pub index_key_bias: DenseWeight,
    pub index_head_weights: QuantWeight,
    pub index_ape: DenseWeight,
    pub index_gates: QuantWeight,
}

pub enum GlmAttentionWeights {
    Kda(KdaWeights),
    Dsa(DsaWeights),
}

#[derive(Debug, Clone, Copy)]
pub struct GlmDenseFfnWeights {
    pub gate: DenseWeight,
    pub up: DenseWeight,
    pub down: DenseWeight,
    pub intermediate_size: usize,
}

pub struct GlmExl3MoeWeights {
    pub router: DenseWeight,
    pub correction_bias: DenseWeight,
    pub shared: GlmDenseFfnWeights,
    pub pointers: Glm53Exl3PointerTables,
    pub intermediate_size: usize,
    pub local_expert_start: usize,
    pub local_expert_end: usize,
}

pub enum GlmFfn {
    Dense(GlmDenseFfnWeights),
    Exl3(GlmExl3MoeWeights),
}

/// One model-wide auxiliary lane for the independent BF16 shared expert.
///
/// The routed EXL3 branch and the shared expert read the same normalized input
/// but do not share scratch or outputs. Running the shared branch on this lane
/// lets GB10 fill routed-expert tail bubbles with dense tensor-core work. Two
/// events join the lane to the compute stream, including during CUDA capture.
pub struct GlmSharedExpertSchedule {
    pub(super) stream: u64,
    pub(super) input_ready: u64,
    pub(super) output_ready: u64,
    pub(super) enabled: bool,
}

impl GlmSharedExpertSchedule {
    pub fn new(gpu: &dyn GpuBackend) -> anyhow::Result<Self> {
        let enabled = std::env::var("ATLAS_GLM_NO_SHARED_EXPERT_OVERLAP")
            .ok()
            .as_deref()
            != Some("1");
        Ok(Self {
            stream: gpu.create_stream()?,
            input_ready: gpu.create_event()?,
            output_ready: gpu.create_event()?,
            enabled,
        })
    }
}

pub(super) struct GlmKernels {
    pub dense_gemv: KernelHandle,
    pub dense_gemm_f32out: KernelHandle,
    pub rms_norm: KernelHandle,
    pub hc_expand: KernelHandle,
    pub hc_pre: KernelHandle,
    pub hc_pre_sqsum: KernelHandle,
    pub hc_pre_finish_norm: KernelHandle,
    pub hc_post: KernelHandle,
    pub hc_mean: KernelHandle,
    pub kda_decode: KernelHandle,
    pub kda_verify_prepare: KernelHandle,
    pub kda_verify_recurrent_tiled: KernelHandle,
    pub kda_verify_norm: KernelHandle,
    pub kda_conv: KernelHandle,
    pub kda_beta_transpose: KernelHandle,
    pub kda_gated_norm: KernelHandle,
    pub kda_split_merged: KernelHandle,
    pub dsa_pool: KernelHandle,
    pub dsa_score: KernelHandle,
    pub dsa_topk: KernelHandle,
    pub dsa_sparse_mla: KernelHandle,
    pub dsa_score_prefill: KernelHandle,
    pub dsa_topk_prefill: KernelHandle,
    pub dsa_sparse_mla_prefill: KernelHandle,
    pub dsa_causal_mla_prefill: KernelHandle,
    pub dsa_sparse_mla_prefill_warp: KernelHandle,
    pub dsa_prefill_tc_scores: KernelHandle,
    pub dsa_prefill_tc_values: KernelHandle,
    pub dsa_latent_append: KernelHandle,
    pub dsa_verify_multi_latent_append: KernelHandle,
    pub dsa_verify_multi_pool: KernelHandle,
    pub dsa_verify_multi_score: KernelHandle,
    pub dsa_verify_multi_sparse_mla: KernelHandle,
    pub dsa_index_norm: KernelHandle,
    pub dsa_absorb_query: KernelHandle,
    pub dsa_expand_value: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    pub moe_topk_sigmoid_batched_f32: KernelHandle,
    pub exl3_bf16_to_fp16: KernelHandle,
    pub exl3_prepare_routes: KernelHandle,
    pub exl3_moe: KernelHandle,
    pub exl3_fat_gather: KernelHandle,
    pub exl3_fat_gate_up: KernelHandle,
    pub exl3_fat_activate: KernelHandle,
    pub exl3_fat_down: KernelHandle,
    pub exl3_fp32_to_bf16: KernelHandle,
}

pub struct Glm5Layer {
    pub(super) layer_idx: usize,
    pub(super) input_norm: DenseWeight,
    pub(super) post_attention_norm: DenseWeight,
    pub(super) attention: GlmAttentionWeights,
    pub(super) ffn: GlmFfn,
    pub(super) hc_attention: GlmHcWeights,
    pub(super) hc_ffn: GlmHcWeights,
    pub(super) shared_expert_schedule: Arc<GlmSharedExpertSchedule>,
    pub(super) kernels: GlmKernels,
}

impl Glm5Layer {
    pub fn new(
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
        Self::from_parts(
            layer_idx,
            input_norm,
            post_attention_norm,
            attention,
            ffn,
            hc_attention,
            hc_ffn,
            shared_expert_schedule,
            gpu,
        )
    }
}
