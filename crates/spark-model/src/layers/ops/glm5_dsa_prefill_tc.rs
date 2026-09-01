// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-shape GB10 tensor-core sparse-MLA prefill launch contracts.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_DSA_PREFILL_TC_MODULE: &str = "glm53_dsa_prefill_tc";
pub const GLM53_DSA_PREFILL_TC_SCORES_ENTRY: &str = "glm53_dsa_prefill_tc_scores";
pub const GLM53_DSA_PREFILL_TC_VALUES_ENTRY: &str = "glm53_dsa_prefill_tc_values";
pub const GLM53_DSA_PREFILL_TC_WIDTH: usize = 2064;
const SCORE_SHARED_BYTES: u32 = 86_016;
const VALUE_SHARED_BYTES: u32 = 20_480;

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaPrefillTcArgs {
    pub query: DevicePtr,
    pub selected_indices: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub scores: DevicePtr,
    pub weights: DevicePtr,
    pub output: DevicePtr,
    pub total_tokens: u32,
    pub num_sequences: u32,
    pub attention_scale: f32,
}

impl Glm53DsaPrefillTcArgs {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.total_tokens > 0,
            "tensor-core DSA prefill requires tokens"
        );
        ensure!(
            self.num_sequences > 0,
            "tensor-core DSA prefill requires sequences"
        );
        ensure!(
            self.attention_scale.is_finite() && self.attention_scale > 0.0,
            "tensor-core DSA prefill scale must be finite and positive"
        );
        for (name, pointer) in [
            ("query", self.query),
            ("selected indices", self.selected_indices),
            ("state table", self.sequence_state_ptrs),
            ("cu_seqlens", self.cu_seqlens),
            ("scores", self.scores),
            ("weights", self.weights),
            ("output", self.output),
        ] {
            ensure!(
                !pointer.is_null(),
                "tensor-core DSA prefill {name} pointer is null"
            );
        }
        Ok(())
    }
}

pub fn glm53_dsa_prefill_tc(
    gpu: &dyn GpuBackend,
    scores_kernel: KernelHandle,
    values_kernel: KernelHandle,
    args: &Glm53DsaPrefillTcArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, scores_kernel)
        .grid([2, args.total_tokens, 1])
        .block([128, 1, 1])
        .shared_mem(SCORE_SHARED_BYTES)
        .arg_ptr(args.query)
        .arg_ptr(args.selected_indices)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.scores)
        .arg_ptr(args.weights)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .arg_f32(args.attention_scale)
        .launch(stream)?;
    KernelLaunch::new(gpu, values_kernel)
        .grid([4, args.total_tokens, 1])
        .block([64, 1, 1])
        .shared_mem(VALUE_SHARED_BYTES)
        .arg_ptr(args.weights)
        .arg_ptr(args.selected_indices)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.output)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_geometry_matches_tp2_checkpoint() {
        assert_eq!(GLM53_DSA_PREFILL_TC_WIDTH, 2064);
        assert!(SCORE_SHARED_BYTES < 96 * 1024);
        assert!(VALUE_SHARED_BYTES < 48 * 1024);
    }
}
