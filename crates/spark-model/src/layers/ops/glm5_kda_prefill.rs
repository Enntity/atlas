// SPDX-License-Identifier: AGPL-3.0-only

//! Native glue around GLM's chunked FlashKDA prefill recurrence.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_KDA_PREFILL_MODULE: &str = "glm53_kda_prefill";
pub const GLM53_KDA_CONV_ENTRY: &str = "glm53_kda_conv_silu_chunk";
pub const GLM53_KDA_BETA_ENTRY: &str = "glm53_kda_beta_transpose";
pub const GLM53_KDA_NORM_ENTRY: &str = "glm53_kda_gated_norm_chunk";
pub const GLM53_KDA_PREFILL_DIM: u32 = 128;
pub const GLM53_KDA_TRANSPOSE_THREADS: u32 = 256;

fn ensure_pointers(pointers: &[(&str, DevicePtr)]) -> Result<()> {
    for (name, pointer) in pointers {
        ensure!(!pointer.is_null(), "GLM KDA prefill {name} pointer is null");
    }
    Ok(())
}

/// In-place depthwise conv4 + SiLU over packed Q/K/V chunks.
pub struct Glm53KdaConvChunkArgs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub query_weight: DevicePtr,
    pub key_weight: DevicePtr,
    pub value_weight: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub num_sequences: u32,
    pub num_heads: u32,
}

impl Glm53KdaConvChunkArgs {
    fn validate(&self) -> Result<u32> {
        ensure!(self.num_sequences > 0, "GLM KDA conv requires sequences");
        ensure!(self.num_heads > 0, "GLM KDA conv requires heads");
        ensure_pointers(&[
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("query_weight", self.query_weight),
            ("key_weight", self.key_weight),
            ("value_weight", self.value_weight),
            ("cu_seqlens", self.cu_seqlens),
            ("sequence_state_ptrs", self.sequence_state_ptrs),
        ])?;
        self.num_sequences
            .checked_mul(self.num_heads)
            .ok_or_else(|| anyhow::anyhow!("GLM KDA conv grid overflows u32"))
    }
}

pub fn glm53_kda_conv_silu_chunk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53KdaConvChunkArgs,
    stream: u64,
) -> Result<()> {
    let sequence_heads = args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([sequence_heads, 1, 1])
        .block([GLM53_KDA_PREFILL_DIM, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.query_weight)
        .arg_ptr(args.key_weight)
        .arg_ptr(args.value_weight)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_u32(args.num_sequences)
        .arg_u32(args.num_heads)
        .launch(stream)
}

pub struct Glm53KdaBetaTransposeArgs {
    pub input: DevicePtr,
    pub output: DevicePtr,
    pub total_tokens: u32,
    pub num_heads: u32,
}

impl Glm53KdaBetaTransposeArgs {
    fn validate(&self) -> Result<u32> {
        ensure!(
            self.total_tokens > 0,
            "GLM KDA beta transpose requires tokens"
        );
        ensure!(self.num_heads > 0, "GLM KDA beta transpose requires heads");
        ensure_pointers(&[("beta input", self.input), ("beta output", self.output)])?;
        self.total_tokens
            .checked_mul(self.num_heads)
            .ok_or_else(|| anyhow::anyhow!("GLM KDA beta element count overflows u32"))
    }
}

pub fn glm53_kda_beta_transpose(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53KdaBetaTransposeArgs,
    stream: u64,
) -> Result<()> {
    let elements = args.validate()?;
    let blocks = elements.div_ceil(GLM53_KDA_TRANSPOSE_THREADS);
    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([GLM53_KDA_TRANSPOSE_THREADS, 1, 1])
        .arg_ptr(args.input)
        .arg_ptr(args.output)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_heads)
        .launch(stream)
}

pub struct Glm53KdaGatedNormArgs {
    pub recurrent_output: DevicePtr,
    pub output_gate: DevicePtr,
    pub norm_weight: DevicePtr,
    pub output: DevicePtr,
    pub token_heads: u32,
    pub norm_epsilon: f32,
}

impl Glm53KdaGatedNormArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.token_heads > 0, "GLM KDA gated norm requires rows");
        ensure!(
            self.norm_epsilon.is_finite() && self.norm_epsilon > 0.0,
            "GLM KDA gated norm epsilon must be finite and positive"
        );
        ensure_pointers(&[
            ("recurrent output", self.recurrent_output),
            ("output gate", self.output_gate),
            ("norm weight", self.norm_weight),
            ("normalized output", self.output),
        ])
    }
}

pub fn glm53_kda_gated_norm_chunk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53KdaGatedNormArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.token_heads, 1, 1])
        .block([GLM53_KDA_PREFILL_DIM, 1, 1])
        .arg_ptr(args.recurrent_output)
        .arg_ptr(args.output_gate)
        .arg_ptr(args.norm_weight)
        .arg_ptr(args.output)
        .arg_u32(args.token_heads)
        .arg_f32(args.norm_epsilon)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_kda_prefill.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    #[test]
    fn production_geometry_and_entries_are_registered() {
        assert_eq!(GLM53_KDA_PREFILL_DIM, 128);
        for entry in [
            GLM53_KDA_CONV_ENTRY,
            GLM53_KDA_BETA_ENTRY,
            GLM53_KDA_NORM_ENTRY,
        ] {
            assert!(SOURCE.contains(&format!("void {entry}(")));
        }
        assert!(REGISTRY.contains("glm53_kda_prefill = \"glm53_kda_prefill\""));
    }

    #[test]
    fn chunk_contracts_reject_empty_or_invalid_inputs() {
        let mut conv = Glm53KdaConvChunkArgs {
            query: DevicePtr(1),
            key: DevicePtr(2),
            value: DevicePtr(3),
            query_weight: DevicePtr(4),
            key_weight: DevicePtr(5),
            value_weight: DevicePtr(6),
            cu_seqlens: DevicePtr(7),
            sequence_state_ptrs: DevicePtr(8),
            num_sequences: 2,
            num_heads: 64,
        };
        assert_eq!(conv.validate().unwrap(), 128);
        conv.query = DevicePtr::NULL;
        assert!(conv.validate().unwrap_err().to_string().contains("query"));

        let mut beta = Glm53KdaBetaTransposeArgs {
            input: DevicePtr(1),
            output: DevicePtr(2),
            total_tokens: 50,
            num_heads: 4,
        };
        assert_eq!(beta.validate().unwrap(), 200);
        beta.total_tokens = 0;
        assert!(beta.validate().unwrap_err().to_string().contains("tokens"));

        let mut norm = Glm53KdaGatedNormArgs {
            recurrent_output: DevicePtr(1),
            output_gate: DevicePtr(2),
            norm_weight: DevicePtr(3),
            output: DevicePtr(4),
            token_heads: 200,
            norm_epsilon: 1.0e-6,
        };
        norm.validate().unwrap();
        norm.norm_epsilon = f32::NAN;
        assert!(norm.validate().unwrap_err().to_string().contains("epsilon"));
    }
}
