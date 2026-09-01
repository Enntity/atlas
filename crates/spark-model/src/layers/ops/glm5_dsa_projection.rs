// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 sparse-MLA projection and cache-append launch contracts.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_DSA_PROJECTION_MODULE: &str = "glm53_dsa_projection";
pub const GLM53_DSA_LATENT_APPEND_ENTRY: &str = "glm53_dsa_latent_append";
pub const GLM53_DSA_INDEX_NORM_ENTRY: &str = "glm53_dsa_index_layernorm";
pub const GLM53_DSA_ABSORB_QUERY_ENTRY: &str = "glm53_dsa_absorb_query_fp8";
pub const GLM53_DSA_EXPAND_VALUE_ENTRY: &str = "glm53_dsa_expand_value_fp8";
pub const GLM53_DSA_ABSORB_QUERY_BF16_ENTRY: &str = "glm53_dsa_absorb_query_bf16";
pub const GLM53_DSA_EXPAND_VALUE_BF16_ENTRY: &str = "glm53_dsa_expand_value_bf16";
pub const GLM53_DSA_MLA_HEADS: u32 = 64;
pub const GLM53_DSA_QK_DIM: u32 = 256;
pub const GLM53_DSA_V_DIM: u32 = 256;
pub const GLM53_DSA_LATENT_DIM: u32 = 512;
pub const GLM53_DSA_INDEX_NORM_DIM: u32 = 128;

fn pointers(values: &[(&str, DevicePtr)]) -> Result<()> {
    for (name, pointer) in values {
        ensure!(!pointer.is_null(), "GLM DSA {name} pointer is null");
    }
    Ok(())
}

pub struct Glm53DsaLatentAppendArgs {
    pub latent: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub positions: DevicePtr,
    pub valid: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub total_tokens: u32,
    pub num_sequences: u32,
    pub latent_capacity: u32,
}

impl Glm53DsaLatentAppendArgs {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.total_tokens > 0,
            "GLM DSA latent append requires tokens"
        );
        ensure!(
            self.num_sequences > 0,
            "GLM DSA latent append requires sequences"
        );
        ensure!(
            self.latent_capacity > 0,
            "GLM DSA latent append requires capacity"
        );
        pointers(&[
            ("latent", self.latent),
            ("cu_seqlens", self.cu_seqlens),
            ("positions", self.positions),
            ("valid", self.valid),
            ("state table", self.sequence_state_ptrs),
        ])
    }
}

pub fn glm53_dsa_latent_append(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaLatentAppendArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.total_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(args.latent)
        .arg_ptr(args.cu_seqlens)
        .arg_ptr(args.positions)
        .arg_ptr(args.valid)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_u32(args.total_tokens)
        .arg_u32(args.num_sequences)
        .arg_u32(args.latent_capacity)
        .launch(stream)
}

pub struct Glm53DsaIndexNormArgs {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub bias: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub epsilon: f32,
}

impl Glm53DsaIndexNormArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.rows > 0, "GLM DSA index LayerNorm requires rows");
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "GLM DSA index LayerNorm epsilon must be positive"
        );
        pointers(&[
            ("index input", self.input),
            ("index weight", self.weight),
            ("index bias", self.bias),
            ("index output", self.output),
        ])
    }
}

pub fn glm53_dsa_index_layernorm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaIndexNormArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.rows, 1, 1])
        .block([GLM53_DSA_INDEX_NORM_DIM, 1, 1])
        .arg_ptr(args.input)
        .arg_ptr(args.weight)
        .arg_ptr(args.bias)
        .arg_ptr(args.output)
        .arg_u32(args.rows)
        .arg_f32(args.epsilon)
        .launch(stream)
}

pub struct Glm53DsaFp8ProjectionArgs {
    pub input: DevicePtr,
    pub kv_b_weight: DevicePtr,
    pub kv_b_scale: DevicePtr,
    pub output: DevicePtr,
    pub num_tokens: u32,
    pub num_heads: u32,
}

impl Glm53DsaFp8ProjectionArgs {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_tokens > 0,
            "GLM DSA FP8 projection requires tokens"
        );
        ensure!(self.num_heads > 0, "GLM DSA FP8 projection requires heads");
        pointers(&[
            ("projection input", self.input),
            ("kv_b weight", self.kv_b_weight),
            ("kv_b scale", self.kv_b_scale),
            ("projection output", self.output),
        ])
    }
}

fn fp8_projection(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaFp8ProjectionArgs,
    threads: u32,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, args.num_tokens, 1])
        .block([threads, 1, 1])
        .arg_ptr(args.input)
        .arg_ptr(args.kv_b_weight)
        .arg_ptr(args.kv_b_scale)
        .arg_ptr(args.output)
        .arg_u32(args.num_tokens)
        .arg_u32(args.num_heads)
        .launch(stream)
}

pub fn glm53_dsa_absorb_query_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaFp8ProjectionArgs,
    stream: u64,
) -> Result<()> {
    fp8_projection(gpu, kernel, args, 256, stream)
}

pub fn glm53_dsa_expand_value_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaFp8ProjectionArgs,
    stream: u64,
) -> Result<()> {
    fp8_projection(gpu, kernel, args, GLM53_DSA_V_DIM, stream)
}

pub struct Glm53DsaBf16ProjectionArgs {
    pub input: DevicePtr,
    pub kv_b_weight: DevicePtr,
    pub output: DevicePtr,
    pub num_tokens: u32,
    pub num_heads: u32,
}

impl Glm53DsaBf16ProjectionArgs {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_tokens > 0,
            "GLM DSA BF16 projection requires tokens"
        );
        ensure!(self.num_heads > 0, "GLM DSA BF16 projection requires heads");
        pointers(&[
            ("projection input", self.input),
            ("kv_b weight", self.kv_b_weight),
            ("projection output", self.output),
        ])
    }
}

fn bf16_projection(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaBf16ProjectionArgs,
    threads: u32,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.num_heads, args.num_tokens, 1])
        .block([threads, 1, 1])
        .arg_ptr(args.input)
        .arg_ptr(args.kv_b_weight)
        .arg_ptr(args.output)
        .arg_u32(args.num_tokens)
        .arg_u32(args.num_heads)
        .launch(stream)
}

pub fn glm53_dsa_absorb_query_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaBf16ProjectionArgs,
    stream: u64,
) -> Result<()> {
    bf16_projection(gpu, kernel, args, 256, stream)
}

pub fn glm53_dsa_expand_value_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53DsaBf16ProjectionArgs,
    stream: u64,
) -> Result<()> {
    bf16_projection(gpu, kernel, args, GLM53_DSA_V_DIM, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_dsa_projection.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    #[test]
    fn official_absorbed_mla_geometry_and_symbols_are_pinned() {
        assert_eq!(GLM53_DSA_MLA_HEADS, 64);
        assert_eq!(GLM53_DSA_QK_DIM, 256);
        assert_eq!(GLM53_DSA_V_DIM, 256);
        assert_eq!(GLM53_DSA_LATENT_DIM, 512);
        for entry in [
            GLM53_DSA_LATENT_APPEND_ENTRY,
            GLM53_DSA_INDEX_NORM_ENTRY,
            GLM53_DSA_ABSORB_QUERY_ENTRY,
            GLM53_DSA_EXPAND_VALUE_ENTRY,
            GLM53_DSA_ABSORB_QUERY_BF16_ENTRY,
            GLM53_DSA_EXPAND_VALUE_BF16_ENTRY,
        ] {
            assert!(SOURCE.contains(&format!("void {entry}(")));
        }
        assert!(REGISTRY.contains("glm53_dsa_projection = \"glm53_dsa_projection\""));
    }

    #[test]
    fn invalid_projection_and_append_contracts_fail_closed() {
        let mut projection = Glm53DsaFp8ProjectionArgs {
            input: DevicePtr(1),
            kv_b_weight: DevicePtr(2),
            kv_b_scale: DevicePtr(3),
            output: DevicePtr(4),
            num_tokens: 2,
            num_heads: 32,
        };
        projection.validate().unwrap();
        projection.num_tokens = 0;
        assert!(
            projection
                .validate()
                .unwrap_err()
                .to_string()
                .contains("tokens")
        );

        let mut append = Glm53DsaLatentAppendArgs {
            latent: DevicePtr(1),
            cu_seqlens: DevicePtr(2),
            positions: DevicePtr(3),
            valid: DevicePtr(4),
            sequence_state_ptrs: DevicePtr(5),
            total_tokens: 2,
            num_sequences: 1,
            latent_capacity: 32,
        };
        append.validate().unwrap();
        append.sequence_state_ptrs = DevicePtr::NULL;
        assert!(
            append
                .validate()
                .unwrap_err()
                .to_string()
                .contains("state table")
        );
    }
}
