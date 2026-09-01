// SPDX-License-Identifier: AGPL-3.0-only

//! Launch contract for GLM-5.3-Flash's single-token KDA recurrence.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_KDA_HEAD_DIM: u32 = 128;
pub const GLM53_KDA_THREADS: u32 = 256;
pub const GLM53_KDA_MODULE: &str = "glm53_kda";
pub const GLM53_KDA_ENTRY: &str = "glm53_kda_decode";
pub const GLM53_KDA_FUSED_ENTRY: &str = "glm53_kda_decode_fused_conv_gate_norm";
pub const GLM53_KDA_VERIFY_ENTRY: &str = "glm53_kda_verify_fused_conv_gate_norm";
pub const GLM53_KDA_VERIFY_TILED_MODULE: &str = "glm53_kda_verify_tiled";
pub const GLM53_KDA_VERIFY_PREPARE_ENTRY: &str = "glm53_kda_verify_prepare";
pub const GLM53_KDA_VERIFY_RECURRENT_ENTRY: &str = "glm53_kda_verify_recurrent_tiled";
pub const GLM53_KDA_VERIFY_NORM_ENTRY: &str = "glm53_kda_verify_norm";
pub const GLM53_KDA_VERIFY_TILES: u32 = 16;
pub const GLM53_KDA_VERIFY_THREADS: u32 = 128;

/// Device buffers for one batched decode step.
///
/// Q/K/V are BF16 `[batch_heads, 128]`, `log_decay` is FP32 with the same
/// shape, `beta` is FP32 `[batch_heads]`, state is FP32
/// `[batch_heads, 128, 128]`, and output is BF16 `[batch_heads, 128]`.
pub struct Glm53KdaDecodeArgs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub log_decay: DevicePtr,
    pub beta: DevicePtr,
    pub state: DevicePtr,
    pub output: DevicePtr,
    pub batch_heads: u32,
}

impl Glm53KdaDecodeArgs {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.batch_heads > 0,
            "GLM KDA decode requires at least one head"
        );
        for (name, pointer) in [
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("log_decay", self.log_decay),
            ("beta", self.beta),
            ("state", self.state),
            ("output", self.output),
        ] {
            ensure!(!pointer.is_null(), "GLM KDA decode {name} pointer is null");
        }
        Ok(())
    }
}

pub fn glm53_kda_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53KdaDecodeArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.batch_heads, 1, 1])
        .block([GLM53_KDA_THREADS, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.log_decay)
        .arg_ptr(args.beta)
        .arg_ptr(args.state)
        .arg_ptr(args.output)
        .arg_u32(args.batch_heads)
        .launch(stream)
}

/// Complete post-projection decode inputs for GLM KDA.
///
/// Model-owned weights use `[num_heads, 128, ...]`; activations use
/// sequence-major `[batch_heads, ...]` storage. `sequence_state_ptrs` is a
/// device `u64[batch, 4]` table in recurrent/Q-conv/K-conv/V-conv order. This
/// keeps scheduler batch order independent from fragmented state-pool slots.
pub struct Glm53KdaFusedDecodeArgs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub query_conv_weight: DevicePtr,
    pub key_conv_weight: DevicePtr,
    pub value_conv_weight: DevicePtr,
    pub sequence_state_ptrs: DevicePtr,
    pub forget_projection: DevicePtr,
    pub dt_bias: DevicePtr,
    pub a_log: DevicePtr,
    pub beta_logit: DevicePtr,
    pub output_gate: DevicePtr,
    pub norm_weight: DevicePtr,
    pub output: DevicePtr,
    pub batch_heads: u32,
    pub num_heads: u32,
    pub norm_epsilon: f32,
}

impl Glm53KdaFusedDecodeArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_heads > 0, "GLM fused KDA requires model heads");
        ensure!(
            self.batch_heads > 0 && self.batch_heads.is_multiple_of(self.num_heads),
            "GLM fused KDA batch_heads must be a positive multiple of num_heads"
        );
        ensure!(
            self.norm_epsilon.is_finite() && self.norm_epsilon > 0.0,
            "GLM fused KDA norm epsilon must be finite and positive"
        );
        for (name, pointer) in [
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("query_conv_weight", self.query_conv_weight),
            ("key_conv_weight", self.key_conv_weight),
            ("value_conv_weight", self.value_conv_weight),
            ("sequence_state_ptrs", self.sequence_state_ptrs),
            ("forget_projection", self.forget_projection),
            ("dt_bias", self.dt_bias),
            ("a_log", self.a_log),
            ("beta_logit", self.beta_logit),
            ("output_gate", self.output_gate),
            ("norm_weight", self.norm_weight),
            ("output", self.output),
        ] {
            ensure!(!pointer.is_null(), "GLM fused KDA {name} pointer is null");
        }
        Ok(())
    }
}

pub fn glm53_kda_fused_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53KdaFusedDecodeArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.batch_heads, 1, 1])
        .block([GLM53_KDA_THREADS, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.query_conv_weight)
        .arg_ptr(args.key_conv_weight)
        .arg_ptr(args.value_conv_weight)
        .arg_ptr(args.sequence_state_ptrs)
        .arg_ptr(args.forget_projection)
        .arg_ptr(args.dt_bias)
        .arg_ptr(args.a_log)
        .arg_ptr(args.beta_logit)
        .arg_ptr(args.output_gate)
        .arg_ptr(args.norm_weight)
        .arg_ptr(args.output)
        .arg_u32(args.batch_heads)
        .arg_u32(args.num_heads)
        .arg_f32(args.norm_epsilon)
        .launch(stream)
}

/// Batched projections for consecutive target-verification rows of one
/// sequence. `state_images` is `u64[1 + snapshot_count][4]`: the mutable
/// current image followed by one exact rollback image per reachable prefix.
pub struct Glm53KdaVerifyArgs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub query_conv_weight: DevicePtr,
    pub key_conv_weight: DevicePtr,
    pub value_conv_weight: DevicePtr,
    pub state_images: DevicePtr,
    pub forget_projection: DevicePtr,
    pub dt_bias: DevicePtr,
    pub a_log: DevicePtr,
    pub beta_logit: DevicePtr,
    pub output_gate: DevicePtr,
    pub norm_weight: DevicePtr,
    pub output: DevicePtr,
    pub num_tokens: u32,
    pub num_sequences: u32,
    pub snapshot_count: u32,
    pub num_heads: u32,
    pub norm_epsilon: f32,
}

impl Glm53KdaVerifyArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.num_tokens > 0, "GLM KDA verify requires tokens");
        ensure!(self.num_sequences > 0, "GLM KDA verify requires sequences");
        ensure!(self.num_heads > 0, "GLM KDA verify requires model heads");
        ensure!(
            self.snapshot_count < self.num_tokens,
            "GLM KDA verify snapshots must exclude the all-accepted row"
        );
        ensure!(
            self.norm_epsilon.is_finite() && self.norm_epsilon > 0.0,
            "GLM KDA verify norm epsilon must be finite and positive"
        );
        for (name, pointer) in [
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("query_conv_weight", self.query_conv_weight),
            ("key_conv_weight", self.key_conv_weight),
            ("value_conv_weight", self.value_conv_weight),
            ("state_images", self.state_images),
            ("forget_projection", self.forget_projection),
            ("dt_bias", self.dt_bias),
            ("a_log", self.a_log),
            ("beta_logit", self.beta_logit),
            ("output_gate", self.output_gate),
            ("norm_weight", self.norm_weight),
            ("output", self.output),
        ] {
            ensure!(!pointer.is_null(), "GLM KDA verify {name} pointer is null");
        }
        Ok(())
    }
}

pub fn glm53_kda_verify(
    gpu: &dyn GpuBackend,
    prepare_kernel: KernelHandle,
    recurrent_kernel: KernelHandle,
    norm_kernel: KernelHandle,
    args: &Glm53KdaVerifyArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    let sequence_heads = args
        .num_heads
        .checked_mul(args.num_sequences)
        .ok_or_else(|| anyhow::anyhow!("GLM KDA verify sequence-head count overflow"))?;
    let token_heads = sequence_heads
        .checked_mul(args.num_tokens)
        .ok_or_else(|| anyhow::anyhow!("GLM KDA verify token-head count overflow"))?;
    let recurrent_tiles = sequence_heads
        .checked_mul(GLM53_KDA_VERIFY_TILES)
        .ok_or_else(|| anyhow::anyhow!("GLM KDA verify tile count overflow"))?;

    KernelLaunch::new(gpu, prepare_kernel)
        .grid([sequence_heads, 1, 1])
        .block([GLM53_KDA_VERIFY_THREADS, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.query_conv_weight)
        .arg_ptr(args.key_conv_weight)
        .arg_ptr(args.value_conv_weight)
        .arg_ptr(args.state_images)
        .arg_u32(args.num_tokens)
        .arg_u32(args.num_sequences)
        .arg_u32(args.snapshot_count)
        .arg_u32(args.num_heads)
        .launch(stream)?;
    KernelLaunch::new(gpu, recurrent_kernel)
        .grid([recurrent_tiles, 1, 1])
        .block([GLM53_KDA_VERIFY_THREADS, 1, 1])
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.state_images)
        .arg_ptr(args.forget_projection)
        .arg_ptr(args.dt_bias)
        .arg_ptr(args.a_log)
        .arg_ptr(args.beta_logit)
        .arg_ptr(args.output)
        .arg_u32(args.num_tokens)
        .arg_u32(args.num_sequences)
        .arg_u32(args.snapshot_count)
        .arg_u32(args.num_heads)
        .launch(stream)?;
    KernelLaunch::new(gpu, norm_kernel)
        .grid([token_heads, 1, 1])
        .block([GLM53_KDA_VERIFY_THREADS, 1, 1])
        .arg_ptr(args.output)
        .arg_ptr(args.output_gate)
        .arg_ptr(args.norm_weight)
        .arg_u32(token_heads)
        .arg_f32(args.norm_epsilon)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUDA_SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_kda_decode.cu");
    const TILED_CUDA_SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_kda_verify_tiled.cu");
    const KERNEL_TOML: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    fn valid_args() -> Glm53KdaDecodeArgs {
        Glm53KdaDecodeArgs {
            query: DevicePtr(1),
            key: DevicePtr(2),
            value: DevicePtr(3),
            log_decay: DevicePtr(4),
            beta: DevicePtr(5),
            state: DevicePtr(6),
            output: DevicePtr(7),
            batch_heads: 64,
        }
    }

    fn valid_fused_args() -> Glm53KdaFusedDecodeArgs {
        Glm53KdaFusedDecodeArgs {
            query: DevicePtr(1),
            key: DevicePtr(2),
            value: DevicePtr(3),
            query_conv_weight: DevicePtr(4),
            key_conv_weight: DevicePtr(5),
            value_conv_weight: DevicePtr(6),
            sequence_state_ptrs: DevicePtr(7),
            forget_projection: DevicePtr(8),
            dt_bias: DevicePtr(9),
            a_log: DevicePtr(10),
            beta_logit: DevicePtr(11),
            output_gate: DevicePtr(12),
            norm_weight: DevicePtr(13),
            output: DevicePtr(14),
            batch_heads: 128,
            num_heads: 64,
            norm_epsilon: 1.0e-6,
        }
    }

    fn valid_verify_args() -> Glm53KdaVerifyArgs {
        Glm53KdaVerifyArgs {
            query: DevicePtr(1),
            key: DevicePtr(2),
            value: DevicePtr(3),
            query_conv_weight: DevicePtr(4),
            key_conv_weight: DevicePtr(5),
            value_conv_weight: DevicePtr(6),
            state_images: DevicePtr(7),
            forget_projection: DevicePtr(8),
            dt_bias: DevicePtr(9),
            a_log: DevicePtr(10),
            beta_logit: DevicePtr(11),
            output_gate: DevicePtr(12),
            norm_weight: DevicePtr(13),
            output: DevicePtr(14),
            num_tokens: 9,
            num_sequences: 1,
            snapshot_count: 8,
            num_heads: 32,
            norm_epsilon: 1.0e-6,
        }
    }

    #[test]
    fn production_geometry_is_fixed_to_official_checkpoint() {
        assert_eq!(GLM53_KDA_HEAD_DIM, 128);
        assert_eq!(GLM53_KDA_THREADS, 256);
        valid_args().validate().unwrap();
    }

    #[test]
    fn rust_and_kernel_registry_names_match_cuda_entry() {
        assert!(CUDA_SOURCE.contains(&format!("void {GLM53_KDA_ENTRY}(")));
        assert!(CUDA_SOURCE.contains(GLM53_KDA_FUSED_ENTRY));
        assert!(CUDA_SOURCE.contains(GLM53_KDA_VERIFY_ENTRY));
        assert!(TILED_CUDA_SOURCE.contains(GLM53_KDA_VERIFY_PREPARE_ENTRY));
        assert!(TILED_CUDA_SOURCE.contains(GLM53_KDA_VERIFY_RECURRENT_ENTRY));
        assert!(TILED_CUDA_SOURCE.contains(GLM53_KDA_VERIFY_NORM_ENTRY));
        assert!(KERNEL_TOML.contains(&format!("{GLM53_KDA_ENTRY} = \"{GLM53_KDA_MODULE}\"")));
        for entry in [
            GLM53_KDA_VERIFY_PREPARE_ENTRY,
            GLM53_KDA_VERIFY_RECURRENT_ENTRY,
            GLM53_KDA_VERIFY_NORM_ENTRY,
        ] {
            assert!(
                KERNEL_TOML.contains(&format!("{entry} = \"{GLM53_KDA_VERIFY_TILED_MODULE}\""))
            );
        }
        assert!(CUDA_SOURCE.contains("beta_logit[(static_cast<unsigned long long>(sequence) *"));
        assert_eq!(GLM53_KDA_VERIFY_TILES * 8, GLM53_KDA_HEAD_DIM);
        valid_verify_args().validate().unwrap();
    }

    #[test]
    fn empty_batch_or_null_buffer_fails_closed() {
        let mut args = valid_args();
        args.batch_heads = 0;
        assert!(
            args.validate()
                .unwrap_err()
                .to_string()
                .contains("one head")
        );

        let mut args = valid_args();
        args.state = DevicePtr::NULL;
        assert!(
            args.validate()
                .unwrap_err()
                .to_string()
                .contains("state pointer")
        );
    }

    #[test]
    fn fused_decode_rejects_partial_sequences_and_bad_epsilon() {
        valid_fused_args().validate().unwrap();

        let mut args = valid_fused_args();
        args.batch_heads = 65;
        assert!(
            args.validate()
                .unwrap_err()
                .to_string()
                .contains("multiple")
        );

        let mut args = valid_fused_args();
        args.norm_epsilon = f32::NAN;
        assert!(args.validate().unwrap_err().to_string().contains("epsilon"));
    }
}
