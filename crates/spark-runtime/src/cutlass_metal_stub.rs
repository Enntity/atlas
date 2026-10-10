// SPDX-License-Identifier: AGPL-3.0-only

//! Metal-build stub of the cuda-only `cutlass` module.
//!
//! The real [`crate::cutlass`] module (CUTLASS NVFP4/BF16 GEMMs + weight
//! packing) is gated behind `feature = "cuda"` because it `#include`s CUTLASS
//! C++ headers that do not exist on macOS. spark-model names these entry
//! points unconditionally, so the metal build (`cargo check --features metal`,
//! cuda off) needs the symbols to resolve even though FP4/FP8 inference never
//! runs there. Unsupported entry points fail explicitly; legacy stubs panic.

use anyhow::Result;

/// Never produced on metal; present so callers that classify CUTLASS errors
/// compile unchanged.
#[derive(Debug)]
pub struct RejectedBeforeLaunch;

impl std::fmt::Display for RejectedBeforeLaunch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CUTLASS rejected the operands before launching")
    }
}

impl std::error::Error for RejectedBeforeLaunch {}

pub fn rejected_before_launch(error: &anyhow::Error) -> bool {
    error.downcast_ref::<RejectedBeforeLaunch>().is_some()
}

pub fn launch_failed(_error: &anyhow::Error) -> bool {
    false
}

/// Preserve the CUDA wrapper ABI while reporting the unavailable backend.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_tuned(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _lda: u32,
    _ldc: u32,
    _config: u32,
    _stream: u64,
) -> Result<()> {
    anyhow::bail!("cutlass::bf16_gemm_tuned requires CUDA")
}

/// Preserve the head-batched GEMM contract on builds without CUDA.
#[allow(clippy::too_many_arguments)]
pub fn bf16_grouped_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _g: u32,
    _n: u32,
    _k: u32,
    _a_stride: u32,
    _c_stride: u32,
    _stream: u64,
) -> Result<()> {
    anyhow::bail!("cutlass::bf16_grouped_gemm_act_weight_t requires CUDA")
}

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn nvfp4_gemm_bf16_act_weight_t(
    _act: u64,
    _weight_packed_t: u64,
    _weight_scale_t: u64,
    _weight_scale_2: f32,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_gemm_bf16_act_weight_t is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_fused(
    _a: u64,
    _sorted_token_ids: u64,
    _gate_packed_ptrs: &[u64],
    _gate_sfb_ptrs: &[u64],
    _gate_scale2_vals: &[f32],
    _up_packed_ptrs: &[u64],
    _up_sfb_ptrs: &[u64],
    _up_scale2_vals: &[f32],
    _c_gate: u64,
    _c_up: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_gate_up_fused is cuda-only (not built for metal)")
}

pub fn nvfp4_grouped_down(
    _a: u64,
    _packed_ptrs: &[u64],
    _sfb_ptrs: &[u64],
    _scale2_vals: &[f32],
    _c: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_down is cuda-only (not built for metal)")
}

pub fn pack_bf16_weight_to_nvfp4_t(
    _weight_bf16: u64,
    _packed_t: u64,
    _scale_t: u64,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_bf16_weight_to_nvfp4_t is cuda-only (not built for metal)")
}

pub fn pack_weight_sfb(
    _scale_in: u64,
    _scale_out: u64,
    _n: u32,
    _k: u32,
    _src_n_major: bool,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_weight_sfb is cuda-only (not built for metal)")
}

pub fn transpose_nvfp4_packed_kton(
    _src_packed_t: u64,
    _dst_packed: u64,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::transpose_nvfp4_packed_kton is cuda-only (not built for metal)")
}

#[cfg(test)]
mod tests {
    #[test]
    fn missing_cuda_gemms_return_backend_errors_without_touching_pointers() {
        let tuned = super::bf16_gemm_tuned(1, 2, 3, 4, 5, 6, 6, 5, 0, 0).unwrap_err();
        assert_eq!(tuned.to_string(), "cutlass::bf16_gemm_tuned requires CUDA");
        let grouped =
            super::bf16_grouped_gemm_act_weight_t(1, 2, 3, 4, 5, 6, 7, 35, 30, 0).unwrap_err();
        assert_eq!(
            grouped.to_string(),
            "cutlass::bf16_grouped_gemm_act_weight_t requires CUDA"
        );
    }
}
