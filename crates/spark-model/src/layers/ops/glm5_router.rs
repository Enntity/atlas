// SPDX-License-Identifier: AGPL-3.0-only

//! FP32 GLM router launch contract.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_ROUTER_MODULE: &str = "glm53_router";
pub const GLM53_ROUTER_ENTRY: &str = "glm53_moe_topk_sigmoid_batched_f32";

#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_topk_sigmoid_batched_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    bias: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    rows: u32,
    stream: u64,
) -> Result<()> {
    ensure!(rows > 0, "GLM router requires rows");
    ensure!(
        num_experts == 288 && top_k == 8,
        "GLM router requires 288 experts and top-k 8"
    );
    ensure!(
        !logits.is_null() && !bias.is_null() && !indices.is_null() && !weights.is_null(),
        "GLM router received a null pointer"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(bias)
        .arg_ptr(indices)
        .arg_ptr(weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_contract_and_fp32_input_are_explicit() {
        let source = include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_router.cu");
        assert!(source.contains("const float* __restrict__ gate_logits"));
        assert!(source.contains(GLM53_ROUTER_ENTRY));
    }
}
