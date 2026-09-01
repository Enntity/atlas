// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-specific residual-stream boundaries.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

pub const GLM53_RESIDUAL_MODULE: &str = "glm53_residual";
pub const GLM53_HC_MEAN_ENTRY: &str = "glm53_hc_mean";

pub fn glm53_hc_mean(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    ensure!(num_tokens > 0, "GLM mHC mean requires tokens");
    ensure!(hidden_size > 0, "GLM mHC mean requires hidden channels");
    ensure!(hc_mult > 0, "GLM mHC mean requires residual streams");
    ensure!(!streams.is_null(), "GLM mHC stream pointer is null");
    ensure!(!output.is_null(), "GLM mHC output pointer is null");
    let elements = num_tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| anyhow::anyhow!("GLM mHC mean element count overflows u32"))?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(output)
        .arg_u32(num_tokens)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_residual.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    #[test]
    fn final_glm_hyper_head_is_registered_as_unweighted_mean() {
        assert!(SOURCE.contains(&format!("void {GLM53_HC_MEAN_ENTRY}(")));
        assert!(REGISTRY.contains("glm53_residual = \"glm53_residual\""));
    }
}
