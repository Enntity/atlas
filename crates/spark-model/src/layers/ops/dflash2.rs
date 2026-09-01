// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM-5.3 DFlash2 dynamic-convolution and selector launches.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

#[allow(clippy::too_many_arguments)]
pub fn dflash2_dynamic_conv2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    output: DevicePtr,
    rows: u32,
    sequence_rows: u32,
    hidden: u32,
    groups: u32,
    stage: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > 0 && sequence_rows > 0 && hidden > 0 && groups > 0,
        "DFlash2 convolution has empty geometry"
    );
    ensure!(
        hidden.is_multiple_of(groups),
        "DFlash2 hidden size is not group-aligned"
    );
    ensure!(stage < 2, "DFlash2 convolution stage must be 0 or 1");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(rows * hidden, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(sequence_rows)
        .arg_u32(hidden)
        .arg_u32(groups)
        .arg_u32(stage)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn dflash2_select_path16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    selector_hidden: DevicePtr,
    predecessor_codebook: DevicePtr,
    successor_codebook: DevicePtr,
    output: DevicePtr,
    candidate_ids: DevicePtr,
    edge_scores: DevicePtr,
    rows: u32,
    vocab: u32,
    rank: u32,
    anchor_token: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > 0 && vocab >= 16 && rank > 0,
        "DFlash2 selector has invalid geometry"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(selector_hidden)
        .arg_ptr(predecessor_codebook)
        .arg_ptr(successor_codebook)
        .arg_ptr(output)
        .arg_ptr(candidate_ids)
        .arg_ptr(edge_scores)
        .arg_u32(rows)
        .arg_u32(vocab)
        .arg_u32(rank)
        .arg_ptr(anchor_token)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/dflash2_native.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");
    const DFLASH_ATTN: &str = include_str!(
        "../../../../../kernels/gb10/glm-5.3-flash/exl3/inferspark_prefill_paged_indirect.cu"
    );

    #[test]
    fn glm_dflash2_kernels_are_native_and_registered() {
        assert!(SOURCE.contains("void dflash2_dynamic_conv2_bf16("));
        assert!(SOURCE.contains("void dflash2_select_path16_bf16("));
        assert!(REGISTRY.contains("dflash2_native = \"dflash2_native\""));
    }

    #[test]
    fn glm_dflash_attention_is_compiled_for_the_drafter_head_width() {
        assert!(DFLASH_ATTN.contains("#define HDIM 128"));
        assert!(DFLASH_ATTN.contains("common/inferspark_prefill_paged_indirect.cu"));
    }
}
