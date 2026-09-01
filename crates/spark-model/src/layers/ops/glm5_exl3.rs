// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

pub const GLM53_EXL3_MODULE: &str = "glm53_exl3_moe";
pub const GLM53_EXL3_ROWS: usize = 7168;
pub const GLM53_EXL3_CONCURRENCY: u32 = 8;
pub const GLM53_EXL3_SHARED_BYTES: u32 = 90 * 1024;
pub const GLM53_EXL3_FAT_CAP: usize = 128;
pub const GLM53_EXL3_FAT_MODULE: &str = "glm53_exl3_fat";
pub const GLM53_EXL3_FAT_SHARED_BYTES: u32 = 13 * 1024;

/// Scratch stride for one persistent expert group.
///
/// The vendor kernel uses `max_tokens_per_expert` only as the distance between
/// the persistent groups' temporary matrices and as a route-count bound.
/// Passing the arena maximum (512) for a one-row decode or a short prefill
/// needlessly spreads those hot matrices across tens of MiB.  The largest
/// possible count for any expert is the number of routed rows, so the exact
/// row count is both safe and the tightest cache footprint.
pub fn glm53_exl3_row_capacity(rows: usize) -> u32 {
    rows.clamp(1, GLM53_EXL3_FAT_CAP) as u32
}

/// Persistent expert groups for the fixed two-GB10 TP2 topology. Six groups
/// place exactly 48 eight-block expert teams on GB10's 48 SMs. A no-weight
/// probe with the checkpoint's exact TP2 tensor shapes and 64 distinct routes
/// measured 1.91 ms at six groups versus 2.28 ms at four; small verifier
/// batches still expose enough independent routed experts to occupy the GPU.
pub fn glm53_exl3_concurrency(_rows: usize) -> u32 {
    6
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3PointerTables {
    pub gate_trellis: DevicePtr,
    pub gate_suh: DevicePtr,
    pub gate_svh: DevicePtr,
    pub up_trellis: DevicePtr,
    pub up_suh: DevicePtr,
    pub up_svh: DevicePtr,
    pub down_trellis: DevicePtr,
    pub down_suh: DevicePtr,
    pub down_svh: DevicePtr,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3Workspace {
    pub hidden_fp16: DevicePtr,
    pub output_fp32: DevicePtr,
    pub temp_state_g: DevicePtr,
    pub temp_state_u: DevicePtr,
    pub temp_intermediate_g: DevicePtr,
    pub temp_intermediate_u: DevicePtr,
    pub expert_count: DevicePtr,
    pub fat_descriptors: DevicePtr,
    pub token_sorted: DevicePtr,
    pub weight_sorted: DevicePtr,
    pub locks: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
pub fn glm53_exl3_prepare_routes(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    topk_ids: DevicePtr,
    topk_weights: DevicePtr,
    workspace: &Glm53Exl3Workspace,
    num_tokens: u32,
    topk: u32,
    num_experts: u32,
    local_start: u32,
    local_end: u32,
    fat_cap: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        num_tokens as usize <= GLM53_EXL3_ROWS,
        "EXL3 route window exceeds {} rows",
        GLM53_EXL3_ROWS
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(topk_weights)
        .arg_ptr(workspace.expert_count)
        .arg_ptr(workspace.fat_descriptors)
        .arg_ptr(workspace.token_sorted)
        .arg_ptr(workspace.weight_sorted)
        .arg_u32(num_tokens)
        .arg_u32(topk)
        .arg_u32(num_experts)
        .arg_u32(local_start)
        .arg_u32(local_end)
        .arg_u32(fat_cap)
        .launch(stream)
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Exl3FatKernels {
    pub gather: KernelHandle,
    pub gate_up: KernelHandle,
    pub activate: KernelHandle,
    pub down: KernelHandle,
}

pub fn glm53_exl3_fat(
    gpu: &dyn GpuBackend,
    kernels: Glm53Exl3FatKernels,
    pointers: &Glm53Exl3PointerTables,
    workspace: &Glm53Exl3Workspace,
    rows: u32,
    hidden: u32,
    intermediate: u32,
    num_experts: u32,
    activation_limit: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > GLM53_EXL3_FAT_CAP as u32,
        "fat EXL3 launch requires oversized routes"
    );
    let gate_up_stride = intermediate * 2;
    let gather = |suh: DevicePtr| {
        KernelLaunch::new(gpu, kernels.gather)
            .grid([48, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(workspace.hidden_fp16)
            .arg_ptr(workspace.temp_state_g)
            .arg_ptr(suh)
            .arg_ptr(workspace.expert_count)
            .arg_ptr(workspace.fat_descriptors)
            .arg_ptr(workspace.token_sorted)
            .arg_u32(hidden)
            .arg_u32(rows)
            .arg_u32(num_experts)
            .launch(stream)
    };
    let gate_up = |trellis: DevicePtr, svh: DevicePtr, column: u32| {
        KernelLaunch::new(gpu, kernels.gate_up)
            .grid([48, 1, 1])
            .block([256, 1, 1])
            .shared_mem(GLM53_EXL3_FAT_SHARED_BYTES)
            .arg_ptr(workspace.temp_state_g)
            .arg_ptr(trellis)
            .arg_ptr(workspace.temp_state_u)
            .arg_ptr(svh)
            .arg_ptr(workspace.expert_count)
            .arg_ptr(workspace.fat_descriptors)
            .arg_u32(hidden)
            .arg_u32(intermediate)
            .arg_u32(gate_up_stride)
            .arg_u32(column)
            .arg_u32(rows)
            .arg_u32(num_experts)
            .launch(stream)
    };
    gather(pointers.gate_suh)?;
    gate_up(pointers.gate_trellis, pointers.gate_svh, 0)?;
    gather(pointers.up_suh)?;
    gate_up(pointers.up_trellis, pointers.up_svh, intermediate)?;
    KernelLaunch::new(gpu, kernels.activate)
        .grid([48, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(workspace.temp_state_u)
        .arg_ptr(workspace.temp_state_g)
        .arg_ptr(pointers.down_suh)
        .arg_ptr(workspace.expert_count)
        .arg_ptr(workspace.fat_descriptors)
        .arg_u32(intermediate)
        .arg_u32(gate_up_stride)
        .arg_f32(activation_limit)
        .arg_u32(rows)
        .arg_u32(num_experts)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.down)
        .grid([48, 1, 1])
        .block([256, 1, 1])
        .shared_mem(GLM53_EXL3_FAT_SHARED_BYTES)
        .arg_ptr(workspace.temp_state_g)
        .arg_ptr(pointers.down_trellis)
        .arg_ptr(workspace.output_fp32)
        .arg_ptr(pointers.down_svh)
        .arg_ptr(workspace.expert_count)
        .arg_ptr(workspace.fat_descriptors)
        .arg_ptr(workspace.token_sorted)
        .arg_ptr(workspace.weight_sorted)
        .arg_u32(intermediate)
        .arg_u32(hidden)
        .arg_u32(rows)
        .arg_u32(num_experts)
        .launch(stream)
}

pub fn glm53_exl3_bf16_to_fp16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    output: DevicePtr,
    elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(elements)
        .launch(stream)
}

pub fn glm53_exl3_fp32_to_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    output: DevicePtr,
    elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(elements)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm53_exl3_moe(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    pointers: &Glm53Exl3PointerTables,
    workspace: &Glm53Exl3Workspace,
    hidden: u32,
    intermediate: u32,
    num_experts: u32,
    topk: u32,
    max_tokens_per_expert: u32,
    concurrency: u32,
    activation_limit: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=GLM53_EXL3_CONCURRENCY).contains(&concurrency),
        "EXL3 concurrency {concurrency} exceeds workspace maximum {GLM53_EXL3_CONCURRENCY}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([8, 1, concurrency])
        .block([512, 1, 1])
        .shared_mem(GLM53_EXL3_SHARED_BYTES)
        .arg_ptr(workspace.hidden_fp16)
        .arg_ptr(workspace.temp_state_g)
        .arg_ptr(workspace.temp_state_u)
        .arg_ptr(workspace.temp_intermediate_g)
        .arg_ptr(workspace.temp_intermediate_u)
        .arg_ptr(workspace.output_fp32)
        .arg_ptr(pointers.gate_trellis)
        .arg_ptr(pointers.gate_suh)
        .arg_ptr(pointers.gate_svh)
        .arg_ptr(pointers.up_trellis)
        .arg_ptr(pointers.up_suh)
        .arg_ptr(pointers.up_svh)
        .arg_ptr(pointers.down_trellis)
        .arg_ptr(pointers.down_suh)
        .arg_ptr(pointers.down_svh)
        .arg_ptr(workspace.expert_count)
        .arg_ptr(workspace.token_sorted)
        .arg_ptr(workspace.weight_sorted)
        .arg_u32(hidden)
        .arg_u32(intermediate)
        .arg_u32(num_experts)
        .arg_u32(topk)
        .arg_u32(max_tokens_per_expert)
        .arg_u32(concurrency)
        .arg_f32(activation_limit)
        .arg_u32(0) // MOE_ACT_SILU
        .arg_u32(4)
        .arg_u32(4)
        .arg_u32(4)
        .arg_ptr(workspace.locks)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    #[test]
    fn checkpoint_specialization_matches_glm_contract() {
        assert_eq!(super::GLM53_EXL3_ROWS, 7168);
        assert_eq!(super::GLM53_EXL3_CONCURRENCY, 8);
        let source =
            include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_exl3_moe.cu");
        assert!(source.contains("glm53_exl3_prepare_routes"));
        assert!(source.contains("#define MOE_TILESIZE_N 256"));
    }

    #[test]
    fn prefill_has_device_compacted_wide_fat_expert_path() {
        let source =
            include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_exl3_fat.cu");
        let registry = include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");
        assert!(source.contains("void glm53_exl3_fat_gather("));
        assert!(source.contains("void glm53_exl3_fat_gemm_gate_up("));
        assert!(source.contains("void glm53_exl3_fat_activate_down_had("));
        assert!(source.contains("void glm53_exl3_fat_gemm_down_scatter("));
        assert!(source.contains("atomicAdd("));
        assert!(source.contains("constexpr int FAT_SLOTS = 288"));
        assert!(registry.contains("glm53_exl3_fat = \"glm53_exl3_fat\""));
    }

    #[test]
    fn scratch_stride_tracks_live_rows_instead_of_arena_maximum() {
        assert_eq!(super::glm53_exl3_row_capacity(1), 1);
        assert_eq!(super::glm53_exl3_row_capacity(4), 4);
        assert_eq!(super::glm53_exl3_row_capacity(128), 128);
        assert_eq!(super::glm53_exl3_row_capacity(212), 128);
        assert_eq!(super::glm53_exl3_row_capacity(512), 128);
        assert_eq!(super::glm53_exl3_row_capacity(1024), 128);
        assert_eq!(super::glm53_exl3_row_capacity(7168), 128);
        assert_eq!(super::glm53_exl3_concurrency(1), 6);
        assert_eq!(super::glm53_exl3_concurrency(16), 6);
        assert_eq!(super::glm53_exl3_concurrency(17), 6);
        assert_eq!(super::glm53_exl3_concurrency(512), 6);
    }
}
