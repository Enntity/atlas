// SPDX-License-Identifier: AGPL-3.0-only

//! Native-FP4 grouped MoE launcher. Split from `moe_grouped_a.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// Grouped W4A4 expert UP GEMM with fused relu^2 (native FP4 tensor cores).
/// A is the pre-quantized NVFP4 latent (packed E2M1 + per-16 E4M3 scales);
/// B comes from the per-expert NVFP4 pointer tables unchanged.
/// Grid: (ceil(n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_relu2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_sf: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_sf)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Native-FP4 grouped GEMM over an activation quantized once outside the
/// output-column grid. A uses the common row-major NVFP4 packed/scale layout.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Native-FP4 prequant GEMM over a compact `(expert, m_tile, n_tile)`
/// worklist. A conservative work-item bound replaces the dense expert grid;
/// excess CTAs exit before entering the unchanged per-tile MMA implementation.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_compact_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_tiles.max(1), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(max_tiles)
        .launch(stream)
}

/// Compact native-FP4 gate and up projections in one projection-multiplexed
/// launch. Grid y selects gate/up; grid x indexes the common device worklist.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_output: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_tiles.max(1), 2, 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_output)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(max_tiles)
        .launch(stream)
}
