// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for GLM-5's pooled semantic-index primitives.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

#[allow(clippy::too_many_arguments)]
pub fn glm_index_layernorm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    values: DevicePtr,
    weight: DevicePtr,
    bias: DevicePtr,
    rows: u32,
    dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([128, 1, 1])
        .shared_mem(128 * std::mem::size_of::<f32>() as u32)
        .arg_ptr(values)
        .arg_ptr(weight)
        .arg_ptr(bias)
        .arg_u32(rows)
        .arg_u32(dim)
        .arg_f32(eps)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_tail_write(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    keys: DevicePtr,
    gates: DevicePtr,
    tail: DevicePtr,
    slots: DevicePtr,
    num_tokens: u32,
    block_size: u32,
    pool_size: u32,
    head_dim: u32,
    tail_block_stride_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(keys)
        .arg_ptr(gates)
        .arg_ptr(tail)
        .arg_ptr(slots)
        .arg_u32(num_tokens)
        .arg_u32(block_size)
        .arg_u32(pool_size)
        .arg_u32(head_dim)
        .arg_u64(tail_block_stride_bytes)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_kpool_finalize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    tail: DevicePtr,
    ape: DevicePtr,
    cache: DevicePtr,
    slots: DevicePtr,
    num_tokens: u32,
    block_size: u32,
    pool_size: u32,
    head_dim: u32,
    tail_block_stride_bytes: u64,
    values_block_stride_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(tail)
        .arg_ptr(ape)
        .arg_ptr(cache)
        .arg_ptr(slots)
        .arg_u32(num_tokens)
        .arg_u32(block_size)
        .arg_u32(pool_size)
        .arg_u32(head_dim)
        .arg_u64(tail_block_stride_bytes)
        .arg_u64(values_block_stride_bytes)
        .launch(stream)
}

pub fn glm_index_fill_causal(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    rows: u32,
    seq_len_start: u32,
    width: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, div_ceil(width, 256), 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(seq_len_start)
        .arg_u32(width)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_logits(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    weights: DevicePtr,
    index_cache: DevicePtr,
    logits: DevicePtr,
    block_table: DevicePtr,
    rows: u32,
    seq_len_start: u32,
    logits_stride: u32,
    index_heads: u32,
    head_dim: u32,
    pool_size: u32,
    cache_block_size: u32,
    index_block_stride_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(logits_stride, 8), rows, 1])
        .block([256, 1, 1])
        .arg_ptr(query)
        .arg_ptr(weights)
        .arg_ptr(index_cache)
        .arg_ptr(logits)
        .arg_ptr(block_table)
        .arg_u32(rows)
        .arg_u32(seq_len_start)
        .arg_u32(logits_stride)
        .arg_u32(index_heads)
        .arg_u32(head_dim)
        .arg_u32(pool_size)
        .arg_u32(cache_block_size)
        .arg_u64(index_block_stride_bytes)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_topk_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    output: DevicePtr,
    rows: u32,
    seq_len_start: u32,
    logits_stride: u32,
    topk_tokens: u32,
    pool_size: u32,
    output_width: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .shared_mem(4 * std::mem::size_of::<u32>() as u32)
        .arg_ptr(logits)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(seq_len_start)
        .arg_u32(logits_stride)
        .arg_u32(topk_tokens)
        .arg_u32(pool_size)
        .arg_u32(output_width)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_sparse_mla_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    token_indices: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    rows: u32,
    num_heads: u32,
    head_dim: u32,
    index_width: u32,
    cache_block_size: u32,
    heads_per_cta: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let heads_per_cta = if heads_per_cta == 8 { 8 } else { 1 };
    // head8 uses scores[8][8], betas[8][8], and three per-head state arrays.
    let shared_floats = if heads_per_cta == 8 { 152 } else { 19 };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_heads, heads_per_cta), rows, 1])
        .block([256, 1, 1])
        .shared_mem(shared_floats * std::mem::size_of::<f32>() as u32)
        .arg_ptr(query)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(token_indices)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(rows)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(index_width)
        .arg_u32(cache_block_size)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}
