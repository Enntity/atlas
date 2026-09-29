// SPDX-License-Identifier: AGPL-3.0-only
//! Launchers for `glm_kv_shard.cu` (`ATLAS_GLM_KV_SHARD=1`): the device half
//! of `spark_runtime::kv_cache::LatentShard`'s block-ownership rule.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MODULE: &str = "glm_kv_shard";

fn kernel(gpu: &dyn GpuBackend, symbol: &'static str) -> Result<KernelHandle> {
    let k = gpu.op_cache().kernel(gpu, MODULE, symbol)?;
    ensure!(k.0 != 0, "GLM KV shard kernel {symbol} is unavailable");
    Ok(k)
}

/// Rank ownership of the pair, as the kernels take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardRank {
    pub rank: u32,
    pub world: u32,
}

/// Global cache-write slots `[n]` i64 → this rank's local slots (`-1` where
/// the peer owns the block).
pub fn glm_kv_shard_map_slots(
    gpu: &dyn GpuBackend,
    slots: DevicePtr,
    local: DevicePtr,
    n: u32,
    block_size: u32,
    owner: ShardRank,
    stream: u64,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_shard_map_slots")?)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(slots)
        .arg_ptr(local)
        .arg_u32(n)
        .arg_u32(block_size)
        .arg_u32(owner.rank)
        .arg_u32(owner.world)
        .launch(stream)
}

/// Selected token IDs `[rows, width]` → this rank's local token IDs
/// (addressed through the shard's identity table), `-1` where the peer owns
/// the block. `selected == None` generates causal IDs: row `r` keeps tokens
/// `[0, causal_start + r + 1)`.
#[allow(clippy::too_many_arguments)]
pub fn glm_kv_shard_localize(
    gpu: &dyn GpuBackend,
    selected: Option<DevicePtr>,
    out: DevicePtr,
    block_table: DevicePtr,
    rows: u32,
    width: u32,
    block_size: u32,
    owner: ShardRank,
    causal_start: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > 0 && rows <= 65535 && width > 0,
        "GLM KV shard localize needs 1..=65535 rows"
    );
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_shard_localize")?)
        .grid([div_ceil(width, 256), rows, 1])
        .block([256, 1, 1])
        .arg_ptr(selected.unwrap_or(DevicePtr::NULL))
        .arg_ptr(out)
        .arg_ptr(block_table)
        .arg_u32(rows)
        .arg_u32(width)
        .arg_u32(block_size)
        .arg_u32(owner.rank)
        .arg_u32(owner.world)
        .arg_u32(causal_start)
        .launch(stream)
}

/// Copy `n` blocks of `block_bytes`: destination block `dst_idx[i]` (or `i`)
/// receives source block `src_idx[i]` (or `i`); `u32` index lists.
#[allow(clippy::too_many_arguments)]
pub fn glm_kv_shard_copy_blocks(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    src_idx: Option<DevicePtr>,
    dst: DevicePtr,
    dst_idx: Option<DevicePtr>,
    n: usize,
    block_bytes: usize,
    stream: u64,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    ensure!(
        block_bytes.is_multiple_of(16)
            && src.0.is_multiple_of(16)
            && dst.0.is_multiple_of(16)
            && u32::try_from(n).is_ok(),
        "GLM KV shard block copy needs 16-byte blocks and bases"
    );
    KernelLaunch::new(gpu, kernel(gpu, "glm_kv_shard_copy_blocks")?)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(src_idx.unwrap_or(DevicePtr::NULL))
        .arg_ptr(dst)
        .arg_ptr(dst_idx.unwrap_or(DevicePtr::NULL))
        .arg_u32(n as u32)
        .arg_u32((block_bytes / 16) as u32)
        .launch(stream)
}
