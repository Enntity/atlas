// SPDX-License-Identifier: AGPL-3.0-only

//! Range guard on the token-id rows `glm_index_split` receives from the peer.
//!
//! The sparse prefill kernels index the block table with a selected id
//! unchecked. The rows this rank selects are in range by construction, but
//! the peer's are whatever the exchange landed: a desynchronized or faulty
//! peer could make this rank read outside its block table and KV pool. So
//! behind each exchange, on the same stream, `glm_index_clamp_ids` rewrites
//! every received id outside `[-1, end)` to -1 (no selection) and counts it
//! in a device word. An in-range id is never written: valid rows keep every
//! byte.
//!
//! `end` is one past the owner's last token position. The chunk's cache write
//! has landed every position below it, so the block table (and an `fp8_g128`
//! owner's BF16 view) covers them, and both ranks hold the same value: it is
//! the mirrored input the split was planned from. A row's own causal extent
//! is a tighter bound, but memory safety does not need it.
//!
//! The guard never waits for the device. [`check_index_split_rows`] reads the
//! count where the chunk's command ends and fails the chunk on this rank:
//! every collective of the command is on the stream by then, so a rank that
//! fails alone leaves its peer nothing to wait for.

use std::cell::Cell;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layer::ForwardContext;

/// Ids per CTA, eight a thread: the fastest grid for a quarter of a 4096-row
/// owner (`scripts/dev/glm_index_split_guard_bench.cu`).
const IDS_PER_CTA: usize = 2048;

thread_local! {
    /// Whether the chunk this thread is running has guarded rows since the
    /// last check.
    static PENDING: Cell<bool> = const { Cell::new(false) };
}

/// The backend's count of clamped ids (a `u32`).
fn counter(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    gpu.op_cache().scratch(gpu, "glm_index_split_guard", 4)
}

/// CTAs over `ids` ids.
fn grid(ids: usize) -> u32 {
    ids.div_ceil(IDS_PER_CTA) as u32
}

/// Clamp the `ids` token ids the peer landed at `received` to `[-1, end)`.
pub(super) fn clamp(
    received: DevicePtr,
    ids: usize,
    end: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let gpu = ctx.gpu;
    let kernel = gpu
        .op_cache()
        .kernel(gpu, "glm_indexer", "glm_index_clamp_ids")?;
    let (count, limit) = (u32::try_from(ids)?, i32::try_from(end)?);
    let clamped = counter(gpu)?;
    // The chunk's first guard starts the count from zero.
    if !PENDING.replace(true) {
        gpu.memset_zero_async(clamped, 4, stream)?;
    }
    KernelLaunch::new(gpu, kernel)
        .grid([grid(ids), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(received)
        .arg_u32(count)
        .arg_i32(limit)
        .arg_ptr(clamped)
        .launch(stream)
}

/// Call on every rank where a prefill chunk's command ends, once all of its
/// work is on `stream`: fails the chunk if the guard clamped an id in it.
/// A chunk that exchanged no rows reads nothing. One that did waits for
/// `stream` here instead of at the next command's broadcast or the first
/// token's sampling, which is the next thing either rank does.
pub fn check_index_split_rows(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    if !PENDING.replace(false) {
        return Ok(());
    }
    let mut clamped = [0u8; 4];
    gpu.copy_d2h_on_stream(counter(gpu)?, &mut clamped, stream)?;
    let clamped = u32::from_le_bytes(clamped);
    ensure!(
        clamped == 0,
        "ATLAS_GLM_INDEX_SPLIT: the peer's index-split rows held {clamped} out-of-range token ids (a desynchronized peer); each was clamped to unselected, so nothing was read out of range"
    );
    Ok(())
}

#[cfg(test)]
#[path = "glm_index_split_guard_tests.rs"]
mod tests;
