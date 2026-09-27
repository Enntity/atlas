// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for the DFlash2 drafter kernels (`kernels/<hw>/common/dflash2.cu`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Kernel `DFLASH2_MAX_TAPS`: widest grouped conv the kernel supports.
pub const DFLASH2_MAX_TAPS: usize = 4;
/// Kernel `DFLASH2_TOPK`: candidates per draft position (compile-time).
pub const DFLASH2_TOPK: usize = 16;
/// Kernel `DFLASH2_MAX_RANK`: selector codebook rank capacity.
pub const DFLASH2_MAX_RANK: usize = 1024;

/// In-place grouped dynamic conv over `rows` block rows (one conv side).
///
/// Kernel: `dflash2_grouped_conv_bf16(x, delta, base, rows, hidden,
/// group_size, taps, block_size, delta_stride, delta_offset)`
/// Grid: (ceil(hidden/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dflash2_grouped_conv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    delta: DevicePtr,
    base: DevicePtr,
    rows: u32,
    hidden: u32,
    group_size: u32,
    taps: u32,
    block_size: u32,
    delta_stride: u32,
    delta_offset: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DFLASH2_MAX_TAPS as u32).contains(&taps),
        "dflash2_grouped_conv: taps={taps} outside 1..={DFLASH2_MAX_TAPS}"
    );
    ensure!(
        group_size > 0 && hidden % group_size == 0 && block_size > 0,
        "dflash2_grouped_conv: group_size={group_size} must divide hidden={hidden}, block_size={block_size} > 0"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(delta)
        .arg_ptr(base)
        .arg_u32(rows)
        .arg_u32(hidden)
        .arg_u32(group_size)
        .arg_u32(taps)
        .arg_u32(block_size)
        .arg_u32(delta_stride)
        .arg_u32(delta_offset)
        .launch(stream)
}

/// Per-row top-[`DFLASH2_TOPK`] of `[rows, vocab]` BF16 logits (descending,
/// ties to the lower id) into `ids` u32 / `vals` f32 `[rows, DFLASH2_TOPK]`.
///
/// Kernel: `dflash2_topk_bf16(logits, ids, vals, vocab)`
/// Grid: (rows, 1, 1)  Block: (256, 1, 1)
pub fn dflash2_topk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    ids: DevicePtr,
    vals: DevicePtr,
    rows: u32,
    vocab: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(ids)
        .arg_ptr(vals)
        .arg_u32(vocab)
        .launch(stream)
}

/// Greedy candidate-selector walk, one block per sequence of `gamma` rows.
/// Writes `tokens[b * gamma + 0]` = top-1 of the anchor row and
/// `tokens[b * gamma + 1..gamma]` = the walked drafts. Depths
/// `1..=ban_depth[b]` never draft one of `end_ids` (`ban_depth` may be null).
///
/// Kernel: `dflash2_selector_walk(cand_ids, cand_vals, hidden, pred, succ,
/// anchors, ban_depth, tokens, gamma, rank, vocab, end0..end3)`
/// Grid: (batch, 1, 1)  Block: (32 * DFLASH2_TOPK = 512, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_walk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    cand_ids: DevicePtr,
    cand_vals: DevicePtr,
    hidden: DevicePtr,
    predecessor_codebook: DevicePtr,
    successor_codebook: DevicePtr,
    anchors: DevicePtr,
    ban_depth: DevicePtr,
    end_ids: [u32; 4],
    tokens: DevicePtr,
    batch: u32,
    gamma: u32,
    rank: u32,
    vocab: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DFLASH2_MAX_RANK as u32).contains(&rank),
        "dflash2_selector_walk: rank={rank} outside 1..={DFLASH2_MAX_RANK}"
    );
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid([batch, 1, 1])
        .block([32 * DFLASH2_TOPK as u32, 1, 1])
        .arg_ptr(cand_ids)
        .arg_ptr(cand_vals)
        .arg_ptr(hidden)
        .arg_ptr(predecessor_codebook)
        .arg_ptr(successor_codebook)
        .arg_ptr(anchors)
        .arg_ptr(ban_depth)
        .arg_ptr(tokens)
        .arg_u32(gamma)
        .arg_u32(rank)
        .arg_u32(vocab);
    for id in end_ids {
        launch = launch.arg_u32(id);
    }
    launch.launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust capacity constants must mirror the kernel's `#define`s.
    #[test]
    fn dflash2_constants_match_kernel_defines() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/common/dflash2.cu"
        ));
        for (name, value) in [
            ("DFLASH2_MAX_TAPS", DFLASH2_MAX_TAPS),
            ("DFLASH2_TOPK", DFLASH2_TOPK),
            ("DFLASH2_MAX_RANK", DFLASH2_MAX_RANK),
        ] {
            let define = format!("#define {name} {value}\n");
            assert!(
                source.contains(&define),
                "kernel is missing `{}`",
                define.trim()
            );
        }
    }
}
