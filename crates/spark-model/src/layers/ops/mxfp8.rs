// SPDX-License-Identifier: AGPL-3.0-only
//! MXFP8 (E4M3 + E8M0 per 32 values) weights for small-row projections.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Values per E8M0 scale block.
pub const MXFP8_BLOCK: usize = 32;

/// Quantize a BF16 `[n, k]` weight into E4M3 `data` (`n*k` bytes) and E8M0
/// `scales` (`n*k/32` bytes).
pub fn mxfp8_quantize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    weight: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 needs K % 32 == 0 (K={k})"
    );
    let blocks = (n * k / MXFP8_BLOCK) as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_u64(blocks)
        .launch(stream)
}

/// `C[m, n] = A[m, k] · dequant(W)[n, k]ᵀ` on the tensor cores; `kernel` is
/// `mxfp8_gemv_tc8/16/32` for up to 8/16/32 rows.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=32).contains(&m) && (k as usize).is_multiple_of(MXFP8_BLOCK),
        "mxfp8_gemv: m={m} k={k} unsupported"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}
