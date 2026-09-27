// SPDX-License-Identifier: AGPL-3.0-only
//! Large-M BF16 projections through CUTLASS tiles with an L2-aware CTA raster.
//!
//! cuBLASLt's heuristic picks an sm80 kernel on GB10 that plateaus near
//! 62 TFLOPS; at K >= 4096 an unswizzled raster re-streams every weight strip
//! from DRAM per row of tiles. `examples/bf16_gemm_bench` measured the configs
//! below at 80-92 TFLOPS on the GLM prefill shapes (M = 4096).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::sync::OnceLock;

/// Rows below this keep cuBLASLt (decode/verify widths).
const MIN_ROWS: u32 = 256;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_BF16_GEMM_CUTLASS").as_deref() == Ok("1"))
}

/// `spark_runtime::cutlass::bf16_gemm_tuned` config for an `[n, k]` weight:
/// narrow N takes 128x256 tiles (swizzle 8), K >= 8192 128x256 (swizzle 4),
/// short K the unswizzled 128x128x32, everything else 128x128x64 (swizzle 8).
fn config(n: u32, k: u32) -> u32 {
    if n <= 1536 {
        5
    } else if k >= 8192 {
        4
    } else if k <= 2048 {
        0
    } else {
        9
    }
}

/// Row-major `out[m, n] = act[m, k] @ weight[n, k]^T`, BF16 in/out with FP32
/// accumulation: CUTLASS for large M when `ATLAS_BF16_GEMM_CUTLASS=1`, else
/// (or if CUTLASS rejects the operands) cuBLASLt.
pub fn bf16_gemm(
    act: DevicePtr,
    weight: u64,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if enabled() && m >= MIN_ROWS {
        let cfg = config(n, k);
        let ok = row_blocks(m, k, cfg).all(|(row, rows)| {
            spark_runtime::cutlass::bf16_gemm_tuned(
                act.0 + u64::from(row) * u64::from(k) * 2,
                weight,
                out.0 + u64::from(row) * u64::from(n) * 2,
                rows,
                n,
                k,
                k,
                n,
                cfg,
                stream,
            )
            .is_ok()
        });
        if ok {
            return Ok(());
        }
    }
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(act.0, weight, out.0, m, n, k, stream)
}

/// Activation bytes the unswizzled raster (config 0) keeps L2-resident while
/// every column of tiles re-reads them: 4096 rows at K 2048. Past this (an
/// 8K prefill chunk) each re-read goes to DRAM, ~7x slower, so wider calls
/// run as consecutive row blocks (rows are independent: identical output).
const UNSWIZZLED_ACT_BYTES: u64 = 16 << 20;

fn row_blocks(m: u32, k: u32, cfg: u32) -> impl Iterator<Item = (u32, u32)> {
    let limit = ((UNSWIZZLED_ACT_BYTES / (u64::from(k) * 2)) as u32 / 128 * 128).max(128);
    // A few scheduling rows past the limit (a 4100-row chunk) stay whole
    // rather than pay a tiny tail launch that re-reads the weight.
    let block = if cfg == 0 && m > limit + 64 {
        m.div_ceil(m.div_ceil(limit)).next_multiple_of(128)
    } else {
        m
    };
    (0..m.div_ceil(block)).map(move |i| (i * block, block.min(m - i * block)))
}

/// Whether [`bf16_gemm`] routes `m` rows through CUTLASS.
pub fn bf16_gemm_cutlass_rows(m: u32) -> bool {
    enabled() && m >= MIN_ROWS
}

/// Exact E4M3 -> BF16 widening of `count` values (`count % 8 == 0`).
pub fn fp8_e4m3_to_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    dst: DevicePtr,
    count: usize,
    stream: u64,
) -> Result<()> {
    ensure!(count % 8 == 0, "E4M3 widening needs count % 8 == 0 ({count})");
    let kernel = gpu.op_cache().kernel(gpu, "fp8_e4m3_to_bf16", "fp8_e4m3_to_bf16")?;
    let count8 = (count / 8) as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(u32::try_from(count8)?, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u64(count8)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::{config, row_blocks};

    #[test]
    fn unswizzled_calls_split_into_l2_resident_row_blocks() {
        let blocks: Vec<_> = row_blocks(8188, 2048, 0).collect();
        assert_eq!(blocks, [(0, 4096), (4096, 4092)]);
        assert_eq!(row_blocks(4100, 2048, 0).collect::<Vec<_>>(), [(0, 4100)]);
        assert_eq!(row_blocks(16388, 2048, 0).count(), 5);
        // Swizzled configs keep one launch.
        assert_eq!(row_blocks(8188, 4096, 9).collect::<Vec<_>>(), [(0, 8188)]);
    }

    #[test]
    fn config_follows_measured_glm_shapes() {
        assert_eq!(config(1536, 4096), 5); // q_a
        assert_eq!(config(576, 4096), 5); // kv_a
        assert_eq!(config(4096, 8192), 4); // o
        assert_eq!(config(8192, 1536), 0); // q_b
        assert_eq!(config(4096, 2048), 0); // shared down
        assert_eq!(config(2048, 4096), 9); // shared gate/up
    }
}
