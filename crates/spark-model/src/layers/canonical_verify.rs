// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_CANONICAL_VERIFY` (default 1; `=0` restores the width-tuned
//! kernels): one row-invariant kernel family per GLM verify op for every row
//! count 1..=[`MAX_ROWS`], so a row's bits depend only on its own inputs.
//!
//! A DFlash verify's width follows the drafter (and an owner-batched verify's
//! total follows the concurrency), so any kernel chosen by row count reaches
//! the target's greedy output. Before this switch a row went through:
//! - W4A16 projections: scalar `w4a16_gemv_batch2/3` at 2..3 rows, the
//!   128-wide-K `w4a16_gemv_tc8` at 4..8 (or the fused K=5 QKV body), the
//!   64-wide-K `tc16`/`tc32` above 8, and the scalar `w4a16_gemv` / prefill
//!   GEMM at one row: four summation orders.
//! - The MoE router: scalar strict-order kernels (`dense_gemm_router`, the C3
//!   rows kernel, BN4 at 5), `dense_gemv_bf16_batchm` lane splits, or the
//!   tensor-core `dense_gemv_bf16_tc16/32` above 8 rows, so near-tied experts
//!   flipped with the width.
//! - The KDA BF16 side projections: `dense_gemv` / batch-M below 9 rows, the
//!   tensor-core tiers above.
//!
//! Canonical mode keeps one family per op, the one the owner-batched (9..32
//! row) verify already ran:
//! - W4A16: `w4a16_gemv_tc8c` (the NT = 1 instance of the `tc16`/`tc32`
//!   template: same 64-wide K chunks, lane runs, MMA order and cross-warp
//!   sum), `tc16`, `tc32` ([`w4a16`]). Their touch, strided and pair twins run
//!   the same bodies.
//! - BF16 GEMV (router, KDA beta/f/g, MLA indexer projections):
//!   `dense_gemv_bf16_tc8` (NT = 1 of the tc16/tc32 template), `tc16`,
//!   `tc32` ([`dense`]); the MLA W_uk / W_uv per-head GEMMs on their grouped
//!   twins ([`dense_grouped`]).
//! - Shared expert: the W4A16 family above, whole or TP-split.
//! - The K=5-only seams (fused QKV, fused dense pairs/triple, fused TP mHC,
//!   K5 mHC cuBLAS, compact K5 routed MoE, deferred K5 shared blend) are off,
//!   and a one-row verify takes the MLA prefill lane like a wider block.
//!
//! A tensor-core MMA's output column depends only on its own B column, and
//! these templates fix every other step of a row's sum (warp chunking,
//! prefetch, cross-warp order) independently of the row count and the row's
//! slot, so rows 1..32 come out bit-identical
//! (`canonical_verify_gpu_tests.rs`). The family needs `ATLAS_W4A16_TC=1`
//! (the tensor-core tiers); without it the switch is inert. Both ranks must
//! run the same value (`model::startup_parity`): the shared expert's TP split
//! follows it.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Widest row count of one canonical launch (an owner-batched verify).
pub const MAX_ROWS: u32 = 32;

/// `ATLAS_GLM_CANONICAL_VERIFY` as set: anything but `0` is on. Read once.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_CANONICAL_VERIFY").as_deref() != Ok("0"))
}

/// Whether the canonical kernels serve: requested, with the tensor-core tiers.
pub fn enabled() -> bool {
    requested() && super::w4a16_gemv_tiers::tc_requested()
}

/// `C[m, n] = A[m, k] · W[n, k]ᵀ` (NVFP4 `W`) on the canonical tensor-core
/// tier for `m` rows, through its PDL touch twin when that is armed
/// (`ATLAS_GLM_DECODE_GEMV_BATCH`, same body).
#[allow(clippy::too_many_arguments)]
pub fn w4a16(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let tier = super::w4a16_gemv_tiers::tc_kernel(m);
    ensure!(
        (1..=MAX_ROWS).contains(&m) && tier.0 != 0,
        "canonical W4A16 GEMV: no tensor-core tier for {m} rows"
    );
    let touch = super::w4a16_gemv_tiers::tc_rows(tier)
        .and_then(ops::w4a16_tc_twin)
        .and_then(ops::gemv_touch);
    match touch {
        Some(touch) => touch.w4a16_tc(gpu, input, weight, output, m, n, k, k / 2, k / 16, stream),
        None => ops::w4a16_gemv_batchm(gpu, tier, input, weight, output, m, n, k, stream),
    }
}

/// The canonical BF16 tensor-core GEMV tier for `m` rows
/// (`dense_gemv_bf16_tc{8,16,32}`), or a zero handle.
pub fn dense_kernel(gpu: &dyn GpuBackend, m: u32) -> KernelHandle {
    static TC: std::sync::OnceLock<[KernelHandle; 3]> = std::sync::OnceLock::new();
    let tiers = TC.get_or_init(|| {
        [8, 16, 32].map(|rows| {
            super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                &format!("dense_gemv_bf16_tc{rows}"),
            )
        })
    });
    match m {
        1..=8 => tiers[0],
        9..=16 => tiers[1],
        17..=MAX_ROWS => tiers[2],
        _ => KernelHandle(0),
    }
}

/// `C[m, n] = A[m, k] · W[n, k]ᵀ` (BF16 `W`), rows `out_stride` apart, on the
/// canonical tensor-core tier for `m` rows.
#[allow(clippy::too_many_arguments)]
pub fn dense(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    let tier = dense_kernel(gpu, m);
    ensure!(
        tier.0 != 0 && k.is_multiple_of(8),
        "canonical BF16 GEMV: no tensor-core tier for {m} rows (k={k})"
    );
    ops::dense_gemv_bf16_tc(
        gpu, tier, input, weight, output, m, n, k, out_stride, stream,
    )
}

/// Per-head `c[:, h*n..] = a[:, h*k..] · W[h]ᵀ` over `g` heads of a BF16
/// `W` `[g, n, k]` (GLM MLA W_uk absorb / W_uv extract), rows `a_stride` /
/// `c_stride` elements apart, on the canonical tensor-core tier for `m` rows
/// (`dense_gemv_bf16_tc{8,16,32}_grouped`: the plain tiers' body per head).
#[allow(clippy::too_many_arguments)]
pub fn dense_grouped(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    weight: DevicePtr,
    c: DevicePtr,
    [m, g, k, n, a_stride, c_stride]: [u32; 6],
    stream: u64,
) -> Result<()> {
    static TC: std::sync::OnceLock<[KernelHandle; 3]> = std::sync::OnceLock::new();
    let tiers = TC.get_or_init(|| {
        [8, 16, 32].map(|rows| {
            super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                &format!("dense_gemv_bf16_tc{rows}_grouped"),
            )
        })
    });
    let tier = match m {
        1..=8 => tiers[0],
        9..=16 => tiers[1],
        17..=MAX_ROWS => tiers[2],
        _ => KernelHandle(0),
    };
    ensure!(
        tier.0 != 0 && k.is_multiple_of(8) && a_stride.is_multiple_of(8),
        "canonical grouped BF16 GEMV: no tier for {m} rows (k={k}, lda={a_stride})"
    );
    spark_runtime::kernel_args::KernelLaunch::new(gpu, tier)
        .grid([n.div_ceil(16), 1, g])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(weight)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_stride)
        .arg_u32(c_stride)
        .launch(stream)
}

#[cfg(test)]
#[path = "canonical_verify_gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use crate::layers::w4a16_gemv_tiers::tc_name;

    #[test]
    fn the_eight_row_tensor_core_tier_is_the_canonical_template_by_default() {
        // `ATLAS_GLM_CANONICAL_VERIFY` unset: on.
        if std::env::var_os("ATLAS_GLM_CANONICAL_VERIFY").is_some() {
            return;
        }
        assert_eq!(tc_name(8, ""), "w4a16_gemv_tc8c");
        assert_eq!(tc_name(8, "_ld"), "w4a16_gemv_tc8c_ld");
        assert_eq!(tc_name(8, "_pair_touch"), "w4a16_gemv_tc8c_pair_touch");
        assert_eq!(tc_name(16, "_touch"), "w4a16_gemv_tc16_touch");
        assert_eq!(tc_name(32, ""), "w4a16_gemv_tc32");
    }
}
