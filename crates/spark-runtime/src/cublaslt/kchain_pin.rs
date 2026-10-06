// SPDX-License-Identifier: AGPL-3.0-only
//! The k-chain pin (`ATLAS_LT_KCHAIN_PIN=1`, default off): a faster cuBLASLt
//! kernel for a large BF16 projection, chosen only where it is byte-identical
//! to the one the heuristic picks.
//!
//! On the runtime image's cuBLASLt (CUDA 13.0, 130000) the heuristic's first
//! choice for a large prefill projection is often the 64x64-tile CUTLASS sm80
//! kernel (algo 21, tile 15) at ~57 TFLOP/s, while the same algorithm with a
//! 256x128 tile (tile 24), offered further down the same heuristic list, runs
//! the same products at ~91: the qwen4_exp TP2 GDN in_proj at 16016 rows is
//! 11.7 -> 7.4 ms. Both are algo 21 without split-K: every output is one FP32
//! accumulator over the m16n8k16 MMAs in increasing k, so the tile only
//! changes which CTA computes it -- byte-identical results
//! (`scripts/dev/qwen4exp_lt_algo_bench.cu pin N K M0 M1 STEP`, inside
//! atlas-release-builder: 0 differing bytes at every M swept for N x K =
//! 8192x2560, 6144x2560, 12288x2560, 4096x2560, 1024x2560, 512x2560,
//! 2560x3072, M = 1024..16384).
//!
//! The rule, applied only to `[N,K]` weights (opT) with M and N >= 2048: if
//! the heuristic's first result is algo 21 with split-K 1 and no reduction,
//! take the first offered algo-21, split-K-1, no-reduction result whose tile
//! is 24 (256x128), else 23 (128x256); otherwise keep the first result. The
//! pick depends only on the library's answer for the shape, so both TP ranks
//! (same library, same GPU) pick the same kernel.

use std::ffi::c_void;

/// Heuristic results requested when the pin may apply.
pub(super) const CANDIDATES: usize = 8;
/// `sizeof(cublasLtMatmulHeuristicResult_t)` (algo[64] + workspaceSize +
/// state + wavesCount + reserved[4]) on CUDA 12/13.
pub(super) const RESULT_BYTES: usize = 96;

const CONFIG_ID: u32 = 0;
const CONFIG_TILE_ID: u32 = 1;
const CONFIG_SPLITK_NUM: u32 = 2;
const CONFIG_REDUCTION_SCHEME: u32 = 3;
/// The CUTLASS sm80 tensor-op GEMM.
const ALGO_SM80: i32 = 21;
/// 256x128, then 128x256.
const PREFERRED_TILES: [i32; 2] = [24, 23];

unsafe extern "C" {
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        buf: *mut c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// `ATLAS_LT_KCHAIN_PIN=1`.
fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_LT_KCHAIN_PIN").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Whether a GEMM of this shape and weight layout asks for candidates.
pub(super) fn applies(m: u32, n: u32, weight_nk: bool) -> bool {
    requested() && weight_nk && m >= 2048 && n >= 2048 && super::version() >= 130000
}

fn attr(algo: *const c_void, which: u32) -> Option<i32> {
    let mut v = 0i32;
    let mut written = 0usize;
    // SAFETY: `algo` points at a heuristic result's algo field (64 bytes,
    // written by cublasLtMatmulAlgoGetHeuristic); `v` is a 4-byte buffer.
    let st = unsafe {
        cublasLtMatmulAlgoConfigGetAttribute(
            algo,
            which,
            &mut v as *mut i32 as *mut c_void,
            4,
            &mut written,
        )
    };
    (st == 0).then_some(v)
}

/// Algo 21 without split-K or a reduction: the in-order k-chain.
fn kchain(algo: *const c_void) -> bool {
    attr(algo, CONFIG_ID) == Some(ALGO_SM80)
        && attr(algo, CONFIG_SPLITK_NUM).is_some_and(|s| s <= 1)
        && attr(algo, CONFIG_REDUCTION_SCHEME) == Some(0)
}

/// Heuristic results to ask for.
pub(super) fn requested_count(pin: bool) -> i32 {
    if pin { CANDIDATES as i32 } else { 1 }
}

/// Byte offset in `results` of the algorithm to run: the pick when `pin`,
/// else the heuristic's first result.
pub(super) fn offset(pin: bool, results: &[u8], returned: i32) -> usize {
    if pin {
        pick(results, returned.max(0) as usize) * RESULT_BYTES
    } else {
        0
    }
}

/// Index of the result to run among `found` heuristic results packed at
/// `RESULT_BYTES` in `results`.
fn pick(results: &[u8], found: usize) -> usize {
    let at = |i: usize| results[i * RESULT_BYTES..].as_ptr() as *const c_void;
    let found = found.min(results.len() / RESULT_BYTES);
    if found == 0 || !kchain(at(0)) {
        return 0;
    }
    PREFERRED_TILES
        .iter()
        .find_map(|&tile| {
            (0..found).find(|&i| kchain(at(i)) && attr(at(i), CONFIG_TILE_ID) == Some(tile))
        })
        .unwrap_or(0)
}
