// SPDX-License-Identifier: AGPL-3.0-only
//! Which cuBLASLt kernel runs an MXFP8 W8A8 GEMM
//! ([`super::fp8_gemm_act_weight_t_mx`]): per `(N, K, row bucket)`, the
//! fastest of the heuristic's in-order candidates, timed once and cached.
//!
//! Why not the heuristic's first pick: it is 2-3x slower than the best
//! offered kernel at some (shape, M) points of FP4-Local's prefill
//! projections, and which points lose moves with M and with the library
//! (GB10, the TP2 GDN qkvz 8192x2560 at 16384 rows: 7.89 vs 4.18 ms on
//! cuBLASLt 13.1.1; 13.0.0, the runtime image's, picks well there). So the
//! choice is per bucket of M (powers of two, 64 .. 16384), not per shape.
//!
//! Why only in-order candidates (no split-K, no reduction scheme): under one
//! such kernel every output is one FP32 accumulation of the block-scaled k32
//! MMAs over K in order, whatever the tile, the row count or the row's
//! position in the pass, so they all give the same bytes, and a bucket may
//! take any of them without breaking the row invariance the qwen4_exp
//! prefill keeps (`ATLAS_QWEN4EXP_PREFILL_ROWINV`: a token's output
//! independent of the pass's height and of where it starts). The heuristic's
//! split-K candidates (splitK -2, reduction 2) are not: they differed from
//! the in-order bytes in 1-17 outputs at many (shape, M) points, and are
//! sometimes its FIRST pick (13.0.0 offers only a split-K kernel for the
//! TP2 indexer 320x2560 at 64 and 100 rows unless asked for in-order ones).
//! `mxfp8_act_test_gpu.rs` (spark-model) checks this byte for byte on both
//! 13.0.0 and 13.1.1: every candidate, M = 1 .. 16384 at odd offsets, every
//! FP4-Local TP2 shape -- in-order candidates never differed.
//!
//! Timing runs on the caller's own operands (the output is rewritten by the
//! real call that follows) and synchronizes `stream`, once per key; never
//! under graph capture. [`fp8_gemm_mx_prepare`] does it at load instead.

use std::sync::Mutex;

use super::fixed_algo::{Operands, Problem, cublasLtMatmulAlgoCheck};
use super::kchain_pin::{RESULT_BYTES, attr, describe};
use super::*;

/// Heuristic results asked for per selection.
pub const MX_CANDIDATES: usize = 16;
/// `sizeof(cublasLtMatmulAlgo_t)`.
const ALGO_BYTES: usize = 64;
/// Row buckets: powers of two in `MIN_BUCKET ..= MAX_BUCKET`.
const MIN_BUCKET: u32 = 64;
const MAX_BUCKET: u32 = 16384;
/// Timed launches per candidate, after `WARM` untimed ones.
const WARM: usize = 2;
const TIMED: usize = 10;

pub(super) type Algo = [u8; ALGO_BYTES];
type Key = (u32, u32, u32);
static CACHE: Mutex<Vec<(Key, Algo)>> = Mutex::new(Vec::new());

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
    fn cuMemsetD8Async(dst: u64, value: u8, count: usize, stream: u64) -> i32;
}

/// The row bucket `m` rows are served from.
pub(super) fn bucket(m: u32) -> u32 {
    m.max(1)
        .checked_next_power_of_two()
        .unwrap_or(MAX_BUCKET)
        .clamp(MIN_BUCKET, MAX_BUCKET)
}

/// cuBLASLt versions ([`super::version`]) on which `mxfp8_gemm_test_gpu.rs`
/// found every in-order candidate byte-identical across row counts and pass
/// offsets: 13.0.0 (the runtime image) and 13.1.1. On any other version
/// every row count shares ONE in-order kernel (the first call's pick), which
/// is row-invariant by itself (`fixed_algo`'s argument), at some speed.
const MX_VERIFIED_LT: [usize; 2] = [130000, 130101];

/// Whether this cuBLASLt may take a kernel per row bucket.
fn per_bucket() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let v = super::version();
        let ok = MX_VERIFIED_LT.contains(&v);
        if !ok {
            tracing::warn!(
                "cuBLASLt MXFP8: {v} is not a version the per-bucket kernel choice was \
                 verified on ({MX_VERIFIED_LT:?}); one kernel serves every row count"
            );
        }
        ok
    })
}

/// The cache key `m` rows of `n x k` are served from.
fn key(n: u32, k: u32, m: u32) -> Key {
    (n, k, if per_bucket() { bucket(m) } else { 0 })
}

/// `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`.
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;

/// The heuristic's algorithms for `p`, best first; `in_order_only` asks for
/// reduction scheme NONE only (a mask of 0), i.e. no split-K. On cuBLASLt
/// 13.0.0 the unmasked heuristic offers ONLY a split-K kernel for some
/// narrow shapes at small M (the TP2 indexer 320x2560 at 64 and 100 rows);
/// masked, it offers in-order ones there too.
pub(super) fn candidates(ctx: &Ctx, p: &Problem, in_order_only: bool) -> Result<Vec<Algo>> {
    let mut results = vec![0u8; RESULT_BYTES * MX_CANDIDATES];
    let mut returned = 0i32;
    // SAFETY: `results` holds MX_CANDIDATES heuristic results; the
    // preference is created and destroyed here.
    unsafe {
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws = ctx.ws_size;
        let mut st = cublasLtMatmulPreferenceSetAttribute(
            pref,
            PREF_MAX_WORKSPACE_BYTES,
            &ws as *const usize as *const c_void,
            std::mem::size_of::<usize>(),
        );
        let none = 0u32;
        if st == 0 && in_order_only {
            st = cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_REDUCTION_SCHEME_MASK,
                &none as *const u32 as *const c_void,
                4,
            );
        }
        if st == 0 {
            st = cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                p.desc,
                p.la,
                p.lb,
                p.ld,
                p.ld,
                pref,
                MX_CANDIDATES as i32,
                results.as_mut_ptr() as *mut c_void,
                &mut returned,
            );
        }
        cublasLtMatmulPreferenceDestroy(pref);
        chk(st, "AlgoGetHeuristic")?;
    }
    let found = (returned.max(0) as usize).min(MX_CANDIDATES);
    Ok((0..found)
        .map(|i| {
            let mut a = [0u8; ALGO_BYTES];
            a.copy_from_slice(&results[i * RESULT_BYTES..i * RESULT_BYTES + ALGO_BYTES]);
            a
        })
        .collect())
}

/// No split-K and no reduction scheme (see the module docs).
pub(super) fn in_order(algo: &Algo) -> bool {
    super::kchain_pin::in_order(algo.as_ptr() as *const c_void)
}

/// `id=.. tile=.. splitK=.. reduction=.. stages=.. swizzle=.. custom=..
/// cluster=..` of `algo` (the heuristic's MXFP8 candidates on GB10 share one
/// id and tile and differ in the rest).
pub(super) fn description(algo: &Algo) -> String {
    let a = algo.as_ptr() as *const c_void;
    let v = |w| attr(a, w).map_or_else(|| "?".to_string(), |v| v.to_string());
    format!(
        "{} stages={} swizzle={} custom={} cluster={}",
        describe(a),
        v(CONFIG_STAGES_ID),
        v(CONFIG_CTA_SWIZZLING),
        v(CONFIG_CUSTOM_OPTION),
        v(CONFIG_CLUSTER_SHAPE_ID)
    )
}

/// `CUBLASLT_ALGO_CONFIG_*` attributes [`description`] adds.
const CONFIG_CTA_SWIZZLING: u32 = 4;
const CONFIG_CUSTOM_OPTION: u32 = 5;
const CONFIG_STAGES_ID: u32 = 6;
const CONFIG_CLUSTER_SHAPE_ID: u32 = 8;

/// Whether `algo` takes problem `p`.
pub(super) fn takes(ctx: &Ctx, p: &Problem, algo: &Algo) -> bool {
    let mut check = [0u8; 128];
    // SAFETY: `algo` is a heuristic result's algo field; `check` holds one
    // heuristic result.
    unsafe {
        cublasLtMatmulAlgoCheck(
            ctx.handle,
            p.desc,
            p.la,
            p.lb,
            p.ld,
            p.ld,
            algo.as_ptr() as *const c_void,
            check.as_mut_ptr() as *mut c_void,
        ) == 0
    }
}

/// `out = act @ weight^T` on `algo`; `[weight, act, out]` device pointers.
pub(super) fn matmul(
    ctx: &Ctx,
    p: &Problem,
    algo: &Algo,
    [weight, act, out]: [u64; 3],
    stream: u64,
) -> Result<()> {
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: the problem's handles outlive the call; the pointers are the
    // caller's device buffers of the problem's shapes.
    let st = unsafe {
        cublasLtMatmul(
            ctx.handle,
            p.desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            p.la,
            act as *const c_void,
            p.lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            p.ld,
            out as *mut c_void,
            p.ld,
            algo.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        )
    };
    chk(st, "Matmul")
}

/// The algorithm for `p` (`m` rows of `n x k`): the cached pick of its
/// bucket, tuned now on `ptrs` when there is none.
pub(super) fn select(
    ctx: &Ctx,
    p: &Problem,
    [m, n, k]: [u32; 3],
    ptrs: [u64; 3],
    stream: u64,
) -> Result<Algo> {
    let key = key(n, k, m);
    let cached = CACHE
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.0 == key)
        .map(|e| e.1);
    if let Some(a) = cached {
        anyhow::ensure!(
            takes(ctx, p, &a),
            "cuBLASLt MXFP8: the {n}x{k} kernel of the {}-row bucket (0: all) ({}) refuses \
             {m} rows",
            key.2,
            description(&a)
        );
        return Ok(a);
    }
    let a = tune(ctx, p, [m, n, k], ptrs, stream)?;
    let mut cache = CACHE.lock().unwrap();
    if !cache.iter().any(|e| e.0 == key) {
        cache.push((key, a));
    }
    Ok(a)
}

/// Milliseconds per launch of `algo` on `p` (median-free: one timed run of
/// TIMED launches after WARM).
fn time(ctx: &Ctx, p: &Problem, algo: &Algo, ptrs: [u64; 3], stream: u64) -> Result<f32> {
    let (mut e0, mut e1) = (0u64, 0u64);
    // SAFETY: plain event create / record / sync / destroy on the caller's
    // stream.
    unsafe {
        chk(cuEventCreate(&mut e0, 0), "EventCreate")?;
        chk(cuEventCreate(&mut e1, 0), "EventCreate")?;
    }
    let run = || -> Result<f32> {
        for _ in 0..WARM {
            matmul(ctx, p, algo, ptrs, stream)?;
        }
        // SAFETY: as above.
        unsafe { chk(cuEventRecord(e0, stream), "EventRecord")? };
        for _ in 0..TIMED {
            matmul(ctx, p, algo, ptrs, stream)?;
        }
        let mut ms = 0f32;
        // SAFETY: as above; `ms` is a 4-byte out-parameter.
        unsafe {
            chk(cuEventRecord(e1, stream), "EventRecord")?;
            chk(cuEventSynchronize(e1), "EventSynchronize")?;
            chk(cuEventElapsedTime(&mut ms, e0, e1), "EventElapsedTime")?;
        }
        Ok(ms / TIMED as f32)
    };
    let ms = run();
    // SAFETY: both events were created above.
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    ms
}

/// The fastest in-order candidate for `p`, timed on `ptrs`.
fn tune(ctx: &Ctx, p: &Problem, [m, n, k]: [u32; 3], ptrs: [u64; 3], stream: u64) -> Result<Algo> {
    let all = candidates(ctx, p, true)?;
    let offered = all.len();
    let fit: Vec<Algo> = all
        .into_iter()
        .filter(|a| in_order(a) && takes(ctx, p, a))
        .collect();
    anyhow::ensure!(
        !fit.is_empty(),
        "cuBLASLt MXFP8: no in-order kernel among {offered} offered for {m}x{n}x{k}"
    );
    let mut times = Vec::with_capacity(fit.len());
    for a in &fit {
        times.push(if fit.len() == 1 {
            0.0
        } else {
            time(ctx, p, a, ptrs, stream)?
        });
    }
    let best = (0..fit.len())
        .min_by(|&i, &j| times[i].total_cmp(&times[j]))
        .expect("non-empty");
    tracing::info!(
        "cuBLASLt MXFP8 {n}x{k}, {}-row bucket (tuned at {m} rows): {} at {:.3} ms \
         ({} in-order of {offered} offered; the first ran {:.3} ms)",
        bucket(m),
        description(&fit[best]),
        times[best],
        fit.len(),
        times[0]
    );
    Ok(fit[best])
}

/// The cached algorithm for `m` rows of `n x k`, described (`None` before
/// its bucket's first GEMM).
pub fn mx_algo_description(n: u32, k: u32, m: u32) -> Option<String> {
    let cache = CACHE.lock().unwrap();
    let a = cache.iter().find(|e| e.0 == key(n, k, m))?.1;
    Some(description(&a))
}

/// Every heuristic candidate for `m` rows of `n x k` on these operands
/// (`in_order_only`: as [`candidates`]): `(description, in_order, ms per
/// launch)` by CUDA events (benches; the output is overwritten). Candidates
/// that refuse the problem are skipped.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_mx_candidate_times(
    act_fp8: u64,
    act_sf: u64,
    weight_fp8: u64,
    weight_sf: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    in_order_only: bool,
    stream: u64,
) -> Result<Vec<(String, bool, f32)>> {
    let ctx = ctx()?;
    let p = Problem::with(m, n, k, Operands::Mxfp8 { weight_sf, act_sf })?;
    let mut times = Vec::new();
    for a in candidates(ctx, &p, in_order_only)? {
        if takes(ctx, &p, &a) {
            let ms = time(ctx, &p, &a, [weight_fp8, act_fp8, out], stream)?;
            times.push((description(&a), in_order(&a), ms));
        }
    }
    Ok(times)
}

/// Tune every row bucket up to `m_max` rows of an `n x k` MXFP8 GEMM now
/// (load time), on scratch operands, so serving never times a candidate.
/// Synchronizes `stream`.
pub fn fp8_gemm_mx_prepare(n: u32, k: u32, m_max: u32, stream: u64) -> Result<()> {
    let ctx = ctx()?;
    let top = bucket(m_max);
    let (rows, kpad) = (
        top.div_ceil(128) as usize * 128,
        (k as usize / 32).div_ceil(4) * 4,
    );
    let n_pad = (n as usize).div_ceil(128) * 128;
    let sizes = [
        (top as usize * k as usize, 0u8),     // act
        (rows * kpad, 127u8),                 // act scales (2^0)
        (n as usize * k as usize, 0u8),       // weight
        (n_pad * kpad, 127u8),                // weight scales
        (top as usize * n as usize * 2, 0u8), // out
    ];
    let mut bufs = [0u64; 5];
    let result = (|| -> Result<()> {
        for (b, &(bytes, fill)) in bufs.iter_mut().zip(&sizes) {
            // SAFETY: allocation and fill of a fresh device buffer.
            unsafe {
                chk(cuMemAlloc_v2(b, bytes), "prepare alloc")?;
                chk(cuMemsetD8Async(*b, fill, bytes, stream), "prepare fill")?;
            }
        }
        let [act, act_sf, weight, weight_sf, out] = bufs;
        let mut m = MIN_BUCKET;
        while m <= top {
            let p = Problem::with(m, n, k, Operands::Mxfp8 { weight_sf, act_sf })?;
            select(ctx, &p, [m, n, k], [weight, act, out], stream)?;
            m *= 2;
        }
        // SAFETY: plain stream sync before the buffers are freed.
        unsafe { chk(cuStreamSynchronize(stream), "prepare sync") }
    })();
    for b in bufs.into_iter().filter(|&b| b != 0) {
        // SAFETY: allocated above, no longer in use (synchronized or failed).
        unsafe {
            cuMemFree_v2(b);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::bucket;

    #[test]
    fn rows_map_to_power_of_two_buckets_within_64_to_16384() {
        assert_eq!(bucket(0), 64);
        assert_eq!(bucket(1), 64);
        assert_eq!(bucket(64), 64);
        assert_eq!(bucket(65), 128);
        assert_eq!(bucket(2048), 2048);
        assert_eq!(bucket(2049), 4096);
        assert_eq!(bucket(16384), 16384);
        assert_eq!(bucket(16046), 16384);
        assert_eq!(bucket(40000), 16384);
        assert_eq!(bucket(u32::MAX), 16384);
    }
}
