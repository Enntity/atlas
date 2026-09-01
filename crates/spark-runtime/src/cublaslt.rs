// SPDX-License-Identifier: AGPL-3.0-only
//! Minimal cuBLASLt FFI for the high-efficiency GEMM path (`ATLAS_CUBLAS_GEMM`).
//!
//! The hand-written mma.sync projection/MoE GEMMs reach only ~30% of the cuBLAS
//! ceiling on GB10 (measured: 32 vs 85 TFLOPS bf16, 152 fp8, on the SSM-qkvz
//! shape 3537×12288×2048). This routes those GEMMs through cuBLASLt instead.
//! BF16 only for now — correctness-clean (no scale-format issues); native fp8
//! block-scaled is the follow-up once the end-to-end win is proven.

use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

// Native FP8 (E4M3) GEMM paths live in the `fp8` sibling (≤500 LoC split);
// re-exported so `spark_runtime::cublaslt::fp8_gemm_*` paths are unchanged.
mod fp8;
pub use fp8::{fp8_gemm_act_weight_t_blkscaled, fp8_gemm_act_weight_t_rowwise};

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_COMPUTE_32F_FAST_TF32: i32 = 77;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const SCALE_MODE_OUTER_VEC_32F: i32 = 3;
const SCALE_MODE_VEC128_32F: i32 = 4;
const SCALE_MODE_BLK128X128_32F: i32 = 5;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;

unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspace: u64,
    ws_size: usize,
    bf16_plans: Mutex<HashMap<(u32, u32, u32), Box<MatmulPlan>>>,
    tf32_plans: Mutex<HashMap<(u32, u32, u32), Box<MatmulPlan>>>,
}

struct MatmulPlan {
    desc: cublasLtMatmulDesc_t,
    layout_weight: cublasLtMatrixLayout_t,
    layout_act: cublasLtMatrixLayout_t,
    layout_out: cublasLtMatrixLayout_t,
    algo: [u8; 128],
}

// Plans are immutable after construction and calls are serialized by the
// scheduler plus the plan-cache mutex. The raw handles are process-context
// objects owned for the process lifetime, like the cuBLASLt handle itself.
unsafe impl Send for MatmulPlan {}
unsafe impl Sync for MatmulPlan {}
// cuBLASLt handle + device workspace are process-global; matmul is invoked
// serially from the single-threaded scheduler forward.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

/// STATIC, DELIBERATELY — CUDA host. This is a workspace allocated in THE
/// process CUDA context (see `atlas_core::cuda_host`, which establishes one
/// per process) and sized by a fixed budget, not by any model's shapes: the
/// bounds below are generous upper limits chosen to fit any realistic serving
/// configuration, so a swap needs no reallocation and re-allocating per model
/// would churn hundreds of megabytes for no change in what is mapped.
///
/// It survives a model swap for the same reason the context does. Nothing in
/// it is derived from a model — no token ids, no weight pointers, no shapes —
/// only scratch the library plans within.
static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let _ = CTX.set(Ctx {
        handle,
        workspace: ws,
        ws_size,
        bf16_plans: Mutex::new(HashMap::new()),
        tf32_plans: Mutex::new(HashMap::new()),
    });
    Ok(CTX.get().unwrap())
}

/// Force cuBLASLt's one-time costs at MODEL LOAD instead of on request 1.
///
/// The lazy `ctx()` means the first GEMM pays `cublasLtCreate`, the 64 MB
/// workspace alloc, and — the expensive part — the library's kernel-image
/// load and heuristic warm-up. Measured on the 35B flagship (2026-08-22,
/// dgx1): the first in-serve request read ~0.9 s slower than warm requests
/// once QKVZ routed through cuBLASLt, and cold TTFT is a headline metric.
/// One 64x64x64 BF16 GEMM here is trivial GPU work and moves that cost to
/// load time, where it overlaps the operator's mental model of "loading".
///
/// Never fails the serve: a pre-warm failure is logged and swallowed — the
/// lazy path remains and request 1 simply pays the old cost.
pub fn prewarm(stream: u64) {
    let r = (|| -> Result<()> {
        let bytes = 64usize * 64 * 2;
        let mut a = 0u64;
        let mut b = 0u64;
        let mut d = 0u64;
        unsafe {
            chk(cuMemAlloc_v2(&mut a, bytes), "prewarm alloc a")?;
            chk(cuMemAlloc_v2(&mut b, bytes), "prewarm alloc b")?;
            chk(cuMemAlloc_v2(&mut d, bytes), "prewarm alloc d")?;
        }
        let res = bf16_gemm_act_weight_t(a, b, d, 64, 64, 64, stream);
        unsafe {
            chk(cuStreamSynchronize(stream), "prewarm sync")?;
            let _ = cuMemFree_v2(a);
            let _ = cuMemFree_v2(b);
            let _ = cuMemFree_v2(d);
        }
        res
    })();
    match r {
        Ok(()) => tracing::info!("cuBLASLt pre-warmed (handle + workspace + kernel images)"),
        Err(e) => tracing::warn!("cuBLASLt pre-warm failed (request 1 pays lazy init): {e}"),
    }
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16 — the standard
/// projection GEMM (activation × transposed weight). Maps to cuBLASLt's
/// column-major convention as `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    let key = (m, n, k);
    let mut plans = ctx.bf16_plans.lock();
    let plan = match plans.entry(key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(Box::new(build_bf16_plan(ctx, m, n, k)?))
        }
    };
    unsafe {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            plan.desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            plan.layout_weight,
            act as *const c_void,
            plan.layout_act,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            plan.layout_out,
            out as *mut c_void,
            plan.layout_out,
            plan.algo.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        chk(status, "Matmul")?;
    }
    Ok(())
}

fn build_bf16_plan(ctx: &Ctx, m: u32, n: u32, k: u32) -> Result<MatmulPlan> {
    build_plan(ctx, m, n, k, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F)
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]T` for FP32 buffers using
/// TF32 tensor cores. GLM mHC is the intended fixed-shape consumer: its BF16
/// checkpoint function matrix is widened exactly once at load, while the FP32
/// residual highway is rounded to TF32 only for this learned 24-row mix GEMM.
pub fn tf32_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    let key = (m, n, k);
    let mut plans = ctx.tf32_plans.lock();
    let plan = match plans.entry(key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => entry.insert(Box::new(build_plan(
            ctx,
            m,
            n,
            k,
            CUDA_R_32F,
            CUDA_R_32F,
            CUBLAS_COMPUTE_32F_FAST_TF32,
        )?)),
    };
    unsafe {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            plan.desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            plan.layout_weight,
            act as *const c_void,
            plan.layout_act,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            plan.layout_out,
            out as *mut c_void,
            plan.layout_out,
            plan.algo.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        chk(status, "TF32 Matmul")?;
    }
    Ok(())
}

fn build_plan(
    ctx: &Ctx,
    m: u32,
    n: u32,
    k: u32,
    input_dtype: i32,
    output_dtype: i32,
    compute_type: i32,
) -> Result<MatmulPlan> {
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, compute_type, CUDA_R_32F),
            "DescCreate",
        )?;
        for (attribute, value, label) in [
            (DESC_TRANSA, CUBLAS_OP_T, "TRANSA"),
            (DESC_TRANSB, CUBLAS_OP_N, "TRANSB"),
        ] {
            chk(
                cublasLtMatmulDescSetAttribute(
                    desc,
                    attribute,
                    &value as *const i32 as *const c_void,
                    size_of::<i32>(),
                ),
                label,
            )?;
        }

        // A = weight row-major [N,K] == column-major [K,N], transposed.
        // B = activation row-major [M,K] == column-major [K,M].
        // D = output row-major [M,N] == column-major [N,M].
        let mut layout_weight: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_act: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut layout_out: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(
                &mut layout_weight,
                input_dtype,
                k as u64,
                n as u64,
                k as i64,
            ),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut layout_act, input_dtype, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut layout_out, output_dtype, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulPreferenceCreate(&mut preference),
            "PrefCreate",
        )?;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                PREF_MAX_WORKSPACE_BYTES,
                &ctx.ws_size as *const usize as *const c_void,
                size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut algo = [0u8; 128];
        let mut returned = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                layout_weight,
                layout_act,
                layout_out,
                layout_out,
                preference,
                1,
                algo.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        cublasLtMatmulPreferenceDestroy(preference);
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        Ok(MatmulPlan {
            desc,
            layout_weight,
            layout_act,
            layout_out,
            algo,
        })
    }
}
