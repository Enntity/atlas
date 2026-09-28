// SPDX-License-Identifier: AGPL-3.0-only
//! Unit-scale E4M3 x E4M3 -> BF16 with FP32 accumulation, eager callers only.
use super::*;
use anyhow::ensure;

#[derive(Default)]
struct Descriptors {
    desc: cublasLtMatmulDesc_t,
    layouts: [cublasLtMatrixLayout_t; 3],
    pref: cublasLtMatmulPreference_t,
}
impl Drop for Descriptors {
    fn drop(&mut self) {
        unsafe {
            if !self.pref.is_null() {
                cublasLtMatmulPreferenceDestroy(self.pref);
            }
            for layout in self.layouts {
                if !layout.is_null() {
                    cublasLtMatrixLayoutDestroy(layout);
                }
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}
// CUDA 13 cublasLt.h: algorithm64, size_t workspace, status32, float waves,
// four reserved int32 values. Exact result layout, rather than packed bytes.
#[repr(C)]
#[derive(Default)]
struct Heuristic {
    algo: [u64; 8],
    workspace: usize,
    state: i32,
    waves: f32,
    reserved: [i32; 4],
}

/// Row-major `out[M,N] = act[M,K] * weight[N,K]^T`. Both operands are already
/// E4M3 with tensor scales exactly one. Calls serialize on the existing model
/// forward stream and share the existing process CUDA64MiB workspace.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_tensorwise(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && k > 0 && k.is_multiple_of(16),
        "FP8 tensorwise geometry"
    );
    let spans = [
        (act, u64::from(m) * u64::from(k)),
        (weight, u64::from(n) * u64::from(k)),
        (
            out,
            u64::from(m)
                .checked_mul(u64::from(n))
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("FP8 tensorwise output overflow"))?,
        ),
    ];
    for (i, &(ptr, bytes)) in spans.iter().enumerate() {
        ensure!(
            ptr > 0 && ptr % 16 == 0 && ptr.checked_add(bytes).is_some(),
            "FP8 tensorwise pointer"
        );
        for &(other, len) in &spans[..i] {
            ensure!(
                ptr >= other + len || other >= ptr + bytes,
                "FP8 tensorwise alias"
            );
        }
    }
    let context = ctx()?;
    let mut h = Descriptors::default();
    unsafe {
        chk(
            cublasLtMatmulDescCreate(&mut h.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "FP8 tensorwise desc",
        )?;
        for (attr, value) in [(DESC_TRANSA, CUBLAS_OP_T), (DESC_TRANSB, CUBLAS_OP_N)] {
            chk(
                cublasLtMatmulDescSetAttribute(
                    h.desc,
                    attr,
                    (&value as *const i32).cast(),
                    size_of::<i32>(),
                ),
                "FP8 tensorwise transpose",
            )?;
        }
        // Header ABI: FAST_ACCUM25 has int8_t payload. Zero preserves FP32
        // accumulation instead of opting into the fast accumulation mode.
        let fast = 0i8;
        chk(
            cublasLtMatmulDescSetAttribute(
                h.desc,
                25,
                (&fast as *const i8).cast(),
                size_of::<i8>(),
            ),
            "FP8 tensorwise fast accumulation",
        )?;
        // NVIDIA cuBLAS documentation explicitly defines NULL A/B scale
        // pointers as scale1. Pass the ADDRESS of the null pointer value.
        let unit: *const c_void = std::ptr::null();
        for attr in [DESC_A_SCALE_POINTER, DESC_B_SCALE_POINTER] {
            chk(
                cublasLtMatmulDescSetAttribute(
                    h.desc,
                    attr,
                    (&unit as *const *const c_void).cast(),
                    size_of::<*const c_void>(),
                ),
                "FP8 tensorwise unit scale",
            )?;
        }
        for (i, (dtype, rows, cols)) in [
            (CUDA_R_8F_E4M3, k, n),
            (CUDA_R_8F_E4M3, k, m),
            (CUDA_R_16BF, n, m),
        ]
        .into_iter()
        .enumerate()
        {
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut h.layouts[i],
                    dtype,
                    u64::from(rows),
                    u64::from(cols),
                    i64::from(rows),
                ),
                "FP8 tensorwise layout",
            )?;
        }
        chk(
            cublasLtMatmulPreferenceCreate(&mut h.pref),
            "FP8 tensorwise preference",
        )?;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                h.pref,
                PREF_MAX_WORKSPACE_BYTES,
                (&context.ws_size as *const usize).cast(),
                size_of::<usize>(),
            ),
            "FP8 tensorwise workspace",
        )?;
        for (attr, bits) in [
            (5, weight | u64::from(k)),
            (6, act | u64::from(k)),
            (7, out | (u64::from(n) * 2)),
            (8, out | (u64::from(n) * 2)),
        ] {
            let alignment = 1u32 << bits.trailing_zeros().min(8);
            chk(
                cublasLtMatmulPreferenceSetAttribute(
                    h.pref,
                    attr,
                    (&alignment as *const u32).cast(),
                    size_of::<u32>(),
                ),
                "FP8 tensorwise alignment",
            )?;
        }
        let mut result = Heuristic::default();
        let mut returned = 0;
        let [la, lb, lc] = h.layouts;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                context.handle,
                h.desc,
                la,
                lb,
                lc,
                lc,
                h.pref,
                1,
                (&mut result as *mut Heuristic).cast(),
                &mut returned,
            ),
            "FP8 tensorwise heuristic",
        )?;
        ensure!(
            returned == 1 && result.state == 0 && result.workspace <= context.ws_size,
            "cuBLASLt no FP8 tensorwise algorithm M={m} N={n} K={k}"
        );
        let alpha = 1f32;
        let beta = 0f32;
        chk(
            cublasLtMatmul(
                context.handle,
                h.desc,
                (&alpha as *const f32).cast(),
                weight as *const c_void,
                la,
                act as *const c_void,
                lb,
                (&beta as *const f32).cast(),
                out as *const c_void,
                lc,
                out as *mut c_void,
                lc,
                result.algo.as_ptr().cast(),
                context.workspace as *mut c_void,
                context.ws_size,
                stream as *mut c_void,
            ),
            "FP8 tensorwise matmul",
        )?;
    }
    Ok(())
}
