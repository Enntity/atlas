// SPDX-License-Identifier: AGPL-3.0-only

//! Row-invariant qwen4_exp prefill (`ATLAS_QWEN4EXP_PREFILL_ROWINV=1`,
//! default off): a token's per-layer outputs are bitwise independent of how
//! many rows share its prefill pass and of where the pass boundaries fall
//! (on the 64-token GDN grid, which the scheduler and the prefix cache keep
//! under the switch). That makes a prompt prefilled in one pass, in chunks,
//! or after a prefix-cache restore give the same logits, and lets several
//! prompts share one pass exactly.
//!
//! What the switch changes, each to an arithmetic that does not look at the
//! row count:
//!
//! * BF16 projections that went to cuBLASLt's first heuristic pick (whose
//!   split-K changes with M): the GDN in_proj, the QSA indexer q/k
//!   projections, every other `ops::bf16_gemm` of the pass. They run on an
//!   in-order k-chain -- cuBLASLt algo 21 without split-K when offered, else
//!   the tile kernel `dense_gemm_bf16_pipelined`, the same bytes
//!   (`cublaslt::kchain_pin`) -- see [`try_bf16_gemm`].
//! * The mHC collapses (`hc_pre`, `hc_head`): the down and injection
//!   projections went to cuBLASLt per slab height, and <= 8 rows took the
//!   decode split path. Every width now takes the prefill fast arm
//!   (`qwen4exp_prefill_hc`: the BF16 stage or post+stage seam, the fused
//!   up + mix -- each a function of its own row) with the down and injection
//!   on `hc_rowinv_down` ([`hc_down`]), the decode collapse's tensor-core
//!   down on BF16 rows: a row is its own mma column, K is split in fixed
//!   slices summed in a fixed order.
//! * Attention runs the paged path from the first chunk on, its FP8
//!   projections on the m128 arm and the BR64 dense kernel at every width
//!   (`qwen3_attention::prefill`).
//! * One-row passes stay on the prefill path (no decode-layer fork), and a
//!   Marconi replay uses the chunked GDN scan, not the token-sequential one.
//!
//! The pass state lives in a thread-local [`Scope`] that `forward_layers`
//! holds around the prefill layers, so decode and verify -- which share the
//! layer code -- are untouched.

use std::cell::Cell;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank_gemm::gemm_raw;
use crate::layers::qwen3_attention::HcLowRank;

/// `ATLAS_QWEN4EXP_PREFILL_ROWINV=1`. Read once.
pub fn on() -> bool {
    spark_runtime::qwen4exp_prefill_rowinv()
}

/// The GDN chunk every pass start must sit on (see the module docs).
pub const PASS_GRANULE: usize = 64;

#[derive(Clone, Copy)]
struct Pass {
    gpu: *const (dyn GpuBackend + 'static),
}

thread_local! {
    static PASS: Cell<Option<Pass>> = const { Cell::new(None) };
}

/// Ends the row-invariant prefill pass when dropped.
pub struct Scope(());

impl Drop for Scope {
    fn drop(&mut self) {
        PASS.with(|p| p.set(None));
    }
}

/// Start a row-invariant prefill pass on this thread (`None` when the switch
/// is off).
pub fn enter(gpu: &dyn GpuBackend) -> Option<Scope> {
    if !on() {
        return None;
    }
    // SAFETY: the pointer is only dereferenced while the returned `Scope`
    // lives, which the caller holds inside the borrow of `gpu`.
    let gpu: &'static dyn GpuBackend = unsafe { std::mem::transmute(gpu) };
    PASS.with(|p| {
        p.set(Some(Pass {
            gpu: gpu as *const _,
        }))
    });
    Some(Scope(()))
}

/// A row-invariant prefill pass is running on this thread.
pub fn active() -> bool {
    PASS.with(|p| p.get().is_some())
}

fn pass_gpu() -> Option<&'static dyn GpuBackend> {
    // SAFETY: see `enter`.
    PASS.with(|p| p.get().map(|p| unsafe { &*p.gpu }))
}

/// `out[m, n] = act[m, k] @ weight[n, k]^T` (BF16) on an in-order k-chain,
/// when a row-invariant pass is running; `Ok(false)` launched nothing.
pub fn try_bf16_gemm(
    act: DevicePtr,
    weight: u64,
    out: DevicePtr,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<bool> {
    let Some(gpu) = pass_gpu() else {
        return Ok(false);
    };
    if spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain_any(
        act.0,
        weight,
        out.0,
        [m, n, k],
        stream,
    )? {
        return Ok(true);
    }
    let kernel = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    gemm_raw(gpu, kernel, act, DevicePtr(weight), out, m, n, k, stream)?;
    Ok(true)
}

/// `hc_rowinv_down` cluster CTAs, rows of `m16` tiles a CTA, rows a group
/// (`HCR_CL`, `HCR_WM`, `HCR_MAX_T` in `qwen4exp_rowinv.cu`).
const HCR_CL: u32 = 8;
const HCR_WM: u32 = 7;
const HCR_MAX_T: u32 = 96;

/// The mHC prefill collapse's down and injection projections, row-invariant
/// (`qwen4exp_rowinv.cu`): `low = bf16(silu(normed x down_w^T / hc))`
/// `[ts, rank]` and, unless `inj_pre` is NULL (the head), `inj_pre =
/// bf16(normed x inject_w^T)` `[ts, hc]`, for the fused up + mix after it.
/// `Ok(false)` outside a row-invariant pass; an error where it cannot run,
/// since any other kernel would break the invariance.
#[allow(clippy::too_many_arguments)]
pub fn hc_down(
    gpu: &dyn GpuBackend,
    w: &HcLowRank,
    normed: DevicePtr,
    low: DevicePtr,
    inj_pre: DevicePtr,
    ts: u32,
    hidden: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<bool> {
    if !active() {
        return Ok(false);
    }
    let rank = w.rank as u32;
    let k = hc_mult * hidden;
    let kernel = crate::layers::try_kernel(gpu, "qwen4exp_rowinv", "hc_rowinv_down");
    anyhow::ensure!(
        kernel.0 != 0 && k.is_multiple_of(32 * HCR_CL * 4) && ts > 0,
        "ATLAS_QWEN4EXP_PREFILL_ROWINV: hc_rowinv_down missing or K = {k} not tiled"
    );
    let rows = rank + if inj_pre.is_null() { 0 } else { hc_mult };
    KernelLaunch::new(gpu, kernel)
        .grid([
            rows.div_ceil(16).div_ceil(HCR_WM),
            HCR_CL,
            ts.div_ceil(HCR_MAX_T),
        ])
        .block([32 * HCR_WM, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down_w)
        .arg_ptr(if inj_pre.is_null() {
            DevicePtr::NULL
        } else {
            w.inject_w
        })
        .arg_ptr(low)
        .arg_ptr(inj_pre)
        .arg_u32(hidden)
        .arg_u32(hc_mult)
        .arg_u32(rank)
        .arg_u32(ts)
        .launch(stream)?;
    Ok(true)
}

/// Whether a pass over `[start, ..)` keeps the GDN grid (logs once if not:
/// the rows after it are then not chunk-invariant).
pub fn check_pass_start(start: usize) {
    if on() && !start.is_multiple_of(PASS_GRANULE) {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_ROWINV: a prefill pass starts at {start}, off the \
                 {PASS_GRANULE}-token GDN grid; its rows are not chunk-invariant (logged once)"
            );
        }
    }
}
