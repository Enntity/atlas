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
//! * The mHC collapses (`hc_pre`, `hc_head`): the GEMM formulation (BF16
//!   `normed`, cuBLASLt down/inject per slab height) and the <= 8-row decode
//!   split path are replaced by the DECODE collapse's own arithmetic at
//!   every width -- FP32 `normed`, `qwen4exp_hc_mma.cu`'s tensor-core down
//!   and finish (row-invariant by construction: a row is its own mma
//!   column) over row groups ([`hc_collapse`]) -- so a prefill row's
//!   collapse is byte for byte its `ATLAS_QWEN4EXP_HC_MMA` decode collapse.
//!   The fused prefill seams (`ATLAS_QWEN4EXP_PREFILL_HC`) are declined.
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
    /// Rows one mHC slab takes in the shared scratch (FP32 `normed` and
    /// `low`), a multiple of the row-kernel groups.
    hc_rows: u32,
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
/// is off). `hc_scratch_bytes`, `hc_dim` and `rank` size the mHC slab.
pub fn enter(
    gpu: &dyn GpuBackend,
    hc_scratch_bytes: usize,
    hc_dim: usize,
    rank: usize,
) -> Option<Scope> {
    if !on() {
        return None;
    }
    let per_row = (hc_dim + rank).max(1) * 4;
    let group = (HCM_ROWS_DN_T * HCM_ROWS_FN_T / 32) as usize; // 192: both groups divide it
    let rows = ((hc_scratch_bytes / per_row).min(2048) / group * group) as u32;
    // SAFETY: the pointer is only dereferenced while the returned `Scope`
    // lives, which the caller holds inside the borrow of `gpu`.
    let gpu: &'static dyn GpuBackend = unsafe { std::mem::transmute(gpu) };
    PASS.with(|p| {
        p.set(Some(Pass {
            gpu: gpu as *const _,
            hc_rows: rows,
        }))
    });
    Some(Scope(()))
}

/// A row-invariant prefill pass is running on this thread.
pub fn active() -> bool {
    PASS.with(|p| p.get().is_some())
}

fn pass() -> Option<(&'static dyn GpuBackend, u32)> {
    // SAFETY: see `enter`.
    PASS.with(|p| p.get().map(|p| (unsafe { &*p.gpu }, p.hc_rows)))
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
    let Some((gpu, _)) = pass() else {
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

/// Rows a group of `hc_mma_down_rows` / `hc_mma_finish_rows` and their
/// weight tiles a CTA (`HCM_ROWS_*`, `HCM_DN_CL` in `qwen4exp_hc_mma.cu`).
const HCM_ROWS_DN_T: u32 = 96;
const HCM_ROWS_FN_T: u32 = 64;
const HCM_ROWS_WM: u32 = 7;
const HCM_DN_CL: u32 = 8;
/// `hc_mma_finish`'s output dims a CTA (`HCM_FN_DW`).
const HCM_FN_DW: u32 = 32;

/// The mHC collapse of `num_tokens` rows, byte for byte the decode collapse
/// (`ATLAS_QWEN4EXP_HC_FAST` + `_HC_MMA`) of each row: per slab of the
/// scratch, `hc_pre_stage_vec` (FP32 `normed`, one 1024-thread RMS a row),
/// then `hc_mma_down_rows` + `hc_mma_finish_rows` (`qwen4exp_hc_mma.cu`, the
/// decode kernels' per-row operation sequence over row groups). `inj_out`
/// NULL is the model-level head. `Ok(false)` outside a row-invariant pass;
/// an error where it cannot run, since any fallback would break the
/// invariance.
#[allow(clippy::too_many_arguments)]
pub fn hc_collapse(
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<bool> {
    let Some((gpu, slab)) = pass() else {
        return Ok(false);
    };
    let hc_dim = hc_mult * hidden_size;
    let rank = w.rank as u32;
    let k_stage = crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_stage_vec");
    let k_down = crate::layers::try_kernel(gpu, "qwen4exp_hc_mma", "hc_mma_down_rows");
    let k_fin = crate::layers::try_kernel(gpu, "qwen4exp_hc_mma", "hc_mma_finish_rows");
    anyhow::ensure!(
        slab >= HCM_ROWS_DN_T
            && !scratch.is_null()
            && hc_mult == 4
            && hc_dim.is_multiple_of(32 * HCM_DN_CL * 4)
            && rank.is_multiple_of(16 * 5)
            && hidden_size.is_multiple_of(HCM_FN_DW)
            && k_stage.0 != 0
            && k_down.0 != 0
            && k_fin.0 != 0,
        "ATLAS_QWEN4EXP_PREFILL_ROWINV: the mHC row kernels do not serve this \
         shape (hidden {hidden_size}, hc {hc_mult}, rank {rank}, slab {slab})"
    );
    let normed = scratch;
    let low = scratch.offset(slab as usize * hc_dim as usize * 4);
    let rows = rank + if inj_out.is_null() { 0 } else { hc_mult };
    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = slab.min(num_tokens - t0);
        let t = t0 as usize;
        // Every stage block computes its row's whole RMS, so the block split
        // is launch geometry only (`hc_stage_split`): one a row once the
        // rows fill the part.
        let split = if ts >= 48 { 1 } else { 8 };
        KernelLaunch::new(gpu, k_stage)
            .grid([ts, split, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams.offset(t * hc_dim as usize * 4))
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;
        let inj = if inj_out.is_null() {
            DevicePtr::NULL
        } else {
            inj_out.offset(t * hc_mult as usize * 4)
        };
        KernelLaunch::new(gpu, k_down)
            .grid([
                rows.div_ceil(16).div_ceil(HCM_ROWS_WM),
                HCM_DN_CL,
                ts.div_ceil(HCM_ROWS_DN_T),
            ])
            .block([32 * HCM_ROWS_WM, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(w.down_w)
            .arg_ptr(if inj.is_null() {
                DevicePtr::NULL
            } else {
                w.inject_w
            })
            .arg_ptr(low)
            .arg_ptr(inj)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(rank)
            .arg_u32(ts)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_fin)
            .grid([hidden_size / HCM_FN_DW, 1, ts.div_ceil(HCM_ROWS_FN_T)])
            .block([128, 1, 1])
            // The stream-mean tile only: the fragments come from global.
            .shared_mem(4 * ts.min(HCM_ROWS_FN_T).div_ceil(8) * 8 * HCM_FN_DW * 4)
            .arg_ptr(normed)
            .arg_ptr(low)
            .arg_ptr(w.up_w)
            .arg_ptr(y_out.offset(t * hidden_size as usize * 2))
            .arg_u32(hidden_size)
            .arg_u32(rank)
            .arg_u32(ts)
            .launch(stream)?;
        t0 += ts;
        // ATLAS_QWEN4EXP_PREFILL_SP_PIPE: rows [0, t0) of `y_out` are set.
        crate::layers::qwen4exp_sp_pipe::slab_done(t0 as usize, stream)?;
    }
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
