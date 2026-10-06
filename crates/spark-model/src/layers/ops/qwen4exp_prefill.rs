// SPDX-License-Identifier: AGPL-3.0-only

//! `qwen4_exp` (Qwen3.8-Flash-Next) PREFILL switches and the launches behind
//! them. Every switch is default OFF and read once per process.
//!
//! | switch                                   | what                                   | exact? |
//! |------------------------------------------|----------------------------------------|--------|
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=1`      | TP2 QSA attention on tensor cores      | = TP1 tc2, != `_g` |
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_LEAN=1`      | TP1 tc2 -> its lean twin (2 CTAs/SM)   | = tc2  |
//!
//! "= X" means byte-identical outputs to kernel X on the same inputs, checked
//! by the `scripts/dev/qwen4exp_*_bench.cu` harnesses on a GB10.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

fn flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true"))
}

/// `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=1`.
pub fn qsa_tc2r_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_QSA_TC2R"))
}

/// `ATLAS_QWEN4EXP_PREFILL_QSA_LEAN=1`.
pub fn qsa_lean_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_QSA_LEAN"))
}

/// One slab of QSA selected-set attention: the arguments every arm takes.
pub struct QsaAttnSlab {
    pub q: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub block_table: DevicePtr,
    pub lists: DevicePtr,
    pub attn_out: DevicePtr,
    pub rows: u32,
    pub first_pos: u32,
    pub topk: u32,
    pub ratio: u32,
    pub block_size: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub inv_sqrt_d: f32,
}

/// Run the slab on a lean tensor-core arm when one is requested and serves
/// this geometry; `Ok(false)` leaves it to the caller's usual arms.
///
/// * One kv head (TP2: 24 q / 2 kv split by heads) under `_QSA_TC2R`:
///   `qsa_prefill_attn_tc2r` puts TWO ROWS in tc2's M tile, one per half.
/// * Two kv heads (TP1) under `_QSA_LEAN`, when tc2 is the arm in force:
///   `qsa_prefill_attn_tc2l`, tc2 with Q in registers and one K buffer.
///
/// Both compile from `qsa_attn_tc2.cu`, so every output byte equals tc2's
/// for the same heads (`scripts/dev/qwen4exp_qsa_tc2r_bench.cu`). Measured on
/// GB10 at 2048 rows / position 14000 / topk 512: 12 heads x 1 kv head, `_g`
/// 21.2 ms -> tc2r 5.85 ms (3.6x); 24 x 2, tc2 20.9 -> tc2l 12.0 ms (1.75x).
pub fn try_qsa_prefill_attn_lean(
    gpu: &dyn GpuBackend,
    s: &QsaAttnSlab,
    tc2_in_force: bool,
    stream: u64,
) -> Result<bool> {
    let shape_ok = s.hd == 256 && s.nkv != 0 && s.nq % s.nkv == 0 && s.nq / s.nkv <= 16;
    let (module, entry, row_pairs) = if s.nkv == 1 && qsa_tc2r_requested() {
        ("qsa_attn_tc2r", "qsa_prefill_attn_tc2r", true)
    } else if s.nkv == 2 && tc2_in_force && qsa_lean_requested() {
        ("qsa_attn_tc2l", "qsa_prefill_attn_tc2l", false)
    } else {
        return Ok(false);
    };
    let k = crate::layers::try_kernel(gpu, module, entry);
    if !shape_ok || k.0 == 0 || s.rows == 0 {
        return Ok(false);
    }
    let ctas = if row_pairs {
        s.rows.div_ceil(2)
    } else {
        s.rows
    };
    let mut l = KernelLaunch::new(gpu, k)
        .grid([1, ctas, 1])
        .block([128, 1, 1])
        .arg_ptr(s.q)
        .arg_ptr(s.k_cache)
        .arg_ptr(s.v_cache)
        .arg_ptr(s.attn_out)
        .arg_ptr(s.block_table)
        .arg_ptr(s.lists)
        .arg_u32(s.first_pos)
        .arg_u32(s.topk)
        .arg_u32(s.ratio)
        .arg_u32(s.block_size)
        .arg_u32(s.nq)
        .arg_u32(s.nkv)
        .arg_u32(s.hd)
        .arg_f32(s.inv_sqrt_d);
    if row_pairs {
        l = l.arg_u32(s.rows);
    }
    l.launch(stream)?;
    Ok(true)
}
