// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_W4_ROWS=1`: the qwen4_exp MTP drafter's NVFP4 draft head
//! (`layers/qwen4exp_draft_head.rs`) over 4..8 rows with the persistent
//! `qwen4exp_w4_rows{M}` (`qwen4exp_w4_rows.cu`) instead of the scalar
//! `w4a16_gemv_batch{M}` tiers. Row `t` is `w4a16_gemv`'s bytes on that row
//! (as the tiers' are), so drafts, confidences and acceptance do not move;
//! the switch is rank-local.
//!
//! GB10, the draft head half a rank under `ATLAS_QWEN4EXP_MTP_DRAFT_TP`
//! (50,000 x 2560, 72 MB), us per call (`scripts/dev/qwen4exp_w4_rows_bench.cu`):
//!
//! | M | 4 | 5 | 6 | 7 | 8 |
//! |---|---|---|---|---|---|
//! | `w4a16_gemv_batch{M}` | 349 | 390 | 535 | 563 | 653 |
//! | `qwen4exp_w4_rows{M}` | 343 | 344 | 342 | 340 | 378 |
//!
//! At 1..3 rows the narrow kernels stay (as fast there).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// Rows the persistent tier serves.
const MIN_ROWS: u32 = 4;
const MAX_ROWS: u32 = 8;
/// Threads a CTA (4 output groups of 64 lanes).
const THREADS: u32 = 256;

/// `ATLAS_QWEN4EXP_W4_ROWS=1`, read once.
pub fn w4_rows_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_W4_ROWS").as_deref() == Ok("1"))
}

/// The `qwen4exp_w4_rows{M}` handles (index M), all 0 when off.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpW4Rows {
    rows: [KernelHandle; MAX_ROWS as usize + 1],
    sms: u32,
}

impl Qwen4ExpW4Rows {
    pub const OFF: Self = Self {
        rows: [KernelHandle(0); MAX_ROWS as usize + 1],
        sms: 0,
    };

    /// Resolved when the switch is on and the target ships the module.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        if !w4_rows_requested() {
            return Self::OFF;
        }
        // Literal names (the kernel-name check reads them).
        let mut rows = Self::OFF.rows;
        rows[4] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows4");
        rows[5] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows5");
        rows[6] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows6");
        rows[7] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows7");
        rows[8] = try_kernel(gpu, "qwen4exp_w4_rows", "qwen4exp_w4_rows8");
        Self {
            rows,
            sms: gpu.sm_count().unwrap_or(0),
        }
    }

    /// Whether `m` rows of a `[n, k]` weight take the persistent tier: the
    /// kernel resolved, `k / 16` even and at most 256 (two lane steps).
    pub fn serves(&self, m: u32, k: u32) -> bool {
        (MIN_ROWS..=MAX_ROWS).contains(&m)
            && self.rows[m as usize].0 != 0
            && self.sms > 0
            && k.is_multiple_of(32)
            && k <= 4096
    }

    /// `m` rows of `input` (`[m, k]`) through the NVFP4 `w` into `output`
    /// (`[m, n]`), row `t` `w4a16_gemv`'s bytes. Gate on [`Self::serves`].
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(self.serves(m, k), "qwen4exp_w4_rows: {m} rows, k={k}");
        // The M rows staged for the whole K: M x lane steps x 4 KiB.
        let smem = m * div_ceil(k / 16, 128) * 4096;
        KernelLaunch::new(gpu, self.rows[m as usize])
            .grid([self.sms.min(div_ceil(n, 16)), 1, 1])
            .block([THREADS, 1, 1])
            .shared_mem(smem)
            .arg_ptr(input)
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    }
}
