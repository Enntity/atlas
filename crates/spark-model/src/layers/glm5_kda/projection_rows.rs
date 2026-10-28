// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Balanced `(first_row, rows)` chunks of at most the batch-M GEMV cap, so an
/// owner-batched verify of `m > 8` rows reuses the exact short-batch kernels.
pub(super) fn verify_row_chunks(m: u32) -> impl Iterator<Item = (u32, u32)> {
    // `ATLAS_GLM_LONG_BATCH_PROJ_ROWS=3` pins owner-exact 3-row chunks.
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| {
        std::env::var("ATLAS_GLM_LONG_BATCH_PROJ_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| (2..=ops::DENSE_GEMV_BATCHM_MAX_M).contains(v))
            .unwrap_or(ops::DENSE_GEMV_BATCHM_MAX_M)
    });
    let chunks = m.div_ceil(cap);
    (0..chunks).map(move |c| {
        let start = c * m / chunks;
        (start, (c + 1) * m / chunks - start)
    })
}
