// SPDX-License-Identifier: AGPL-3.0-only

//! NVFP4 n-gram rows (Qwen3.8-Flash-Next-FP4-Local): the packed row layout
//! the row cache holds, and its stride guard.
//!
//! The checkpoint ships every shard as two tensors — `weight` U8 `[R, 80]`
//! (E2M1 code pairs, low nibble first) and `weight_scale` F8_E4M3 `[R, 10]`
//! (one scale per 16 values) — plus one `weight_scale_2` F32. The loader
//! repacks them once (`spark_storage::ngram_pack`) into rows of
//!
//! ```text
//! [ head_dim/2 B codes | head_dim/16 B scales | zero pad to a 16 B multiple ]
//!   80                   10                     6        = 96 B at dim 160
//! ```
//!
//! so a cache miss is ONE NVMe read, and `batched_embed_nvfp4` dequantizes on
//! gather exactly as the FP8 path does: `(e2m1 * e4m3) * global` in f32, then
//! BF16. The 16 B multiple keeps every slot 16 B-aligned for vector loads; it
//! costs 6 B a row (1.9 GB over 320M rows, on disk only).

use anyhow::Result;

/// Values per E4M3 block scale (ModelOpt NVFP4 `group_size`).
pub const GROUP: usize = 16;

/// Packed code bytes per row.
pub const fn code_bytes(head_dim: usize) -> usize {
    head_dim / 2
}

/// Block-scale bytes per row.
pub const fn scale_bytes(head_dim: usize) -> usize {
    head_dim / GROUP
}

/// Bytes per packed row: codes + scales, rounded up to 16.
pub const fn packed_stride(head_dim: usize) -> usize {
    (code_bytes(head_dim) + scale_bytes(head_dim)).next_multiple_of(16)
}

/// Refuse a cache whose geometry is not the packed NVFP4 row of `head_dim`,
/// or that also carries an FP8 scale array (which only the FP8 gather reads,
/// and which would mean two components disagree about what the bytes are).
// Reached from `PleLayer::new` only under cuda; the tests run it everywhere.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn check_packed_stride(stride: usize, head_dim: usize, scaled: bool) -> Result<()> {
    anyhow::ensure!(
        !scaled,
        "PLE: NVFP4 rows carry their own block scales, but the cache also has \
         an FP8 scale array"
    );
    anyhow::ensure!(
        head_dim > 0 && head_dim.is_multiple_of(GROUP),
        "PLE: NVFP4 head_dim {head_dim} is not a multiple of {GROUP}"
    );
    anyhow::ensure!(
        stride == packed_stride(head_dim),
        "PLE: NVFP4 row stride {stride} != packed_stride({head_dim}) = {}",
        packed_stride(head_dim)
    );
    Ok(())
}

/// CPU dequant of packed rows to BF16 bits — the oracle the GPU gather is held
/// to. `atlas_core::numeric::nvfp4_dequant` is the same `(e2m1 * block) *
/// global` f32 product the reference computes; this only slices the packed row.
#[cfg(test)]
pub(crate) fn dequant_packed_bf16(rows: &[u8], head_dim: usize, global: f32) -> Vec<u16> {
    let stride = packed_stride(head_dim);
    let (cb, sb) = (code_bytes(head_dim), scale_bytes(head_dim));
    rows.chunks_exact(stride)
        .flat_map(|r| {
            atlas_core::numeric::nvfp4_dequant(
                &r[..cb],
                &r[cb..cb + sb],
                global,
                1,
                head_dim,
                GROUP,
            )
            .expect("NVFP4 row geometry")
        })
        .map(atlas_core::numeric::f32_to_bf16)
        .collect()
}

#[cfg(test)]
#[path = "ngram_nvfp4_tests.rs"]
mod tests;
