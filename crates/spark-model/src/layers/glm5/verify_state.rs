// SPDX-License-Identifier: AGPL-3.0-only

//! Exact GLM sparse-index speculative-verification state images.

use anyhow::Result;

use crate::layer::ForwardContext;

pub(crate) fn copy_dsa_tail(
    source: crate::layer::GlmSparseMlaStatePointers,
    target: crate::layer::GlmSparseMlaStatePointers,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let tail_bytes =
        ctx.config.index_kpool.saturating_sub(1) * ctx.config.index_head_dim * size_of::<u16>();
    for (src, dst, bytes) in [
        (source.tail_keys, target.tail_keys, tail_bytes),
        (source.tail_gates, target.tail_gates, tail_bytes),
        (
            source.tail_metadata,
            target.tail_metadata,
            4 * size_of::<i32>(),
        ),
    ] {
        ctx.gpu.copy_d2d_async(src, dst, bytes, stream)?;
    }
    Ok(())
}
