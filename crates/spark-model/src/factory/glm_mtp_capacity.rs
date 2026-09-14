// SPDX-License-Identifier: AGPL-3.0-only
//! Full retained contexts include the verifier's transient draft inputs.
use anyhow::{Context, Result, ensure};

pub(super) fn validate_target_pool(
    context: usize,
    drafts: usize,
    block_size: usize,
    owners: usize,
    available_blocks: usize,
) -> Result<()> {
    ensure!(
        context > 0 && drafts > 0 && block_size > 0 && owners > 0,
        "GLM repaired target KV capacity needs positive context/depth/block/owners"
    );
    let rows = context
        .checked_add(drafts)
        .context("GLM target KV row overflow")?;
    let required = rows
        .div_ceil(block_size)
        .checked_mul(owners)
        .context("GLM target KV owner capacity overflow")?;
    ensure!(
        available_blocks >= required,
        "GLM repaired target KV needs {required} blocks for {owners} contexts of {context} plus {drafts} speculative rows, but only {available_blocks} blocks fit; increase the memory budget without enabling KV overcommit"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_target_pool as check;

    #[test]
    fn c4_32k_requires_four_speculative_overflow_blocks() {
        assert!(check(32768, 2, 16, 4, 8196).is_ok());
        for blocks in [0, 8192, 8195] {
            assert!(check(32768, 2, 16, 4, blocks).is_err());
        }
    }

    #[test]
    fn block_aligned_and_partial_contexts_use_exact_row_capacity() {
        assert!(check(2044, 4, 16, 1, 128).is_ok());
        assert!(check(2044, 4, 16, 1, 127).is_err());
        assert!(check(32767, 2, 16, 4, 8192).is_err());
        assert!(check(32766, 2, 16, 4, 8192).is_ok());
    }

    #[test]
    fn invalid_or_overflowing_geometry_fails_closed() {
        for args in [
            (0, 2, 16, 4),
            (32768, 0, 16, 4),
            (32768, 2, 0, 4),
            (32768, 2, 16, 0),
            (usize::MAX, 2, 16, 4),
            (32768, 2, 16, usize::MAX),
        ] {
            assert!(check(args.0, args.1, args.2, args.3, usize::MAX).is_err());
        }
    }
}
