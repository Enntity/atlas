// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, anyhow};

/// Rows one DFlash verify may hold: the drafter's block of `num_drafts`, or
/// a long copy (`speculative::long_copy`) where those are on.
pub(crate) fn dflash_verify_rows(active: bool, num_drafts: usize) -> Result<usize> {
    if !active {
        return Ok(0);
    }
    num_drafts
        .max(crate::speculative::long_copy::drafts())
        .checked_add(1)
        .ok_or_else(|| anyhow!("DFlash/DSpark verify-row count overflow for K={num_drafts}"))
}
