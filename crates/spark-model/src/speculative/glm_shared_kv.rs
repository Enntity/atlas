// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit physical shared KV capacity, independent of the served window.
use anyhow::{Context, Result, ensure};

pub const ENV: &str = "ATLAS_GLM_SHARED_KV_TOKENS";

pub fn parse(raw: Option<&str>) -> Result<Option<usize>> {
    raw.map(|value| {
        ensure!(
            !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
            "{ENV} must be a positive integer"
        );
        let tokens = value
            .parse::<usize>()
            .context("shared KV token count overflow")?;
        ensure!(tokens > 0, "{ENV} must be positive");
        Ok(tokens)
    })
    .transpose()
}

/// Reserve the scheduler's pending row plus transient speculative spill.
/// This intentionally retains the scheduler's conservative full-block boundary.
pub fn commitment(tokens: usize, block_size: usize, drafts: usize) -> Result<usize> {
    ensure!(block_size > 0, "shared KV block size must be positive");
    (tokens / block_size)
        .checked_add(1)
        .and_then(|n| n.checked_add(drafts.div_ceil(block_size)))
        .context("shared KV commitment overflow")
}

/// All counts include the permanent target dummy exactly once, while each
/// retained owner includes its own transient verifier spill.
pub fn physical_blocks(
    tokens: usize,
    context: usize,
    owners: usize,
    drafts: usize,
    block_size: usize,
) -> Result<usize> {
    ensure!(
        context > 0 && owners > 0 && drafts > 0 && block_size > 0,
        "shared KV geometry must be positive"
    );
    let short = super::glm_repair_policy::repair_context(context);
    let minimum = commitment(context, block_size, drafts)?.max(
        commitment(short, block_size, drafts)?
            .checked_mul(owners)
            .context("shared KV owner capacity overflow")?,
    );
    let usable = tokens.div_ceil(block_size);
    ensure!(
        usable >= minimum,
        "{ENV} provides {usable} usable blocks, needs at least {minimum} for one full context or {owners} repaired contexts"
    );
    usable
        .checked_add(1)
        .context("shared KV dummy capacity overflow")
}

pub fn validate_watermark(raw: Option<&str>, context: usize) -> Result<()> {
    if let Some(raw) = raw {
        let watermark = raw
            .parse::<usize>()
            .context("invalid KV admission watermark")?;
        ensure!(
            watermark >= context,
            "{ENV} requires the full generation reservation; ATLAS_KV_ADMIT_WATERMARK must be unset or >= served context"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_pool_covers_one_long_or_four_short_not_four_long() {
        let physical = physical_blocks(270336, 262144, 4, 2, 16).unwrap();
        assert_eq!(physical, 16897);
        let usable = physical - 1;
        assert!(commitment(262144, 16, 2).unwrap() <= usable);
        assert!(4 * commitment(32768, 16, 2).unwrap() <= usable);
        assert!(2 * commitment(262144, 16, 2).unwrap() > usable);
    }
    #[test]
    fn exact_capacity_accounts_for_dummy_and_spill() {
        let minimum = commitment(262144, 16, 2).unwrap();
        assert_eq!(
            physical_blocks(minimum * 16, 262144, 4, 2, 16).unwrap(),
            minimum + 1
        );
        assert!(physical_blocks((minimum - 1) * 16, 262144, 4, 2, 16).is_err());
    }
    #[test]
    fn invalid_and_overflowing_inputs_fail_closed() {
        for raw in ["", "0", "-1", "1.5", " 16", "99999999999999999999999999"] {
            assert!(parse(Some(raw)).is_err());
        }
        assert_eq!(parse(None).unwrap(), None);
        assert_eq!(parse(Some("270336")).unwrap(), Some(270336));
        assert!(physical_blocks(1, 0, 4, 2, 16).is_err());
        assert!(physical_blocks(usize::MAX, 262144, usize::MAX, 2, 16).is_err());
        assert!(physical_blocks(usize::MAX, 262144, 4, 2, 0).is_err());
        assert!(commitment(usize::MAX, 1, 2).is_err());
    }
    #[test]
    fn shared_policy_cannot_reduce_generation_reservations() {
        assert!(validate_watermark(None, 262144).is_ok());
        assert!(validate_watermark(Some("262144"), 262144).is_ok());
        for value in ["0", "32768", "invalid"] {
            assert!(validate_watermark(Some(value), 262144).is_err());
        }
    }
}
