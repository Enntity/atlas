// SPDX-License-Identifier: AGPL-3.0-only
//! Full retained contexts include the verifier's transient draft inputs.
//!
//! In long-context MTP2 C4 the budget-derived pool can exceed what the retained
//! owners need, and the surplus breaches the host reserve. Operators who want
//! the pool narrowed to the exact verified geometry opt in with [`CAP_FLAG`];
//! every other model/mode keeps the budget-derived count.
use anyhow::{Context, Result, ensure};

const CAP_FLAG: &str = "ATLAS_GLM_KV_CAP_TO_CONTEXTS";

/// Parse the cap opt-in. Same shape as the neighbouring GLM repair gates:
/// absent or `0` is off, `1` is on, anything else is a hard error.
pub(super) fn parse_cap(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("{CAP_FLAG} must be 0 or 1"),
    }
}

/// Geometry-derived block count for the selected retained owners, or `None`
/// when the cap does not apply.
///
/// The cap only applies to the selected GLM model with repair and long-context
/// enabled and no HSS: any other model or mode with the flag set is a
/// configuration error, not a silent no-op. Pure — the caller owns the env — so
/// tests never mutate process-global state.
#[allow(clippy::too_many_arguments)]
pub(super) fn capped_blocks(
    cap_requested: bool,
    model_is_glm5_next: bool,
    repair_long_context: bool,
    hss: bool,
    context: usize,
    drafts: usize,
    block_size: usize,
    owners: usize,
    available_blocks: usize,
) -> Result<Option<usize>> {
    if !cap_requested {
        return Ok(None);
    }
    ensure!(
        model_is_glm5_next && repair_long_context && !hss,
        "{CAP_FLAG}=1 selects the repaired GLM long-context target pool only \
         (requires model_type=glm5_next, repair + long context enabled, no HSS)"
    );
    let required = required_blocks(context, drafts, block_size, owners)?;
    ensure!(
        available_blocks >= required,
        "GLM repaired target KV needs {required} blocks for {owners} contexts of \
         {context} plus {drafts} speculative rows, but only {available_blocks} \
         blocks fit; increase the memory budget without enabling KV overcommit"
    );
    Ok(Some(required))
}

/// Blocks the repaired verifier needs for every selected retained owner.
fn required_blocks(
    context: usize,
    drafts: usize,
    block_size: usize,
    owners: usize,
) -> Result<usize> {
    ensure!(
        context > 0 && drafts > 0 && block_size > 0 && owners > 0,
        "GLM repaired target KV capacity needs positive context/depth/block/owners"
    );
    let rows = context
        .checked_add(drafts)
        .context("GLM target KV row overflow")?;
    rows.div_ceil(block_size)
        .checked_mul(owners)
        .context("GLM target KV owner capacity overflow")
}

pub(super) fn validate_target_pool(
    context: usize,
    drafts: usize,
    block_size: usize,
    owners: usize,
    available_blocks: usize,
) -> Result<()> {
    let required = required_blocks(context, drafts, block_size, owners)?;
    ensure!(
        available_blocks >= required,
        "GLM repaired target KV needs {required} blocks for {owners} contexts of {context} plus {drafts} speculative rows, but only {available_blocks} blocks fit; increase the memory budget without enabling KV overcommit"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{capped_blocks, validate_target_pool};

    /// The long-context MTP2 C4 shape the cap exists for: 36K context, two
    /// speculative rows, 16-token blocks, four retained owners.
    fn capped(available: usize) -> anyhow::Result<Option<usize>> {
        capped_blocks(true, true, true, false, 36864, 2, 16, 4, available)
    }

    #[test]
    fn c4_32k_requires_four_speculative_overflow_blocks() {
        for blocks in [0, 8192, 8195] {
            assert!(validate_target_pool(32768, 2, 16, 4, blocks).is_err());
        }
        assert!(validate_target_pool(32768, 2, 16, 4, 8196).is_ok());
    }

    #[test]
    fn block_aligned_and_partial_contexts_use_exact_row_capacity() {
        assert!(validate_target_pool(2044, 4, 16, 1, 128).is_ok());
        assert!(validate_target_pool(2044, 4, 16, 1, 127).is_err());
        assert!(validate_target_pool(32767, 2, 16, 4, 8192).is_err());
        assert!(validate_target_pool(32766, 2, 16, 4, 8192).is_ok());
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
            assert!(validate_target_pool(args.0, args.1, args.2, args.3, usize::MAX).is_err());
        }
    }

    #[test]
    fn default_off_leaves_the_budget_derived_pool_untouched() {
        for value in [None, Some("0")] {
            assert!(!super::parse_cap(value).unwrap());
        }
        // Surplus budget is only narrowed when the operator asks for it.
        assert_eq!(
            capped_blocks(false, true, true, false, 36864, 2, 16, 4, 10_000).unwrap(),
            None
        );
        // Off is off even for geometry the cap would reject when on.
        assert_eq!(
            capped_blocks(false, false, false, true, 0, 0, 0, 0, 0).unwrap(),
            None
        );
    }

    #[test]
    fn opt_in_narrows_the_surplus_to_the_exact_context_capacity() {
        assert_eq!(capped(10_000).unwrap(), Some(9220));
        assert_eq!(capped(usize::MAX).unwrap(), Some(9220));
    }

    #[test]
    fn exact_fit_budget_keeps_every_block() {
        assert_eq!(capped(9220).unwrap(), Some(9220));
    }

    #[test]
    fn insufficient_budget_fails_before_the_cap_is_applied() {
        for available in [0, 9219] {
            let error = capped(available).unwrap_err().to_string();
            assert!(error.contains("needs 9220 blocks"), "{error}");
        }
    }

    #[test]
    fn invalid_geometry_and_overflow_fail_closed_under_the_cap() {
        for args in [
            (0, 2, 16, 4),
            (36864, 0, 16, 4),
            (36864, 2, 0, 4),
            (36864, 2, 16, 0),
            (usize::MAX, 2, 16, 4),
            (36864, 2, 16, usize::MAX),
        ] {
            assert!(
                capped_blocks(
                    true,
                    true,
                    true,
                    false,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    usize::MAX
                )
                .is_err()
            );
        }
    }

    #[test]
    fn activation_is_rejected_outside_the_repaired_glm_long_context_lane() {
        for (model, repair_long_context, hss) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            let error = capped_blocks(
                true,
                model,
                repair_long_context,
                hss,
                36864,
                2,
                16,
                4,
                usize::MAX,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("ATLAS_GLM_KV_CAP_TO_CONTEXTS=1"), "{error}");
        }
        assert!(super::parse_cap(Some("true")).is_err());
        assert!(super::parse_cap(Some("")).is_err());
    }
}
