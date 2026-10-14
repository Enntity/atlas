// SPDX-License-Identifier: AGPL-3.0-only

//! Verify rows for long copy drafts (spark-server `copy_drafts`,
//! `ATLAS_DFLASH_COPY_LONG_MAX`).
//!
//! A copy of the prompt may propose more drafts than the DFlash drafter's
//! block (γ − 1). Off (the default) nothing here changes a size. On, every
//! pool a lone DFlash verify writes per row is sized for `long + 1` rows
//! instead of γ: the SSM verify intermediates and the preflight reserve
//! (`dspark_pool::dflash_verify_rows`, `ssm_reserve`), the DFlash hidden-save
//! rows and the proposer's capture lifecycle (both from the same row count),
//! and the GLM KDA layers' FFN takes the exact grouped prefill MoE for the
//! wider block (`glm_independent::ffn_rows_selected`) instead of the per-token
//! loop. The drafter's own block stays γ.

use std::sync::OnceLock;

/// The longest copy: 31 drafts, a 32-row verify, the widest canonical
/// verify (`canonical_verify`) and verify-metadata layout.
pub const MAX_LONG_COPY_DRAFTS: usize = 31;

/// Drafts a long copy may propose (0 = off), from `var` (an environment
/// lookup): `ATLAS_DFLASH_COPY_LONG_MAX` in `8..=31` (past the scheduler's
/// seven-draft block, `dflash_width::MAX_DRAFTS`), with copy drafts
/// (`ATLAS_DFLASH_COPY_DRAFTS=1`) and the confidence width that prices their
/// rows (`ATLAS_DFLASH_CONF_WIDTH=1`) on.
pub fn parse_drafts(var: impl Fn(&str) -> Option<String>) -> usize {
    let on = |name: &str| var(name).as_deref() == Some("1");
    if !on("ATLAS_DFLASH_COPY_DRAFTS") || !on("ATLAS_DFLASH_CONF_WIDTH") {
        return 0;
    }
    var("ATLAS_DFLASH_COPY_LONG_MAX")
        .and_then(|v| v.trim().parse().ok())
        .filter(|v| (8..=MAX_LONG_COPY_DRAFTS).contains(v))
        .unwrap_or(0)
}

/// [`parse_drafts`] of the process environment, read once.
pub fn drafts() -> usize {
    static DRAFTS: OnceLock<usize> = OnceLock::new();
    *DRAFTS.get_or_init(|| parse_drafts(|name| std::env::var(name).ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> usize {
        parse_drafts(|name| {
            pairs
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| (*v).into())
        })
    }

    #[test]
    fn long_copies_need_copies_and_the_confidence_width() {
        let copies = ("ATLAS_DFLASH_COPY_DRAFTS", "1");
        let conf = ("ATLAS_DFLASH_CONF_WIDTH", "1");
        assert_eq!(env(&[]), 0);
        assert_eq!(env(&[copies, conf]), 0);
        assert_eq!(
            env(&[copies, conf, ("ATLAS_DFLASH_COPY_LONG_MAX", "15")]),
            15
        );
        assert_eq!(
            env(&[copies, conf, ("ATLAS_DFLASH_COPY_LONG_MAX", " 31 ")]),
            31
        );
        assert_eq!(env(&[conf, ("ATLAS_DFLASH_COPY_LONG_MAX", "15")]), 0);
        assert_eq!(env(&[copies, ("ATLAS_DFLASH_COPY_LONG_MAX", "15")]), 0);
        for bad in ["7", "32", "x", ""] {
            assert_eq!(env(&[copies, conf, ("ATLAS_DFLASH_COPY_LONG_MAX", bad)]), 0);
        }
    }
}
