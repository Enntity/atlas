// SPDX-License-Identifier: AGPL-3.0-only

//! CPU reference for GLM sparse-index pool selection and raw-token expansion.

use std::cmp::Ordering;

use anyhow::{Result, bail};

/// Select visible complete pools, expand each to raw token positions, then
/// append the visible incomplete tail exactly once.
///
/// `pool_scores[i]` corresponds to raw tokens
/// `[i * kpool, i * kpool + kpool)`. NaNs are treated as invisible. Ties are
/// resolved by lower pool index, giving deterministic CPU/GPU parity vectors.
pub fn select_token_indices(
    pool_scores: &[f32],
    visible_complete_pools: usize,
    top_pools: usize,
    kpool: usize,
    visible_tokens: usize,
    always_select_tail: bool,
) -> Result<Vec<usize>> {
    if kpool == 0 {
        bail!("GLM sparse-index kpool must be greater than zero");
    }
    let complete = visible_complete_pools.min(pool_scores.len());
    if complete.saturating_mul(kpool) > visible_tokens {
        bail!("visible pool range exceeds visible token range");
    }

    let mut pools: Vec<usize> = (0..complete)
        .filter(|&pool| pool_scores[pool].is_finite())
        .collect();
    pools.sort_unstable_by(|&left, &right| {
        pool_scores[right]
            .partial_cmp(&pool_scores[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    pools.truncate(top_pools);

    let tail_start = complete * kpool;
    let tail_len = visible_tokens.saturating_sub(tail_start).min(kpool - 1);
    let tail_extra = usize::from(always_select_tail) * tail_len;
    let mut selected = Vec::with_capacity(pools.len() * kpool + tail_extra);
    for pool in pools {
        let start = pool * kpool;
        selected.extend(start..start + kpool);
    }
    if always_select_tail {
        selected.extend(tail_start..tail_start + tail_len);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::select_token_indices;

    #[test]
    fn expands_top_pools_and_appends_incomplete_tail() {
        let selected = select_token_indices(&[0.2, 0.9, 0.5], 3, 2, 4, 14, true).unwrap();
        assert_eq!(selected, vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn tie_break_and_nan_filter_are_deterministic() {
        let selected = select_token_indices(&[1.0, f32::NAN, 1.0], 3, 2, 4, 12, true).unwrap();
        assert_eq!(selected, vec![0, 1, 2, 3, 8, 9, 10, 11]);
    }

    #[test]
    fn boundary_lengths_never_duplicate_tail() {
        assert_eq!(
            select_token_indices(&[], 0, 512, 4, 3, true).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(
            select_token_indices(&[1.0], 1, 512, 4, 4, true).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            select_token_indices(&[1.0], 1, 512, 4, 7, true).unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn refuses_pool_visibility_beyond_causal_tokens() {
        assert!(select_token_indices(&[1.0], 1, 1, 4, 3, true).is_err());
    }
}
