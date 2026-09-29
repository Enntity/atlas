// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `paged_glm` dense-selection exactness.

use super::dense_selection_is_exact;

#[test]
fn dense_reference_stops_at_the_semantic_topk_boundary() {
    assert!(dense_selection_is_exact(2048, 2048));
    assert!(!dense_selection_is_exact(2049, 2048));
    assert!(!dense_selection_is_exact(1, 0));
}
