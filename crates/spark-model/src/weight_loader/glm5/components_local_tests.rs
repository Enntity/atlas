// SPDX-License-Identifier: AGPL-3.0-only

use super::unified_moe_layout_enabled;

#[test]
fn unified_moe_layout_is_explicitly_opt_in() {
    assert!(unified_moe_layout_enabled(Some("1")));
    assert!(unified_moe_layout_enabled(Some("TRUE")));
    assert!(!unified_moe_layout_enabled(None));
    assert!(!unified_moe_layout_enabled(Some("0")));
    assert!(!unified_moe_layout_enabled(Some("full")));
}
