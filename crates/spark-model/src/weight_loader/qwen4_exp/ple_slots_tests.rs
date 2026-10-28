// SPDX-License-Identifier: AGPL-3.0-only

use super::derived_slots;

/// The fixed 65536 this replaced assumed a 2048-token chunk; the default
/// serve config presents 8193, and a prefill pins tokens x heads rows.
/// Default warm lookahead is 2 spans (`ATLAS_PLE_WARM_AHEAD`), so the
/// derived count covers span + lookahead.
#[test]
fn derived_slots_cover_the_default_chunk() {
    assert_eq!(derived_slots(8193, 16), 524_288); // (8193 + 16386) x 16, rounded up
    assert_eq!(derived_slots(2048, 16), 131_072); // (2048 + 4096) x 16, exact
    assert_eq!(derived_slots(4096, 16), 262_144); // (4096 + 8192) x 16, exact
    assert!(derived_slots(20481, 16) >= 20481 * 16);
}
