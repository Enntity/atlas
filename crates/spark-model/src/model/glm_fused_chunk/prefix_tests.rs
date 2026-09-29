// SPDX-License-Identifier: AGPL-3.0-only

//! When verify owners may ride a pass under the prefix cache (host-side only).

use super::ride_is_cold;

#[test]
fn chunk_zero_never_carries_owners() {
    // Its lookup has not run when the scheduler decides.
    assert!(!ride_is_cold(0, 0, 0, 0));
    assert!(!ride_is_cold(0, 15_872, 0, 0));
}

#[test]
fn cold_later_chunks_and_their_tail_passes_carry_owners() {
    assert!(ride_is_cold(8_192, 8_192, 0, 0));
    assert!(ride_is_cold(8_192, 15_872, 0, 0));
    // A skip ending exactly at the pass leaves every row computed.
    assert!(ride_is_cold(8_192, 8_192, 8_192, 8_192));
}

#[test]
fn a_skip_or_cached_kv_reaching_the_pass_refuses() {
    // Marconi restore inside the chunk: rows skipped.
    assert!(!ride_is_cold(40_960, 40_960, 44_928, 44_992));
    // Snapshot below the chunk, cached K/V inside it: a write floor.
    assert!(!ride_is_cold(40_960, 40_960, 32_768, 44_992));
    // Hit without a snapshot: recompute over shared blocks.
    assert!(!ride_is_cold(8_192, 8_192, 0, 12_288));
}

#[test]
fn a_warm_turn_rides_its_tail_pass_once_past_the_match() {
    // 46_000-token turn over a 44_992-token match (bs 64): the last chunk
    // [40_960, 46_000) splits at 45_888, past the match.
    assert!(ride_is_cold(40_960, 45_888, 44_928, 44_992));
    // A 45_050-token turn splits at 44_928, below the match: refused.
    assert!(!ride_is_cold(40_960, 44_928, 44_928, 44_992));
}
