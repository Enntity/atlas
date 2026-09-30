// SPDX-License-Identifier: AGPL-3.0-only

//! How a sequence takes cached KV blocks.
//!
//! Invariant: no two sequences ever hold a writable reference to one KV
//! block. A block is shared, with the prefix cache or another sequence, only
//! while every position in it is a committed token that no holder writes
//! again. Three rules keep it:
//!
//! * A match covers whole blocks only (`PrefixCache::lookup_whole_blocks`).
//!   A sequence writes from its matched length on, so the first block it
//!   writes is one it allocated.
//! * The cache publishes whole blocks only (`RadixTree` insert), so the block
//!   a live sequence still appends to, or rewrites during a speculative
//!   verify, is referenced by that sequence alone.
//! * A prefill that resumes from a recurrent-state snapshot below its matched
//!   length writes nothing below that length (the `cached_prefix_tokens`
//!   write floor).
//!
//! One write into shared blocks remains. A hybrid-SSM prefill that matched a
//! prefix but found no snapshot to resume from recomputes the whole prompt
//! and rewrites the matched rows. Those values are a function of the same
//! token prefix the rows already hold, so holders read an equivalent row.

use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixMatch;

use super::block_mgmt::reuse_prefix_match_disk_ids;
use crate::traits::SequenceState;

/// Take the sequence's references on a prefix match and start its block
/// table with the matched blocks. Every prefill path acquires its cached
/// prefix here.
pub(crate) fn adopt_prefix_match(
    seq: &mut SequenceState,
    prefix_match: &PrefixMatch,
    kv_cache: &mut PagedKvCache,
) {
    debug_assert_eq!(
        prefix_match.matched_tokens,
        prefix_match.matched_blocks.len() * kv_cache.block_size(),
        "a prefix match must cover whole blocks"
    );
    seq.cached_prefix_tokens = prefix_match.matched_tokens;
    seq.cached_prefix_blocks = prefix_match.matched_blocks.len();
    for &block in &prefix_match.matched_blocks {
        kv_cache.inc_ref(block);
        seq.block_table.push(block);
    }
    reuse_prefix_match_disk_ids(
        &prefix_match.matched_disk_block_ids,
        &mut seq.disk_block_ids,
    );
}

#[cfg(test)]
#[path = "prefix_share_tests.rs"]
mod tests;
