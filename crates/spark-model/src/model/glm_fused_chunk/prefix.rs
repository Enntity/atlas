// SPDX-License-Identifier: AGPL-3.0-only

//! A fused chunk under the prefix cache.
//!
//! The owners' rows sit after the chunk's rows, and the passenger layer path
//! (`prefill_with_glm_passengers`) runs the chunk as a cold prefill: every
//! row computed, every row's K/V written. Three prefix-cache behaviours
//! change a chunk's rows, and each is either avoided or refused upfront:
//!
//! - **Chunk 0's lookup.** A hit skips rows (`proc_range`) and, on a Marconi
//!   restore, leaves a KV write floor over the replay window, which the
//!   passenger path does not take. The outcome is unknown before the lookup,
//!   so chunk 0 never carries owners; later chunks read the recorded outcome
//!   (`marconi_skip_to`, `cached_prefix_tokens`).
//! - **Inherited skips.** A later chunk skips rows below `marconi_skip_to`
//!   and must not rewrite K/V below `cached_prefix_tokens` (shared radix
//!   blocks, whose recomputed values would not be bit-equal). Owners ride
//!   only a pass lying wholly past both, which computes like a cold chunk
//!   (`gdn_exact_replay` is read only by GDN layers, not GLM's KDA).
//! - **The tail-checkpoint split** (`prefill_tail_split`) runs the last chunk
//!   as `[start, cut)` + `[cut, len)`. Owners ride the tail pass, the shape
//!   of an unsplit last fused chunk: an ordinary pass and its prefill
//!   checkpoint run first, and nothing but the chunk's finalize runs between
//!   the owners' traversal and their tails, as before.
//!
//! The owners' own rows are unaffected by the cache: their blocks come from
//! `ensure_blocks_through_decode` exactly as in the separate owner-batched
//! verify (eviction releases only the cache's own reference, so no live
//! row's block is freed), and the chunk's cache inserts and snapshot saves
//! cover the chunk's sequence alone.

use super::super::TransformerModel;
use crate::traits::SequenceState;

/// Whether owners may ride a pass starting at `ride_start` of a chunk starting
/// at `chunk_start` (`ride_start >= chunk_start`), given chunk 0's recorded
/// prefix decision: past chunk 0, with no skip or cached K/V reaching the
/// pass.
pub(in crate::model) fn ride_is_cold(
    chunk_start: usize,
    ride_start: usize,
    marconi_skip_to: usize,
    cached_prefix_tokens: usize,
) -> bool {
    chunk_start > 0 && marconi_skip_to <= ride_start && cached_prefix_tokens <= ride_start
}

impl TransformerModel {
    /// The prefix-cache half of `glm_fused_chunk_supported`: the pass the
    /// owners would ride (the chunk, or its tail after the tail split)
    /// computes as a cold chunk. Decided from the request, the configuration
    /// and chunk 0's prefix decision, which every rank's ordinary chunks
    /// already rely on agreeing (the lookup min-reduces the match).
    pub(in crate::model) fn glm_fused_prefix_ok(
        &self,
        prompt: &[u32],
        seq: &SequenceState,
        chunk_start: usize,
        chunk_len: usize,
    ) -> bool {
        if !self.prefix_cache.is_active() {
            return true;
        }
        let is_last = chunk_start + chunk_len >= prompt.len();
        let ride = self
            .prefill_tail_split(prompt, chunk_start, is_last)
            .unwrap_or(chunk_start);
        ride_is_cold(
            chunk_start,
            ride,
            seq.marconi_skip_to,
            seq.cached_prefix_tokens,
        )
    }

    /// Runtime twin of the gate, inside the pass the owners ride (after the
    /// lookup): every row of the chunk is computed, and under the prefix
    /// cache the pass lies past chunk 0's cached prefix.
    pub(in crate::model) fn glm_fused_pass_ok(
        &self,
        seq: &SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
    ) -> bool {
        proc_start == chunk_start
            && proc_count == chunk_len
            && (!self.prefix_cache.is_active()
                || ride_is_cold(
                    chunk_start,
                    chunk_start,
                    seq.marconi_skip_to,
                    seq.cached_prefix_tokens,
                ))
    }
}

#[cfg(test)]
#[path = "prefix_tests.rs"]
mod tests;
