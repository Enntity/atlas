// SPDX-License-Identifier: AGPL-3.0-only

/// `warm_ahead` term is the prefill prefetch window (`warm.rs`): without
/// slack beside the span's pins, the worker's rows would have to evict
/// each other — or the span's — before the gather reads them. The old
/// `max_batch_tokens` derivation predates the span loop and would
/// over-provision the arena 4x at a 32K --max-prefill-tokens.
#[cfg(feature = "cuda")]
pub(super) fn derived_slots(scratch_tokens: usize, ngram_heads: usize) -> usize {
    (scratch_tokens
        .saturating_add(crate::layers::ple::warm_ahead_tokens(scratch_tokens))
        .saturating_mul(ngram_heads)
        .next_power_of_two())
    .max(65536)
}

/// The forward pipeline's scratch width. The six `[span, hc*H]` buffers cost
/// `scratch * 10240 * 14` bytes — ~1.18 GB at the 8192 default — whatever the
/// forward width: the llama.cpp ubatch discipline, applied to the one
/// subsystem whose scratch used to scale with `--max-prefill-tokens`.
#[cfg(feature = "cuda")]
pub(super) fn chunk_from_env() -> usize {
    std::env::var("ATLAS_PLE_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(8192)
}

#[cfg(feature = "cuda")]
pub(super) fn slots_from_env(scratch_tokens: usize, ngram_heads: usize) -> (usize, &'static str) {
    match std::env::var("ATLAS_PLE_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(n) if n > 0 => (n, "ATLAS_PLE_CACHE_SLOTS"),
        _ => (
            derived_slots(scratch_tokens, ngram_heads),
            "(span + warm_ahead)*heads rounded up",
        ),
    }
}
