// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_PREFILL_BUSY_TOKENS=n` (default 0 = off): while sequences are
//! decoding, prefill chunks are at most `n` tokens; with none decoding they
//! keep `--max-prefill-tokens`.
//!
//! A chunk holds every decoding sequence for its whole run, so the chunk size
//! trades a long prompt's TTFT against the other streams' token gaps. With
//! this cap the arena and the idle chunk can be sized for prefill (larger
//! chunks give the routed experts more rows a weight read: Inkling 2 Mini's
//! 32K-token prompt took 7.79 s at 4,096-token chunks and 7.42 s at 8,192)
//! while decode keeps the short chunk's gaps. A prompt that arrives while an
//! idle prefill runs waits for at most one idle chunk, as it does today.
//! Chunk boundaries move with load, which changes nothing a request's text
//! depends on beyond the prefill numerics any chunk size has.

use std::sync::OnceLock;

/// The chunk cap the scheduler hands this tick's prefill phases.
pub(super) fn cap(max_prefill_tokens: usize, decoding: bool) -> usize {
    cap_with(busy_tokens(), max_prefill_tokens, decoding)
}

fn cap_with(busy: usize, max_prefill_tokens: usize, decoding: bool) -> usize {
    if decoding && busy > 0 && max_prefill_tokens > 0 {
        busy.min(max_prefill_tokens)
    } else {
        max_prefill_tokens
    }
}

fn busy_tokens() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("ATLAS_PREFILL_BUSY_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if n > 0 {
            tracing::info!("ATLAS_PREFILL_BUSY_TOKENS={n}: prefill chunks of at most {n} tokens while sequences decode");
        }
        n
    })
}

#[cfg(test)]
mod tests {
    use super::cap_with;

    #[test]
    fn decoding_caps_the_chunk() {
        assert_eq!(cap_with(4096, 8192, true), 4096);
        // never above the configured chunk the arena is sized for
        assert_eq!(cap_with(16384, 8192, true), 8192);
    }

    #[test]
    fn idle_or_off_keeps_the_configured_chunk() {
        assert_eq!(cap_with(4096, 8192, false), 8192);
        assert_eq!(cap_with(0, 8192, true), 8192);
        // unchunked prefill stays unchunked
        assert_eq!(cap_with(4096, 0, true), 0);
    }
}
