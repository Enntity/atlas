// SPDX-License-Identifier: AGPL-3.0-only

//! Where `prefill_chunk_dispatch` splits a prompt's last chunk for the
//! warm-turn tail checkpoint.

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// The token at which the last chunk `[chunk_start, len)` of `tokens`
    /// splits into `[chunk_start, cut)` + `[cut, len)`, or `None` when it
    /// runs as one pass. A pure function of the tokens, the configuration
    /// and the environment, so every rank splits alike. Locks the KV cache
    /// briefly for its block size.
    pub(in crate::model) fn prefill_tail_split(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        is_last_chunk: bool,
    ) -> Option<usize> {
        // Tail-checkpoint split (issue #15 follow-up, 2026-07-02): a warm
        // multi-turn hit matches the radix at BLOCK granularity, and the
        // divergence point sits at/near the previous prompt end (the chat
        // template's generation-only suffix — e.g. Qwen's forced empty
        // <think> block — is absent from the re-rendered history), so the
        // next turn's `matched` lands at floor(divergence/bs)*bs, which is
        // the prompt's last full-block boundary OR one block below it (when
        // the template suffix crosses that boundary; measured: both occur).
        // Snapshot eligibility requires snap_tok <= matched — the leaf
        // snapshot (at `total`) is PAST both, making warm turns recompute the
        // full SSM state (or fall back an entire turn to the previous tail,
        // measured 1.3-3.2k-token replays). Split the final chunk ONCE, one
        // block below the last block boundary under `total`: that position is
        // <= both possible match points, so the snapshot
        // `prefill_b_save_checkpoint` saves there (independent of
        // --ssm-checkpoint-interval) is always eligible and the warm replay
        // is <= 2 blocks, folded into the suffix prefill pass. A single cut
        // costs one extra small pass at save time (a cut at the boundary
        // itself would need a second pass and is redundant — measured
        // +~160ms/turn for two cuts vs <=31-token replay for one).
        //
        // The extra pass costs ~150ms on this class of MoE model (a tiny-M
        // pass still sweeps most activated expert weights), which is -7% on
        // a cold 2k prefill — so on single-GPU the split only fires when the
        // radix already holds a prefix of this prompt (peek is read-only):
        // single-shot requests never pay; conversations pay from turn 2
        // onward, where the cost is amortized against the warm win. Known
        // residual: turn 2 of a conversation still recomputes the full SSM
        // state (its cold turn 1 saved no tail checkpoint). On EP>1 the
        // split is unconditional instead: rank-local radix contents diverge,
        // and chunk sequences must be deterministic on (tokens, config)
        // across ranks (bug #33 invariant). Skipped for vision prompts (pad
        // runs must not straddle chunk boundaries) and non-SSM models
        // (KV-only cache hits need no snapshot).
        if !(is_last_chunk
            && self.config.num_ssm_layers() > 0
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
            && !self.tokens_have_vision_pad(tokens))
        {
            return None;
        }
        let bs = self.kv_cache.lock().block_size();
        // UNCONDITIONAL. This used to additionally require
        // `ep_active || peek_matched_tokens(..) > 0`, i.e. it split only on a
        // WARM request (radix already populated) — which made the prompt take a
        // DIFFERENT SHAPE cold vs warm: one N-token pass when cold, two passes
        // [0..cut) + [cut..N) when warm. BF16 accumulation is not associative,
        // so the two shapes produce different hidden states, and at temperature 0
        // a near-tied argmax flips. Same prompt, same seed, different answer.
        //
        // MEASURED on Puzzle-75B (fresh container, temp 0, exact-hit shortcut
        // bypassed so this split is the ONLY cold/warm difference):
        //   34-token prompt (cut=16, split fires warm) : cold "17 barrels"
        //                                                warm "15 barrels"  DIVERGE
        //   16-token prompt (cut<=0, split IMPOSSIBLE) : cold == warm       MATCH
        // The cut threshold predicts the divergence exactly.
        //
        // The invariant is already stated for EP>1 in the comment above —
        // "chunk sequences must be deterministic on (tokens, config)" — it was
        // just never enforced on a single rank. Splitting unconditionally makes
        // cold and warm identical by construction, at the cost of one extra pass
        // on cold prefills that cross the cut.
        //
        // `ATLAS_NO_TAIL_SPLIT=1` disables the split entirely (same-binary A/B).
        // That is the OTHER way to satisfy the invariant — always one pass — and
        // it keeps the single-pass numerics, at the cost of the warm-turn tail
        // checkpoint this split exists to create.
        if std::env::var("ATLAS_NO_TAIL_SPLIT").as_deref() == Ok("1") {
            return None;
        }
        tail_split_cut(tokens.len(), chunk_start, bs)
    }
}

/// One block below the last block boundary strictly under `total`, when that
/// cut falls strictly inside `(chunk_start, total)`. The tail pass
/// `[cut, total)` then holds at least `bs + 1` tokens.
pub(in crate::model) fn tail_split_cut(
    total: usize,
    chunk_start: usize,
    bs: usize,
) -> Option<usize> {
    let cut = ((total.saturating_sub(1) / bs) * bs).saturating_sub(bs);
    (cut > chunk_start && cut < total).then_some(cut)
}

#[cfg(test)]
#[path = "tail_split_tests.rs"]
mod tests;
