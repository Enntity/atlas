// SPDX-License-Identifier: AGPL-3.0-only

//! The fixed overhead of a prefix-cached ("warm") turn: what a request pays
//! before its first token that does not depend on its new tokens.
//!
//! # A warm turn, step by step
//!
//! `N` prompt tokens, of which the radix tree matches `matched` and an SSM
//! snapshot restores `restored <= matched`; chunks of `C` tokens
//! (`--max-prefill-tokens`), 16-token blocks. Both ranks run every model step
//! in lockstep; the worker follows the head's commands. Cost class in
//! brackets: `fixed` per request, `chunk` per prefill chunk, `pass` per chunk
//! that computes, `cached` per cached token or block, `new` per new row.
//!
//! 1. API thread: parse, render the template, tokenize [cached + new, host].
//!    Outside the scheduler's `TTFT=`; a client sees it.
//! 2. Scheduler: allocate the sequence, send the request preamble (native
//!    fence, vision state) [fixed, a few 4-byte broadcasts].
//! 3. Per chunk, the head sends the chunk command: slot, command, chunk
//!    length, chunk start, prompt length [chunk, five 4-byte broadcasts, each
//!    a stream sync and a device read on the worker], then the whole prompt
//!    [chunk x cached: a pageable copy to the device, one broadcast, a sync
//!    and a read of `4 N` bytes on the worker]. `ATLAS_GLM_PROMPT_DELTA`
//!    sends only what the worker does not hold from an earlier command.
//! 4. Per chunk, both ranks: zero the whole buffer arena [chunk, fixed size:
//!    about 17 ms at an 8K-row arena], embed the chunk [chunk x chunk rows].
//!    A chunk below `restored` then computes nothing: it reserves its blocks
//!    (one min-vote, two 4-byte broadcasts), appends its tokens and returns.
//!    `ATLAS_GLM_WARM_SKIP_CACHED` does not zero or embed for such a chunk
//!    (`prefill_b::warm`; multi-rank worlds).
//! 5. Chunk 0 only: the radix walk and its references [cached blocks, host],
//!    the match min-vote [fixed], the restore-depth agreement with
//!    `ATLAS_GLM_PC_EVICT` or `_BRANCH` [fixed, one to three votes], the
//!    snapshot restore [fixed: one copy of the SSM state per rank].
//! 6. The chunk holding `restored` and every later one computes rows
//!    `[max(start, restored), end)`: the block-table upload [cached blocks on
//!    the first pass, new blocks after], positions and slots [new], a stream
//!    sync, then all layers [pass: fixed launches and collectives per layer;
//!    new: the weights each row's experts sweep, which dominates; new x
//!    context: attention and the index over the cached rows].
//! 7. The last chunk is split at the tail cut (`pc_policy::tail_cut`), so a
//!    warm turn runs two passes: `[restored, cut)`, then the checkpoint save
//!    at `cut` [fixed: one state copy, and the radix insert over the cached
//!    blocks], then `[cut, N)`. Each pass pays step 4 again. A turn whose
//!    restore depth is `cut` already has an empty first half, which
//!    `ATLAS_GLM_WARM_SKIP_CACHED` makes free.
//! 8. Final norm and LM head on the last row [fixed: one sweep of the head],
//!    the radix insert of the prompt [cached blocks, host], then the
//!    scheduler reads the logits and samples [fixed].
//!
//! `ATLAS_PROFILE_PREFILL` logs each chunk's host submit time and the
//! scheduler's `Done:` line the TTFT. Nothing sums a request, or times its
//! prompt transfer, lookup and finish.
//!
//! # What is not removed, and why
//!
//! * The arena zero of a chunk that computes. The arena is shared scratch
//!   with layouts that do not follow the row count (the index logits of a
//!   row group over the whole context, the FlashKDA workspace, the MLA
//!   absorbed query), and `decode_a` documents a path that read rows it had
//!   not written. Zeroing "the rows this chunk uses" is therefore not the
//!   zero every pass starts from today; it needs a measured bound on what a
//!   pass dirties.
//! * The second pass of the tail split. One pass over `[restored, N)` runs
//!   the same math with other GEMM shapes, MoE groups and KDA pieces
//!   (`pc_policy`, "Accumulation order"), and the checkpoint at `cut` would
//!   need the KDA recurrence split in-pass, which only the `atlas_scale`
//!   build has (`midchunk_capture`).
//! * The full-chunk embed of a chunk that computes only its tail. Its rows
//!   past the computed ones are what a kernel reading beyond its rows sees
//!   today; skipping it would change them.
//!
//! # Rank parity
//!
//! `ATLAS_GLM_PROMPT_DELTA` changes the head's command words, so both ranks
//! must run with the same value (the launcher's startup agreement must carry
//! it: [`prompt_delta_requested`]). `ATLAS_GLM_WARM_SKIP_CACHED` adds no
//! command and no collective: a rank without it only does the work the other
//! skips.

use std::sync::Arc;

use anyhow::{Result, ensure};
use parking_lot::Mutex;

use super::types::TransformerModel;

#[cfg(test)]
#[path = "warm_turn_tests.rs"]
mod tests;

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// `ATLAS_GLM_PROMPT_DELTA=1`: see "Rank parity" above.
pub(in crate::model) fn prompt_delta_requested() -> bool {
    env_on("ATLAS_GLM_PROMPT_DELTA")
}

/// The warm-turn switches and the state they keep, one per model.
pub(in crate::model) struct WarmTurn {
    /// `ATLAS_GLM_PROMPT_DELTA=1`: a prefill command carries the prompt as
    /// its difference from the prompt a slot already holds.
    pub(in crate::model) prompt_delta: bool,
    /// `ATLAS_GLM_WARM_SKIP_CACHED=1`: a chunk that computes nothing does not
    /// zero the arena or embed (`prefill_b::warm`).
    pub(in crate::model) skip_cached: bool,
    /// The prompts of the prefill commands this rank sent or received.
    prompts: Mutex<PromptMirrors>,
}

impl WarmTurn {
    pub(in crate::model) fn from_env() -> Self {
        Self {
            prompt_delta: prompt_delta_requested(),
            skip_cached: env_on("ATLAS_GLM_WARM_SKIP_CACHED"),
            prompts: Mutex::default(),
        }
    }
}

/// A rank's copy of the prompt a slot's last prefill command carried, with
/// the hash both ranks compare.
#[derive(Clone)]
struct PromptMirror {
    tokens: Arc<Vec<u32>>,
    hash: u32,
}

impl Default for PromptMirror {
    fn default() -> Self {
        Self::of(Vec::new())
    }
}

impl PromptMirror {
    fn of(tokens: Vec<u32>) -> Self {
        Self {
            hash: prompt_hash(&tokens),
            tokens: Arc::new(tokens),
        }
    }
}

/// FNV-1a over the length and the token words: what the worker checks its
/// rebuilt prompt against.
pub(in crate::model) fn prompt_hash(tokens: &[u32]) -> u32 {
    let step = |h: u32, w: u32| (h ^ w).wrapping_mul(0x0100_0193);
    tokens
        .iter()
        .fold(step(0x811c_9dc5, tokens.len() as u32), |h, &t| step(h, t))
}

/// Tokens `last` and `next` share from position 0.
pub(super) fn common_prefix(last: &[u32], next: &[u32]) -> usize {
    last.iter().zip(next).take_while(|(a, b)| a == b).count()
}

/// The mirrors of every slot, by the head's slot number. A chunk command of
/// the prompt a slot already holds shares all of it; the next turn of a
/// conversation shares its previous prompt, in whichever slot that ran;
/// interleaved prefills of several slots each keep their own.
#[derive(Default)]
struct PromptMirrors(Vec<PromptMirror>);

/// The words that announce a prompt: its slot, the slot whose mirror it
/// extends, the tokens it shares with that mirror, and its hash.
pub(super) const ANNOUNCE_WORDS: usize = 4;

/// Slots a mirror table may grow to: far above any `--max-num-seqs`, so a
/// word that is not a slot fails instead of allocating.
const MAX_MIRROR_SLOTS: usize = 4096;

impl PromptMirrors {
    fn slot(&mut self, slot: usize) -> &mut PromptMirror {
        if self.0.len() <= slot {
            self.0.resize_with(slot + 1, PromptMirror::default);
        }
        &mut self.0[slot]
    }

    /// Head: make `next` the prompt of `slot`. Returns the announce words and
    /// where the unsent suffix of `next` starts. The base is the mirror that
    /// shares the most tokens, `slot`'s own on a tie.
    fn advance(&mut self, slot: usize, next: &[u32]) -> ([u32; ANNOUNCE_WORDS], usize) {
        let own = common_prefix(&self.slot(slot).tokens, next);
        let shared = |m: &PromptMirror| common_prefix(&m.tokens, next);
        let (from, common) = (0..self.0.len())
            .map(|i| (i, shared(&self.0[i])))
            .fold((slot, own), |best, x| if x.1 > best.1 { x } else { best });
        let base = self.0[from].clone();
        let mirror = self.slot(slot);
        *mirror = if common == next.len() && base.tokens.len() == common {
            base
        } else {
            PromptMirror::of(next.to_vec())
        };
        (
            [slot as u32, from as u32, common as u32, mirror.hash],
            common,
        )
    }

    /// Worker: how many tokens of a `full_len` prompt the head still sends
    /// after the announce `words`.
    fn suffix_len(&self, words: [u32; ANNOUNCE_WORDS], full_len: usize) -> Result<usize> {
        let [slot, from, common, _] = words.map(|w| w as usize);
        let held = self.0.get(from).map_or(0, |m| m.tokens.len());
        ensure!(
            slot < MAX_MIRROR_SLOTS && common <= full_len && common <= held,
            "prompt delta: the head shares {common} tokens of slot {slot}'s {full_len}-token \
             prompt with slot {from}, where this rank holds {held} (the ranks are out of step)"
        );
        Ok(full_len - common)
    }

    /// Worker: rebuild the announced prompt from the mirror it extends and
    /// the received `suffix`, and check it against the head's hash.
    fn rebuild(&mut self, words: [u32; ANNOUNCE_WORDS], suffix: &[u32]) -> Result<Arc<Vec<u32>>> {
        let [slot, from, common, _] = words.map(|w| w as usize);
        self.suffix_len(words, common + suffix.len())?;
        let hash = words[3];
        let base = self.0.get(from).cloned().unwrap_or_default();
        let mirror = if suffix.is_empty() && base.tokens.len() == common {
            base
        } else {
            PromptMirror::of([&base.tokens[..common], suffix].concat())
        };
        ensure!(
            mirror.hash == hash,
            "prompt delta: this rank rebuilt a prompt with hash {:#010x}, the head sent \
             {hash:#010x} (the ranks are out of step)",
            mirror.hash
        );
        *self.slot(slot) = mirror.clone();
        Ok(mirror.tokens)
    }
}

impl TransformerModel {
    /// Head: send the full prompt `tokens` of the prefill command for slot
    /// `seq_id` whose length word went out just before. With the switch off
    /// this is the bulk broadcast it replaces.
    pub(in crate::model) fn ep_broadcast_prompt_dispatch(
        &self,
        seq_id: u32,
        tokens: &[u32],
    ) -> Result<()> {
        if self.comm.is_none() {
            return Ok(());
        }
        if self.warm.prompt_delta {
            let mut mirrors = self.warm.prompts.lock();
            let (words, from) = mirrors.advance(seq_id as usize, tokens);
            self.ep_broadcast_tokens(&words)?;
            if from < tokens.len() {
                self.ep_broadcast_tokens(&tokens[from..])?;
            }
        } else {
            self.ep_broadcast_tokens(tokens)?;
        }
        Ok(())
    }

    /// Worker: the `full_len`-token prompt of a prefill command.
    pub(in crate::model) fn ep_recv_prompt(&self, full_len: usize) -> Result<Arc<Vec<u32>>> {
        if self.warm.prompt_delta {
            let words = self.ep_broadcast_tokens(&[0u32; ANNOUNCE_WORDS])?;
            let words: [u32; ANNOUNCE_WORDS] = words[..].try_into()?;
            let mut mirrors = self.warm.prompts.lock();
            let suffix = match mirrors.suffix_len(words, full_len)? {
                0 => Vec::new(),
                n => self.ep_broadcast_tokens(&vec![0u32; n])?,
            };
            mirrors.rebuild(words, &suffix)
        } else {
            Ok(Arc::new(self.ep_broadcast_tokens(&vec![0u32; full_len])?))
        }
    }
}
