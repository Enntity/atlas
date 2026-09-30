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
//!    and a read of `4 N` bytes on the worker].
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
//! `ATLAS_GLM_WARM_SKIP_CACHED` adds no command and no collective: a rank
//! without it only does the work the other skips.

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// The warm-turn switches and the state they keep, one per model.
pub(in crate::model) struct WarmTurn {
    /// `ATLAS_GLM_WARM_SKIP_CACHED=1`: a chunk that computes nothing does not
    /// zero the arena or embed (`prefill_b::warm`).
    pub(in crate::model) skip_cached: bool,
}

impl WarmTurn {
    pub(in crate::model) fn from_env() -> Self {
        Self {
            skip_cached: env_on("ATLAS_GLM_WARM_SKIP_CACHED"),
        }
    }
}
