// SPDX-License-Identifier: AGPL-3.0-only

//! The arena zero and chunk embed of a prefill chunk, and the chunk that
//! needs neither (`ATLAS_GLM_WARM_SKIP_CACHED=1`, default off).
//!
//! A chunk that lies entirely below the restore depth of a warm prefill
//! computes nothing: it reserves its blocks, appends its tokens and returns
//! (`ProcRange::EarlyReturn`). It still zeroed the whole arena and embedded
//! its rows first, because both come before the prefix lookup that decides
//! it. A 45K-token warm turn at 8K-row chunks has five such chunks, a
//! 512K-token one over sixty, and each costs a full arena zero (about 17 ms
//! on GB10) plus an embed of up to a chunk of rows.
//!
//! With the switch, in a multi-rank world, the lookup runs first and a chunk
//! with nothing to compute skips both. Nothing reads what they would have
//! written: no layer runs in that chunk, and the next chunk that computes
//! zeroes the whole arena and embeds its own rows before its first kernel,
//! as every multi-rank chunk does. So every pass that runs sees the same
//! arena, byte for byte, as without the switch; only a decode step of
//! another sequence interleaved between two such chunks finds other
//! leftovers in the arena, as it does after any other request's chunk.
//! The lookup itself touches no arena buffer (its collectives use the
//! command word buffer, its restore the SSM pools), so moving it ahead of
//! the zero changes nothing it reads or writes.
//!
//! A chunk that computes is unchanged: full zero, full embed, then the
//! re-embed of its uncached rows (`proc_range`).

use std::time::{Duration, Instant};

use anyhow::Result;

use super::super::super::types::TransformerModel;
use super::super::super::warm_turn::{PHASES, RequestShape};
use crate::traits::SequenceState;

/// Whether chunk `[start, start + len)` computes nothing: it is not the last
/// chunk (whose final row always runs, for the logits) and a snapshot or
/// cache skip (`skip`) covers it through `skip_to`. This is the condition
/// under which `prefill_b_proc_range` returns `EarlyReturn`.
pub(super) fn fully_cached(
    skip: bool,
    skip_to: usize,
    start: usize,
    len: usize,
    is_last: bool,
) -> bool {
    skip && !is_last && skip_to >= start + len
}

impl TransformerModel {
    /// Whether a chunk looks its prefix up before it zeroes and embeds: the
    /// switch, in a multi-rank world. A single rank zeroes once per request,
    /// at chunk 0, and later chunks find the rows earlier ones embedded; a
    /// chunk 0 that skipped would change what they find, so it keeps its
    /// order.
    pub(super) fn warm_lookup_first(&self) -> bool {
        self.warm.skip_cached && self.comm.is_some()
    }

    /// Zero the arena and embed the chunk; returns the two host spans.
    ///
    /// EP=2: zero ALL buffers on every chunk (NCCL defense-in-depth).
    /// EP=1, first chunk (chunk_start==0): zero only buffers whose stale
    /// contents can affect prefill; the remaining scratch buffers are
    /// overwritten before read by embedding + layer forward.
    /// EP=1, subsequent chunks: skip zeroing — buffers are overwritten by embedding
    /// + layer forward before read. Saves 7 memsets × (chunks-1) per prefill.
    pub(super) fn prefill_b_zero_and_embed(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<[Duration; 2]> {
        let t0 = Instant::now();
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }
        self.warm_trace_sync(stream)?;
        let zero = t0.elapsed();
        // ── Phase 1+1b: embed chunk + vision pad overlay ──
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;
        self.warm_trace_sync(stream)?;
        Ok([zero, t0.elapsed() - zero])
    }

    /// `ATLAS_GLM_WARM_TRACE`: drain `stream`, so the span being timed holds
    /// the device time of its own launches. Nothing with the switch off.
    pub(super) fn warm_trace_sync(&self, stream: u64) -> Result<()> {
        if self.warm.trace {
            self.gpu.synchronize(stream)?;
        }
        Ok(())
    }

    /// `ATLAS_GLM_WARM_TRACE`: add a chunk that began at `began` and computed
    /// `rows` rows to its request's trace, and log the request's line after
    /// the last chunk. `pre` is the zero and embed spans; `marks` are the
    /// lookup span and the times since `began` at which the lookup (with a
    /// late zero and embed), the block reservation, the metadata and the
    /// forward were done. What follows the forward is the finish.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn warm_trace_chunk(
        &self,
        seq: &SequenceState,
        prompt: usize,
        began: Instant,
        rows: usize,
        pre: [Duration; 2],
        marks: [Duration; 5],
        is_last: bool,
        stream: u64,
    ) -> Result<()> {
        if !self.warm.trace {
            return Ok(());
        }
        self.gpu.synchronize(stream)?;
        let [lookup, looked, reserved, meta, forward] = marks;
        let spans: [Duration; PHASES.len() - 1] = [
            pre[0],
            pre[1],
            lookup,
            reserved.saturating_sub(looked),
            meta.saturating_sub(reserved),
            forward.saturating_sub(meta),
            began.elapsed().saturating_sub(forward),
        ];
        let shape = is_last.then(|| RequestShape {
            rank: self.comm.as_ref().map_or(0, |c| c.rank()),
            prompt,
            matched: seq.cached_prefix_tokens,
            restored: seq.marconi_skip_to,
        });
        if let Some(line) = self
            .warm
            .note_chunk(seq.slot_idx, began, rows, spans, shape)
        {
            tracing::info!("{line}");
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "warm_tests.rs"]
mod tests;
