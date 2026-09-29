// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2b: compute the effective processing range within this chunk
//! after Marconi/prefix-cache skip. May early-return when the entire
//! chunk is covered by cache and is_last_chunk == false.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

pub(in crate::model) enum ProcRange {
    /// Process this many tokens; phase 3+ run normally.
    Compute {
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
    },
    /// Whole chunk cached and not last — caller returns immediately.
    EarlyReturn(DevicePtr),
}

impl TransformerModel {
    pub(in crate::model) fn prefill_b_proc_range(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        kv_write_start: usize,
        marconi_skip: bool,
        hidden_dst: DevicePtr,
        stream: u64,
    ) -> Result<ProcRange> {
        let h = self.config.hidden_size;
        // DEFECT 1 fix: re-embed into the caller-provided per-stream hidden slot
        // (the proc_off slot in the kernel-batched path), NOT the offset-0 base
        // buffer. Single-stream / per-stream-fallback callers pass the base
        // (`buffers.hidden_states()`) ⇒ byte-identical.
        let hidden = hidden_dst;

        // Stale-V prefix-cache fix: track the contiguous prefix (from token 0)
        // whose paged K/V is guaranteed fully written for this sequence.
        //   • At chunk 0 the reused prefix-cache match (`kv_write_start` tokens
        //     when `marconi_skip`) is the only pre-validated KV; reset the
        //     accumulator to it (0 on a cold prefill).
        //   • Each chunk that runs the real prefill path writes KV for
        //     `[effective_seq_len_start, effective_seq_len_start + proc_count)`,
        //     extending the contiguous valid prefix to its end.
        //   • The `proc_count == 1` last-chunk decode shortcut writes only the
        //     single re-embedded last token and therefore does NOT extend the
        //     contiguous valid prefix past `kv_write_start` — any trailing
        //     complete blocks it "treats as cached" but never wrote must not be
        //     inserted into the prefix cache (handled by the insert-side cap in
        //     `finalize_last`/`save_checkpoint`).
        if chunk_start == 0 {
            seq.kv_valid_tokens = if marconi_skip { kv_write_start } else { 0 };
        }

        if marconi_skip && kv_write_start > chunk_start {
            // Skip cached tokens within this chunk
            let skip_in_chunk = (kv_write_start - chunk_start).min(chunk_len);
            if skip_in_chunk >= chunk_len {
                // Entire chunk is cached — skip computation, just update state.
                // Don't add tokens here; the normal path at step 5 handles it.
                seq.seq_len = chunk_start + chunk_len;
                if is_last_chunk {
                    // Need to process at least the last token for logits.
                    // Re-embed just the last token into hidden[0].
                    let last_tok = tokens[chunk_start + chunk_len - 1];
                    // SAFETY: 4 == `size_of::<u32>()` bytes over the single,
                    // fully initialised `last_tok` local on the line above (an
                    // in-bounds copy out of `tokens`).
                    let last_tok_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(&last_tok as *const u32 as *const u8, 4)
                    };
                    let token_id_dev = self.buffers.scratch();
                    self.gpu
                        .copy_h2d_async(last_tok_bytes, token_id_dev, stream)?;
                    if self.has_ngram_embedding() {
                        let last = chunk_start + chunk_len;
                        let cs = last.saturating_sub(self.ngram_lookbehind() + 1);
                        self.embed_tokens_fused(&tokens[cs..last], 1, hidden, stream)?;
                    } else {
                        ops::batched_embed(
                            self.gpu.as_ref(),
                            self.batched_embed_kernel,
                            token_id_dev,
                            self.embed_tokens.weight,
                            hidden,
                            1,
                            h as u32,
                            stream,
                        )?;
                    }
                    self.scale_embeddings(hidden, 1usize, stream)?;
                    Ok(ProcRange::Compute {
                        proc_start: chunk_start + chunk_len - 1,
                        proc_count: 1,
                        effective_seq_len_start: chunk_start + chunk_len - 1,
                    })
                } else {
                    Ok(ProcRange::EarlyReturn(DevicePtr::NULL))
                }
            } else {
                // Re-embed only uncached portion
                let uncached_start = chunk_start + skip_in_chunk;
                let uncached_count = chunk_len - skip_in_chunk;
                let uncached_tokens = &tokens[uncached_start..uncached_start + uncached_count];
                // SAFETY: `uncached_tokens` is sliced on the line above with an
                // END bound of `uncached_start + uncached_count`, so its length
                // IS `uncached_count` (an out-of-range range panics in that
                // slice index first) and the byte length is
                // `uncached_tokens.len() * size_of::<u32>()` over a live `&[u32]`.
                let token_ids_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        uncached_tokens.as_ptr() as *const u8,
                        uncached_count * 4,
                    )
                };
                let token_ids_dev = self.buffers.scratch();
                self.gpu
                    .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
                if self.has_ngram_embedding() {
                    let cs = uncached_start.saturating_sub(self.ngram_lookbehind());
                    self.embed_tokens_fused(
                        &tokens[cs..uncached_start + uncached_count],
                        uncached_count,
                        hidden,
                        stream,
                    )?;
                } else {
                    ops::batched_embed(
                        self.gpu.as_ref(),
                        self.batched_embed_kernel,
                        token_ids_dev,
                        self.embed_tokens.weight,
                        hidden,
                        uncached_count as u32,
                        h as u32,
                        stream,
                    )?;
                }
                self.scale_embeddings(hidden, uncached_count, stream)?;
                // Real prefill path: KV written for [uncached_start, end).
                seq.kv_valid_tokens = seq.kv_valid_tokens.max(uncached_start + uncached_count);
                Ok(ProcRange::Compute {
                    proc_start: uncached_start,
                    proc_count: uncached_count,
                    effective_seq_len_start: uncached_start,
                })
            }
        } else {
            // Full-chunk prefill path: KV written for [chunk_start, end).
            seq.kv_valid_tokens = seq.kv_valid_tokens.max(chunk_start + chunk_len);
            Ok(ProcRange::Compute {
                proc_start: chunk_start,
                proc_count: chunk_len,
                effective_seq_len_start: chunk_start,
            })
        }
    }
}

/// The per-pass KV write floor the prefill layers take: rows below it keep
/// the K/V already in the pass's slots. On a Marconi warm hit the pass
/// replays SSM state over positions whose K/V live in shared prefix-cache
/// blocks (below `cached_prefix_tokens`), which must not be rewritten.
pub(in crate::model) fn layer_kv_write_floor(
    marconi_skip: bool,
    cached_prefix_tokens: usize,
    effective_seq_len_start: usize,
    proc_count: usize,
    kv_write_start: usize,
) -> usize {
    if marconi_skip {
        cached_prefix_tokens
            .saturating_sub(effective_seq_len_start)
            .min(proc_count)
    } else {
        kv_write_start
    }
}

#[cfg(test)]
mod floor_tests {
    use super::layer_kv_write_floor;
    use crate::model::glm_fused_chunk::prefix::ride_is_cold;

    #[test]
    fn the_floor_covers_the_cached_rows_of_a_warm_pass() {
        assert_eq!(layer_kv_write_floor(false, 0, 8_192, 4_096, 0), 0);
        assert_eq!(
            layer_kv_write_floor(true, 44_992, 44_928, 1_072, 44_928),
            64
        );
        assert_eq!(layer_kv_write_floor(true, 44_992, 40_960, 64, 44_928), 64);
        assert_eq!(layer_kv_write_floor(true, 44_992, 45_888, 112, 44_928), 0);
    }

    /// A pass the fused gate admits (chunk 1+, inheriting chunk 0's decision
    /// as `prefix_lookup` hands it on) computes every row and takes no floor,
    /// which the passenger layer path relies on.
    #[test]
    fn an_admitted_pass_is_computed_whole_with_no_floor() {
        let points = [0usize, 64, 4_032, 4_096, 4_160, 8_192, 12_288];
        for start in [64usize, 4_096, 8_192] {
            for len in [2usize, 64, 4_096] {
                for &skip_to in &points {
                    for &cached in points.iter().filter(|&&c| c >= skip_to) {
                        if !ride_is_cold(start, start, skip_to, cached) {
                            continue;
                        }
                        let marconi_skip = skip_to > 0;
                        let kv_write_start = if marconi_skip { skip_to } else { 0 };
                        // `prefill_b_proc_range` skips rows only past the start.
                        assert!(!(marconi_skip && kv_write_start > start));
                        let floor =
                            layer_kv_write_floor(marconi_skip, cached, start, len, kv_write_start);
                        assert_eq!(floor, 0, "start {start} skip {skip_to} cached {cached}");
                    }
                }
            }
        }
    }
}
