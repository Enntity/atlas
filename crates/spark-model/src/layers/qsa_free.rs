// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence lifecycle of the QSA indexer carry: allocation, growth and
//! release, kept TOGETHER because the pair is the invariant that matters
//! (they drifted apart once already: alloc existed, free did not). Split from
//! `qsa.rs` for the 500-LoC cap. See `TransformerLayer::free_state` for why
//! release exists at all: `DevicePtr` has no `Drop` and the backend sweeps
//! only at process exit.
//!
//! The carry grows with the sequence instead of being sized for
//! `--max-seq-len` up front. At 262144 the up-front size was 80 MB per layer
//! and sequence (12 layers x 4 sequences = 3.8 GB) that no KV budget counted,
//! taken from the host's headroom the moment a request arrived. Growth copies
//! the live prefix into a larger buffer: the bytes every kernel reads are the
//! same, so the numerics are untouched.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{QsaIndexer, QsaSeqState};

/// Growth granule in tokens (a multiple of every pooling ratio in use).
const QSA_GRANULE: usize = 4096;

impl QsaIndexer {
    /// Device bytes one token costs this layer's carry: its raw key plus its
    /// share of a pooled block key. What the KV budget charges per token.
    pub fn bytes_per_token(&self) -> usize {
        let hd = self.hd as usize;
        hd * 2 + (hd * 2).div_ceil(self.ratio as usize)
    }

    /// The most positions a carry can hold (the served context).
    pub fn capacity(&self) -> usize {
        self.max_tokens
    }

    /// Free this sequence's buffers. Zeroed so a second teardown is a no-op.
    /// See `TransformerLayer::free_state`.
    pub fn free_seq_state(&self, st: &mut QsaSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [&mut st.raw_keys, &mut st.block_keys] {
            if p.0 != 0 {
                gpu.free(*p)?;
                *p = DevicePtr(0);
            }
        }
        st.cap = 0;
        Ok(())
    }

    /// An empty carry; [`Self::reserve`] gives it room.
    pub fn new_seq_state(&self, _gpu: &dyn GpuBackend) -> Result<QsaSeqState> {
        Ok(QsaSeqState {
            ingested: 0,
            pooled: 0,
            table_len: 0,
            cap: 0,
            raw_keys: DevicePtr(0),
            block_keys: DevicePtr(0),
        })
    }

    /// Make room for `tokens` keys (positions `0..tokens`). Grows by at least
    /// half again, in `QSA_GRANULE` steps, up to the served capacity, and
    /// keeps the ingested keys and pooled blocks. Never inside a graph
    /// capture: the buffers move.
    pub fn reserve(
        &self,
        st: &mut QsaSeqState,
        tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if tokens <= st.cap {
            return Ok(());
        }
        anyhow::ensure!(
            tokens <= self.max_tokens,
            "QSA: {tokens} tokens exceeds the indexer capacity {} — it derives \
             from --max-seq-len (ATLAS_QSA_MAX_TOKENS overrides)",
            self.max_tokens
        );
        anyhow::ensure!(
            !gpu.stream_is_capturing(stream),
            "QSA: carry growth to {tokens} tokens inside a graph capture"
        );
        let cap = tokens
            .max(st.cap + st.cap / 2)
            .next_multiple_of(QSA_GRANULE)
            .min(self.max_tokens);
        let hd = self.hd as usize;
        let ratio = self.ratio as usize;
        let raw = gpu.alloc(cap * hd * 2)?;
        let block = match gpu.alloc((cap / ratio).max(1) * hd * 2) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(raw);
                return Err(e);
            }
        };
        if st.cap > 0 {
            let (ingested, pooled) = (st.ingested.min(st.cap), st.pooled.min(st.cap / ratio));
            if ingested > 0 {
                gpu.copy_d2d_async(st.raw_keys, raw, ingested * hd * 2, stream)?;
            }
            if pooled > 0 {
                gpu.copy_d2d_async(st.block_keys, block, pooled * hd * 2, stream)?;
            }
            // The copies (and any reader queued before them) must finish
            // before the old buffers can be released.
            gpu.synchronize(stream)?;
            self.free_seq_state(st, gpu)?;
        }
        st.raw_keys = raw;
        st.block_keys = block;
        st.cap = cap;
        Ok(())
    }
}

#[cfg(test)]
#[path = "qsa_free_tests.rs"]
mod tests;
