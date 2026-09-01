// SPDX-License-Identifier: AGPL-3.0-only

//! Target-hidden capture used to seed per-request DFlash prompt context.
//!
//! The ordinary prefill paths expose one sequence at hidden row zero. Q12
//! packs several sequences into one `[sum(proc_count), hidden]` buffer. Both
//! paths must copy every configured capture layer into the sequence-private
//! DFlash accumulator before the next target layer overwrites those rows.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layer::BatchedAttnMetadata;
use crate::traits::SequenceState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CaptureSpan {
    source_row: usize,
    position_start: usize,
    row_count: usize,
}

fn packed_capture_spans(cu_seqlens: &[i32], position_starts: &[usize]) -> Result<Vec<CaptureSpan>> {
    ensure!(
        cu_seqlens.len() == position_starts.len() + 1,
        "DFlash batched capture metadata is not aligned"
    );
    let mut spans = Vec::with_capacity(position_starts.len());
    for (index, &position_start) in position_starts.iter().enumerate() {
        let source_row = usize::try_from(cu_seqlens[index])?;
        let source_end = usize::try_from(cu_seqlens[index + 1])?;
        ensure!(
            source_end >= source_row,
            "DFlash batched capture cu_seqlens are not monotonic"
        );
        spans.push(CaptureSpan {
            source_row,
            position_start,
            row_count: source_end - source_row,
        });
    }
    Ok(spans)
}

impl TransformerModel {
    fn dflash_capture_slot(&self, layer_idx: usize) -> Option<usize> {
        if self.comm.as_ref().is_some_and(|comm| comm.rank() != 0) {
            return None;
        }
        self.dflash_capture_layers
            .iter()
            .position(|&capture_layer| capture_layer == layer_idx)
    }

    fn capture_dflash_rows(
        &self,
        seq: &mut SequenceState,
        slot_idx: usize,
        position_start: usize,
        proc_count: usize,
        source: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some(state) = seq.proposer_state.as_mut().and_then(|state| {
            state
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        }) else {
            return Ok(());
        };
        let hidden_size = self.config.hidden_size;
        let row_bytes = hidden_size * 2;
        let capture_width = self.dflash_capture_layers.len() * row_bytes;
        for row in 0..proc_count {
            let position = position_start + row;
            if position >= state.max_ctx_len {
                break;
            }
            let destination = state
                .ctx_hidden_acc
                .offset(position * capture_width + slot_idx * row_bytes);
            self.gpu.copy_d2d_async(
                source.offset(row * row_bytes),
                destination,
                row_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Capture one sequence whose processed rows start at hidden row zero.
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut SequenceState,
        layer_idx: usize,
        position_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(slot_idx) = self.dflash_capture_slot(layer_idx) else {
            return Ok(());
        };
        self.capture_dflash_rows(
            seq,
            slot_idx,
            position_start,
            proc_count,
            self.buffers.hidden_states(),
            stream,
        )
    }

    /// Capture a prompt span whose rows begin at `source_row` in a larger
    /// heterogeneous target batch.  Used by the GLM DFlash/prefill co-dispatch
    /// path; unlike the ordinary single-stream helper it does not assume row 0.
    pub(super) fn try_dflash_prefill_capture_from_row_layer(
        &self,
        seq: &mut SequenceState,
        layer_idx: usize,
        position_start: usize,
        proc_count: usize,
        source_row: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(slot_idx) = self.dflash_capture_slot(layer_idx) else {
            return Ok(());
        };
        let row_bytes = self.config.hidden_size * size_of::<u16>();
        self.capture_dflash_rows(
            seq,
            slot_idx,
            position_start,
            proc_count,
            self.buffers.hidden_states().offset(source_row * row_bytes),
            stream,
        )
    }

    /// Capture every request from Q12's packed hidden rows before the next
    /// target layer overwrites them.
    pub(super) fn try_dflash_batched_prefill_capture_layer(
        &self,
        seqs: &mut [&mut SequenceState],
        layer_idx: usize,
        meta: &BatchedAttnMetadata,
        position_starts: &[usize],
        stream: u64,
    ) -> Result<()> {
        let Some(slot_idx) = self.dflash_capture_slot(layer_idx) else {
            return Ok(());
        };
        ensure!(
            seqs.len() == position_starts.len(),
            "DFlash batched capture sequence count is not aligned"
        );
        let spans = packed_capture_spans(&meta.cu_seqlens_host, position_starts)?;
        if self.stats.once("log:dflash_batched_prefill_capture") {
            tracing::info!(
                sequences = seqs.len(),
                rows = meta.total_tokens,
                "DFlash Q12 prompt capture engaged: packed target features are copied into each request's private context"
            );
        }
        let row_bytes = self.config.hidden_size * 2;
        for (seq, span) in seqs.iter_mut().zip(spans) {
            self.capture_dflash_rows(
                seq,
                slot_idx,
                span.position_start,
                span.row_count,
                self.buffers
                    .hidden_states()
                    .offset(span.source_row * row_bytes),
                stream,
            )?;
        }
        Ok(())
    }

    /// Mark the prompt feature rows as available to DFlash after prefill.
    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty()
            || self.comm.as_ref().is_some_and(|comm| comm.rank() != 0)
        {
            return Ok(());
        }
        if let Some(state) = seq.proposer_state.as_mut().and_then(|state| {
            state
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        }) {
            let new_len = (chunk_start + proc_count).min(state.max_ctx_len);
            state.ctx_len = new_len;
            state.ctx_positions = (0..new_len).map(|position| position as i32).collect();
            // Prefill capture already populated the hidden row for the final
            // prompt token.  The first proposal receives the same row through
            // `target_hidden_stack`; appending it again would create
            // `[..., prompt_len - 1, prompt_len - 1]`, shift every subsequent
            // DFlash context slot by one, and poison acceptance after the easy
            // opening tokens.  Consume the existing one-shot append guard so
            // the first proposal only projects the rows captured above.
            state.skip_next_decode_append = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CaptureSpan, packed_capture_spans};

    #[test]
    fn packed_capture_maps_each_request_to_its_own_rows() {
        let spans = packed_capture_spans(&[0, 27, 54], &[0, 0]).unwrap();
        assert_eq!(
            spans,
            vec![
                CaptureSpan {
                    source_row: 0,
                    position_start: 0,
                    row_count: 27,
                },
                CaptureSpan {
                    source_row: 27,
                    position_start: 0,
                    row_count: 27,
                },
            ]
        );
    }

    #[test]
    fn packed_capture_preserves_ragged_absolute_positions() {
        let spans = packed_capture_spans(&[0, 3, 8], &[11, 41]).unwrap();
        assert_eq!(
            spans,
            vec![
                CaptureSpan {
                    source_row: 0,
                    position_start: 11,
                    row_count: 3,
                },
                CaptureSpan {
                    source_row: 3,
                    position_start: 41,
                    row_count: 5,
                },
            ]
        );
    }

    #[test]
    fn packed_capture_rejects_misaligned_metadata() {
        assert!(packed_capture_spans(&[0, 3], &[0, 3]).is_err());
        assert!(packed_capture_spans(&[0, -1], &[0]).is_err());
    }
}
