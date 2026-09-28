// SPDX-License-Identifier: AGPL-3.0-only
//! Complete cold F0, no selected scheduler admission or process containment.
use super::*;
use spark_runtime::gpu::DevicePtr;

impl TransformerModel {
    fn paired_cold_extent(&self, start: usize, rows: usize, full: usize) -> Result<()> {
        let bytes = full.checked_mul(4).context("paired F0 payload overflow")?;
        ensure!(
            start == 0
                && rows == full
                && (2..=1024).contains(&full)
                && full <= self.buffers.max_batch_tokens()
                && full <= self.mtp_prefill_capacity
                && bytes <= self.buffers.sizes().scratch,
            "paired F0 requires bounded complete cold chunk"
        );
        Ok(())
    }
    pub(in crate::model) fn paired_validate_cold_prefill(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
    ) -> Result<()> {
        self.paired_wire_profile(0)?;
        self.paired_cold_extent(0, tokens.len(), tokens.len())?;
        self.paired_prefill_preflight(
            tokens,
            seq,
            0,
            tokens.len(),
            true,
            self.gpu.default_stream(),
        )
    }
    pub(in crate::model) fn paired_send_cold_prefill(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
    ) -> Result<DevicePtr> {
        self.paired_validate_cold_prefill(seq, tokens)?;
        let slot = u32::try_from(seq.slot_idx)?;
        let rows = u32::try_from(tokens.len())?;
        // From the first header attempt, uncertainty is terminal. T2 still owns
        // process containment and the worker's earlier preamble/slot resolution.
        (|| {
            self.ep_broadcast_disable_mtp_for_seq(slot, seq.disable_mtp)?;
            self.ep_broadcast_seq_and_cmd(slot, 0xfffffff0, true)?;
            self.ep_broadcast_u32(rows)?;
            self.ep_broadcast_u32(0)?;
            self.ep_broadcast_u32(rows)?;
            self.ep_broadcast_tokens(tokens)?;
            // Match the worker's actual two-root prefix agreement even when cold.
            self.prefill_chunk(
                tokens,
                seq,
                0,
                tokens.len(),
                true,
                self.gpu.default_stream(),
            )
        })()
        .map_err(|error| self.paired_transport_error(error))
    }
    pub(in crate::model) fn paired_receive_cold_prefill(
        &self,
        seq: &mut SequenceState,
    ) -> Result<()> {
        (|| {
            self.paired_wire_profile(1)?;
            let rows = self.ep_broadcast_u32(0)? as usize;
            let start = self.ep_broadcast_u32(0)? as usize;
            let full = self.ep_broadcast_u32(0)? as usize;
            // Validate untrusted metadata before allocating the receive vector.
            self.paired_cold_extent(start, rows, full)?;
            let tokens = self.ep_broadcast_tokens(&vec![0; full])?;
            self.paired_prefill_preflight(
                &tokens,
                seq,
                start,
                rows,
                true,
                self.gpu.default_stream(),
            )?;
            self.prefill_chunk(&tokens, seq, start, rows, true, self.gpu.default_stream())?;
            // Selected ownership does not use legacy best-effort normalization.
            Ok(())
        })()
        .map_err(|error| self.paired_transport_error(error))
    }
}
