// SPDX-License-Identifier: AGPL-3.0-only

//! Packed rank-symmetric protocol for GLM native multi-sequence prefill.

use anyhow::{Context, Result, ensure};

use super::types::TransformerModel;
use crate::traits::{EP_PREFILL_BATCH_CMD, Model, PrefillSlice, SequenceState};

const PAYLOAD_VERSION: u32 = 1;
const RECORD_HEADER_WORDS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrefillRecord {
    seq_id: usize,
    chunk_start: usize,
    chunk_len: usize,
    tokens: Vec<u32>,
}

fn encode_records(records: &[PrefillRecord]) -> Result<Vec<u32>> {
    ensure!(!records.is_empty(), "EP prefill batch cannot be empty");
    let mut payload = Vec::new();
    payload.push(PAYLOAD_VERSION);
    payload.push(u32::try_from(records.len()).context("EP prefill stream count exceeds u32")?);
    let mut previous = None;
    for record in records {
        if let Some(previous) = previous {
            ensure!(
                record.seq_id > previous,
                "EP prefill stream slots must be strictly increasing"
            );
        }
        ensure!(
            record.chunk_len > 0 && record.chunk_start + record.chunk_len <= record.tokens.len(),
            "EP prefill record has invalid chunk bounds"
        );
        payload.extend([
            u32::try_from(record.seq_id).context("EP prefill slot exceeds u32")?,
            u32::try_from(record.chunk_start).context("EP prefill offset exceeds u32")?,
            u32::try_from(record.chunk_len).context("EP prefill chunk exceeds u32")?,
            u32::try_from(record.tokens.len()).context("EP prefill prompt exceeds u32")?,
        ]);
        payload.extend_from_slice(&record.tokens);
        previous = Some(record.seq_id);
    }
    Ok(payload)
}

fn decode_records(payload: &[u32]) -> Result<Vec<PrefillRecord>> {
    ensure!(payload.len() >= 2, "EP prefill payload is truncated");
    ensure!(
        payload[0] == PAYLOAD_VERSION,
        "unsupported EP prefill payload version {}",
        payload[0]
    );
    let count = payload[1] as usize;
    ensure!(count > 0, "EP prefill payload cannot contain zero streams");
    let mut cursor = 2usize;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        ensure!(
            cursor + RECORD_HEADER_WORDS <= payload.len(),
            "EP prefill record header is truncated"
        );
        let seq_id = payload[cursor] as usize;
        let chunk_start = payload[cursor + 1] as usize;
        let chunk_len = payload[cursor + 2] as usize;
        let full_len = payload[cursor + 3] as usize;
        cursor += RECORD_HEADER_WORDS;
        ensure!(
            cursor + full_len <= payload.len(),
            "EP prefill token payload is truncated"
        );
        let tokens = payload[cursor..cursor + full_len].to_vec();
        cursor += full_len;
        records.push(PrefillRecord {
            seq_id,
            chunk_start,
            chunk_len,
            tokens,
        });
    }
    ensure!(
        cursor == payload.len(),
        "EP prefill payload has trailing words"
    );
    // Reuse the encoder's ordering/bounds validation without retaining its copy.
    let _ = encode_records(&records)?;
    Ok(records)
}

impl TransformerModel {
    pub(super) fn ep_broadcast_prefill_batch_dispatch(
        &self,
        streams: &[PrefillSlice<'_>],
    ) -> Result<()> {
        ensure!(
            self.config.model_type == "glm5_next",
            "native EP batched prefill is GLM-5.3-only"
        );
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        let records = streams
            .iter()
            .map(|slice| PrefillRecord {
                seq_id: slice.seq.slot_idx,
                chunk_start: slice.chunk_start,
                chunk_len: slice.chunk_len,
                tokens: slice.prompt_tokens.to_vec(),
            })
            .collect::<Vec<_>>();
        let payload = encode_records(&records)?;
        self.ep_broadcast_seq_and_cmd(0, EP_PREFILL_BATCH_CMD, self.ep_protocol_v2)?;
        self.ep_broadcast_u32(
            u32::try_from(payload.len()).context("EP prefill payload exceeds u32")?,
        )?;
        let _ = self.ep_broadcast_tokens(&payload)?;
        Ok(())
    }

    pub(super) fn ep_worker_prefill_batch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let payload_len = self.ep_broadcast_u32(0)? as usize;
        ensure!(payload_len > 0, "EP prefill payload length is zero");
        let payload = self.ep_broadcast_tokens(&vec![0u32; payload_len])?;
        let records = decode_records(&payload)?;

        let mut slices = Vec::with_capacity(records.len());
        let mut record_index = 0usize;
        for (slot_idx, slot) in slots.iter_mut().enumerate() {
            if record_index >= records.len() || records[record_index].seq_id != slot_idx {
                continue;
            }
            let record = &records[record_index];
            let seq = slot.as_mut().with_context(|| {
                format!("EP prefill batch addressed unallocated slot {slot_idx}")
            })?;
            slices.push(PrefillSlice {
                prompt_tokens: &record.tokens,
                seq,
                chunk_start: record.chunk_start,
                chunk_len: record.chunk_len,
                is_last_chunk: record.chunk_start + record.chunk_len >= record.tokens.len(),
            });
            record_index += 1;
        }
        ensure!(
            record_index == records.len(),
            "EP prefill batch contains a slot outside worker capacity"
        );
        let _ = Model::prefill_batch_chunk(self, &mut slices, self.gpu.default_stream())?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records() -> Vec<PrefillRecord> {
        vec![
            PrefillRecord {
                seq_id: 1,
                chunk_start: 0,
                chunk_len: 3,
                tokens: vec![10, 11, 12, 13],
            },
            PrefillRecord {
                seq_id: 3,
                chunk_start: 4,
                chunk_len: 2,
                tokens: vec![20, 21, 22, 23, 24, 25],
            },
        ]
    }

    #[test]
    fn packed_prefill_payload_round_trips_ragged_offsets() {
        let expected = records();
        let encoded = encode_records(&expected).unwrap();
        assert_eq!(decode_records(&encoded).unwrap(), expected);
    }

    #[test]
    fn packed_prefill_payload_rejects_duplicate_or_reordered_slots() {
        let mut bad = records();
        bad[1].seq_id = 1;
        assert!(encode_records(&bad).is_err());
    }

    #[test]
    fn packed_prefill_payload_rejects_truncation() {
        let mut encoded = encode_records(&records()).unwrap();
        encoded.pop();
        assert!(decode_records(&encoded).is_err());
    }
}
