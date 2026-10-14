// SPDX-License-Identifier: AGPL-3.0-only
//! Request identity transport is independent of numerical prefill payloads.
use super::types::TransformerModel;
use crate::traits::SequenceState;
use anyhow::{Result, ensure};

/// Binds the addressed worker sequence's request identity. Upstream's
/// `0xFFFF_FFF8` is `EP_CMD_CACHE_SEQUENCE` (`trait_impl::finish_leaf`).
pub(in crate::model) const EP_CMD_REQUEST_ID: u32 = 0xFFFF_FFF9;
const MAX_ID_BYTES: usize = 16 * 1024;
const NO_REQUEST_ID: u32 = u32::MAX;

fn encode(request_id: Option<&str>) -> Result<(u32, Vec<u32>)> {
    let Some(request_id) = request_id else {
        return Ok((NO_REQUEST_ID, vec![]));
    };
    let bytes = request_id.as_bytes();
    ensure!(
        bytes.len() <= MAX_ID_BYTES,
        "request-id wire payload exceeds limit"
    );
    let words = bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect();
    Ok((u32::try_from(bytes.len())?, words))
}

fn decode(length: u32, words: &[u32]) -> Result<Option<String>> {
    if length == NO_REQUEST_ID {
        ensure!(words.is_empty(), "absent request-id must have no payload");
        return Ok(None);
    }
    let length = length as usize;
    ensure!(
        length <= MAX_ID_BYTES,
        "request-id wire payload exceeds limit"
    );
    ensure!(
        words.len() == length.div_ceil(4),
        "request-id wire extent mismatch"
    );
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    ensure!(
        bytes[length..].iter().all(|byte| *byte == 0),
        "request-id wire padding is nonzero"
    );
    Ok(Some(String::from_utf8(bytes[..length].to_vec())?))
}

impl TransformerModel {
    pub(super) fn send_request_id(&self, slot: u32, request_id: Option<&str>) -> Result<()> {
        // Same gate as `ep_broadcast_seq_and_cmd`: no worker loop, no packet.
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        let (length, words) = encode(request_id)?;
        ensure!(
            words.len() * 4 <= self.buffers.sizes().scratch,
            "request-id wire payload exceeds scratch capacity"
        );
        self.ep_broadcast_seq_and_cmd(slot, EP_CMD_REQUEST_ID, self.ep_protocol_v2)?;
        self.ep_broadcast_u32(length)?;
        if !words.is_empty() {
            self.ep_broadcast_tokens(&words)?;
        }
        Ok(())
    }

    pub(super) fn receive_request_id(&self, seq: &mut SequenceState) -> Result<()> {
        let length = self.ep_broadcast_u32(0)?;
        ensure!(
            length == NO_REQUEST_ID || length as usize <= MAX_ID_BYTES,
            "request-id wire payload exceeds limit"
        );
        let count = if length == NO_REQUEST_ID {
            0
        } else {
            (length as usize).div_ceil(4)
        };
        ensure!(
            count * 4 <= self.buffers.sizes().scratch,
            "request-id wire payload exceeds scratch capacity"
        );
        let words = if count == 0 {
            vec![]
        } else {
            self.ep_broadcast_tokens(&vec![0; count])?
        };
        seq.request_id = decode(length, &words)?.map(Into::into);
        Ok(())
    }
}

#[cfg(test)]
#[path = "request_identity_tests.rs"]
mod tests;
