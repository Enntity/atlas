// SPDX-License-Identifier: AGPL-3.0-only
//! Per-block header for codec-compressed KV spill blocks.
//! Format (little-endian): [codec_id: u8][uncompressed_len: u32][compressed_len: u32][payload...]

use crate::CodecId;

pub const HEADER_SIZE: usize = 9; // 1 + 4 + 4

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHeader {
    pub codec_id: CodecId,
    pub uncompressed_len: u32,
    pub compressed_len: u32,
}

impl BlockHeader {
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0] = self.codec_id as u8;
        b[1..5].copy_from_slice(&self.uncompressed_len.to_le_bytes());
        b[5..9].copy_from_slice(&self.compressed_len.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < HEADER_SIZE {
            return None;
        }
        let codec_id = match b[0] {
            0 => CodecId::None,
            1 => CodecId::DeltaLz4,
            2 => CodecId::DeltaZstd,
            3 => CodecId::DoubleDeltaLz4,
            4 => CodecId::DoubleDeltaZstd,
            5 => CodecId::GorillaZstd,
            6 => CodecId::T64Lz4,
            7 => CodecId::Alplz4,
            8 => CodecId::FpcZstd,
            _ => return None,
        };
        let uncompressed_len = u32::from_le_bytes(b[1..5].try_into().unwrap());
        let compressed_len = u32::from_le_bytes(b[5..9].try_into().unwrap());
        Some(Self {
            codec_id,
            uncompressed_len,
            compressed_len,
        })
    }
}

/// Decode payload given full buffer (header + compressed payload).
pub fn decode_with_header(input: &[u8]) -> Option<(Vec<u8>, CodecId)> {
    let hdr = BlockHeader::decode(input)?;
    let payload = input.get(HEADER_SIZE..)?;
    if payload.len() < hdr.compressed_len as usize {
        return None;
    }
    let compressed = &payload[..hdr.compressed_len as usize];
    let decoded = if hdr.codec_id == CodecId::None {
        compressed.to_vec()
    } else {
        // Full pipeline: prep inverse + backend inverse with known output size.
        let prepped =
            crate::codec::back_inverse(crate::codec::id_to_pipe(hdr.codec_id)?, compressed)?;
        crate::codec::unprep(crate::codec::id_to_pipe(hdr.codec_id)?, &prepped)
    };
    // For Lz4-backed pipelines, the prepped bytes are the uncompressed prepped stream;
    // unprep yields final. The above already handles it. If only backend decompression
    // was needed (no prep), we'd need out_len; the pipeline structure handles it.
    Some((decoded, hdr.codec_id))
}

/// Build header + payload for writing.
pub fn encode_with_header(cid: CodecId, uncompressed: usize, compressed: &[u8]) -> Vec<u8> {
    let hdr = BlockHeader {
        codec_id: cid,
        uncompressed_len: uncompressed as u32,
        compressed_len: compressed.len() as u32,
    };
    let mut out = Vec::with_capacity(HEADER_SIZE + compressed.len());
    out.extend_from_slice(&hdr.encode());
    out.extend_from_slice(compressed);
    out
}
