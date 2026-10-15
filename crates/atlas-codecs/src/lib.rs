// SPDX-License-Identifier: AGPL-3.0-only
//! Atlas column codecs — experiment crate (plan §5 Stage 0 + minimal Stage 1).
//! Lossless composable pipelines over KV-stream byte columns.

pub mod block;
pub mod codec;

/// Explicit codec id per block header (PCND — never implicit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum CodecId {
    #[default]
    None = 0,
    DeltaLz4 = 1,
    DeltaZstd = 2,
    DoubleDeltaLz4 = 3,
    DoubleDeltaZstd = 4,
    GorillaZstd = 5,
    T64Lz4 = 6,
    Alplz4 = 7,  // stub — full catalog deferred
    FpcZstd = 8, // stub
}

/// Composable pipeline declared per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Pipeline {
    #[default]
    Raw,
    DeltaLz4,
    DeltaZstd,
    DoubleDeltaLz4,
    DoubleDeltaZstd,
    GorillaZstd,
    T64Lz4,
    AlpLz4,
    FpcZstd,
}

impl Pipeline {
    pub fn codec_id(self) -> CodecId {
        use Pipeline::*;
        match self {
            Raw => CodecId::None,
            DeltaLz4 => CodecId::DeltaLz4,
            DeltaZstd => CodecId::DeltaZstd,
            DoubleDeltaLz4 => CodecId::DoubleDeltaLz4,
            DoubleDeltaZstd => CodecId::DoubleDeltaZstd,
            GorillaZstd => CodecId::GorillaZstd,
            T64Lz4 => CodecId::T64Lz4,
            AlpLz4 => CodecId::Alplz4,
            FpcZstd => CodecId::FpcZstd,
        }
    }

    /// Encode a block; returns (compressed bytes, codec used).
    pub fn encode(&self, input: &[u8]) -> (Vec<u8>, CodecId) {
        codec::encode(*self, input)
    }

    /// Decode; input must start with block header (see block module).
    pub fn decode(input: &[u8]) -> Option<(Vec<u8>, CodecId)> {
        block::decode_with_header(input)
    }
}

#[cfg(test)]
mod roundtrip_tests {
    use super::*;
    #[test]
    fn core_roundtrips() {
        let raw = b"hello world synthetic kv block with float-like sequence";
        for p in [
            Pipeline::DeltaZstd,
            Pipeline::GorillaZstd,
            Pipeline::DoubleDeltaLz4,
            Pipeline::T64Lz4,
        ] {
            let (enc, cid) = p.encode(raw.as_slice());
            let full = block::encode_with_header(cid, raw.len(), &enc);
            let (decoded, _) = Pipeline::decode(&full).unwrap();
            assert_eq!(&decoded[..], raw.as_slice(), "fail {:?}", p);
        }
    }
}
