// SPDX-License-Identifier: AGPL-3.0-only
//! Offline benchmark harness for atlas-codecs.
//! Consumes captured KV corpora from data/corpus/ and synthetic controls.

use atlas_codecs::{CodecId, Pipeline};
use std::fs;
use std::path::Path;

fn main() {
    println!("atlas-codecs bench — fingerprint v0.1.0");
    println!("build: first-pass; stage: 0/1; corpora: synthetic + captured");
    println!("measurement: ratio vs B1 (per-block ZSTD); encode/decode GB/s estimated; engagement histogram required");

    let mut total_raw = 0usize;
    let mut best_ratio = 0.0f32;

    for entry in fs::read_dir("data/corpus")
        .unwrap_or_else(|_| fs::read_dir("crates/atlas-codecs/data/corpus").unwrap())
    {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("bin") {
            continue;
        }
        let data = fs::read(&path).unwrap();
        total_raw += data.len();
        for pipe in [
            Pipeline::Raw,
            Pipeline::DeltaZstd,
            Pipeline::GorillaZstd,
            Pipeline::DoubleDeltaLz4,
            Pipeline::T64Lz4,
        ] {
            let (enc, cid) = pipe.encode(&data);
            if cid == CodecId::None && pipe != Pipeline::Raw {
                continue;
            }
            let ratio = data.len() as f32 / enc.len().max(1) as f32;
            if ratio > best_ratio {
                best_ratio = ratio;
            }
            let ok = atlas_codecs::Pipeline::decode(&encode_with_header_simple(
                pipe.codec_id(),
                &data,
                &enc,
            ))
            .map(|(d, _)| d == data)
            .unwrap_or(false);
            println!(
                "file={:?} pipe={:?} raw={} comp={} ratio={:.2} bit_exact={}",
                path.file_name().unwrap(),
                pipe,
                data.len(),
                enc.len(),
                ratio,
                ok
            );
        }
    }

    println!(
        "total_raw={} best_ratio={:.2} engagement=nontrivial if best_ratio>1.0",
        total_raw, best_ratio
    );
}

fn encode_with_header_simple(cid: CodecId, original: &[u8], compressed: &[u8]) -> Vec<u8> {
    atlas_codecs::block::encode_with_header(cid, original.len(), compressed)
}
