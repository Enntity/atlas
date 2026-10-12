// SPDX-License-Identifier: AGPL-3.0-only
//! Diagnostic-only C1 local o_proj binding capture at weight load.
//!
//! `ATLAS_LOCAL_O_PROJ_BINDING_DIR` plus an explicit
//! `ATLAS_LOCAL_O_PROJ_BINDING_LAYERS=0` gate every write; with the
//! directory unset there is zero behavioral difference. The emitted BF16
//! weight is the **common comparison space** for
//! `atlas-local-o-proj-binding-v1`: the resident EXL3 shard is reconstructed
//! (`reference_reconstruct`, then round-to-nearest-even BF16), mirroring
//! vLLM's load-time EXL3 dequant on the reference engine. These bytes are
//! not claimed to be the exact bits the native GEMM multiplies — that claim
//! is what the self-anchor gate tests at GPU time; the actual resident
//! quantized bytes are preserved verbatim in the `resident` sub-record.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::{
    Exl3LinearView, reference_reconstruct,
    tp::{Exl3DeviceShard, OwnedExl3Shard, TpShardPlan},
};
use std::{
    ffi::OsString,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

pub(crate) const DIR_ENV: &str = "ATLAS_LOCAL_O_PROJ_BINDING_DIR";
pub(crate) const LAYERS_ENV: &str = "ATLAS_LOCAL_O_PROJ_BINDING_LAYERS";
pub(crate) const LOAD_SCHEMA: &str = "atlas-local-o-proj-load-v1";
pub(crate) const BINDING_SCHEMA: &str = "atlas-local-o-proj-binding-v1";
const RESIDENT_FORM: &str = "exl3-trellis-mul1";
const RESIDENT_DTYPE: &str = "exl3-shard-pack-v1";
const DEQUANT_RECIPE: &str = "exl3-mul1-reference-reconstruct-rne-bf16";

pub(crate) fn weight_file(rank: usize) -> String {
    format!("o-proj-weight-L0-rank{rank}.npy")
}

pub(crate) fn resident_file(rank: usize) -> String {
    format!("o-proj-resident-L0-rank{rank}.bin")
}

pub(crate) fn load_file(rank: usize) -> String {
    format!("o-proj-load-L0-rank{rank}.json")
}

/// Explicit opt-in: directory plus a layers CSV that selects layer 0 only.
/// Returns `None` for non-selected layers, `Some(dir)` for layer 0.
pub(crate) fn gate_dir(layer: usize) -> Result<Option<PathBuf>> {
    parse_gate(
        std::env::var_os(DIR_ENV),
        std::env::var(LAYERS_ENV).ok(),
        layer,
    )
}

fn parse_gate(
    dir: Option<OsString>,
    layers: Option<String>,
    layer: usize,
) -> Result<Option<PathBuf>> {
    let Some(dir) = dir else {
        return Ok(None);
    };
    let layers = layers.context(format!(
        "{LAYERS_ENV} must be explicit when {DIR_ENV} is set"
    ))?;
    ensure!(
        layers.trim() == "0",
        "local o_proj binding supports explicit layer0 only, got {layers:?}"
    );
    let dir = PathBuf::from(dir);
    ensure!(
        dir.is_dir(),
        "{DIR_ENV} must be an existing directory, got {}",
        dir.display()
    );
    Ok((layer == 0).then_some(dir))
}

/// Round one finite F32 value to BF16 bits, round-to-nearest-even; the same
/// contract as `local_o_proj_replay.f32_to_bf16`, fail-closed on non-finite
/// values and values whose exponent overflows BF16.
pub(crate) fn round_bf16(value: f32) -> Result<u16> {
    ensure!(
        value.is_finite(),
        "BF16 reconstruction requires finite values"
    );
    let words = value.to_bits();
    let rounded = words.wrapping_add(0x7fff).wrapping_add((words >> 16) & 1);
    let bits = (rounded >> 16) as u16;
    ensure!(
        bits & 0x7f80 != 0x7f80,
        "BF16 reconstruction overflowed the exponent: {value}"
    );
    Ok(bits)
}

/// Little-endian uint16 NPY encoding (`descr '<u2'`, C order) of a
/// `[rows, cols]` BF16-bit matrix, 64-byte header aligned per NPY v1.
pub(crate) fn npy_u16_le(rows: usize, cols: usize, data: &[u16]) -> Vec<u8> {
    let mut header = "{'descr': '<u2', 'fortran_order': False, 'shape': (".to_string();
    header.push_str(&format!("{rows}, {cols}), }}"));
    let prefix = 10; // magic (6) + version (2) + header length (2)
    let pad = (64 - (prefix + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(prefix + header.len() + data.len() * 2);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for word in data {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("local o_proj binding write {}", path.display()))?
        .write_all(bytes)?;
    Ok(())
}

/// Fail closed when a load record misses any `atlas-local-o-proj-binding-v1`
/// field the forward capture must forward verbatim (or binds another rank).
pub(crate) fn validate_load_record(record: &serde_json::Value, rank: usize) -> Result<()> {
    ensure!(
        record.get("schema").and_then(|v| v.as_str()) == Some(LOAD_SCHEMA),
        "local o_proj load record schema mismatch"
    );
    ensure!(
        record.get("layer").and_then(|v| v.as_u64()) == Some(0),
        "local o_proj load record must bind layer 0"
    );
    ensure!(
        record.get("rank").and_then(|v| v.as_u64()) == Some(rank as u64),
        "local o_proj load record rank mismatch"
    );
    record
        .get("module")
        .and_then(|v| v.as_str())
        .filter(|m| !m.is_empty())
        .context("local o_proj load record module missing")?;
    let projection = record.get("projection").context("projection missing")?;
    ensure!(
        projection.get("tp_pre_reduce") == Some(&serde_json::json!(true)),
        "projection must bind the local pre-TP output"
    );
    for key in [
        "weight_layout",
        "input_dtype",
        "weight_dtype",
        "accumulator_dtype",
        "output_dtype",
    ] {
        ensure!(
            projection
                .get(key)
                .and_then(|v| v.as_str())
                .is_some_and(|v| !v.is_empty()),
            "projection {key} missing"
        );
    }
    let weight = record
        .pointer("/files/weight")
        .context("files.weight missing")?;
    for key in ["file", "dtype", "sha256"] {
        ensure!(
            weight
                .get(key)
                .and_then(|v| v.as_str())
                .is_some_and(|v| !v.is_empty()),
            "files.weight {key} missing"
        );
    }
    ensure!(
        weight
            .get("shape")
            .and_then(|v| v.as_array())
            .is_some_and(|s| s.len() == 2),
        "files.weight shape missing"
    );
    let resident = record.get("resident").context("resident missing")?;
    for key in ["form", "file", "dtype", "sha256"] {
        ensure!(
            resident
                .get(key)
                .and_then(|v| v.as_str())
                .is_some_and(|v| !v.is_empty()),
            "resident {key} missing"
        );
    }
    ensure!(
        resident
            .pointer("/recipe/name")
            .and_then(|v| v.as_str())
            .is_some_and(|v| !v.is_empty()),
        "resident recipe name missing"
    );
    resident
        .pointer("/recipe/inputs")
        .and_then(|v| v.as_object())
        .context("resident recipe inputs missing")?;
    Ok(())
}

/// Serialize the layer-0 local o_proj weight pair for one rank: the BF16
/// common-comparison-space matrix plus the verbatim resident shard bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_exl3(
    gpu: &dyn GpuBackend,
    dir: &Path,
    layer: usize,
    rank: usize,
    world: usize,
    module: &str,
    view: &Exl3LinearView<'_>,
    plan: &TpShardPlan,
    shard: &Exl3DeviceShard,
    stream: u64,
) -> Result<serde_json::Value> {
    ensure!(layer == 0, "local o_proj binding captures layer 0 only");
    let buffers = shard.buffers()?;
    let sizes = [
        plan.trellis_byte_size(),
        plan.in_features() * 2,
        plan.out_features() * 2,
    ];
    let mut resident = Vec::with_capacity(sizes.iter().sum::<usize>());
    for (ptr, bytes) in buffers.iter().zip(sizes) {
        let mut host = vec![0u8; bytes];
        gpu.copy_d2h_on_stream(*ptr, &mut host, stream)?;
        resident.extend_from_slice(&host);
    }
    gpu.synchronize(stream)?;
    let resident_sha = sha256_hex(&resident);
    write_new(&dir.join(resident_file(rank)), &resident)?;

    // Checkpoint-materialized view of the same shard bytes the device
    // upload copied, reconstructed to the common comparison space.
    let owned: OwnedExl3Shard = plan.materialize(view)?;
    let host_view = owned.view()?;
    let dense = reference_reconstruct(&host_view)
        .context("EXL3 o_proj reconstruction for the binding record")?;
    let k = plan.in_features();
    let n = plan.out_features();
    ensure!(
        dense.len() == k * n,
        "EXL3 o_proj reconstruction geometry mismatch"
    );
    let mut bits = Vec::with_capacity(dense.len());
    for value in dense {
        bits.push(round_bf16(value)?);
    }
    let npy = npy_u16_le(k, n, &bits);
    let weight_sha = sha256_hex(&npy);
    write_new(&dir.join(weight_file(rank)), &npy)?;

    let record = serde_json::json!({
        "schema": LOAD_SCHEMA,
        "layer": layer,
        "rank": rank,
        "tp_world": world,
        "module": module,
        "projection": {
            "module": module,
            "tp_pre_reduce": true,
            "weight_layout": "in-out",
            "input_dtype": "bf16",
            "weight_dtype": "bf16",
            "accumulator_dtype": "f32",
            "output_dtype": "bf16",
        },
        "files": {
            "weight": {
                "file": weight_file(rank),
                "shape": [k, n],
                "dtype": "bf16-le",
                "sha256": weight_sha,
            }
        },
        "resident": {
            "form": RESIDENT_FORM,
            "file": resident_file(rank),
            "dtype": RESIDENT_DTYPE,
            "sha256": resident_sha,
            "buffers": [
                {"name": "trellis", "bytes": sizes[0]},
                {"name": "input_scale", "bytes": sizes[1]},
                {"name": "output_scale", "bytes": sizes[2]},
            ],
            "recipe": {
                "name": DEQUANT_RECIPE,
                "inputs": {"resident_sha256": resident_sha},
            },
        },
    });
    validate_load_record(&record, rank)?;
    write_new(
        &dir.join(load_file(rank)),
        serde_json::to_string(&record)?.as_bytes(),
    )?;
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_requires_explicit_layers_and_existing_directory() {
        let dir = std::env::temp_dir();
        let missing = PathBuf::from("/nonexistent-o-proj-binding-dir");
        assert_eq!(parse_gate(None, None, 0).unwrap(), None);
        assert!(parse_gate(Some(dir.clone().into_os_string()), None, 0).is_err());
        assert!(parse_gate(Some(dir.clone().into_os_string()), Some("1".into()), 0).is_err());
        assert!(parse_gate(Some(dir.clone().into_os_string()), Some("0,3".into()), 0).is_err());
        assert!(parse_gate(Some(missing.into_os_string()), Some("0".into()), 0).is_err());
        assert_eq!(
            parse_gate(Some(dir.clone().into_os_string()), Some("0".into()), 0).unwrap(),
            Some(dir.clone())
        );
        assert_eq!(
            parse_gate(Some(dir.clone().into_os_string()), Some("0".into()), 1).unwrap(),
            None
        );
    }

    #[test]
    fn bf16_rounding_is_nearest_even_and_fails_closed() {
        assert_eq!(round_bf16(1.0).unwrap(), 0x3f80);
        assert_eq!(round_bf16(-2.5).unwrap(), 0xc020);
        // Exact halfway cases (lower16 == 0x8000): round half to even — an
        // even retained mantissa stays, an odd one carries upward.
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8000)).unwrap(), 0x3f80);
        assert_eq!(round_bf16(f32::from_bits(0x3f81_8000)).unwrap(), 0x3f82);
        // Strictly below/above the halfway point.
        assert_eq!(round_bf16(f32::from_bits(0x3f80_7fff)).unwrap(), 0x3f80);
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8001)).unwrap(), 0x3f81);
        assert_eq!(round_bf16(f32::from_bits(0x3f80_bfff)).unwrap(), 0x3f81);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            assert!(round_bf16(invalid).is_err(), "{invalid} must fail closed");
        }
    }

    #[test]
    fn npy_encoding_is_v1_aligned_little_endian_uint16() {
        let data = [0x3f80u16, 0x4000, 0xc000, 0x0001];
        let bytes = npy_u16_le(2, 2, &data);
        assert!(bytes.starts_with(b"\x93NUMPY\x01\x00"));
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header.contains("'descr': '<u2'"));
        assert!(header.contains("'fortran_order': False"));
        assert!(header.contains("'shape': (2, 2)"));
        assert!(header.ends_with('\n'));
        assert_eq!(
            (10 + header_len) % 64,
            0,
            "NPY v1 headers are 64-byte aligned"
        );
        let payload = &bytes[10 + header_len..];
        let expect: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(payload, expect);
        // Leave a fixed-path artefact for the cross-language check
        // (python3 numpy must np.load the exact bytes).
        std::fs::write(
            std::env::temp_dir().join("atlas-o-proj-npy-crosscheck.npy"),
            &bytes,
        )
        .unwrap();
    }

    #[test]
    fn load_record_round_trips_and_fails_closed_on_missing_fields() {
        let record = serde_json::json!({
            "schema": LOAD_SCHEMA,
            "layer": 0,
            "rank": 1,
            "tp_world": 2,
            "module": "model.layers.0.self_attn.o_proj",
            "projection": {
                "module": "model.layers.0.self_attn.o_proj",
                "tp_pre_reduce": true,
                "weight_layout": "in-out",
                "input_dtype": "bf16",
                "weight_dtype": "bf16",
                "accumulator_dtype": "f32",
                "output_dtype": "bf16",
            },
            "files": {"weight": {"file": "w.npy", "shape": [4, 6],
                                  "dtype": "bf16-le", "sha256": "ab".repeat(32)}},
            "resident": {"form": RESIDENT_FORM, "file": "r.bin",
                         "dtype": RESIDENT_DTYPE, "sha256": "cd".repeat(32),
                         "recipe": {"name": DEQUANT_RECIPE,
                                    "inputs": {"resident_sha256": "cd".repeat(32)}}},
        });
        validate_load_record(&record, 1).unwrap();
        // Fail closed on every removed top-level or nested required field.
        for path in [
            "/schema",
            "/layer",
            "/rank",
            "/module",
            "/projection",
            "/projection/tp_pre_reduce",
            "/projection/weight_layout",
            "/projection/input_dtype",
            "/projection/weight_dtype",
            "/projection/accumulator_dtype",
            "/projection/output_dtype",
            "/files/weight",
            "/files/weight/file",
            "/files/weight/shape",
            "/files/weight/dtype",
            "/files/weight/sha256",
            "/resident",
            "/resident/form",
            "/resident/file",
            "/resident/dtype",
            "/resident/sha256",
            "/resident/recipe",
            "/resident/recipe/name",
            "/resident/recipe/inputs",
        ] {
            let mut broken = record.clone();
            remove_path(&mut broken, path);
            assert!(
                validate_load_record(&broken, 1).is_err(),
                "missing {path} must fail closed"
            );
        }
        assert!(
            validate_load_record(&record, 0).is_err(),
            "rank mismatch fails closed"
        );
    }

    fn remove_path(value: &mut serde_json::Value, path: &str) {
        let mut segments = path.trim_start_matches('/').split('/');
        let last = segments.next_back().unwrap();
        let mut node = value;
        for segment in segments {
            node = &mut node[segment];
        }
        match node {
            serde_json::Value::Object(map) => {
                map.remove(last);
            }
            serde_json::Value::Array(list) => {
                list.remove(last.parse::<usize>().unwrap());
            }
            other => *other = serde_json::Value::Null,
        }
    }
}
