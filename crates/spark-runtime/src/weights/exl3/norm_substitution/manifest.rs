// SPDX-License-Identifier: AGPL-3.0-only

//! Fixture manifest parse, validation and payload staging (spec D1).
//!
//! Split from `../norm_substitution.rs` (file-size cap); the runtime knob,
//! substitution and record commit stay there. Host-side only: reads
//! `$fixture_root/manifest.json` plus every declared payload, validates the
//! schema (PCND — any error aborts the load), and returns the staged variant
//! with entries keyed by the spec's uniqueness tuple
//! `(layer, tp_rank, invocation, tensor)`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, ensure};

use super::{MANIFEST_NAME, MANIFEST_SCHEMA, VARIANTS, hex_sha256};

/// One validated manifest entry: declaration plus staged payload bytes.
pub(super) struct FixtureEntry {
    pub(super) request_id: String,
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    pub(super) file: String,
    pub(super) sha256: String,
    pub(super) bytes: Vec<u8>,
}

/// Uniqueness-tuple key from the D1 contract.
pub(super) type EntryKey = (usize, usize, usize, String);

/// Parse and validate `$fixture_root/manifest.json`, verify every payload's
/// sha256 and size, and stage the payload bytes in host memory. The fixture
/// directory is read-only to this module.
pub(super) fn load(
    fixture_root: &Path,
    world: usize,
) -> Result<(String, BTreeMap<EntryKey, FixtureEntry>)> {
    let manifest_path = fixture_root.join(MANIFEST_NAME);
    let raw = std::fs::read(&manifest_path)
        .with_context(|| format!("norm substitution manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&raw)
        .context("norm substitution manifest is not valid JSON for its schema")?;
    ensure!(
        manifest.schema == MANIFEST_SCHEMA,
        "norm substitution manifest schema mismatch: {:?}",
        manifest.schema
    );
    ensure!(
        VARIANTS.contains(&manifest.variant.as_str()),
        "norm substitution manifest variant invalid: {:?}",
        manifest.variant
    );
    ensure!(
        !manifest.source.capture_root.is_empty()
            && !manifest.source.engine.is_empty()
            && is_sha256_hex(&manifest.source.schedule_sha256),
        "norm substitution manifest source invalid"
    );
    let mut entries = BTreeMap::new();
    let mut keys = BTreeSet::new();
    for entry in &manifest.entries {
        ensure!(
            entry.layer == 0,
            "norm substitution entry layer must be 0 (assumption A12), got {}",
            entry.layer
        );
        ensure!(
            entry.tp_world == world && entry.tp_rank < world,
            "norm substitution entry topology mismatch: tp_rank {} tp_world {} (runtime world {})",
            entry.tp_rank,
            entry.tp_world,
            world
        );
        ensure!(
            !entry.request_id.is_empty(),
            "norm substitution entry request_id empty"
        );
        ensure!(
            safe_identifier(&entry.stage) && safe_identifier(&entry.tensor),
            "norm substitution entry stage/tensor invalid"
        );
        let width = dtype_width(&entry.dtype)
            .with_context(|| format!("norm substitution entry dtype invalid: {:?}", entry.dtype))?;
        ensure!(
            entry.shape.len() == 2 && entry.shape.iter().all(|n| *n > 0),
            "norm substitution entry shape invalid: {:?}",
            entry.shape
        );
        ensure!(
            safe_filename(&entry.file),
            "norm substitution entry file invalid: {:?}",
            entry.file
        );
        ensure!(
            is_sha256_hex(&entry.sha256),
            "norm substitution entry sha256 invalid: {:?}",
            entry.sha256
        );
        let key = (
            entry.layer,
            entry.tp_rank,
            entry.invocation,
            entry.tensor.clone(),
        );
        ensure!(
            keys.insert(key.clone()),
            "norm substitution duplicate entry: {key:?}"
        );
        let payload_path = fixture_root.join(&entry.file);
        let bytes = std::fs::read(&payload_path)
            .with_context(|| format!("norm substitution payload {}", payload_path.display()))?;
        let expected = entry
            .shape
            .iter()
            .try_fold(width, |size, n| size.checked_mul(*n))
            .context("norm substitution entry bytes overflow")?;
        ensure!(
            bytes.len() == expected,
            "norm substitution payload size mismatch for {:?}: {} bytes, declared {}",
            entry.file,
            bytes.len(),
            expected
        );
        ensure!(
            hex_sha256(&bytes) == entry.sha256,
            "norm substitution payload sha256 mismatch for {:?}",
            entry.file
        );
        entries.insert(
            key,
            FixtureEntry {
                request_id: entry.request_id.clone(),
                dtype: entry.dtype.clone(),
                shape: entry.shape.clone(),
                file: entry.file.clone(),
                sha256: entry.sha256.clone(),
                bytes,
            },
        );
    }
    Ok((manifest.variant, entries))
}

#[derive(serde::Deserialize)]
struct Manifest {
    schema: String,
    variant: String,
    source: Source,
    entries: Vec<Entry>,
}

#[derive(serde::Deserialize)]
struct Source {
    capture_root: String,
    engine: String,
    schedule_sha256: String,
}

#[derive(serde::Deserialize)]
struct Entry {
    request_id: String,
    layer: usize,
    tp_rank: usize,
    tp_world: usize,
    invocation: usize,
    stage: String,
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    file: String,
    sha256: String,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn safe_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn safe_filename(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && Path::new(value).file_name() == Some(std::ffi::OsStr::new(value))
}

pub(super) fn dtype_width(label: &str) -> Option<usize> {
    match label {
        "bf16-le" => Some(2),
        "f32-le" | "i32-le" => Some(4),
        _ => None,
    }
}
