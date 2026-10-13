// SPDX-License-Identifier: AGPL-3.0-only

//! C2 norm-output substitution knob — the diagnostic I/O boundary (SBIO).
//!
//! Manifest parse/validation, fixture payload staging, `(layer, tp_rank,
//! invocation, tensor)` lookup and hashing all live here. The GLM-5 KDA
//! layer engages this module only at the layer-0 `gated` output
//! (`layers/glm5_kda/forward_attention.rs`, assumption A12) and refuses the
//! multi-sequence path while the knob is engaged (assumption A9).
//!
//! Off by default: `ATLAS_EXL3_NORM_SUBSTITUTE_DIR` unset/empty means zero
//! file I/O and zero device copies. Engaged: the manifest is parsed and its
//! payloads staged once per model load (pattern:
//! `spark-model/src/exl3_gate_perturbation.rs`); any validation error aborts
//! the load (PCND — no defaults). Contract:
//! `docs/plans/c2-tooling-spec.md` D1; semantics authority
//! `docs/plans/q2-norm-causality-plan.md` §"Experiment C2".

use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::exl3::ffn_io::{DiagnosticDtype, DiagnosticTensor, Exl3FfnRouter};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fixture directory (`$DIR/manifest.json` + payloads). Unset/empty = off.
pub const ENV_SUBSTITUTE_DIR: &str = "ATLAS_EXL3_NORM_SUBSTITUTE_DIR";
/// Records directory: REQUIRED when the knob is engaged (one fresh directory
/// per attempt; see the D1 contract in `docs/plans/c2-tooling-spec.md`). The
/// fixture directory is immutable — records never land in it.
pub const ENV_RECORD_DIR: &str = "ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR";
/// Fixture manifest schema (spec D1).
pub const MANIFEST_SCHEMA: &str = "atlas-norm-substitution-fixtures-v1";
/// Substitution record metadata schema (spec D1).
pub const RECORD_SCHEMA: &str = "atlas-norm-substitution-v1";
/// Record stage passed to `Exl3FfnRouter::capture_with_metadata` (spec D1).
pub const RECORD_STAGE: &str = "norm_substitution";
/// Fixture manifest file name inside `ENV_SUBSTITUTE_DIR` (assumption A10).
const MANIFEST_NAME: &str = "manifest.json";
/// The one substituted tensor: the layer-0 `kda_sigmoid_gated_norm` output.
pub const TENSOR: &str = "gated";

/// Graph interlock shared with `Exl3FfnRouter::from_env_named` (assumption
/// A11): fixture bytes must never be baked into a CUDA graph.
const GRAPH_FLAGS: [&str; 3] = [
    "ATLAS_NO_DECODE_GRAPHS",
    "ATLAS_NO_DECODE_GRAPHS_MULTISEQ",
    "ATLAS_DEBUG_NO_GRAPH",
];

/// Fixture `variant` values (spec D1); recorded in every substitution record.
const VARIANTS: [&str; 2] = ["reference-exact", "v24-corrected"];

/// One validated manifest entry: declaration plus staged payload bytes.
struct FixtureEntry {
    request_id: String,
    dtype: String,
    shape: Vec<usize>,
    file: String,
    sha256: String,
    bytes: Vec<u8>,
}

/// An engaged substitution awaiting its record. The pre-image hash is taken
/// before the fixture bytes replace `gated`; the record is committed after
/// the post-substitution `gated_norm` capture (spec D1 order: kernel → D2H
/// pre-image → sha256 → H2D fixture → `gated_norm` capture → record).
pub struct PendingSubstitution<'a> {
    records: &'a Exl3FfnRouter,
    invocation: usize,
    request_id: String,
    variant: String,
    shape: Vec<usize>,
    fixture_file: String,
    fixture_sha256: String,
    pre_sha256: String,
}

impl std::fmt::Debug for PendingSubstitution<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSubstitution")
            .field("invocation", &self.invocation)
            .field("request_id", &self.request_id)
            .field("variant", &self.variant)
            .field("shape", &self.shape)
            .field("fixture_file", &self.fixture_file)
            .field("fixture_sha256", &self.fixture_sha256)
            .field("pre_sha256", &self.pre_sha256)
            .finish()
    }
}

impl PendingSubstitution<'_> {
    /// Atomic `norm_substitution` record through
    /// `Exl3FfnRouter::capture_with_metadata`. The captured `gated` payload
    /// is POST-substitution (`payload_stage: post-substitution`); the
    /// pre-substitution bytes survive only as their sha256 in the metadata.
    pub fn record(&self, gpu: &dyn GpuBackend, gated: DevicePtr, stream: u64) -> Result<()> {
        self.records.capture_with_metadata(
            gpu,
            self.invocation,
            RECORD_STAGE,
            &[DiagnosticTensor {
                name: TENSOR,
                dtype: DiagnosticDtype::Bf16,
                shape: self.shape.clone(),
                ptr: gated,
            }],
            stream,
            &serde_json::json!({
                "schema": RECORD_SCHEMA,
                "payload_stage": "post-substitution",
                "variant": self.variant,
                "request_id": self.request_id,
                "invocation": self.invocation,
                "shape": self.shape,
                "fixture_file": self.fixture_file,
                "fixture_sha256": self.fixture_sha256,
                "pre_substitution_sha256": self.pre_sha256,
            }),
        )
    }
}

/// Parsed, validated and staged substitution fixture set for one model load.
pub struct NormSubstitution {
    variant: String,
    /// Keyed by the spec's uniqueness tuple `(layer, tp_rank, invocation,
    /// tensor)`; `layer` is validated to 0 at load (assumption A12).
    entries: BTreeMap<(usize, usize, usize, String), FixtureEntry>,
    rank: usize,
    records: Exl3FfnRouter,
    sequence: AtomicUsize,
}

impl NormSubstitution {
    /// Model-load entry point (pattern: `exl3_gate_perturbation.rs`). Unset
    /// or empty env = off. Otherwise the graph interlock is required, the
    /// record directory is required (PCND), and any validation error aborts
    /// the load.
    pub fn from_env(rank: usize, world: usize) -> Result<Option<Self>> {
        let Some(dir) = nonempty_var(ENV_SUBSTITUTE_DIR) else {
            // Off: the record directory is never read.
            return Ok(None);
        };
        for flag in GRAPH_FLAGS {
            ensure!(
                std::env::var(flag).as_deref() == Ok("1"),
                "{ENV_SUBSTITUTE_DIR} requires {flag}=1"
            );
        }
        let record_dir = nonempty_var(ENV_RECORD_DIR)
            .with_context(|| format!("{ENV_SUBSTITUTE_DIR} requires explicit {ENV_RECORD_DIR}"))?;
        Self::load(&dir, &record_dir, rank, world).map(Some)
    }

    /// Parse and validate `$dir/manifest.json`, verify every payload's
    /// sha256 and size, and stage the payload bytes in host memory. The
    /// fixture directory is read-only to this module; records are rooted at
    /// `record_dir` (which must already exist and must be neither `dir`
    /// itself nor nested under it — `Path::starts_with` compares whole path
    /// components, so a shared name prefix like `/x/foo` vs `/x/foobar`
    /// stays legal).
    pub fn load(dir: &Path, record_dir: &Path, rank: usize, world: usize) -> Result<Self> {
        ensure!(
            world > 0 && rank < world,
            "norm substitution topology invalid"
        );
        let fixture_root = dir
            .canonicalize()
            .with_context(|| format!("norm substitution fixture directory {}", dir.display()))?;
        let records_root = record_dir.canonicalize().with_context(|| {
            format!(
                "norm substitution record directory {} (must already exist)",
                record_dir.display()
            )
        })?;
        ensure!(
            !records_root.starts_with(&fixture_root),
            "norm substitution record directory must not be the fixture directory \
             or nested under it"
        );
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
            let width = dtype_width(&entry.dtype).with_context(|| {
                format!("norm substitution entry dtype invalid: {:?}", entry.dtype)
            })?;
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
        let records = Exl3FfnRouter::new(record_dir.to_path_buf(), 0, rank, world)?;
        Ok(Self {
            variant: manifest.variant,
            entries,
            rank,
            records,
            sequence: AtomicUsize::new(0),
        })
    }

    /// One forward invocation at `layer` on this rank: when `layer == 0`
    /// (assumption A12) the full `gated` tensor is replaced by the fixture
    /// entry for the current invocation — D2H pre-image → sha256 → H2D
    /// staged fixture bytes. An engaged process with no entry for the
    /// invocation, a missing or mismatched request-id binding, or a
    /// shape/dtype mismatch is a run error (spec D1 verification contract).
    #[allow(clippy::too_many_arguments)]
    pub fn substitute(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        gated: DevicePtr,
        dtype: DiagnosticDtype,
        shape: &[usize],
        request_id: Option<&str>,
        stream: u64,
    ) -> Result<Option<PendingSubstitution<'_>>> {
        if layer != 0 {
            // A12: later layers are unchanged by this experiment.
            return Ok(None);
        }
        let invocation = self.sequence.fetch_add(1, Ordering::SeqCst);
        let key = (0usize, self.rank, invocation, TENSOR.to_string());
        let entry = self.entries.get(&key).with_context(|| {
            format!(
                "norm substitution has no fixture entry for invocation {invocation} on rank {}",
                self.rank
            )
        })?;
        let bound = request_id
            .context("norm substitution requires the ForwardContext request-id binding")?;
        ensure!(
            bound == entry.request_id,
            "norm substitution request-id mismatch: live {bound:?}, fixture {:?}",
            entry.request_id
        );
        ensure!(
            shape == entry.shape,
            "norm substitution shape mismatch at invocation {invocation}: live {shape:?}, fixture {:?}",
            entry.shape
        );
        ensure!(
            dtype.label() == entry.dtype,
            "norm substitution dtype mismatch at invocation {invocation}: live {}, fixture {:?}",
            dtype.label(),
            entry.dtype
        );
        let mut pre = vec![0u8; entry.bytes.len()];
        gpu.copy_d2h_on_stream(gated, &mut pre, stream)
            .context("norm substitution pre-image readback")?;
        let pre_sha256 = hex_sha256(&pre);
        gpu.copy_h2d_async(&entry.bytes, gated, stream)
            .context("norm substitution fixture upload")?;
        Ok(Some(PendingSubstitution {
            records: &self.records,
            invocation,
            request_id: entry.request_id.clone(),
            variant: self.variant.clone(),
            shape: entry.shape.clone(),
            fixture_file: entry.file.clone(),
            fixture_sha256: entry.sha256.clone(),
            pre_sha256,
        }))
    }

    /// Assumption A9: the frozen comparison profile admits one sequence, so
    /// the multi-sequence KDA path must error (never substitute) while the
    /// knob is engaged.
    pub fn reject_multi_seq(&self) -> Result<()> {
        anyhow::bail!(
            "{ENV_SUBSTITUTE_DIR} does not support the multi-sequence KDA path \
             (assumption A9: one-sequence admission)"
        )
    }
}

/// Engagement seam for `glm5_kda/forward_attention.rs`. `None` = knob off:
/// zero file I/O and zero device copies.
#[allow(clippy::too_many_arguments)]
pub fn engage<'a>(
    gpu: &dyn GpuBackend,
    substitution: Option<&'a NormSubstitution>,
    layer: usize,
    gated: DevicePtr,
    dtype: DiagnosticDtype,
    shape: &[usize],
    request_id: Option<&str>,
    stream: u64,
) -> Result<Option<PendingSubstitution<'a>>> {
    match substitution {
        Some(s) => s.substitute(gpu, layer, gated, dtype, shape, request_id, stream),
        None => Ok(None),
    }
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

fn nonempty_var(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
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

fn dtype_width(label: &str) -> Option<usize> {
    match label {
        "bf16-le" => Some(2),
        "f32-le" | "i32-le" => Some(4),
        _ => None,
    }
}

#[cfg(test)]
#[path = "norm_substitution_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "norm_substitution_env_tests.rs"]
mod env_tests;
