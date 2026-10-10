// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 filesystem I/O boundary. Mapped checkpoint bytes remain in their native
//! dtypes; companion tensors can reside in different indexed shards.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapOptions};
use safetensors::{
    SafeTensors,
    tensor::{Metadata, TensorView},
};
use serde::Deserialize;

use super::Exl3LinearView;

#[derive(Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

struct MappedShard {
    #[cfg(target_os = "linux")]
    file: File,
    bytes: Mmap,
    metadata: Metadata,
    data_start: usize,
}

/// Read-only indexed EXL3 checkpoint IORouter.
///
/// The checkpoint files must remain immutable while this router exists, as
/// required by memory-mapped safetensors loading. Opening maps files and parses
/// each shard's metadata once; it does not materialize or upload weight payloads.
pub struct Exl3CheckpointRouter {
    weight_map: BTreeMap<String, String>,
    shards: BTreeMap<String, MappedShard>,
}

impl Exl3CheckpointRouter {
    /// Open an indexed model directory with explicit EXL3/mul1 metadata.
    /// `original_quantization_config` describes pre-conversion weights and is
    /// deliberately never consulted to select the checkpoint's format.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let root = model_dir
            .as_ref()
            .canonicalize()
            .context("EXL3 model directory")?;
        let config: serde_json::Value = serde_json::from_reader(
            File::open(root.join("config.json")).context("EXL3 config.json")?,
        )
        .context("EXL3 config JSON")?;
        let quant = config
            .get("quantization_config")
            .context("EXL3 requires quantization_config")?;
        ensure!(
            quant.get("quant_method").and_then(|v| v.as_str()) == Some("exl3"),
            "EXL3 router requires quantization_config.quant_method=exl3"
        );
        ensure!(
            quant.get("codebook").and_then(|v| v.as_str()) == Some("mul1"),
            "EXL3 router requires explicit codebook=mul1"
        );
        let index: CheckpointIndex = serde_json::from_reader(
            File::open(root.join("model.safetensors.index.json"))
                .context("EXL3 safetensors index")?,
        )
        .context("EXL3 safetensors index JSON")?;
        ensure!(
            !index.weight_map.is_empty(),
            "EXL3 checkpoint index has no tensors"
        );
        let mut shards = BTreeMap::new();
        let hf_blobs = hf_blob_directory(&root)?;
        for shard_name in index.weight_map.values() {
            if shards.contains_key(shard_name) {
                continue;
            }
            validate_shard_name(shard_name)?;
            let path = root
                .join(shard_name)
                .canonicalize()
                .with_context(|| format!("EXL3 shard {shard_name}"))?;
            ensure!(
                path.parent() == Some(root.as_path())
                    || hf_blobs.as_deref().is_some_and(|blobs| {
                        path.parent() == Some(blobs)
                            && path
                                .file_name()
                                .and_then(|s| s.to_str())
                                .is_some_and(|s| hex_name(s, 64))
                    }),
                "EXL3 shard {shard_name} escapes model directory"
            );
            let file = File::open(&path).with_context(|| format!("EXL3 open {shard_name}"))?;
            // SAFETY: this read-only checkpoint boundary requires immutable
            // model files for the lifetime of the router, as documented above.
            let bytes = unsafe { MmapOptions::new().map(&file) }
                .with_context(|| format!("EXL3 mmap {shard_name}"))?;
            let (header_len, metadata) = SafeTensors::read_metadata(&bytes)
                .with_context(|| format!("EXL3 metadata {shard_name}"))?;
            shards.insert(
                shard_name.clone(),
                MappedShard {
                    #[cfg(target_os = "linux")]
                    file,
                    bytes,
                    metadata,
                    data_start: header_len + 8,
                },
            );
        }
        for (name, shard_name) in &index.weight_map {
            ensure!(
                shards[shard_name].metadata.info(name).is_some(),
                "EXL3 index tensor {name} is missing from shard {shard_name}"
            );
        }
        Ok(Self {
            weight_map: index.weight_map,
            shards,
        })
    }

    /// Release mapped page residency after rank-local uploads on Linux UMA.
    /// Call between reads; immutable views remain valid and refault checkpoint
    /// bytes on subsequent access. Other platforms perform no cache advice.
    /// File-cache advice is best effort through the shared weight I/O helper.
    pub fn evict_cached_pages(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        for (name, shard) in &self.shards {
            // SAFETY: these are read-only file-backed mappings of immutable
            // checkpoint files. DONTNEED releases residency without changing
            // logical bytes; existing borrowed views refault the same data.
            unsafe {
                shard
                    .bytes
                    .unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
            }
            .with_context(|| format!("EXL3 evict mapped pages {name}"))?;
            crate::weights::evict_page_cache(&shard.file);
        }
        Ok(())
    }

    /// Borrow and validate a complete modern EXL3 group without copying bytes.
    pub fn linear_view(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Exl3LinearView<'_>> {
        for suffix in ["su", "sv", "mcg", "scale", "bias", "weight"] {
            let name = format!("{prefix}.{suffix}");
            ensure!(
                !self
                    .shards
                    .values()
                    .any(|shard| shard.metadata.info(&name).is_some()),
                "EXL3 unsupported or conflicting companion {name}"
            );
        }
        let view = |suffix: &str| self.tensor_view(&format!("{prefix}.{suffix}"));
        Exl3LinearView::new(
            view("trellis")?,
            view("suh")?,
            view("svh")?,
            view("mul1")?,
            in_features,
            out_features,
        )
        .with_context(|| format!("EXL3 linear {prefix}"))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    /// Physical checkpoint dimensions, including any EXL3 H128 padding.
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor_view(name)?.shape().to_vec())
    }

    pub fn tensor_dtype(&self, name: &str) -> Result<safetensors::Dtype> {
        Ok(self.tensor_view(name)?.dtype())
    }

    pub(super) fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        let shard_name = self
            .weight_map
            .get(name)
            .with_context(|| format!("EXL3 missing companion {name}"))?;
        let shard = &self.shards[shard_name];
        let info = shard
            .metadata
            .info(name)
            .with_context(|| format!("EXL3 missing tensor {name}"))?;
        let start = shard.data_start + info.data_offsets.0;
        let end = shard.data_start + info.data_offsets.1;
        TensorView::new(info.dtype, info.shape.clone(), &shard.bytes[start..end])
            .with_context(|| format!("EXL3 tensor view {name}"))
    }
}

// HF snapshots use relative symlinks to LFS files in the same repository's
// sibling blobs directory. This exception does not authorize arbitrary sibling
// files, another snapshot, or a blobs directory symlink outside the repository.
fn hf_blob_directory(root: &Path) -> Result<Option<PathBuf>> {
    let Some(snapshots) = root.parent() else {
        return Ok(None);
    };
    let Some(repository) = snapshots.parent() else {
        return Ok(None);
    };
    if snapshots.file_name().and_then(|s| s.to_str()) != Some("snapshots")
        || !root
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| hex_name(s, 40))
        || !repository
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.starts_with("models--"))
    {
        return Ok(None);
    }
    let blobs = repository
        .join("blobs")
        .canonicalize()
        .context("EXL3 HF blob directory")?;
    ensure!(
        blobs.parent() == Some(repository),
        "EXL3 HF blob directory escapes repository"
    );
    Ok(Some(blobs))
}

fn hex_name(name: &str, len: usize) -> bool {
    name.len() == len
        && name
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_shard_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(_)))
            && components.next().is_none()
            && !name.contains('\\')
            && Path::new(name)
                .extension()
                .is_some_and(|extension| extension == "safetensors"),
        "EXL3 unsafe shard name {name:?}"
    );
    Ok(())
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
