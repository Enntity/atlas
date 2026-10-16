// SPDX-License-Identifier: AGPL-3.0-only

//! The NVFP4 n-gram table of Qwen3.8-Flash-Next-FP4-Local, opened through a
//! one-time packed copy.
//!
//! ```text
//! {lp}.ple_embedding.ngram_embedding.shard_{0..127}.weight        [R, 80] U8
//! {lp}.ple_embedding.ngram_embedding.shard_{0..127}.weight_scale  [R, 10] F8_E4M3
//! {lp}.ple_embedding.ngram_embedding.weight_scale_2               [1]     F32
//! ```
//!
//! A row's codes and its scales sit in two tensors at unrelated offsets, so a
//! cache miss read in place would be two NVMe reads. Per-IO latency is the
//! PLE stall on Spark (the APST wake behind `ATLAS_PLE_NVME_KEEPALIVE_MS`;
//! ~74 us per serial fault, 22K faults on a 4.6K-token prefill), not bytes,
//! so the planes are interleaved once (`spark_storage::ngram_pack`) into
//! 96-byte rows (`layers::ple::ngram_nvfp4`) and the cache opens that file as
//! one contiguous table: one read per miss, 28.8 GB on disk next to the
//! 26.8 GB checkpoint table, no extra memory. The checkpoint directory is not
//! written; the packed copy goes to `ATLAS_PLE_PACKED_DIR`, defaulting to
//! `atlas-ple/` under the Hugging Face hub cache, beside `atlas-ngram/` (the
//! n-gram speculation table `serve_load` persists there).

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};
use spark_storage::ngram_pack::{PackSpec, Plane, open_or_pack};

use crate::layers::ple::ngram_nvfp4::{code_bytes, packed_stride, scale_bytes};

/// Open the packed NVFP4 table for the `shards` (each `shard_{i}.weight`'s
/// file and offset, `packed_cols` bytes per row), building the packed copy
/// first if there is no valid one.
pub(super) fn open(
    store: &WeightStore,
    lp: &str,
    shards: &[(std::path::PathBuf, u64)],
    rows_per: u64,
    packed_cols: usize,
    slots: usize,
    gpu: &dyn GpuBackend,
) -> Result<spark_storage::NgramRowCache> {
    let head_dim = packed_cols * 2;
    let base = format!("{lp}.ple_embedding.ngram_embedding");
    let scales = scale_shards(store, &base, shards.len(), rows_per, head_dim)?;
    let global = crate::weight_map::scalar_f32(store, &format!("{base}.weight_scale_2"), gpu)
        .context("PLE: NVFP4 weight_scale_2")?;
    anyhow::ensure!(
        global.is_finite() && global > 0.0,
        "PLE: NVFP4 weight_scale_2 is {global}"
    );
    let spec = PackSpec {
        planes: vec![
            Plane {
                shards,
                row_bytes: code_bytes(head_dim),
            },
            Plane {
                shards: &scales,
                row_bytes: scale_bytes(head_dim),
            },
        ],
        rows_per_shard: rows_per,
        stride: packed_stride(head_dim),
        tag: "nvfp4g16",
    };
    let dir = packed_dir()?;
    let table = open_or_pack(&spec, &dir).with_context(|| {
        format!(
            "PLE: packing the NVFP4 n-gram table into {} (set ATLAS_PLE_PACKED_DIR \
             to a writable directory with ~29 GB free)",
            dir.display()
        )
    })?;
    let mut cache = spark_storage::NgramRowCache::open_at(
        &table.path,
        table.base_offset,
        None,
        table.rows_total,
        table.stride,
        slots,
    )
    .context("PLE: NVFP4 row cache")?;
    cache.set_nvfp4_global_scale(global)?;
    tracing::info!(
        "PLE n-gram table: NVFP4 (E2M1 + E4M3/16, global {global:.6e}), {} shards \
         read through packed {} ({} B rows, one NVMe read per miss)",
        shards.len(),
        table.path.display(),
        table.stride
    );
    Ok(cache)
}

/// `shard_{i}.weight_scale` for every shard: deferred (they are 3.2 GB), E4M3,
/// one row per code row.
fn scale_shards(
    store: &WeightStore,
    base: &str,
    n: usize,
    rows_per: u64,
    head_dim: usize,
) -> Result<Vec<(std::path::PathBuf, u64)>> {
    (0..n)
        .map(|i| {
            let name = format!("{base}.shard_{i}.weight_scale");
            let d = store.deferred(&name).with_context(|| {
                format!("PLE: NVFP4 shard {i} has codes but `{name}` was not deferred")
            })?;
            anyhow::ensure!(
                d.dtype == WeightDtype::FP8E4M3
                    && d.shape == [rows_per as usize, scale_bytes(head_dim)],
                "PLE: `{name}` is {:?} {:?}, expected F8_E4M3 [{rows_per}, {}]",
                d.dtype,
                d.shape,
                scale_bytes(head_dim)
            );
            Ok((d.path.clone(), d.offset))
        })
        .collect()
}

/// `ATLAS_PLE_PACKED_DIR`, else `atlas-ple/` under the hub cache root, resolved
/// with `spark-server`'s `resolve_cache_root` precedence (minus `--cache-dir`,
/// which the loader never sees): `$HF_HUB_CACHE`, `$HF_HOME/hub`,
/// `~/.cache/huggingface/hub`.
fn packed_dir() -> Result<std::path::PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
    if let Some(d) = env("ATLAS_PLE_PACKED_DIR") {
        return Ok(d.into());
    }
    let root = match (env("HF_HUB_CACHE"), env("HF_HOME"), env("HOME")) {
        (Some(c), _, _) => std::path::PathBuf::from(c),
        (None, Some(h), _) => std::path::PathBuf::from(h).join("hub"),
        (None, None, Some(h)) => std::path::PathBuf::from(h).join(".cache/huggingface/hub"),
        _ => anyhow::bail!("PLE: set ATLAS_PLE_PACKED_DIR (no HF_HUB_CACHE, HF_HOME or HOME)"),
    };
    Ok(root.join("atlas-ple"))
}
