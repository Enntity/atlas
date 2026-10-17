// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in: FP4-Local's KDA q/k/v/o computed on the checkpoint's own MXFP8
//! bytes instead of production's runtime NVFP4 twins (default off).
//!
//! `ATLAS_GLM_FP4L_KDA_MXFP8_DECODE=1`: every decode, multi-sequence decode
//! and verify row (1..=32 a launch, any count chunked) reads the rank's MXFP8
//! twin through `ops::mxfp8_gemv_rows`. Exact weights, and one row-invariant
//! kernel family for every row count, so a verify row and the same row
//! decoded agree bit for bit, as on the canonical W4A16 tier. It reads ~1.03
//! B/param where NVFP4 reads ~0.56 (34 layers x 4 projections: ~1 GB more
//! per rank per step) and adds the twin (~2.4 GB per rank) beside the NVFP4
//! copies prefill still reads. Tiled under `ATLAS_GLM_MXFP8_TILED`. On a
//! checkpoint that does not ship the projection MXFP8 it is inert.

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::tp_shard::TpShardKind;
use crate::weight_map::{Mxfp8Weight, mxfp8_cols_twin, mxfp8_rows_twin, raw_mxfp8};

/// The switch(es), as set (each `0`/unset or `1`; anything else refused).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct KdaMxPolicy {
    pub decode: bool,
}

fn flag(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be 0 or 1, got {other:?}"),
    }
}

impl KdaMxPolicy {
    pub(super) const DECODE: &'static str = "ATLAS_GLM_FP4L_KDA_MXFP8_DECODE";

    pub(super) fn parse(decode: Option<&str>) -> Result<Self> {
        Ok(Self {
            decode: flag(Self::DECODE, decode)?,
        })
    }

    pub(super) fn from_env() -> Result<Self> {
        Self::parse(std::env::var(Self::DECODE).ok().as_deref())
    }
}

/// What a KDA projection reads from its MXFP8 twin.
#[derive(Debug, Default)]
pub(super) struct KdaMx {
    /// The decode / verify twin (`decode`).
    pub decode: Option<Mxfp8Weight>,
}

/// This rank's MXFP8 twin of KDA projection `stem` (full `[n, k]`, sharded by
/// `kind`) under `policy`, cut from the checkpoint's bytes; `None` when the
/// policy is off or the checkpoint ships `stem` in another format. The twin's
/// row (column) slice is the matching slice of the checkpoint tensor, as
/// `mxfp8_load` cuts it, and the same rows the BF16 TP shard keeps.
/// Synchronizes the default stream.
pub(super) fn kda_twin(
    policy: KdaMxPolicy,
    store: &WeightStore,
    stem: &str,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<KdaMx>> {
    if !policy.decode {
        return Ok(None);
    }
    let Some(raw) = raw_mxfp8(store, stem)? else {
        return Ok(None);
    };
    let stream = gpu.default_stream();
    let (mut twin, n, k) = match kind {
        TpShardKind::ColumnParallel => {
            mxfp8_rows_twin(gpu, &[(raw, &[raw.n][..])], tp_rank, tp_size, stream)?
        }
        TpShardKind::RowParallel => mxfp8_cols_twin(gpu, raw, tp_rank, tp_size, stream)?,
        TpShardKind::Replicated => bail!("{stem}: KDA projections are TP-sharded"),
    };
    crate::layers::ops::mxfp8_tile_if_requested(gpu, &mut twin, n, k, stream)?;
    Ok(Some(KdaMx { decode: Some(twin) }))
}

#[cfg(test)]
#[path = "kda_mx_tests.rs"]
mod tests;
