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
//! copies prefill still reads. Tiled under `ATLAS_GLM_MXFP8_TILED`.
//!
//! `ATLAS_GLM_FP4L_KDA_W8A8_PREFILL=1`: every prefill projection is the
//! cuBLASLt MXFP8 W8A8 GEMM on the twin's E4M3 bytes (`ops::MxW8a8`,
//! NVIDIA's recipe: exact weights, activations rounded to E4M3 per 32 values,
//! ~2-3e-2 relative from BF16 activations), row-invariant end to end. The
//! transposed NVFP4 prefill twins are not built and the LT FP8 copies are
//! never taken. GLM's TP2 shape (4096 x 4096) is in `cublaslt::MX_SWEEP`
//! but not yet swept, so it runs one in-order kernel for every row count.
//!
//! Both on: nothing reads NVFP4, so the layer keeps only the MXFP8 twin and
//! the W8A8 scale image (~1.06 B/param, below production's NVFP4 decode +
//! transposed prefill twins at ~1.1) and the load skips the BF16 expansion.
//!
//! On a checkpoint that does not ship the projection MXFP8 both are inert.

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::ops::MxW8a8;
use crate::tp_shard::TpShardKind;
use crate::weight_map::{Mxfp8Weight, mxfp8_cols_twin, mxfp8_rows_twin, raw_mxfp8};

/// The switch(es), as set (each `0`/unset or `1`; anything else refused).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct KdaMxPolicy {
    pub decode: bool,
    pub w8a8_prefill: bool,
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
    pub(super) const W8A8: &'static str = "ATLAS_GLM_FP4L_KDA_W8A8_PREFILL";

    pub(super) fn parse(decode: Option<&str>, w8a8: Option<&str>) -> Result<Self> {
        Ok(Self {
            decode: flag(Self::DECODE, decode)?,
            w8a8_prefill: flag(Self::W8A8, w8a8)?,
        })
    }

    pub(super) fn from_env() -> Result<Self> {
        let var = |name| std::env::var(name).ok();
        Self::parse(var(Self::DECODE).as_deref(), var(Self::W8A8).as_deref())
    }
}

/// What a KDA projection reads from its MXFP8 twin.
#[derive(Debug, Default)]
pub(super) struct KdaMx {
    /// The decode / verify twin (`decode`).
    pub decode: Option<Mxfp8Weight>,
    /// The prefill W8A8 operand over the same E4M3 bytes (`w8a8_prefill`).
    pub w8a8: Option<MxW8a8>,
}

impl KdaMx {
    /// Whether these twins serve decode, verify AND prefill: nothing reads
    /// the projection's NVFP4 copies, so none are built.
    pub(super) fn serves_all(&self) -> bool {
        self.decode.is_some() && self.w8a8.is_some()
    }
}

/// This rank's MXFP8 twin of KDA projection `stem` (full `[n, k]`, sharded by
/// `kind`) under `policy`, cut from the checkpoint's bytes; `None` when the
/// policy is off or the checkpoint ships `stem` in another format. The twin's
/// row (column) slice is the matching slice of the checkpoint tensor, as
/// `mxfp8_load` cuts it, and the same rows the BF16 TP shard keeps.
/// `m_max` bounds the W8A8 row buckets tuned at load. Synchronizes the
/// default stream.
#[allow(clippy::too_many_arguments)]
pub(super) fn kda_twin(
    policy: KdaMxPolicy,
    store: &WeightStore,
    stem: &str,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    m_max: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<KdaMx>> {
    if !policy.decode && !policy.w8a8_prefill {
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
    let w8a8 = if policy.w8a8_prefill {
        // Reads the twin's E4M3 bytes in place, untiled.
        match MxW8a8::from_twin(gpu, &twin, [n, k], m_max, stream) {
            Ok(w) => Some(w),
            Err(e) => {
                gpu.free(twin.data)?;
                gpu.free(twin.scales)?;
                return Err(e.context(format!("{stem}: MXFP8 W8A8 operand")));
            }
        }
    } else {
        crate::layers::ops::mxfp8_tile_if_requested(gpu, &mut twin, n, k, stream)?;
        None
    };
    if !policy.decode {
        // W8A8 alone reads the E4M3 bytes (its operand) and its own tiled
        // scale image: the row-major scales have no reader.
        gpu.free(twin.scales)?;
    }
    Ok(Some(KdaMx {
        decode: policy.decode.then_some(twin),
        w8a8,
    }))
}

#[cfg(test)]
#[path = "kda_mx_tests.rs"]
mod tests;
