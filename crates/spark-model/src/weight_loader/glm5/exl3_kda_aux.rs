// SPDX-License-Identifier: AGPL-3.0-only

//! Merged KDA auxiliary rows `[B; FA; GA]` for the single reference-ordered
//! GEMM. The merged base is the sole owner; B/FA/GA become interior views that
//! are never registered or freed independently.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::dense::{Exl3DenseDeviceTensor, Exl3DenseDtype};

use super::{Exl3KdaDenseSlot as Slot, Glm5Exl3KdaDenseResources};
use crate::weight_map::DenseWeight;

const MERGED: [Slot; 3] = [Slot::BProj, Slot::FAProj, Slot::GAProj];

/// Concatenate B, FA, GA, then retire the originals. Originals stay owned
/// until the copy succeeds; a failed retirement keeps that original in `dense`
/// so the caller's rollback retries it.
pub(super) fn merge_auxiliary(
    dense: &mut Vec<(Slot, Exl3DenseDeviceTensor)>,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let find = |slot| {
        dense
            .iter()
            .find(|(key, _)| *key == slot)
            .map(|(_, tensor)| tensor)
            .context("EXL3 KDA auxiliary companion missing before merge")
    };
    let parts = [find(Slot::BProj)?, find(Slot::FAProj)?, find(Slot::GAProj)?];
    ensure!(
        parts.iter().all(|t| t.dtype() == Exl3DenseDtype::Bf16)
            && parts[1].shape() == parts[2].shape(),
        "EXL3 KDA merged auxiliary requires BF16 B and matching FA/GA"
    );
    let merged = Exl3DenseDeviceTensor::concat_rows(gpu, &parts)?;
    dense.push((Slot::AuxMerged, merged));
    let mut first = None;
    for slot in MERGED {
        let index = dense
            .iter()
            .position(|(key, _)| *key == slot)
            .context("EXL3 KDA auxiliary companion vanished during merge")?;
        let (key, mut tensor) = dense.remove(index);
        if let Err(error) = tensor.release(gpu) {
            dense.push((key, tensor));
            first.get_or_insert(error);
        }
    }
    first.map_or(Ok(()), |e| {
        Err(e.context("EXL3 KDA auxiliary retirement failed"))
    })
}

impl Glm5Exl3KdaDenseResources {
    /// Interior view of the merged base: B at row 0, FA after `heads` rows,
    /// GA after another `dim` rows. `dim` is FB's inner width.
    pub(super) fn merged_view(&self, slot: Slot) -> Result<DenseWeight> {
        let merged = self.tensor(Slot::AuxMerged)?;
        let dim = *self
            .tensor(Slot::FBProj)?
            .shape()
            .get(1)
            .context("EXL3 KDA FB projection must be a matrix")?;
        let [rows, inner] = merged.shape() else {
            anyhow::bail!("EXL3 KDA merged auxiliary must be a matrix");
        };
        let heads = rows
            .checked_sub(2 * dim)
            .filter(|&h| h > 0)
            .context("EXL3 KDA merged auxiliary rows do not cover FA/GA")?;
        let row = match slot {
            Slot::BProj => 0,
            Slot::FAProj => heads,
            Slot::GAProj => heads + dim,
            _ => anyhow::bail!("EXL3 KDA slot {slot:?} is not merged"),
        };
        Ok(DenseWeight {
            weight: merged.ptr()?.offset(row * inner * 2),
        })
    }
}
