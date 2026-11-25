// SPDX-License-Identifier: AGPL-3.0-only

//! LoopCD's reference coda pass (`atlas_core::config::LoopCdMode`; oMLX
//! `Inkling2TextModel.__call__`, jundot/omlx#5640, Apache-2.0).
//!
//! The second pass's restart visit stashes the first pass: its output hidden
//! and the blocks it pushed. After the model's last visit has left the final
//! output in `hidden`, the reference visits (the coda layers again, each with
//! its own KV and conv state) set it aside, restore the first pass and run;
//! the last one leaves the reference rows in `Shared::loopcd_ref` and puts
//! the final output back, so the model's final norm and head see the rows
//! they always did. Everything here is within one forward, so graph capture,
//! verify journals and prefix snapshots carry the reference visits like any
//! other.

use std::sync::atomic::Ordering;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layer::Inkling2Layer;
use super::schedule::LoopCdRole;

impl Inkling2Layer {
    /// Before this visit runs over `n` rows of `hidden`.
    pub(super) fn loopcd_enter(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let (s, d) = (&self.shared, &self.dims);
        let rb = n * d.hidden * 2;
        let bstride = d.block_stride() * 2;
        let (first, count) = d.loopcd_stash;
        let stack = |j: usize| s.blocks.offset((first + j) * bstride);
        let stash = |j: usize| s.ref_blocks.offset(j * bstride);
        match self.visit.loopcd {
            LoopCdRole::SaveFirstPass => {
                gpu.copy_d2d_async(hidden, s.ref_h, rb, stream)?;
                for j in 0..count {
                    gpu.copy_d2d_async(stack(j), stash(j), rb, stream)?;
                }
            }
            LoopCdRole::Reference { enter: true, .. } => {
                gpu.copy_d2d_async(hidden, s.keep_final, rb, stream)?;
                gpu.copy_d2d_async(s.ref_h, hidden, rb, stream)?;
                for j in 0..count {
                    gpu.copy_d2d_async(stash(j), stack(j), rb, stream)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Whether this visit ends the reference pass.
    pub(super) fn loopcd_leaves(&self) -> bool {
        matches!(self.visit.loopcd, LoopCdRole::Reference { leave: true, .. })
    }

    /// The last reference visit, its rows cast into `loopcd_ref`: put the
    /// final output back and say where the forward's rows were.
    pub(super) fn loopcd_leave(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let rb = n * self.dims.hidden * 2;
        gpu.copy_d2d_async(self.shared.keep_final, hidden, rb, stream)?;
        self.loopcd_rows.store(hidden.0, Ordering::Relaxed);
        Ok(())
    }

    /// LoopCD's reference rows for the model's head (`model::loopcd`):
    /// `None` unless this is the last reference visit.
    pub(crate) fn loopcd_reference(&self) -> Option<crate::model::LoopCdRows> {
        self.loopcd_leaves().then(|| crate::model::LoopCdRows {
            forward_hidden: self.loopcd_rows.clone(),
            rows: self.shared.loopcd_ref,
            max_rows: self.dims.max_rows,
        })
    }
}
