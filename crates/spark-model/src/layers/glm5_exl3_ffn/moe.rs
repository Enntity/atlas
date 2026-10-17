// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-native routed and shared EXL3 experts, with layer-shared scratch.

use super::tables::ExpertTable;
use super::*;
use spark_runtime::weights::exl3::dense::{Exl3DenseDeviceTensor, Exl3DenseDtype};

mod coop;
mod coop_fat;
mod overlap;
pub(super) use overlap::{overlap_side_stream, shared_overlap};
mod forward;
mod policy;
mod project;
mod setup;
pub(super) use policy::RoutedPrecisionPolicy;

pub struct NativeMoeFfn {
    pub(super) layer: Option<usize>,
    pub(super) capture: Option<diagnostics::Exl3FfnRouter>,
    pub(super) shared: NativeDenseFfn,
    tables: Vec<ExpertTable>,
    companions: Vec<Exl3DenseDeviceTensor>,
    buffers: [DevicePtr; 11],
    scratch: MoeScratchBorrow,
    owned_scratch: Option<MoeScratch>,
    kernels: [KernelHandle; 9],
    activation: Exl3FfnActivation,
    precision: RoutedPrecisionPolicy,
    pub(super) hidden: usize,
    intermediate: usize,
    experts: usize,
    topk: usize,
    normalize: bool,
    scale: f32,
    pub(super) rank: usize,
    pub(super) world: usize,
    capacity: usize,
    stream: u64,
    active: bool,
    /// Gate control: skip the shared expert (combine gets has_shared=0).
    drop_shared: bool,
    /// Gate control: BF16-round the running routed sum per expert.
    round_routed_accum: bool,
    /// `ATLAS_EXL3_COOP_MOE=1` thin path; None keeps the existing kernels.
    coop: Option<coop::CoopMoe>,
    /// `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1` grouped prefill path; None keeps
    /// the existing kernels.
    coop_fat: Option<coop_fat::CoopFat>,
    /// `ATLAS_EXL3_SHARED_OVERLAP=1`: the shared expert runs on its own side
    /// stream (it was built bound to it), forked after the router and
    /// joined before its output is consumed. None: same stream, in order.
    overlap: Option<overlap::SharedOverlap>,
}
impl NativeMoeFfn {
    pub fn validate(&self, rows: usize, stream: u64, rank: usize, world: usize) -> Result<()> {
        ensure!(
            self.active
                && rows <= self.capacity
                && stream == self.stream
                && rank == self.rank
                && world == self.world,
            "EXL3 MoE lifetime/capacity/stream/topology mismatch"
        );
        self.shared
            .validate(rows, self.shared.stream, rank, world)?;
        ensure!(
            self.shared.stream == stream || self.overlap.is_some(),
            "EXL3 shared expert stream differs without an overlap fork/join"
        );
        self.scratch
            .validate(self.scratch.geometry(), rows, stream)?;
        ensure!(
            self.buffers.iter().all(|p| !p.is_null()),
            "EXL3 MoE scratch has been released"
        );
        for c in &self.companions {
            c.ptr()?;
        }
        Ok(())
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for NativeMoeFfn {
    fn label(&self) -> &'static str {
        "GLM EXL3 native MoE"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        self.active = false;
        let mut error = None;
        retain_error(&mut error, self.shared.release(gpu));
        for t in &mut self.tables {
            retain_error(&mut error, t.release(gpu));
        }
        for c in &mut self.companions {
            retain_error(&mut error, c.release(gpu));
        }
        if let Some(owner) = &mut self.owned_scratch {
            retain_error(&mut error, owner.release(gpu));
        }
        if let Some(coop) = self.coop.take() {
            retain_error(&mut error, gpu.free(coop.locks));
            if let Some(mr) = coop.multirow {
                retain_error(&mut error, gpu.free(mr.groups));
            }
        }
        if let Some(coop_fat) = self.coop_fat.take() {
            retain_error(&mut error, coop_fat.release(gpu));
        }
        if let Some(overlap) = self.overlap.take() {
            retain_error(&mut error, overlap.release(gpu));
        }
        finish(error)
    }
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod load_tests;
