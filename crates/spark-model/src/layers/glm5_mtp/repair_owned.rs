// SPDX-License-Identifier: AGPL-3.0-only
//! Request-owned prompt tail and accepted-row staging for serial C4 repair.
use super::*;
use crate::speculative::glm_repair::RepairSpan;
use anyhow::{Context, ensure};

pub(crate) fn enabled() -> bool {
    crate::speculative::glm_repair_policy::enabled()
        && std::env::var("ATLAS_GLM_MTP_LONG_CONTEXT").as_deref() == Ok("1")
}

pub(crate) fn permits_capacity(capacity: u32) -> bool {
    capacity == 1 || (enabled() && capacity == 4)
}

pub(super) struct OwnedRepairRows {
    pub(super) ptr: DevicePtr,
    row_bytes: usize,
    generation: u64,
    prompt: usize,
}

impl OwnedRepairRows {
    pub(super) fn validate(&self, generation: u64, prompt: usize, row_bytes: usize) -> Result<()> {
        ensure!(
            self.ptr.0 != 0
                && self.ptr.0.is_multiple_of(2)
                && generation != 0
                && generation == self.generation
                && prompt > 0
                && prompt == self.prompt
                && row_bytes == self.row_bytes,
            "GLM retained prompt tail ownership mismatch"
        );
        Ok(())
    }
    pub(super) fn staging(&self) -> RepairSpan {
        RepairSpan {
            ptr: self.ptr.offset(self.row_bytes),
            bytes: self.row_bytes * 2,
        }
    }
}

impl Glm5MtpProposerState {
    pub(crate) fn repair_capture_generation(
        &self,
        generation: u64,
        prompt: usize,
    ) -> Result<Option<u64>> {
        let Some(rows) = &self.repair_owned else {
            return Ok(None);
        };
        rows.validate(generation, prompt, rows.row_bytes)?;
        Ok(Some(rows.generation))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn retain_repair_prompt_tail(
        &mut self,
        gpu: &dyn GpuBackend,
        source: DevicePtr,
        generation: u64,
        prompt: usize,
        row_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.paired.is_none()
                && self.repair_owned.is_none()
                && matches!(self.repair, repair_state::RepairPhase::Capture)
                && generation != 0
                && prompt > 1
                && self.seq_len == prompt - 1
                && row_bytes > 0
                && source.0 != 0,
            "GLM retained tail requires a completed owned eager prompt primer"
        );
        let bytes = row_bytes
            .checked_mul(3)
            .context("GLM retained row allocation overflow")?;
        let ptr = gpu.alloc(bytes)?;
        if let Err(error) = gpu
            .copy_d2d_async(source, ptr, row_bytes, stream)
            .and_then(|_| gpu.synchronize(stream))
        {
            let _ = gpu.free(ptr);
            self.repair = repair_state::RepairPhase::Failed;
            return Err(error);
        }
        self.repair_owned = Some(OwnedRepairRows {
            ptr,
            row_bytes,
            generation,
            prompt,
        });
        Ok(())
    }

    pub(super) fn free_repair_owned(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(rows) = self.repair_owned.take() {
            gpu.free(rows.ptr)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "repair_owned_tests.rs"]
mod tests;
