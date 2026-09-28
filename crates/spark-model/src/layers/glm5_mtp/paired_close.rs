// SPDX-License-Identifier: AGPL-3.0-only
//! Invalidate readers before fallible model cleanup; completion failures stick.
use super::*;

impl Glm5MtpHead {
    pub(super) fn paired_retire(
        &self,
        state: &mut dyn ProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<usize>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired retirement requires actual GLM state")?;
        let mut pool = self.paired.as_ref().context("paired pool missing")?.lock();
        pool.begin_retire(state, gpu)
    }

    pub(super) fn paired_close(&self, gpu: &dyn GpuBackend, secondary_stream: u64) -> Result<()> {
        let owner = self.paired.as_ref().context("paired pool missing")?;
        let slab = {
            let mut pool = owner.lock();
            ensure!(
                pool.backend == gpu as *const dyn GpuBackend as *const () as usize,
                "foreign paired close backend"
            );
            if pool.closed {
                ensure!(
                    !pool.close_failed,
                    "paired close previously failed; no retry or sweep"
                );
                return Ok(());
            }
            pool.closed = true;
            // Pessimistic until both completion and the one free attempt succeed.
            pool.close_failed = true;
            for slot in &mut pool.slots {
                slot.retiring = true;
            }
            pool.slab
        };
        // No pool lock may span a backend callback. Unknown completion forbids
        // freeing any model owner or invoking its bulk sweep in release_pools.
        gpu.synchronize(gpu.default_stream())
            .context("paired close completion failed; model release prohibited")?;
        // A failed record_event may leave secondary writes absent from the
        // event's history. Join the actual model stream, not only that event.
        if secondary_stream != gpu.default_stream() {
            gpu.synchronize(secondary_stream)
                .context("paired secondary close completion failed; model release prohibited")?;
        }
        gpu.free(slab)
            .context("paired slab free failed; no retry")?;
        owner.lock().close_failed = false;
        Ok(())
    }
}
