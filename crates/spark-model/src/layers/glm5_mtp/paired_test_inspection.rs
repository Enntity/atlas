// SPDX-License-Identifier: AGPL-3.0-only
//! Read-only actual-owner inspection, shared by unit and explicit dependency tests.
use super::*;

impl Glm5MtpHead {
    pub(crate) fn paired_test_free_blocks(&self) -> usize {
        self.kv_cache.lock().num_free_blocks()
    }
    pub(crate) fn paired_test_kv_rows(
        &self,
        state: &dyn ProposerState,
        backend: &dyn GpuBackend,
        rows: usize,
    ) -> Result<Vec<(DevicePtr, DevicePtr)>> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("inspection requires actual GLM state")?;
        self.validate_paired_live(state, backend)?;
        let pool = self
            .paired
            .as_ref()
            .context("inspection requires paired pool")?
            .lock();
        let slot = &pool.slots[state.paired.as_ref().expect("validated lease").slot];
        let cache = self.kv_cache.lock();
        ensure!(
            rows <= slot.blocks.len() * cache.block_size(),
            "inspection exceeds canonical reserve"
        );
        let k_stride = cache.k_block_stride_bytes_for_layer(0) / cache.block_size();
        let v_stride = cache.v_block_stride_bytes_for_layer(0) / cache.block_size();
        Ok((0..rows)
            .map(|row| {
                let block = slot.blocks[row / cache.block_size()];
                let offset = row % cache.block_size();
                (
                    cache.k_cache_ptr(0, block).offset(offset * k_stride),
                    cache.v_cache_ptr(0, block).offset(offset * v_stride),
                )
            })
            .collect())
    }
}
