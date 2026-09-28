// SPDX-License-Identifier: AGPL-3.0-only
//! Bind actual K5 writers and accepted-prefix copies to the same target slot.
use super::*;
use crate::layer::SsmLayerState;

impl TransformerModel {
    pub(in crate::model) fn paired_ssm_bindings(&self, seq: &SequenceState) -> Result<()> {
        let pool = &self.ssm_pool;
        let count = self.config.num_ssm_layers();
        ensure!(
            count == pool.num_ssm_layers && seq.layer_states.len() == self.layers.len(),
            "paired target layer/SSM pool shape changed"
        );
        if count == 0 {
            return Ok(());
        }
        pool.require_verify_rollback_supported()?;
        let slot = seq.slot_idx;
        ensure!(
            pool.has_mtp
                && slot < pool.mtp_slots
                && slot < pool.max_slots
                && pool.h_inter_counts.len() > slot
                && pool.h_inter_offsets.len() > slot
                && pool.h_inter_count(slot) >= 4
                && pool.num_intermediates >= 5
                && pool.h_state_pools.len() == count
                && pool.conv_state_pools.len() == count
                && pool.h_intermediate_pools.len() == count
                && pool.conv_intermediate_pools.len() == count
                && seq.ssm_slot_idx() == Some(slot)
                && !pool.slot_is_free(slot)
                && seq
                    .ssm_slot
                    .as_ref()
                    .is_some_and(|guard| guard.belongs_to(pool))
                && pool.h_stored_bytes == pool.h_bytes
                && pool.h_bytes == self.config.ssm_h_state_bytes()
                && pool.conv_bytes == self.config.ssm_conv_state_bytes(),
            "paired K5 requires an actual claimed FP32 snapshot slot with K5 capacity"
        );
        let mut index = 0;
        for (layer, state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(layer) != atlas_core::config::LayerType::LinearAttention {
                continue;
            }
            let state = state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .context("paired target SSM state missing")?;
            ensure!(
                !state.h_is_f16
                    && state.h_prefill_stage.is_none()
                    && state.h_state == pool.h_state(index, slot)
                    && state.conv_state == pool.conv_state(index, slot)
                    && state.h_state_intermediates.len() >= 4
                    && state.conv_state_intermediates.len() >= 5,
                "paired actual target SSM destinations differ from slot owners"
            );
            for row in 0..4 {
                ensure!(
                    state.h_state_intermediates[row] == pool.h_intermediate(index, slot, row),
                    "paired target H snapshot differs from actual slot"
                );
            }
            for row in 0..5 {
                ensure!(
                    state.conv_state_intermediates[row] == pool.conv_intermediate(index, slot, row),
                    "paired target conv snapshot differs from actual slot"
                );
            }
            for (ptr, bytes) in [
                (state.h_state, pool.h_stored_bytes),
                (state.conv_state, pool.conv_bytes),
            ] {
                ensure!(
                    !ptr.is_null()
                        && ptr.0.is_multiple_of(4)
                        && bytes > 0
                        && ptr.0.checked_add(bytes as u64).is_some(),
                    "paired target SSM span invalid"
                );
            }
            index += 1;
        }
        ensure!(index == count, "paired actual SSM state coverage mismatch");
        Ok(())
    }
}
