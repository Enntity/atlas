// SPDX-License-Identifier: AGPL-3.0-only
//! Selected inner Result boundaries retain owners until fallible work completes.
use super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::{Model, SequenceState};
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

impl TransformerModel {
    pub(super) fn free_paired_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.retire_paired_sequence(seq).map_err(|error| {
            // Even a rejected foreign/missing-view cleanup cannot leave a held
            // guard whose later Drop recycles an index after selected failure.
            if let Some(guard) = seq.ssm_slot.as_mut() {
                guard.take();
            }
            self.paired_ownership_error(error)
        })
    }

    fn retire_paired_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        let capability = self
            .paired_handoff()
            .context("paired cleanup capability missing")?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("paired cleanup state missing")?;
        let Some(slot) = capability.retire(state.as_mut(), self.gpu.as_ref())? else {
            // No lease is not authority to use the old public slot number.
            // A genuinely completed old object is inert even after replacement.
            ensure!(
                seq.ssm_slot_idx().is_none()
                    && seq.block_table.is_empty()
                    && seq.disk_block_ids.is_empty()
                    && seq.disk_last_offloaded_per_layer.iter().all(|v| *v == 0)
                    && seq.chunked_prefill_meta.is_none()
                    && seq.acquired_adapter_slot < 0
                    && seq.cached_prefix_tokens == 0
                    && seq.cached_prefix_blocks == 0
                    && seq.prefix_ref_tokens.is_empty()
                    && seq.marconi_exact_snap.is_none()
                    && seq.layer_states.iter().all(|state| {
                        state
                            .as_any()
                            .downcast_ref::<SsmLayerState>()
                            .is_none_or(|ssm| {
                                ssm.h_state.is_null()
                                    && ssm.conv_state.is_null()
                                    && ssm.h_prefill_stage.is_none()
                                    && ssm.h_state_checkpoint.is_none()
                                    && ssm.conv_state_checkpoint.is_none()
                                    && ssm.h_state_intermediates.is_empty()
                                    && ssm.conv_state_intermediates.is_empty()
                            })
                    }),
                "paired no-lease cleanup still retains target resources"
            );
            return Ok(());
        };
        let guard = seq
            .ssm_slot
            .as_mut()
            .context("paired live target guard missing")?;
        ensure!(
            guard.belongs_to(&self.ssm_pool) && guard.idx() == Some(slot) && seq.slot_idx == slot,
            "paired cleanup target/private guard ownership mismatch"
        );
        guard.take();
        ensure!(
            seq.adapter_id == 0
                && seq.adapter_slot < 0
                && seq.acquired_adapter_slot < 0
                && !self.prefix_cache.is_active()
                && seq.cached_prefix_tokens == 0
                && seq.cached_prefix_blocks == 0
                && seq.prefix_ref_tokens.is_empty()
                && seq.marconi_exact_snap.is_none()
                && seq.disk_block_ids.is_empty()
                && seq.disk_last_offloaded_per_layer.iter().all(|v| *v == 0)
                && self.kv_cache.lock().config().cache_blocks_per_seq.is_none(),
            "paired cleanup requires base, uncached, fixed-slot target ownership"
        );
        let stream = self.gpu.default_stream();
        self.gpu.stream_wait_event(stream, self.secondary_event)?;
        self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
        self.gpu.synchronize(stream)?;
        for graph_map in [&self.verify_kgamma_graph, &self.fused_graph] {
            loop {
                let graph = {
                    let mut cache = graph_map.lock();
                    let key = cache.keys().find(|key| key.0 == slot).copied();
                    key.and_then(|key| cache.remove(&key))
                };
                let Some(graph) = graph else { break };
                // Remove before the single attempt; unknown destruction is not retryable.
                self.gpu.destroy_graph(graph)?;
            }
        }
        self.free_chunked_prefill_meta(seq)?;
        self.proposer
            .as_ref()
            .context("paired proposer missing")?
            .free_state(
                self.gpu.as_ref(),
                seq.proposer_state
                    .as_mut()
                    .expect("validated state")
                    .as_mut(),
            )?;
        // All remaining transitions are host-only. No resource returns before
        // the last fallible private completion operation has succeeded.
        self.kv_cache.lock().free_blocks(&seq.block_table);
        seq.block_table.clear();
        for state in &mut seq.layer_states {
            if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                ssm.h_state = DevicePtr(0);
                ssm.conv_state = DevicePtr(0);
                ssm.h_prefill_stage = None;
                ssm.h_state_checkpoint = None;
                ssm.conv_state_checkpoint = None;
                ssm.h_state_intermediates.clear();
                ssm.conv_state_intermediates.clear();
            }
        }
        self.ssm_pool.release_slot(slot);
        Ok(())
    }

    pub(super) fn paired_ownership_error(&self, error: anyhow::Error) -> anyhow::Error {
        let latched = self
            .paired_handoff()
            .context("paired ownership capability missing")
            .and_then(|capability| capability.fail_session(self.gpu.as_ref()));
        error.context(format!(
            "paired ownership failure is terminal; latch={latched:?}"
        ))
    }

    pub(super) fn paired_replace_worker_slot(
        &self,
        slots: &mut [Option<SequenceState>],
        slot: usize,
    ) -> Result<bool> {
        (|| {
            if let Some(old) = slots[slot].as_mut() {
                self.free_sequence(old)?;
            }
            let replacement = self.alloc_sequence_owned(Some(slot))?;
            slots[slot] = Some(replacement);
            Ok(true)
        })()
        .map_err(|error| self.paired_ownership_error(error))
    }
}
