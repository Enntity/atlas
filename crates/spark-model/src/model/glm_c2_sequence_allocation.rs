// SPDX-License-Identifier: AGPL-3.0-only
//! Keep the actual target guard outside fallible initialization until publication.
use super::types::TransformerModel;
use crate::layer::{LayerState, SsmLayerState};
use crate::speculative::ProposerState;
use crate::traits::SequenceState;
use anyhow::{Result, ensure};
use atlas_core::config::LayerType;

struct AllocationParts {
    layer_states: Vec<Box<dyn LayerState>>,
    proposer_state: Option<Box<dyn ProposerState>>,
}

impl TransformerModel {
    pub(super) fn alloc_sequence_owned(
        &self,
        expected_slot: Option<usize>,
    ) -> Result<SequenceState> {
        let paired = self.paired_handoff();
        let candidate = paired
            .map(|capability| capability.validate_allocation(self.gpu.as_ref()))
            .transpose()?;
        let mut slot_guard = self.ssm_pool.claim_guarded()?;
        let slot = slot_guard.idx().expect("claim_guarded owns a slot");
        let result = (|| {
            if let Some(candidate) = candidate {
                ensure!(
                    candidate == slot && expected_slot.is_none_or(|expected| expected == slot),
                    "paired allocation private/target/addressed slot mismatch"
                );
            }
            self.initialize_sequence_parts(slot)
        })();
        let AllocationParts {
            layer_states,
            proposer_state,
        } = match result {
            Ok(parts) => parts,
            Err(error) => {
                if paired.is_some() {
                    slot_guard.take();
                    return Err(self.paired_ownership_error(error));
                }
                return Err(error);
            }
        };
        // No graph invalidation needed — pool addresses are stable across sequences.

        // Phase 6.1.d critical fix: pre-size disk_last_offloaded_per_layer
        // to the model's attention-layer count. The vector stays empty
        // when HSS isn't engaged (cache_blocks_per_seq is None) — the
        // helper short-circuits before reading from it. Sized here once
        // so the layer-0 offload helper doesn't need to grow a Vec on
        // every sequence's first decode step.
        let num_attn_layers = self.config.num_attention_layers();
        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,          // default: defer to installed active adapter
            acquired_adapter_slot: -1, // Task #25: no ref held until prefill acquires
            src_lang_id: 0,            // NLLB-only per-request lang (0 = deployment default)
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states,
            proposer_state,
            slot_idx: slot,
            ssm_slot: Some(slot_guard),
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            mtp_capture_gen: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            cached_prefix_blocks: 0,
            prefix_ref_tokens: Vec::new(),
            prefix_lookup_applied: false,
            prefix_lookup_skip: false,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: vec![0; num_attn_layers],
        })
    }

    fn initialize_sequence_parts(&self, slot: usize) -> Result<AllocationParts> {
        // Zero SSM state to prevent stale h_state/conv_state from prior
        // sequences corrupting the recurrent computation during prefill.
        // CRITICAL: use Atlas's own stream (not stream 0) because Atlas's stream
        // is CU_STREAM_NON_BLOCKING and does NOT synchronize with stream 0.
        // Using stream 0 would race with the subsequent prefill kernel.
        let stream = self.gpu.default_stream();
        self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
        // Ensure zero completes before any prefill kernels touch this slot.
        self.gpu.synchronize(stream)?;
        // Worker ranks do not own a model-specific proposer, but they still
        // run the distributed target verify and must attach the rollback
        // buffers allocated in the shared SSM pool.
        let has_mtp = self.ssm_pool.has_mtp;

        // ATLAS_MTP_DRAFTER_PREFILL: a fresh sequence invalidates the
        // whole-prompt hidden capture — without this, a warm-restored prefill
        // (no chunks computed) would pair the NEW prompt's tokens with the
        // PREVIOUS sequence's captured hiddens in the drafter prefill.
        self.mtp_prefill_capture_len
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // ATLAS_MTP_CARRY_DRAFTER: the position-indexed hidden interval is
        // per-sequence by construction. Resetting it here is what makes the
        // carry path immune to the latent cross-sequence stale-hidden bug that
        // the legacy `captured >= prompt_len` guard still has: a warm-turn
        // append can only ever read rows THIS sequence's prefill wrote.
        *self.mtp_store_range.lock() = (0, 0);

        // Build layer states: SSM layers point into the pool (fixed addresses),
        // attention layers use their own alloc_state (EmptyLayerState).
        // When MTP is available, pre-allocate checkpoint + K=2 intermediate
        // buffers so CUDA graph capture doesn't trigger lazy allocation.
        let mut ssm_layer_idx = 0usize;
        let mut layer_states: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                // Layer-independent (one FP32 staging blob per SLOT), so it is
                // the same pointer for every SSM layer of this sequence.
                let stage = self.ssm_pool.h_prefill_stage(slot);
                let mut ssm_state = SsmLayerState {
                    h_state: self.ssm_pool.h_state(ssm_layer_idx, slot),
                    conv_state: self.ssm_pool.conv_state(ssm_layer_idx, slot),
                    h_state_checkpoint: None,
                    conv_state_checkpoint: None,
                    h_state_intermediates: Vec::new(),
                    conv_state_intermediates: Vec::new(),
                    // A freshly allocated slot has just been zeroed, and zero
                    // is zero in both formats. Which format it then HOLDS is
                    // decided by the pool width, not by the phase: under the
                    // stage-3 f16-SIZED pool the slot is physically 2
                    // bytes/element and prefill stages its FP32 work
                    // elsewhere, so the slot is FP16 from here on and the
                    // decode mixer has nothing to do. Under an FP32-sized
                    // pool prefill writes FP32 in place and FP32 is the truth
                    // until the mixer converts.
                    h_is_f16: stage.is_some(),
                    h_prefill_stage: stage,
                };

                if has_mtp {
                    // Use pool-based fixed addresses (stable across sequence
                    // lifetimes → CUDA graph can replay without stale pointers).
                    ssm_state.h_state_checkpoint =
                        Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, slot));
                    ssm_state.conv_state_checkpoint =
                        Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, slot));

                    // Tiered pools: H count is per-SLOT (h_inter_count),
                    // conv count is uniform. The vec lengths are the
                    // capacity gates every verify arm checks before writing.
                    for t in 0..self.ssm_pool.h_inter_count(slot) {
                        ssm_state
                            .h_state_intermediates
                            .push(self.ssm_pool.h_intermediate(ssm_layer_idx, slot, t));
                    }
                    for t in 0..self.ssm_pool.num_intermediates {
                        ssm_state
                            .conv_state_intermediates
                            .push(self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, t));
                    }
                }

                layer_states.push(Box::new(ssm_state));
                ssm_layer_idx += 1;
            } else {
                layer_states.push(layer.alloc_state(self.gpu.as_ref())?);
            }
        }

        // Zero SSM states for the new sequence.
        // Synchronous reset: memset + stream sync ensures zero is visible
        // before any subsequent kernel reads the state.
        self.ssm_pool.reset_slot(slot, self.gpu.as_ref())?;
        // Double-check: explicit sync to guarantee zero is complete
        self.gpu.synchronize(self.gpu.default_stream())?;

        // Allocate MTP proposer state (owns its own KV cache block table)
        let proposer_state = match &self.proposer {
            Some(p) => Some(p.alloc_state(self.gpu.as_ref())?),
            None => None,
        };

        Ok(AllocationParts {
            layer_states,
            proposer_state,
        })
    }
}
