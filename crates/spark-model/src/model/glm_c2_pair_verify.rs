// SPDX-License-Identifier: AGPL-3.0-only
//! Actual fixed-pair target traversal; transport owns post-header terminal policy.
use super::{GlmPairedInput, TransformerModel};
use crate::layer::glm_pair_verify::{GlmPairLayerInput, GlmPairWorkspace, ROW_BYTES};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::speculative::ProposerState;
use crate::traits::SequenceState;
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

impl TransformerModel {
    /// Both actual states are checked before either take, then restored on every
    /// fallible return. Inputs retain the original Model-owned normalized arena.
    pub(in crate::model) fn paired_with_states<T>(
        &self,
        seqs: &mut [&mut SequenceState; 2],
        body: impl FnOnce(
            &[GlmPairedInput<'_>; 2],
            [&mut dyn ProposerState; 2],
            &ForwardContext<'_>,
        ) -> Result<T>,
    ) -> Result<T> {
        ensure!(
            seqs.iter().all(|seq| seq.proposer_state.is_some()),
            "actual pair proposer state missing"
        );
        let mut state0 = seqs[0]
            .proposer_state
            .take()
            .expect("checked actual state0");
        let mut state1 = seqs[1]
            .proposer_state
            .take()
            .expect("checked actual state1");
        let result = (|| {
            let inputs = [self.paired_input(seqs[0])?, self.paired_input(seqs[1])?];
            body(
                &inputs,
                [state0.as_mut(), state1.as_mut()],
                &self.glm_repair_context(),
            )
        })();
        seqs[0].proposer_state = Some(state0);
        seqs[1].proposer_state = Some(state1);
        result
    }

    pub(in crate::model) fn paired_compute_verify(
        &self,
        mut seqs: [&mut SequenceState; 2],
        tokens: &[[u32; 5]; 2],
    ) -> Result<[[u32; 5]; 2]> {
        let mode = self
            .glm_pair_verify_mode
            .context("pair compute mode was not admitted")?;
        crate::layers::glm5_mtp::Glm5MtpHead::validate_fixed_pair_slots(
            [seqs[0].slot_idx, seqs[1].slot_idx],
            self.paired_owner_capacity()?,
        )?;
        for owner in 0..2 {
            self.paired_validate_verify(seqs[owner], &tokens[owner])?;
        }
        let stream = self.gpu.default_stream();
        let context = self.glm_repair_context();
        let mut workspace = GlmPairWorkspace::new(&context, mode)?;
        workspace.validate_context(&context, stream)?;
        let max_blocks = self.max_blocks_per_seq as usize;
        ensure!(
            (1..=128).contains(&max_blocks),
            "pair metadata block capacity"
        );
        let metadata_stride = (768 + 5 * max_blocks * 4).next_multiple_of(256);
        let metadata_end = 32768 + 2 * metadata_stride;
        let logits_bytes = self
            .config
            .vocab_size
            .checked_mul(20)
            .context("pair logits overflow")?;
        ensure!(
            metadata_end <= 49152
                && self.buffers.sizes().scratch >= metadata_end
                && self.buffers.sizes().logits >= logits_bytes
                && self.buffers.sizes().token_ids >= 40
                && self.lora.is_none(),
            "pair target metadata/logits/token arena or base-only profile invalid"
        );
        let bases = [seqs[0].seq_len, seqs[1].seq_len];
        // Existing per-owner preflight checked both ends before these bounded sums.
        let positions: [[usize; 5]; 2] =
            std::array::from_fn(|owner| std::array::from_fn(|row| bases[owner] + row));
        let capability = self
            .paired_handoff()
            .context("pair compute capability missing")?;
        self.paired_with_states(&mut seqs, |inputs, states, ctx| {
            capability.begin_verify_pair(inputs, tokens, states, ctx)
        })?;

        let mut cache = self.kv_cache.lock();
        let needed0 = self.paired_target_budget(seqs[0], &cache)?.1;
        let needed1 = self.paired_target_budget(seqs[1], &cache)?.1;
        let additional = needed0.saturating_sub(seqs[0].block_table.len())
            + needed1.saturating_sub(seqs[1].block_table.len());
        ensure!(
            additional <= cache.num_free_blocks(),
            "pair aggregate target budget exhausted"
        );
        for seq in &mut seqs {
            ensure!(
                self.paired_allocate_target(seq, &mut cache, 5, stream)?,
                "actual pair allocator capability disappeared"
            );
        }
        ensure!(
            !seqs[0]
                .block_table
                .iter()
                .any(|block| seqs[1].block_table.contains(block)),
            "pair target owners alias physical cache blocks"
        );
        for owner in 0..2 {
            for row in 0..5 {
                self.embed(
                    tokens[owner][row],
                    self.buffers
                        .hidden_states()
                        .offset((owner * 5 + row) * ROW_BYTES),
                    stream,
                )?;
            }
        }
        let token_bytes: Vec<u8> = tokens
            .iter()
            .flatten()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        self.gpu
            .copy_h2d_async(&token_bytes, self.buffers.token_ids(), stream)?;

        // Retain initialized upload storage through the final synchronous read.
        let mut metadata_host = [vec![0u8; metadata_stride], vec![0u8; metadata_stride]];
        let mut contexts = [self.glm_repair_context(), self.glm_repair_context()];
        for owner in 0..2 {
            let bytes = &mut metadata_host[owner];
            for row in 0..5 {
                let position = positions[owner][row];
                let block =
                    self.paired_physical_block(seqs[owner], position / cache.block_size(), true)?;
                let slot = block as i64 * cache.block_size() as i64
                    + (position % cache.block_size()) as i64;
                bytes[row * 4..row * 4 + 4].copy_from_slice(&(position as u32).to_le_bytes());
                bytes[256 + row * 8..256 + row * 8 + 8].copy_from_slice(&slot.to_le_bytes());
                bytes[512 + row * 4..512 + row * 4 + 4]
                    .copy_from_slice(&((position + 1) as i32).to_le_bytes());
                for (column, &block) in seqs[owner].block_table.iter().enumerate() {
                    let offset = 768 + (row * max_blocks + column) * 4;
                    bytes[offset..offset + 4].copy_from_slice(&(block as i32).to_le_bytes());
                }
            }
            let base = self
                .buffers
                .scratch()
                .offset(32768 + owner * metadata_stride);
            self.gpu.copy_h2d_async(bytes, base, stream)?;
            contexts[owner].attn_metadata = Some(AttnMetadataDev {
                positions: base,
                positions_h: base,
                positions_w: base,
                slot: base.offset(256),
                seq_len: base.offset(512),
                block_table: base.offset(768),
                max_blocks_per_seq: self.max_blocks_per_seq,
                num_seqs: 5,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            });
            contexts[owner].token_ids = Some(self.buffers.token_ids().offset(owner * 20));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let [seq0, seq1] = &mut seqs;
            let owners = [
                GlmPairLayerInput {
                    hidden: self.buffers.hidden_states(),
                    state: seq0.layer_states[index].as_mut(),
                    positions: &positions[0],
                    block_table: &seq0.block_table,
                },
                GlmPairLayerInput {
                    hidden: self.buffers.hidden_states().offset(5 * ROW_BYTES),
                    state: seq1.layer_states[index].as_mut(),
                    positions: &positions[1],
                    block_table: &seq1.block_table,
                },
            ];
            layer.decode_glm_pair_verify(
                owners,
                &mut cache,
                &mut workspace,
                [&contexts[0], &contexts[1]],
                stream,
            )?;
        }
        // Preserve the two established M5 finalization paths and their BF16 ABI.
        for owner in 0..2 {
            let normalized = self.buffers.norm_output().offset(owner * 5 * ROW_BYTES);
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                self.buffers.hidden_states().offset(owner * 5 * ROW_BYTES),
                &self.final_norm,
                normalized,
                5,
                self.config.hidden_size as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
            self.lm_head_batched(
                normalized,
                5,
                self.buffers
                    .logits()
                    .offset(owner * 5 * self.config.vocab_size * 2),
                stream,
            )?;
        }
        for row in 0..10 {
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                self.buffers
                    .logits()
                    .offset(row * self.config.vocab_size * 2),
                self.buffers.scratch().offset(row * 4),
                self.config.vocab_size as u32,
                stream,
            )?;
        }
        let mut bytes = [0u8; 40];
        self.gpu.copy_d2h(self.buffers.scratch(), &mut bytes)?;
        let predictions = std::array::from_fn(|owner| {
            std::array::from_fn(|row| {
                let offset = (owner * 5 + row) * 4;
                u32::from_le_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .expect("fixed four-byte token"),
                )
            })
        });
        drop(cache);
        for owner in 0..2 {
            seqs[owner].tokens.extend_from_slice(&tokens[owner]);
            seqs[owner].seq_len = bases[owner] + 5;
        }
        self.paired_with_states(&mut seqs, |inputs, states, ctx| {
            capability.publish_verify_pair(inputs, tokens, &predictions, states, ctx)
        })?;
        Ok(predictions)
    }
}
