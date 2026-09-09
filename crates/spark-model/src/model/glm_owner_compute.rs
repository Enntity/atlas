// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded wider target traversal; caller owns issued-command terminal policy.
use super::{GlmPairedInput, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext,
    glm_owner_verify::{GlmOwnerBatchShape, GlmOwnerBatchWorkspace},
    glm_pair_verify::{GlmPairLayerInput, ROW_BYTES},
};
use crate::{
    layers::ops,
    speculative::ProposerState,
    traits::{Model, SequenceState},
};
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

impl TransformerModel {
    pub(in crate::model) fn owner_finish_verify(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState],
        tokens: &[[u32; 5]],
        accepted: &[usize],
    ) -> Result<()> {
        // The producer has already issued: even malformed local verdicts fail
        // the entire retained session, never reopen it for another proposal.
        (|| {
            ensure!(
                seqs.len() == shape.owners()
                    && tokens.len() == shape.owners()
                    && accepted.len() == shape.owners()
                    && accepted.iter().all(|&a| a <= 4),
                "owner verdict count mismatch"
            );
            match shape.owners() {
                3 => self.owner_finish_fixed::<3>(shape, seqs.try_into()?, tokens, accepted),
                4 => self.owner_finish_fixed::<4>(shape, seqs.try_into()?, tokens, accepted),
                _ => anyhow::bail!("invalid checked owner verdict shape"),
            }
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    fn owner_finish_fixed<const N: usize>(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState; N],
        tokens: &[[u32; 5]],
        accepted: &[usize],
    ) -> Result<()> {
        let mut bases = [0usize; N];
        let capacity = self.paired_owner_capacity()?;
        for (index, seq) in seqs.iter().enumerate() {
            ensure!(
                seq.slot_idx < capacity && (index == 0 || seqs[index - 1].slot_idx < seq.slot_idx),
                "owner verdict physical slots changed"
            );
            bases[index] = seq
                .seq_len
                .checked_sub(5)
                .context("owner verdict lacks five rows")?;
            ensure!(
                seq.tokens.len() == seq.seq_len
                    && seq.tokens.get(bases[index]..) == Some(tokens[index].as_slice()),
                "owner verdict canonical issued append changed"
            );
            self.paired_ssm_bindings(seq)?;
        }
        for owner in 0..N {
            seqs[owner].seq_len = bases[owner] + accepted[owner] + 1;
            seqs[owner].tokens.truncate(seqs[owner].seq_len);
        }
        let capability = self
            .paired_handoff()
            .context("owner verdict capability missing")?;
        self.owner_with_states(seqs, |inputs, states, ctx| {
            capability.record_verify_owners(shape, inputs, &bases, tokens, accepted, states, ctx)
        })?;
        for owner in 0..N {
            self.trim_proposer_state(seqs[owner], accepted[owner], 0)?;
            self.commit_accepted_prefix(seqs[owner], accepted[owner] + 1, 5)?;
        }
        self.sync_secondary()?;
        crate::speculative::glm_paired_execution::GlmPairedExecution::check_communication_health(
            self,
        )
    }

    pub(in crate::model) fn owner_compute_verify(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState],
        tokens: &[[u32; 5]],
    ) -> Result<[[u32; 5]; 4]> {
        ensure!(
            seqs.len() == shape.owners() && tokens.len() == shape.owners(),
            "owner model count mismatch"
        );
        match shape.owners() {
            3 => self.owner_compute_fixed::<3>(shape, seqs.try_into()?, tokens),
            4 => self.owner_compute_fixed::<4>(shape, seqs.try_into()?, tokens),
            _ => anyhow::bail!("invalid checked owner shape"),
        }
    }

    /// Take only after checking every state; restore on every fallible return.
    pub(in crate::model) fn owner_with_states<const N: usize, T>(
        &self,
        seqs: &mut [&mut SequenceState; N],
        body: impl FnOnce(
            &[GlmPairedInput<'_>; N],
            &mut [&mut dyn ProposerState; N],
            &ForwardContext<'_>,
        ) -> Result<T>,
    ) -> Result<T> {
        ensure!(
            matches!(N, 3 | 4) && seqs.iter().all(|s| s.proposer_state.is_some()),
            "owner proposer state missing"
        );
        let mut states: [Box<dyn ProposerState>; N] = std::array::from_fn(|i| {
            seqs[i]
                .proposer_state
                .take()
                .expect("all actual states checked")
        });
        let result = (|| {
            let mut inputs = std::array::from_fn::<_, N, _>(|_| None);
            for (index, seq) in seqs.iter().enumerate() {
                inputs[index] = Some(self.paired_input(seq)?);
            }
            let inputs = inputs.map(|input| input.expect("all actual inputs checked"));
            let mut borrowed: [&mut dyn ProposerState; N] = states
                .each_mut()
                .map(|state| &mut **state as &mut dyn ProposerState);
            body(&inputs, &mut borrowed, &self.glm_repair_context())
        })();
        for (seq, state) in seqs.iter_mut().zip(states) {
            seq.proposer_state = Some(state);
        }
        result
    }

    fn owner_compute_fixed<const N: usize>(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState; N],
        tokens: &[[u32; 5]],
    ) -> Result<[[u32; 5]; 4]> {
        let context = self.glm_repair_context();
        let stream = self.gpu.default_stream();
        let mut workspace = GlmOwnerBatchWorkspace::new(&context, shape)?;
        workspace.scratch.validate_context(&context, stream)?;
        ensure!(
            shape.owners() == N && self.glm_pair_verify_mode.is_some() && self.lora.is_none(),
            "wider compute requires admitted base paired model"
        );
        let capacity = self.paired_owner_capacity()?;
        let max_blocks = self.max_blocks_per_seq as usize;
        ensure!(
            (1..=128).contains(&max_blocks),
            "owner metadata block capacity"
        );
        let stride = (768 + 5 * max_blocks * 4).next_multiple_of(256);
        ensure!(
            stride <= 3328
                && 32768 + N * stride <= 49152
                && self.buffers.sizes().scratch >= 49152
                && self.buffers.sizes().logits
                    >= self
                        .config
                        .vocab_size
                        .checked_mul(N * 10)
                        .context("owner logits overflow")?
                && self.buffers.sizes().token_ids >= N * 20,
            "owner metadata/logits/token arena capacity"
        );
        for (index, seq) in seqs.iter().enumerate() {
            ensure!(
                seq.slot_idx < capacity && (index == 0 || seqs[index - 1].slot_idx < seq.slot_idx),
                "owner slots must be canonical distinct physical slots"
            );
            self.paired_validate_verify(seq, &tokens[index])?;
            let end = seq
                .seq_len
                .checked_add(5)
                .context("owner target end overflow")?;
            ensure!(
                seq.tokens.capacity() >= end && seq.block_table.capacity() >= max_blocks,
                "owner host storage must be reserved before issue"
            );
        }
        for layer in &self.layers {
            layer.validate_glm_owner_verify(&context, shape, stream)?;
        }
        let mut cache = self.kv_cache.lock();
        let mut additional = 0usize;
        for (index, seq) in seqs.iter().enumerate() {
            additional += self
                .paired_target_budget(seq, &cache)?
                .1
                .saturating_sub(seq.block_table.len());
            ensure!(
                seqs[..index].iter().all(|other| seq
                    .block_table
                    .iter()
                    .all(|block| !other.block_table.contains(block))),
                "owner historical target cache maps alias"
            );
        }
        ensure!(
            additional <= cache.num_free_blocks(),
            "owner aggregate target budget exhausted"
        );
        drop(cache);
        let bases: [usize; N] = std::array::from_fn(|i| seqs[i].seq_len);
        let positions: [[usize; 5]; N] =
            std::array::from_fn(|i| std::array::from_fn(|row| bases[i] + row));
        let capability = self
            .paired_handoff()
            .context("owner compute capability missing")?;
        self.owner_with_states(seqs, |inputs, states, ctx| {
            capability.begin_verify_owners(shape, inputs, tokens, states, ctx)
        })?;
        cache = self.kv_cache.lock();
        for seq in seqs.iter_mut() {
            ensure!(
                self.paired_allocate_target(seq, &mut cache, 5, stream)?,
                "owner target allocator disappeared"
            );
        }
        for (index, seq) in seqs.iter().enumerate() {
            ensure!(
                seqs[..index].iter().all(|other| seq
                    .block_table
                    .iter()
                    .all(|block| !other.block_table.contains(block))),
                "owner allocated target cache maps alias"
            );
        }
        let mut token_bytes = [0u8; 80];
        for owner in 0..N {
            for row in 0..5 {
                self.embed(
                    tokens[owner][row],
                    self.buffers
                        .hidden_states()
                        .offset((owner * 5 + row) * ROW_BYTES),
                    stream,
                )?;
                let offset = (owner * 5 + row) * 4;
                token_bytes[offset..offset + 4].copy_from_slice(&tokens[owner][row].to_le_bytes());
            }
        }
        self.gpu
            .copy_h2d_async(&token_bytes[..N * 20], self.buffers.token_ids(), stream)?;
        // Initialized bounded upload storage survives through the synchronous read.
        let mut metadata_host = [[0u8; 3328]; N];
        let mut contexts = std::array::from_fn::<_, N, _>(|_| self.glm_repair_context());
        for owner in 0..N {
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
            let base = self.buffers.scratch().offset(32768 + owner * stride);
            self.gpu.copy_h2d_async(&bytes[..stride], base, stream)?;
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
            let mut ordinal = 0;
            let mut owners = seqs.each_mut().map(|seq| {
                let owner = ordinal;
                ordinal += 1;
                let seq = &mut **seq;
                GlmPairLayerInput {
                    hidden: self.buffers.hidden_states().offset(owner * 5 * ROW_BYTES),
                    state: seq.layer_states[index].as_mut(),
                    positions: &positions[owner],
                    block_table: &seq.block_table,
                }
            });
            layer.decode_glm_owner_verify(
                &mut owners,
                &mut cache,
                &mut workspace,
                &contexts.each_ref(),
                stream,
            )?;
        }
        // Preserve each owner's established K5 normalization/projection arithmetic.
        for owner in 0..N {
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
        for row in 0..N * 5 {
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
        let mut bytes = [0u8; 80];
        self.gpu
            .copy_d2h(self.buffers.scratch(), &mut bytes[..N * 20])?;
        let mut predictions = [[0u32; 5]; 4];
        for (owner, prediction) in predictions[..N].iter_mut().enumerate() {
            for (row, token) in prediction.iter_mut().enumerate() {
                let offset = (owner * 5 + row) * 4;
                *token = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
            }
        }
        drop(cache);
        for owner in 0..N {
            seqs[owner].tokens.extend_from_slice(&tokens[owner]);
            seqs[owner].seq_len = bases[owner] + 5;
        }
        self.owner_with_states(seqs, |inputs, states, ctx| {
            capability.publish_verify_owners(shape, inputs, tokens, &predictions[..N], states, ctx)
        })?;
        Ok(predictions)
    }
}
