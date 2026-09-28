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
                5 => self.owner_finish_fixed::<5>(shape, seqs.try_into()?, tokens, accepted),
                6 => self.owner_finish_fixed::<6>(shape, seqs.try_into()?, tokens, accepted),
                7 => self.owner_finish_fixed::<7>(shape, seqs.try_into()?, tokens, accepted),
                8 => self.owner_finish_fixed::<8>(shape, seqs.try_into()?, tokens, accepted),
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
        let borrowed = seqs.each_ref().map(|seq| &**seq);
        let bases = self.owner_verdict_preflight(shape, &borrowed, tokens, accepted)?;
        for owner in 0..N {
            seqs[owner].seq_len = bases[owner] + accepted[owner] + 1;
            seqs[owner].tokens.truncate(seqs[owner].seq_len);
        }
        let capability = self
            .paired_handoff()
            .context("owner verdict capability missing")?;
        self.owner_with_states(seqs, |inputs, states, ctx| {
            capability.record_verify_owners(
                shape,
                inputs,
                &bases[..N],
                tokens,
                accepted,
                states,
                ctx,
            )
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
    ) -> Result<[[u32; 5]; 8]> {
        ensure!(
            seqs.len() == shape.owners() && tokens.len() == shape.owners(),
            "owner model count mismatch"
        );
        match shape.owners() {
            3 => self.owner_compute_fixed::<3>(shape, seqs.try_into()?, tokens),
            4 => self.owner_compute_fixed::<4>(shape, seqs.try_into()?, tokens),
            5 => self.owner_compute_fixed::<5>(shape, seqs.try_into()?, tokens),
            6 => self.owner_compute_fixed::<6>(shape, seqs.try_into()?, tokens),
            7 => self.owner_compute_fixed::<7>(shape, seqs.try_into()?, tokens),
            8 => self.owner_compute_fixed::<8>(shape, seqs.try_into()?, tokens),
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
            (3..=8).contains(&N) && seqs.iter().all(|s| s.proposer_state.is_some()),
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
    ) -> Result<[[u32; 5]; 8]> {
        let context = self.glm_repair_context();
        let stream = self.gpu.default_stream();
        let mut workspace = GlmOwnerBatchWorkspace::new(&context, shape)?;
        let borrowed = seqs.each_ref().map(|seq| &**seq);
        let metadata = self.owner_compute_preflight(shape, &borrowed, tokens)?;
        let stride = metadata.stride;
        let max_blocks = self.max_blocks_per_seq as usize;
        let bases: [usize; N] = std::array::from_fn(|i| seqs[i].seq_len);
        let positions: [[usize; 5]; N] =
            std::array::from_fn(|i| std::array::from_fn(|row| bases[i] + row));
        let capability = self
            .paired_handoff()
            .context("owner compute capability missing")?;
        self.owner_with_states(seqs, |inputs, states, ctx| {
            capability.begin_verify_owners(shape, inputs, tokens, states, ctx)
        })?;
        let mut cache = self.kv_cache.lock();
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
        let mut token_bytes = [0u8; 8 * 5 * 4];
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
            let base = self.buffers.scratch().offset(metadata.offset(owner)?);
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
        let mut bytes = [0u8; 8 * 5 * 4];
        self.gpu
            .copy_d2h(self.buffers.scratch(), &mut bytes[..N * 20])?;
        let mut predictions = [[0u32; 5]; 8];
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
