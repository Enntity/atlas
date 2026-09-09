// SPDX-License-Identifier: AGPL-3.0-only
//! Shared immutable checks for cold selection and every E7 pre-dispatch.
use super::TransformerModel;
use crate::layer::glm_owner_verify::{GlmOwnerBatchShape, GlmOwnerBatchWorkspace};
use crate::traits::SequenceState;
use anyhow::{Context, Result, ensure};

impl TransformerModel {
    pub(in crate::model) fn owner_verdict_preflight(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &[&SequenceState],
        tokens: &[[u32; 5]],
        accepted: &[usize],
    ) -> Result<[usize; 4]> {
        ensure!(
            seqs.len() == shape.owners()
                && tokens.len() == shape.owners()
                && accepted.len() == shape.owners()
                && accepted.iter().all(|&a| a <= 4),
            "owner verdict count mismatch"
        );
        let mut bases = [0usize; 4];
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
        Ok(bases)
    }

    pub(in crate::model) fn validate_glm_owner_compute(
        &self,
        shape: GlmOwnerBatchShape,
    ) -> Result<usize> {
        let context = self.glm_repair_context();
        let stream = self.gpu.default_stream();
        let workspace = GlmOwnerBatchWorkspace::new(&context, shape)?;
        workspace.scratch.validate_context(&context, stream)?;
        ensure!(
            self.glm_pair_verify_mode.is_some()
                && self.lora.is_none()
                && self.config.adapter_max_rank == 0
                && self.config.dflash_capture_layers.is_empty()
                && self.config.vision.is_none()
                && self.layers.len() == self.config.num_hidden_layers
                && shape.owners() <= self.paired_owner_capacity()?,
            "wider compute requires complete admitted base paired model"
        );
        let max_blocks = self.max_blocks_per_seq as usize;
        ensure!(
            (1..=128).contains(&max_blocks),
            "owner metadata block capacity"
        );
        let stride = (768 + 5 * max_blocks * 4).next_multiple_of(256);
        ensure!(
            stride <= 3328
                && 32768 + shape.owners() * stride <= 49152
                && self.buffers.sizes().scratch >= 49152
                && self.buffers.sizes().logits
                    >= self
                        .config
                        .vocab_size
                        .checked_mul(shape.rows() * 2)
                        .context("owner logits overflow")?
                && self.buffers.sizes().token_ids >= shape.rows() * 4,
            "owner metadata/logits/token arena capacity"
        );
        for layer in &self.layers {
            layer.validate_glm_owner_verify(&context, shape, stream)?;
        }
        Ok(stride)
    }

    pub(in crate::model) fn owner_compute_preflight(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &[&SequenceState],
        tokens: &[[u32; 5]],
    ) -> Result<usize> {
        ensure!(
            seqs.len() == shape.owners() && tokens.len() == shape.owners(),
            "owner model count mismatch"
        );
        let stride = self.validate_glm_owner_compute(shape)?;
        let capacity = self.paired_owner_capacity()?;
        let max_blocks = self.max_blocks_per_seq as usize;
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
        let cache = self.kv_cache.lock();
        let mut additional = 0usize;
        for (index, seq) in seqs.iter().enumerate() {
            additional = additional
                .checked_add(
                    self.paired_target_budget(seq, &cache)?
                        .1
                        .saturating_sub(seq.block_table.len()),
                )
                .context("owner aggregate target budget overflow")?;
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
        Ok(stride)
    }
}
