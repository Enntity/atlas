// SPDX-License-Identifier: AGPL-3.0-only

//! Replicated packed EXL3 vocabulary head for the DFlash2 drafter.
//!
//! The target's own head (`Glm5Exl3VocabHead`) holds only this rank's half
//! of the vocabulary and all-gathers logits across TP ranks. The drafter
//! runs on both ranks without exchanging anything, so each rank uploads BOTH
//! TP2 halves of the packed `lm_head` and projects the drafter rows through
//! each with the exact projection, numerics profile and BF16 output the
//! target head uses (`launch_local_bf16`): the drafter's logits are the same
//! values the target head would produce for those rows. No decoded BF16
//! matrix exists; the two packed halves cost ~0.5 GB per rank.

use anyhow::{Context, Result, ensure};
use atlas_core::{config::ModelConfig, scope::ModelResource};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::{
    Exl3CheckpointRouter,
    tp::{Exl3DeviceShard, ShardAxis, TpShardPlan},
};

use crate::layers::dflash_head::DraftLogitsHead;
use crate::weight_loader::{Glm5Exl3Projection, glm53_tp2_packed_numerics};

/// TP world the packed head numerics are pinned to.
const HALVES: usize = 2;

pub struct Exl3ReplicatedVocabHead {
    halves: Vec<Glm5Exl3Projection>,
    /// `[capacity, local]` BF16 per half.
    scratch: Vec<DevicePtr>,
    local: usize,
    vocab: usize,
    capacity: usize,
    stream: u64,
}

impl Exl3ReplicatedVocabHead {
    /// Upload both TP2 halves of `lm_head` on this rank. `vocab` is the
    /// drafter's logits row width; `capacity` the drafter block rows (γ).
    pub fn load(
        checkpoint: &Exl3CheckpointRouter,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        vocab: usize,
        capacity: usize,
        stream: u64,
    ) -> Result<Self> {
        let shape = checkpoint.tensor_shape("lm_head.svh")?;
        ensure!(
            shape.len() == 1 && shape[0] >= vocab && vocab > 0,
            "EXL3 DFlash head: physical vocabulary {shape:?} does not cover drafter vocab {vocab}"
        );
        let view = checkpoint.linear_view("lm_head", config.hidden_size, shape[0])?;
        let mut head = Self {
            halves: Vec::with_capacity(HALVES),
            scratch: Vec::with_capacity(HALVES),
            local: 0,
            vocab,
            capacity,
            stream,
        };
        let setup = (|| -> Result<()> {
            for rank in 0..HALVES {
                let plan = TpShardPlan::new(&view, ShardAxis::Output, rank, HALVES)?;
                let numerics = glm53_tp2_packed_numerics(
                    plan.world(),
                    plan.in_features(),
                    plan.out_features(),
                    plan.bits(),
                    plan.out_features(),
                    0,
                )?;
                let local = plan.out_features();
                ensure!(
                    head.local == 0 || head.local == local,
                    "EXL3 DFlash head halves differ in width"
                );
                head.local = local;
                let shard = Exl3DeviceShard::upload(gpu, &view, &plan)?;
                head.halves.push(Glm5Exl3Projection::new_with_numerics(
                    gpu, shard, capacity, stream, numerics,
                )?);
                let bytes = capacity
                    .checked_mul(local)
                    .and_then(|n| n.checked_mul(2))
                    .context("EXL3 DFlash head scratch overflow")?;
                head.scratch.push(gpu.alloc(bytes)?);
            }
            ensure!(
                head.local < vocab && vocab <= HALVES * head.local,
                "EXL3 DFlash head: drafter vocab {vocab} must span both halves of width {}",
                head.local
            );
            Ok(())
        })();
        match setup {
            Ok(()) => Ok(head),
            Err(error) => {
                let cleanup = head.release(gpu);
                Err(match cleanup {
                    Ok(()) => error,
                    Err(other) => error.context(format!("EXL3 DFlash head cleanup: {other:#}")),
                })
            }
        }
    }

    pub fn bytes_per_step(&self) -> usize {
        self.halves
            .iter()
            .map(|h| h.shard().plan().trellis_byte_size())
            .sum()
    }
}

impl DraftLogitsHead for Exl3ReplicatedVocabHead {
    fn project(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows <= self.capacity && stream == self.stream,
            "EXL3 DFlash head: {rows} rows on stream {stream:#x} (capacity {}, stream {:#x})",
            self.capacity,
            self.stream
        );
        for (half, scratch) in self.halves.iter().zip(&self.scratch) {
            half.launch_local_bf16(gpu, input, *scratch, rows)?;
        }
        // Rank-major halves -> `[rows, vocab]`; trim padding off the tail.
        for row in 0..rows {
            for (index, scratch) in self.scratch.iter().enumerate() {
                let start = index * self.local;
                let width = self.local.min(self.vocab - start);
                gpu.copy_d2d_async(
                    scratch.offset(row * self.local * 2),
                    out.offset((row * self.vocab + start) * 2),
                    width * 2,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Exl3ReplicatedVocabHead {
    fn label(&self) -> &'static str {
        "GLM EXL3 DFlash replicated vocabulary head"
    }

    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut error = None;
        for ptr in self.scratch.iter_mut() {
            if !ptr.is_null() {
                match gpu.free(*ptr) {
                    Ok(()) => *ptr = DevicePtr::NULL,
                    Err(e) => {
                        error.get_or_insert(e);
                    }
                }
            }
        }
        for half in self.halves.iter_mut() {
            if let Err(e) = half.release(gpu) {
                error.get_or_insert(e);
            }
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
