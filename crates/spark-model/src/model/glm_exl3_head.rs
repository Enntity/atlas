// SPDX-License-Identifier: AGPL-3.0-only

//! Packed EXL3 BF16 vocabulary projection with F32 logit transport and TP gather.
use super::types::TransformerModel;
use crate::weight_loader::Glm5Exl3Projection;
use anyhow::{Context, Result, ensure};
use atlas_core::{config::ModelConfig, scope::ModelResource};
use spark_comm::CommBackend;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
    weights::exl3::{
        Exl3CheckpointRouter,
        native::{Exl3LinearNumerics, Exl3LinearWorkspace},
        tp::{Exl3DeviceShard, ShardAxis, TpShardPlan},
    },
};

/// Owns packed head weights, F32 local/gather/global buffers and linear scratch.
/// Physical vocabulary comes from the checkpoint; active vocabulary from config.
/// Each rank retains complete H128 blocks. Padding is trimmed only after gather.
pub struct Glm5Exl3VocabHead {
    projection: Glm5Exl3Projection,
    buffers: [DevicePtr; 3],
    local_bf16: DevicePtr,
    reorder: KernelHandle,
    convert: KernelHandle,
    transport: KernelHandle,
    active_vocab: usize,
    active: bool,
    capture: Option<spark_runtime::weights::exl3::logits_io::Exl3LogitsRouter>,
}

impl Glm5Exl3VocabHead {
    pub fn load(
        checkpoint: &Exl3CheckpointRouter,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        capacity: usize,
        stream: u64,
    ) -> Result<Self> {
        ensure!(
            config.model_type == "glm5_next",
            "EXL3 vocabulary head requires GLM-5"
        );
        ensure!(
            config.tp_world_size > 0 && config.tp_rank < config.tp_world_size,
            "EXL3 vocabulary head requires explicit TP topology"
        );
        ensure!(
            config.ep_world_size == 1 && config.ep_rank == 0 && !config.expert_tp,
            "EXL3 vocabulary head requires pure TP"
        );
        let shape = checkpoint.tensor_shape("lm_head.svh")?;
        ensure!(
            shape.len() == 1 && shape[0] >= config.vocab_size && config.vocab_size > 0,
            "EXL3 physical vocabulary must cover active vocabulary"
        );
        let view = checkpoint.linear_view("lm_head", config.hidden_size, shape[0])?;
        let plan = TpShardPlan::new(
            &view,
            ShardAxis::Output,
            config.tp_rank,
            config.tp_world_size,
        )?;
        let capture =
            spark_runtime::weights::exl3::logits_io::Exl3LogitsRouter::from_env(config.tp_rank)?;
        let numerics = crate::weight_loader::glm53_tp2_packed_numerics(
            plan.world(),
            plan.in_features(),
            plan.out_features(),
            plan.bits(),
            plan.out_features(),
            0,
        )?;
        Exl3LinearWorkspace::preflight_numerics(
            gpu,
            plan.in_features(),
            plan.out_features(),
            capacity,
            numerics,
        )?;
        let shard = Exl3DeviceShard::upload(gpu, &view, &plan)?;
        let mut head = Self::new(gpu, shard, config.vocab_size, capacity, stream, numerics)?;
        head.capture = capture;
        Ok(head)
    }

    fn new(
        gpu: &dyn GpuBackend,
        shard: Exl3DeviceShard,
        active_vocab: usize,
        capacity: usize,
        stream: u64,
        numerics: Exl3LinearNumerics,
    ) -> Result<Self> {
        let projection =
            Glm5Exl3Projection::new_with_numerics(gpu, shard, capacity, stream, numerics)?;
        let mut head = Self {
            projection,
            buffers: [DevicePtr::NULL; 3],
            local_bf16: DevicePtr::NULL,
            reorder: KernelHandle(0),
            convert: KernelHandle(0),
            transport: KernelHandle(0),
            active_vocab,
            active: true,
            capture: None,
        };
        let setup = (|| {
            let plan = head.projection.shard().plan();
            let local = plan.out_features();
            let physical = local
                .checked_mul(plan.world())
                .context("EXL3 physical vocabulary overflow")?;
            ensure!(
                plan.axis() == ShardAxis::Output && active_vocab > 0 && active_vocab <= physical,
                "EXL3 vocabulary requires output-sharded physical weights"
            );
            ensure!(
                active_vocab <= u32::MAX as usize,
                "EXL3 active vocabulary exceeds kernel ABI"
            );
            head.reorder = gpu.kernel("exl3", "exl3_vocab_gather_reorder")?;
            head.convert = gpu.kernel("exl3", "exl3_logits_to_bf16")?;
            head.transport = gpu.kernel("residual_add", "bf16_to_f32")?;
            let local_elements = capacity
                .checked_mul(local)
                .context("EXL3 local vocabulary capacity overflow")?;
            ensure!(
                local_elements <= u32::MAX as usize,
                "EXL3 vocabulary transport exceeds kernel ABI"
            );
            head.local_bf16 = gpu.alloc(
                local_elements
                    .checked_mul(2)
                    .context("EXL3 local BF16 vocabulary size overflow")?,
            )?;
            for (slot, width) in [local, physical, active_vocab].into_iter().enumerate() {
                let bytes = capacity
                    .checked_mul(width)
                    .and_then(|n| n.checked_mul(4))
                    .context("EXL3 vocabulary buffer size overflow")?;
                head.buffers[slot] = gpu.alloc(bytes)?;
            }
            Ok(())
        })();
        match setup {
            Ok(()) => Ok(head),
            Err(error) => {
                let cleanup = head.release(gpu);
                Err(match cleanup {
                    Ok(()) => error,
                    Err(other) => error.context(format!("EXL3 vocabulary cleanup: {other:#}")),
                })
            }
        }
    }

    pub fn logits_buffer(&self) -> DevicePtr {
        self.buffers[2]
    }
    pub fn capacity(&self) -> usize {
        self.projection.capacity()
    }
    pub fn vocab_size(&self) -> usize {
        self.active_vocab
    }

    /// BF16 head output -> `[rows,active_vocab]` F32 pre-softmax transport.
    /// All ranks call this in the same order on their explicit compute stream.
    pub fn forward(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        ensure!(self.active, "EXL3 vocabulary teardown has started");
        let plan = self.projection.shard().plan();
        self.projection.validate_local_forward(
            rows,
            plan.in_features(),
            plan.out_features(),
            stream,
            plan.rank(),
            plan.world(),
        )?;
        let collective = if plan.world() > 1 {
            let comm = comm.context("EXL3 vocabulary requires TP communicator")?;
            ensure!(
                comm.rank() == plan.rank() && comm.world_size() == plan.world(),
                "EXL3 vocabulary communicator topology mismatch"
            );
            Some(comm)
        } else {
            None
        };
        if rows == 0 {
            return Ok(());
        }
        ensure!(
            !input.is_null()
                && !output.is_null()
                && input.0.is_multiple_of(2)
                && output.0.is_multiple_of(4),
            "EXL3 vocabulary requires non-null aligned activations/logits"
        );
        ensure!(
            input != self.local_bf16
                && output != self.local_bf16
                && !self.buffers.contains(&input)
                && !self.buffers[..2].contains(&output)
                && input != output,
            "EXL3 vocabulary operands alias scratch"
        );
        let local_bytes = rows
            .checked_mul(plan.out_features())
            .and_then(|n| n.checked_mul(4))
            .context("EXL3 vocabulary gather bytes overflow")?;
        let count = rows
            .checked_mul(self.active_vocab)
            .context("EXL3 vocabulary logits size overflow")?;
        let blocks = u32::try_from(count.div_ceil(256)).context("EXL3 vocabulary grid overflow")?;
        self.projection
            .launch_local_bf16(gpu, input, self.local_bf16, rows)?;
        let local_count = u32::try_from(
            rows.checked_mul(plan.out_features())
                .context("EXL3 local vocabulary rows overflow")?,
        )
        .context("EXL3 vocabulary transport exceeds kernel ABI")?;
        KernelLaunch::new(gpu, self.transport)
            .grid([local_count.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.local_bf16)
            .arg_ptr(self.buffers[0])
            .arg_u32(local_count)
            .launch(stream)?;
        if let Some(comm) = collective {
            comm.all_gather_on_stream(self.buffers[0].0, self.buffers[1].0, local_bytes, stream)?;
        } else {
            gpu.copy_d2d_async(self.buffers[0], self.buffers[1], local_bytes, stream)?;
        }
        KernelLaunch::new(gpu, self.reorder)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buffers[1])
            .arg_ptr(output)
            .arg_u32(rows as u32)
            .arg_u32(plan.out_features() as u32)
            .arg_u32(self.active_vocab as u32)
            .launch(stream)?;
        if let Some(capture) = &self.capture {
            capture.capture(gpu, output, rows, self.active_vocab, stream)?;
        }
        Ok(())
    }

    fn write_bf16(
        &self,
        gpu: &dyn GpuBackend,
        output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let count = rows
            .checked_mul(self.active_vocab)
            .context("EXL3 logits conversion overflow")?;
        if count == 0 {
            return Ok(());
        }
        let blocks =
            u32::try_from(count.div_ceil(256)).context("EXL3 logits conversion grid overflow")?;
        KernelLaunch::new(gpu, self.convert)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.logits_buffer())
            .arg_ptr(output)
            .arg_u64(count as u64)
            .launch(stream)
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3VocabHead {
    fn label(&self) -> &'static str {
        "GLM EXL3 vocabulary head"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        self.active = false;
        let mut error = None;
        for ptr in self
            .buffers
            .iter_mut()
            .rev()
            .chain(std::iter::once(&mut self.local_bf16))
        {
            if ptr.is_null() {
                continue;
            }
            match gpu.free(*ptr) {
                Ok(()) => *ptr = DevicePtr::NULL,
                Err(e) => {
                    error.get_or_insert(e);
                }
            }
        }
        if let Err(e) = self.projection.release(gpu) {
            error.get_or_insert(e);
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl TransformerModel {
    /// Factory installs the native head after the generic model constructor.
    pub fn attach_exl3_vocab_head(&mut self, mut head: Glm5Exl3VocabHead) -> Result<()> {
        let check: Result<()> = (|| {
            ensure!(
                self.exl3_vocab_head.is_none(),
                "EXL3 vocabulary head already attached"
            );
            ensure!(
                self.lm_head_fp8.is_none() && self.lm_head_nvfp4.is_none(),
                "EXL3 vocabulary head conflicts with runtime head quantization"
            );
            ensure!(
                head.vocab_size() == self.config.vocab_size,
                "EXL3 vocabulary head active size disagrees with model"
            );
            ensure!(
                self.config.final_logit_softcapping == 0.0 && self.overlays.is_none(),
                "EXL3 vocabulary head does not support logit softcap or overlays"
            );
            let plan = head.projection.shard().plan();
            head.projection.validate_local_forward(
                0,
                self.config.hidden_size,
                plan.out_features(),
                self.gpu.default_stream(),
                self.config.tp_rank,
                self.config.tp_world_size,
            )?;
            Ok(())
        })();
        if let Err(error) = check {
            let cleanup = head.release(self.gpu.as_ref());
            return Err(match cleanup {
                Ok(()) => error,
                Err(other) => error.context(format!("EXL3 head attach cleanup: {other:#}")),
            });
        }
        self.exl3_vocab_head = Some(head);
        Ok(())
    }

    /// Raw F32 buffer for validation, containing the last native head batch.
    pub fn exl3_logits_buffer_f32(&self) -> Option<DevicePtr> {
        self.exl3_vocab_head
            .as_ref()
            .map(Glm5Exl3VocabHead::logits_buffer)
    }

    pub(super) fn project_exl3_vocab(
        &self,
        input: DevicePtr,
        output_bf16: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(head) = &self.exl3_vocab_head else {
            return Ok(false);
        };
        head.forward(
            self.gpu.as_ref(),
            input,
            head.logits_buffer(),
            rows,
            self.comm.as_deref(),
            stream,
        )?;
        head.write_bf16(self.gpu.as_ref(), output_bf16, rows, stream)?;
        Ok(true)
    }
}

#[cfg(test)]
#[path = "glm_exl3_head_tests.rs"]
mod tests;
