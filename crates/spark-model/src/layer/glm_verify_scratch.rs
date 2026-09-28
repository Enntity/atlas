// SPDX-License-Identifier: AGPL-3.0-only
//! Checked original-arena dead tails shared by pair and three-through-eight-owner traversal.
//! No allocation, FFN mode selection, or sequence/transaction authority.
use super::{
    ForwardContext,
    glm_pair_verify::{GlmPairLayerInput, OWNER_HIGHWAY_BYTES, OWNER_NORM_BYTES, ROW_BYTES},
};
use anyhow::{Result, ensure};
use spark_runtime::{
    buffers::BufferArena,
    gpu::{DevicePtr, GpuBackend},
};

pub(crate) struct GlmVerifyScratch<'a> {
    owners: usize,
    next_layer: usize,
    norm_saved: DevicePtr,
    highway_saved: DevicePtr,
    post_saved: DevicePtr,
    comb_saved: DevicePtr,
    arena: &'a BufferArena,
    gpu: &'a dyn GpuBackend,
}

impl<'a> GlmVerifyScratch<'a> {
    pub(crate) fn new(ctx: &ForwardContext<'a>, owners: usize) -> Result<Self> {
        ensure!(
            (2..=8).contains(&owners),
            "GLM scratch owner count must be2..8"
        );
        // The bounded count makes every fixed-row byte product below fit usize;
        // real device span ends are separately checked for address overflow.
        ensure!(
            ctx.config.model_type == "glm5_next"
                && ctx.config.hidden_size == 4096
                && ctx.config.hc_mult == 4,
            "GLM pair workspace requires H4096/hc4"
        );
        let b = ctx.buffers;
        let s = b.sizes();
        ensure!(
            b.max_batch_tokens() >= owners * 10
                && s.hidden_states >= owners * 5 * ROW_BYTES
                && s.norm_output >= owners * 10 * ROW_BYTES
                && s.attn_output >= owners * 5 * ROW_BYTES
                && s.moe_output >= owners * 5 * ROW_BYTES
                && s.hc_streams >= (owners + 1) * OWNER_HIGHWAY_BYTES
                && s.hc_post >= (owners + 1) * 5 * 4 * 4
                && s.hc_comb >= (owners + 1) * 5 * 4 * 4 * 4,
            "GLM pair working arena/tail capacity"
        );
        let spans = [
            (b.norm_output(), s.norm_output),
            (b.hc_streams(), s.hc_streams),
            (b.hc_post(), s.hc_post),
            (b.hc_comb(), s.hc_comb),
            (b.hidden_states(), s.hidden_states),
            (b.attn_output(), s.attn_output),
            (b.moe_output(), s.moe_output),
        ];
        for (i, &(ptr, bytes)) in spans.iter().enumerate() {
            ensure!(
                !ptr.is_null() && ptr.0.is_multiple_of(4),
                "GLM pair arena pointer/alignment"
            );
            let end = ptr
                .0
                .checked_add(bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("GLM pair arena address overflow"))?;
            for &(other, len) in &spans[..i] {
                let other_end = other
                    .0
                    .checked_add(len as u64)
                    .ok_or_else(|| anyhow::anyhow!("GLM pair arena address overflow"))?;
                ensure!(
                    end <= other.0 || other_end <= ptr.0,
                    "GLM pair arena owners alias"
                );
            }
        }
        Ok(Self {
            owners,
            next_layer: 0,
            norm_saved: b.norm_output().offset(owners * 5 * ROW_BYTES),
            highway_saved: b.hc_streams().offset(OWNER_HIGHWAY_BYTES),
            post_saved: b.hc_post().offset(5 * 4 * 4),
            comb_saved: b.hc_comb().offset(5 * 4 * 4 * 4),
            arena: b,
            gpu: ctx.gpu,
        })
    }

    pub(crate) fn owner_count(&self) -> usize {
        self.owners
    }

    pub(crate) fn validate_context(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        ensure!(std::ptr::eq(self.arena, ctx.buffers)
            && std::ptr::addr_eq(self.gpu, ctx.gpu)
            && stream == self.gpu.default_stream()
            && !ctx.graph_capture && !ctx.profile && !self.gpu.stream_is_capturing(stream)
            && ctx.ssm_batch.is_none() && ctx.routed_lora_layers.is_none()
            && ctx.config.model_type == "glm5_next" && ctx.config.hidden_size == 4096
            && ctx.config.hc_mult == 4 && ctx.config.tp_world_size == 2
            && ctx.config.ep_world_size == 2 && ctx.config.tp_rank == ctx.config.ep_rank
            && ctx.comm.is_some_and(|comm| comm.world_size() == 2 && comm.rank() == ctx.config.ep_rank),
            "GLM pair requires the original eager arena/backend/default stream and TP2/EP2 context");
        Ok(())
    }

    pub(crate) fn begin_layer(
        &self,
        layer: usize,
        owners: &[GlmPairLayerInput<'_>],
        ctx: &[&ForwardContext],
        stream: u64,
    ) -> Result<()> {
        ensure!(self.next_layer == layer, "GLM pair layer order changed");
        ensure!(
            owners.len() == self.owners && ctx.len() == self.owners,
            "GLM scratch owner/context count changed"
        );
        for (index, owner) in owners.iter().enumerate() {
            self.validate_context(ctx[index], stream)?;
            ensure!(
                std::ptr::eq(ctx[0].config, ctx[index].config)
                    && owner.hidden == self.arena.hidden_states().offset(index * OWNER_NORM_BYTES)
                    && owner.positions[4] < 2048
                    && owner
                        .positions
                        .iter()
                        .enumerate()
                        .all(|(row, &position)| owner.positions[0].checked_add(row)
                            == Some(position)),
                "GLM pair owner rows/config changed"
            );
        }
        Ok(())
    }

    pub(crate) fn restore_highway(&self, owner: usize, stream: u64) -> Result<()> {
        ensure!(owner < self.owners, "GLM pair owner index");
        if self.next_layer != 0 {
            self.gpu.copy_d2d_async(
                self.highway_saved.offset(owner * OWNER_HIGHWAY_BYTES),
                self.arena.hc_streams(),
                OWNER_HIGHWAY_BYTES,
                stream,
            )?;
        }
        Ok(())
    }

    pub(crate) fn save_attention(&self, owner: usize, stream: u64) -> Result<()> {
        self.copy_ffn_state(owner, true, true, stream)
    }

    pub(crate) fn restore_ffn(&self, owner: usize, norm: bool, stream: u64) -> Result<()> {
        self.copy_ffn_state(owner, false, norm, stream)
    }

    fn copy_ffn_state(&self, owner: usize, save: bool, norm: bool, stream: u64) -> Result<()> {
        ensure!(owner < self.owners, "GLM pair owner index");
        for (index, (working, saved, bytes)) in [
            (self.arena.norm_output(), self.norm_saved, OWNER_NORM_BYTES),
            (
                self.arena.hc_streams(),
                self.highway_saved,
                OWNER_HIGHWAY_BYTES,
            ),
            (self.arena.hc_post(), self.post_saved, 80),
            (self.arena.hc_comb(), self.comb_saved, 320),
        ]
        .into_iter()
        .enumerate()
        {
            if index == 0 && !norm {
                continue;
            }
            let saved = saved.offset(owner * bytes);
            let (src, dst) = if save {
                (working, saved)
            } else {
                (saved, working)
            };
            self.gpu.copy_d2d_async(src, dst, bytes, stream)?;
        }
        Ok(())
    }

    pub(crate) fn pack_norms(&self, stream: u64) -> Result<()> {
        self.gpu.copy_d2d_async(
            self.norm_saved,
            self.arena.norm_output(),
            self.owners * OWNER_NORM_BYTES,
            stream,
        )
    }

    pub(crate) fn save_highway(&self, owner: usize, stream: u64) -> Result<()> {
        ensure!(owner < self.owners, "GLM pair owner index");
        self.gpu.copy_d2d_async(
            self.arena.hc_streams(),
            self.highway_saved.offset(owner * OWNER_HIGHWAY_BYTES),
            OWNER_HIGHWAY_BYTES,
            stream,
        )
    }

    pub(crate) fn finish_layer(&mut self) {
        self.next_layer += 1;
    }
}
