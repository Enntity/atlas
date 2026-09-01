// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::Glm5Layer;
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

impl TransformerLayer for Glm5Layer {
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_rows(hidden, 1, &[state], &[seq_len], ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_verify_rows(hidden, num_tokens, seq_len, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_rows(hidden, num_tokens, seq_len_start, state, ctx, stream)
    }

    fn prefill_glm_batched(
        &self,
        hidden_stacked: DevicePtr,
        _residual_stacked: DevicePtr,
        states: &[&(dyn LayerState + '_)],
        meta: &crate::layer::BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_rows_batched(hidden_stacked, states, meta, ctx, stream)
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state_refs = states
            .iter()
            .map(|state| &**state as &(dyn LayerState + '_))
            .collect::<Vec<_>>();
        self.forward_rows(hidden, num_seqs, &state_refs, seq_lens, ctx, stream)
    }

    fn decode_verify_glm_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        rows_per_seq: usize,
        seq_lens: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_verify_rows_multi(hidden, rows_per_seq, seq_lens, states, ctx, stream)
    }

    fn decode_verify_glm_with_prefill(
        &self,
        hidden: DevicePtr,
        verify_rows: usize,
        verify_seq_len: usize,
        verify_state: &mut (dyn LayerState + 'static),
        prefill_rows: usize,
        prefill_seq_len: usize,
        prefill_state: &mut (dyn LayerState + 'static),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_verify_with_prefill(
            hidden,
            verify_rows,
            verify_seq_len,
            verify_state,
            prefill_rows,
            prefill_seq_len,
            prefill_state,
            ctx,
            stream,
        )
    }

    fn alloc_state(
        &self,
        _gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) -> Result<Box<dyn LayerState>> {
        anyhow::bail!("GLM layer state must come from the fixed-address KDA/DSA sequence pool")
    }

    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        let _ = (gpu, config);
        // EXL3 stays in its native packed layout for decode and prefill.
        Ok(())
    }
}
