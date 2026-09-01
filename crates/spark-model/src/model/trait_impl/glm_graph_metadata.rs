// SPDX-License-Identifier: AGPL-3.0-only

//! Graph-safe dynamic metadata staging for GLM-5.3 multi-sequence decode.

use anyhow::{Context, Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::gpu::GpuBackend;

use super::super::types::TransformerModel;
use crate::layer::{GlmSparseMlaLayerState, KdaLayerState, LayerState};
use crate::traits::SequenceState;

impl TransformerModel {
    /// Refresh every GLM state-pointer table before graph capture/replay.
    ///
    /// The graph captures only stable device table addresses. Pointer values
    /// remain dynamic because slot vectors change as requests arrive and
    /// finish. No host allocation or H2D copy is therefore captured.
    pub(super) fn stage_glm_graph_metadata(
        &self,
        seqs: &[&mut SequenceState],
        dispatch_n: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.ssm_pool.is_glm() {
            return Ok(());
        }
        let slots = seqs
            .iter()
            .map(|seq| {
                seq.ssm_slot_idx()
                    .context("GLM graph sequence has no fixed state slot")
            })
            .collect::<Result<Vec<_>>>()?;
        self.stage_glm_graph_metadata_slots(slots, dispatch_n, stream)
    }

    /// Single-sequence graph replay uses the same indirect state tables as
    /// batched decode. Native prefill may have overwritten those tables with
    /// another cohort since the graph was captured, so refresh them on every
    /// replay just like `decode_batch` does.
    pub(super) fn stage_glm_graph_metadata_single(
        &self,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        if !self.ssm_pool.is_glm() {
            return Ok(());
        }
        let slot = seq
            .ssm_slot_idx()
            .context("GLM graph sequence has no fixed state slot")?;
        self.stage_glm_graph_metadata_slots(vec![slot], 1, stream)
    }

    /// Refresh the layer-specific state images consumed by one K-row GLM
    /// speculative-verification graph. Unlike ordinary decode metadata, KDA
    /// verification needs the canonical state followed by K-1 rollback
    /// images, while DSA consumes the canonical table and writes its rollback
    /// images through fixed graph nodes. `upload_state_table` deliberately
    /// suppresses H2D work during capture, so failing to stage this exact
    /// image leaves zero/stale pointers in the captured graph.
    pub(super) fn stage_glm_verify_graph_metadata(
        &self,
        seq: &SequenceState,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.ssm_pool.is_glm() {
            return Ok(());
        }
        ensure!(rows > 0, "GLM verify graph requires at least one row");
        let layout = self.buffers.glm_layout();
        let workspace = self.buffers.glm_workspace();
        ensure!(!workspace.is_null(), "GLM workspace is not allocated");
        let tables_len = layout
            .state_ptrs_stride
            .checked_mul(layout.state_ptrs_layers)
            .context("GLM verify graph metadata size overflow")?;
        let mut table_bytes = vec![0u8; tables_len];

        for (layer_idx, state) in seq.layer_states.iter().enumerate() {
            let values = match self.config.layer_type(layer_idx) {
                LayerType::LinearAttention => {
                    let state = state
                        .as_any()
                        .downcast_ref::<KdaLayerState>()
                        .context("GLM verify graph received incompatible KDA state")?;
                    ensure!(
                        state.intermediates.len() >= rows.saturating_sub(1),
                        "GLM verify graph has insufficient KDA rollback images"
                    );
                    std::iter::once(state.current)
                        .chain(
                            state
                                .intermediates
                                .iter()
                                .copied()
                                .take(rows.saturating_sub(1)),
                        )
                        .flat_map(|image| {
                            [
                                image.recurrent.0,
                                image.q_conv.0,
                                image.k_conv.0,
                                image.v_conv.0,
                            ]
                        })
                        .collect::<Vec<_>>()
                }
                LayerType::FullAttention => {
                    let state = state
                        .as_any()
                        .downcast_ref::<GlmSparseMlaLayerState>()
                        .context("GLM verify graph received incompatible DSA state")?;
                    let image = state.current;
                    vec![
                        image.latent_cache.0,
                        image.pooled_keys.0,
                        image.tail_keys.0,
                        image.tail_gates.0,
                        image.tail_metadata.0,
                    ]
                }
                other => anyhow::bail!("unsupported GLM layer type {other:?} at layer {layer_idx}"),
            };
            let layer_offset = layout
                .state_table_offset(layer_idx)?
                .checked_sub(layout.state_ptrs)
                .context("GLM verify state table precedes table arena")?;
            let bytes_len = values.len() * size_of::<u64>();
            ensure!(
                bytes_len <= layout.state_ptrs_stride,
                "GLM verify layer state table exceeds its stride"
            );
            for (value, target) in values
                .iter()
                .zip(table_bytes[layer_offset..layer_offset + bytes_len].chunks_exact_mut(8))
            {
                target.copy_from_slice(&value.to_le_bytes());
            }
        }
        self.gpu
            .copy_h2d_async(&table_bytes, workspace.offset(layout.state_ptrs), stream)?;

        // DSA verify advances one causal row at a time inside the graph.
        let cu = [0i32, 1i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        self.gpu
            .copy_h2d_async(&cu, workspace.offset(layout.cu_seqlens_i32), stream)?;
        self.gpu
            .memset_async(workspace.offset(layout.valid), 1, 1, stream)?;
        Ok(())
    }

    /// Refresh the fixed-address state tables consumed by one N x K DFlash
    /// target graph. KDA stores each sequence's current image followed by its
    /// rollback images. DSA stores all current five-pointer images first, then
    /// `[depth, sequence]` tail-snapshot destinations for the depth-major
    /// verifier. Keeping every pointer value outside capture makes one graph
    /// reusable across slot assignments with the same `(N,K,pool-bucket)`.
    pub(super) fn stage_glm_verify_multi_graph_metadata(
        &self,
        seqs: &[&mut SequenceState],
        rows_per_sequence: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.ssm_pool.is_glm() {
            return Ok(());
        }
        ensure!(
            seqs.len() > 1,
            "GLM multi-verify graph requires concurrency"
        );
        ensure!(
            rows_per_sequence > 0,
            "GLM multi-verify graph requires rows"
        );
        let snapshot_count = rows_per_sequence.saturating_sub(1);
        let layout = self.buffers.glm_layout();
        let workspace = self.buffers.glm_workspace();
        ensure!(!workspace.is_null(), "GLM workspace is not allocated");
        let tables_len = layout
            .state_ptrs_stride
            .checked_mul(layout.state_ptrs_layers)
            .context("GLM multi-verify metadata size overflow")?;
        let mut table_bytes = vec![0u8; tables_len];

        for layer_idx in 0..self.layers.len() {
            let mut values = Vec::new();
            match self.config.layer_type(layer_idx) {
                LayerType::LinearAttention => {
                    values.reserve(seqs.len() * rows_per_sequence * 4);
                    for seq in seqs {
                        let state = seq.layer_states[layer_idx]
                            .as_any()
                            .downcast_ref::<KdaLayerState>()
                            .context("GLM multi-verify graph received incompatible KDA state")?;
                        ensure!(
                            state.intermediates.len() >= snapshot_count,
                            "GLM multi-verify graph has insufficient KDA rollback images"
                        );
                        for image in std::iter::once(state.current)
                            .chain(state.intermediates.iter().copied().take(snapshot_count))
                        {
                            values.extend([
                                image.recurrent.0,
                                image.q_conv.0,
                                image.k_conv.0,
                                image.v_conv.0,
                            ]);
                        }
                    }
                }
                LayerType::FullAttention => {
                    values.reserve(seqs.len() * (5 + snapshot_count * 3));
                    for seq in seqs {
                        let state = seq.layer_states[layer_idx]
                            .as_any()
                            .downcast_ref::<GlmSparseMlaLayerState>()
                            .context("GLM multi-verify graph received incompatible DSA state")?;
                        ensure!(
                            state.intermediates.len() >= snapshot_count,
                            "GLM multi-verify graph has insufficient DSA rollback images"
                        );
                        let current = state.current;
                        values.extend([
                            current.latent_cache.0,
                            current.pooled_keys.0,
                            current.tail_keys.0,
                            current.tail_gates.0,
                            current.tail_metadata.0,
                        ]);
                    }
                    for depth in 0..snapshot_count {
                        for seq in seqs {
                            let state = seq.layer_states[layer_idx]
                                .as_any()
                                .downcast_ref::<GlmSparseMlaLayerState>()
                                .context(
                                    "GLM multi-verify graph received incompatible DSA state",
                                )?;
                            let snapshot = state.intermediates[depth];
                            values.extend([
                                snapshot.tail_keys.0,
                                snapshot.tail_gates.0,
                                snapshot.tail_metadata.0,
                            ]);
                        }
                    }
                }
                other => anyhow::bail!("unsupported GLM layer type {other:?} at layer {layer_idx}"),
            }
            let layer_offset = layout
                .state_table_offset(layer_idx)?
                .checked_sub(layout.state_ptrs)
                .context("GLM multi-verify state table precedes table arena")?;
            let bytes_len = values.len() * size_of::<u64>();
            ensure!(
                bytes_len <= layout.state_ptrs_stride,
                "GLM multi-verify layer state table exceeds its stride"
            );
            for (value, target) in values
                .iter()
                .zip(table_bytes[layer_offset..layer_offset + bytes_len].chunks_exact_mut(8))
            {
                target.copy_from_slice(&value.to_le_bytes());
            }
        }
        self.gpu
            .copy_h2d_async(&table_bytes, workspace.offset(layout.state_ptrs), stream)?;
        self.gpu
            .memset_async(workspace.offset(layout.valid), 1, seqs.len(), stream)?;
        Ok(())
    }

    fn stage_glm_graph_metadata_slots(
        &self,
        mut slots: Vec<usize>,
        dispatch_n: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            dispatch_n >= slots.len(),
            "GLM graph dispatch width is too small"
        );
        let layout = self.buffers.glm_layout();
        ensure!(
            dispatch_n <= layout.max_batch_size,
            "GLM graph dispatch width exceeds workspace"
        );
        let workspace = self.buffers.glm_workspace();
        ensure!(!workspace.is_null(), "GLM workspace is not allocated");
        slots.resize(dispatch_n, self.ssm_pool.dummy_slot());

        // All layer tables occupy one contiguous stable arena.  Build the
        // complete image on the host and issue one H2D rather than 45 tiny
        // copies per decode token.  Besides reducing launch/driver overhead,
        // this makes the table refresh atomic with respect to subsequent graph
        // replay on the same stream.
        let tables_len = layout
            .state_ptrs_stride
            .checked_mul(layout.state_ptrs_layers)
            .context("GLM graph metadata size overflow")?;
        let mut table_bytes = vec![0u8; tables_len];
        let mut kda_idx = 0usize;
        let mut dsa_idx = 0usize;
        for layer_idx in 0..self.layers.len() {
            let mut values = Vec::with_capacity(dispatch_n * 5);
            match self.config.layer_type(layer_idx) {
                LayerType::LinearAttention => {
                    for &slot in &slots {
                        let state = self.ssm_pool.kda_state(kda_idx, slot);
                        values.extend([
                            state.recurrent.0,
                            state.q_conv.0,
                            state.k_conv.0,
                            state.v_conv.0,
                        ]);
                    }
                    kda_idx += 1;
                }
                LayerType::FullAttention => {
                    for &slot in &slots {
                        let state = self.ssm_pool.glm_dsa_state(dsa_idx, slot);
                        values.extend([
                            state.latent_cache.0,
                            state.pooled_keys.0,
                            state.tail_keys.0,
                            state.tail_gates.0,
                            state.tail_metadata.0,
                        ]);
                    }
                    dsa_idx += 1;
                }
                other => anyhow::bail!("unsupported GLM layer type {other:?} at layer {layer_idx}"),
            }
            let layer_offset = layout
                .state_table_offset(layer_idx)?
                .checked_sub(layout.state_ptrs)
                .context("GLM state table precedes table arena")?;
            let bytes_len = values.len() * std::mem::size_of::<u64>();
            ensure!(
                bytes_len <= layout.state_ptrs_stride,
                "GLM layer state table exceeds its stride"
            );
            for (value, target) in values
                .iter()
                .zip(table_bytes[layer_offset..layer_offset + bytes_len].chunks_exact_mut(8))
            {
                target.copy_from_slice(&value.to_le_bytes());
            }
        }
        self.gpu
            .copy_h2d_async(&table_bytes, workspace.offset(layout.state_ptrs), stream)?;

        // Shared DSA metadata is identical for every sparse-attention layer.
        let cu = (0..=dispatch_n as i32)
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        self.gpu
            .copy_h2d_async(&cu, workspace.offset(layout.cu_seqlens_i32), stream)?;
        self.gpu
            .memset_async(workspace.offset(layout.valid), 1, dispatch_n, stream)?;
        Ok(())
    }
}
