// SPDX-License-Identifier: AGPL-3.0-only

//! Exact GLM KDA + sparse-MLA speculative state checkpoint/rollback.

use anyhow::{Context, Result, ensure};
use atlas_core::config::LayerType;

use super::super::types::TransformerModel;
use crate::layer::{
    GlmSparseMlaLayerState, GlmSparseMlaStatePointers, KdaLayerState, KdaStatePointers,
};
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn glm_checkpoint_dispatch(
        &self,
        seq: &mut SequenceState,
        stream: u64,
        record_event: bool,
    ) -> Result<()> {
        for (layer_idx, layer_state) in seq.layer_states.iter().enumerate() {
            match self.config.layer_type(layer_idx) {
                LayerType::LinearAttention => {
                    let state = layer_state
                        .as_any()
                        .downcast_ref::<KdaLayerState>()
                        .context("GLM checkpoint expected KdaLayerState")?;
                    let target = state
                        .checkpoint
                        .context("GLM KDA verify checkpoint was not reserved")?;
                    self.copy_kda_image(state.current, target, stream)?;
                }
                LayerType::FullAttention => {
                    let state = layer_state
                        .as_any()
                        .downcast_ref::<GlmSparseMlaLayerState>()
                        .context("GLM checkpoint expected sparse-MLA state")?;
                    let target = state
                        .checkpoint
                        .context("GLM DSA verify checkpoint was not reserved")?;
                    self.copy_dsa_image(state.current, target, stream)?;
                }
                other => anyhow::bail!(
                    "GLM checkpoint encountered unsupported layer type {other:?} at {layer_idx}"
                ),
            }
        }
        if record_event {
            self.gpu.record_event(self.secondary_event, stream)?;
        } else {
            self.gpu.synchronize(stream)?;
        }
        Ok(())
    }

    pub(super) fn glm_restore_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
        checkpoint_after: bool,
        record_event: bool,
    ) -> Result<()> {
        // Validate every source before issuing a single GPU copy; an error
        // must never leave half the GLM layers at a different token boundary.
        for (layer_idx, layer_state) in seq.layer_states.iter().enumerate() {
            let available = match self.config.layer_type(layer_idx) {
                LayerType::LinearAttention => layer_state
                    .as_any()
                    .downcast_ref::<KdaLayerState>()
                    .context("GLM rollback expected KdaLayerState")?
                    .intermediates
                    .len(),
                LayerType::FullAttention => layer_state
                    .as_any()
                    .downcast_ref::<GlmSparseMlaLayerState>()
                    .context("GLM rollback expected sparse-MLA state")?
                    .intermediates
                    .len(),
                other => anyhow::bail!(
                    "GLM rollback encountered unsupported layer type {other:?} at {layer_idx}"
                ),
            };
            ensure!(
                num_accepted == 0 || num_accepted <= available,
                "GLM rollback target {num_accepted} exceeds layer {layer_idx} snapshot capacity {available}"
            );
        }

        for (layer_idx, layer_state) in seq.layer_states.iter().enumerate() {
            match self.config.layer_type(layer_idx) {
                LayerType::LinearAttention => {
                    let state = layer_state
                        .as_any()
                        .downcast_ref::<KdaLayerState>()
                        .context("GLM rollback expected KdaLayerState")?;
                    let source = if num_accepted == 0 {
                        state.checkpoint.context("GLM KDA checkpoint missing")?
                    } else {
                        state.intermediates[num_accepted - 1]
                    };
                    self.copy_kda_image(source, state.current, stream)?;
                    if checkpoint_after {
                        self.copy_kda_image(
                            state.current,
                            state.checkpoint.context("GLM KDA checkpoint missing")?,
                            stream,
                        )?;
                    }
                }
                LayerType::FullAttention => {
                    let state = layer_state
                        .as_any()
                        .downcast_ref::<GlmSparseMlaLayerState>()
                        .context("GLM rollback expected sparse-MLA state")?;
                    let source = if num_accepted == 0 {
                        state.checkpoint.context("GLM DSA checkpoint missing")?
                    } else {
                        state.intermediates[num_accepted - 1]
                    };
                    self.copy_dsa_image(source, state.current, stream)?;
                    if checkpoint_after {
                        self.copy_dsa_image(
                            state.current,
                            state.checkpoint.context("GLM DSA checkpoint missing")?,
                            stream,
                        )?;
                    }
                }
                other => anyhow::bail!(
                    "GLM rollback encountered unsupported layer type {other:?} at {layer_idx}"
                ),
            }
        }
        if record_event {
            self.gpu.record_event(self.secondary_event, stream)?;
        }
        Ok(())
    }

    pub(super) fn glm_commit_accepted_prefix_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(
            num_accepted <= k,
            "GLM commit accepts {num_accepted} rows from width {k}"
        );
        if num_accepted == k {
            return Ok(());
        }
        ensure!(
            num_accepted > 0,
            "GLM commit needs at least the verified anchor row"
        );
        self.glm_restore_dispatch(seq, num_accepted, self.secondary_stream, false, true)
    }

    fn copy_kda_image(
        &self,
        source: KdaStatePointers,
        target: KdaStatePointers,
        stream: u64,
    ) -> Result<()> {
        self.gpu.copy_d2d_async(
            source.recurrent,
            target.recurrent,
            self.ssm_pool.h_stored_bytes,
            stream,
        )?;
        self.gpu.copy_d2d_async(
            source.q_conv,
            target.q_conv,
            self.ssm_pool.conv_bytes,
            stream,
        )?;
        Ok(())
    }

    fn copy_dsa_image(
        &self,
        source: GlmSparseMlaStatePointers,
        target: GlmSparseMlaStatePointers,
        stream: u64,
    ) -> Result<()> {
        let tail_bytes = self.config.index_kpool.saturating_sub(1)
            * self.config.index_head_dim
            * size_of::<u16>();
        for (src, dst, bytes) in [
            (source.tail_keys, target.tail_keys, tail_bytes),
            (source.tail_gates, target.tail_gates, tail_bytes),
            (
                source.tail_metadata,
                target.tail_metadata,
                4 * size_of::<i32>(),
            ),
        ] {
            self.gpu.copy_d2d_async(src, dst, bytes, stream)?;
        }
        Ok(())
    }
}
