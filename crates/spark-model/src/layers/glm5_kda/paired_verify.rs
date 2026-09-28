// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded temporal owners; original scratch is reused only after saving mHC.

use super::*;
use crate::layer::glm_pair_verify::{GlmPairLayerInput, GlmPairWorkspace, OWNER_NORM_BYTES};
use crate::layer::{glm_verify_ffn::GlmVerifyFfn, glm_verify_scratch::GlmVerifyScratch};

impl Glm5KdaLayer {
    pub(super) fn pair_supported(&self) -> bool {
        !self.ffn.is_none()
            && self.hidden_size == 4096
            && self.heads == 32
            && self.dim == 128
            && self.conv_width == 4
            && self.hc.hc_mult == 4
    }

    pub(super) fn validate_pair(
        &self,
        workspace: &GlmPairWorkspace,
        contexts: [&ForwardContext; 2],
        stream: u64,
    ) -> Result<()> {
        self.validate_temporal(
            &workspace.scratch,
            &contexts,
            GlmVerifyFfn::Pair(workspace.mode),
            stream,
        )
    }

    pub(super) fn validate_temporal(
        &self,
        workspace: &GlmVerifyScratch,
        contexts: &[&ForwardContext],
        mode: GlmVerifyFfn,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            contexts.len() == mode.owners() && workspace.owner_count() == mode.owners(),
            "GLM temporal KDA owner/workspace/context count mismatch"
        );
        ensure!(
            self.pair_supported(),
            "GLM pair KDA geometry/FFN unsupported"
        );
        ensure!(
            !mode.is_two_k5() || verify_batched_ffn_enabled(),
            "GLM pair TwoK5 requires the existing batched K5 FFN dispatch"
        );
        for ctx in contexts {
            workspace.validate_context(ctx, stream)?;
            let s = ctx.buffers.sizes();
            ensure!(
                s.qkv_output >= 5 * (3 * 4096 + 32 + 2 * 128) * 2
                    && s.ssm_deinterleaved >= 2 * 5 * 4096 * 2
                    && s.ssm_qkvz >= 3 * 5 * 4096 * 2
                    && s.ssm_conv_out_f32 >= 3 * 5 * 4096 * 2,
                "GLM pair KDA working scratch capacity"
            );
            mode.validate(&self.ffn, ctx.buffers.norm_output(), ctx, stream)?;
        }
        for k in [
            self.rms_norm_k,
            self.dense_gemv_batchm_k,
            self.w4a16_gemv_batchm.kernel(5),
            self.pack_k,
            self.conv_prefill_k,
            self.recurrent_k,
            self.gated_norm_k,
            self.hc_pre_k,
            self.hc_pre_from_raw_mix_k,
            self.hc_post_k,
            self.hc_expand_k,
            self.hc_contract_k,
        ] {
            ensure!(k.0 != 0, "GLM pair KDA required handle missing");
        }
        for weight in [
            &self.weights.q_proj,
            &self.weights.k_proj,
            &self.weights.v_proj,
            &self.weights.o_proj,
        ] {
            ensure!(
                !weight.nvfp4.weight.is_null() && !weight.nvfp4.weight_scale.is_null(),
                "GLM pair KDA projection weight missing"
            );
        }
        Ok(())
    }

    pub(super) fn decode_pair(
        &self,
        owners: [GlmPairLayerInput<'_>; 2],
        workspace: &mut GlmPairWorkspace,
        contexts: [&ForwardContext; 2],
        stream: u64,
    ) -> Result<()> {
        let mode = GlmVerifyFfn::Pair(workspace.mode);
        self.decode_temporal(
            &mut { owners },
            &mut workspace.scratch,
            &contexts,
            mode,
            stream,
        )
    }

    pub(super) fn decode_temporal(
        &self,
        owners: &mut [GlmPairLayerInput<'_>],
        workspace: &mut GlmVerifyScratch,
        contexts: &[&ForwardContext],
        mode: GlmVerifyFfn,
        stream: u64,
    ) -> Result<()> {
        self.validate_temporal(workspace, contexts, mode, stream)?;
        workspace.begin_layer(self.layer_idx, owners, contexts, stream)?;
        // Validate every snapshot set before advancing any canonical state.
        let mut storage = [(DevicePtr::NULL, 0usize); 88];
        let spans = &mut storage[..owners.len() * 11];
        for (owner, input) in owners.iter().enumerate() {
            let s = input
                .state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM pair KDA state type"))?;
            ensure!(
                !s.h_is_f16
                    && s.h_prefill_stage.is_none()
                    && s.h_state_intermediates.len() >= 4
                    && s.conv_state_intermediates.len() >= 5,
                "GLM pair KDA needs each owner's FP32 K5 snapshots"
            );
            spans[owner * 11] = (s.h_state, self.h_state_bytes);
            spans[owner * 11 + 1] = (s.conv_state, self.conv_state_bytes);
            for row in 0..4 {
                spans[owner * 11 + 2 + row] = (s.h_state_intermediates[row], self.h_state_bytes);
            }
            for row in 0..5 {
                spans[owner * 11 + 6 + row] =
                    (s.conv_state_intermediates[row], self.conv_state_bytes);
            }
        }
        for (i, &(ptr, bytes)) in spans.iter().enumerate() {
            ensure!(
                !ptr.is_null() && ptr.0.is_multiple_of(4),
                "GLM pair KDA state span"
            );
            let end = ptr
                .0
                .checked_add(bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("KDA state overflow"))?;
            for &(other, len) in &spans[..i] {
                let other_end = other
                    .0
                    .checked_add(len as u64)
                    .ok_or_else(|| anyhow::anyhow!("KDA state overflow"))?;
                ensure!(
                    end <= other.0 || other_end <= ptr.0,
                    "GLM pair KDA state/snapshot alias"
                );
            }
        }
        let mut phases = [None, None, None, None, None, None, None, None];
        for (owner, input) in owners.iter_mut().enumerate() {
            workspace.restore_highway(owner, stream)?;
            phases[owner] = Some(self.forward_attention(
                input.hidden,
                input.state,
                5,
                false,
                true,
                contexts[owner],
                stream,
            )?);
            workspace.save_attention(owner, stream)?;
        }
        workspace.pack_norms(stream)?;
        let joint = mode.forward(
            &self.ffn,
            contexts[0].buffers.norm_output(),
            contexts[0],
            stream,
        )?;
        for (owner, phase) in phases.into_iter().take(owners.len()).enumerate() {
            workspace.restore_ffn(owner, joint.is_none(), stream)?;
            let mut phase = phase.expect("all KDA attention phases completed");
            if let Some(output) = joint {
                phase.normed = contexts[owner]
                    .buffers
                    .norm_output()
                    .offset(owner * OWNER_NORM_BYTES);
                self.forward_ffn_post(
                    phase,
                    output.offset(owner * OWNER_NORM_BYTES),
                    None,
                    contexts[owner],
                    stream,
                )?;
            } else {
                self.forward_ffn(phase, contexts[owner], stream)?;
            }
            workspace.save_highway(owner, stream)?;
        }
        workspace.finish_layer();
        Ok(())
    }
}
