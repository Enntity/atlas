// SPDX-License-Identifier: AGPL-3.0-only

//! Existing interleaved/fused snapshot paths, kept in their original stream order.

use super::*;

impl Glm5KdaLayer {
    pub(super) fn forward_recurrent(
        &self,
        projected: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        state: &mut SsmLayerState,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let m = tokens as u32;
        let p = self.heads * self.dim;
        let bf16 = 2usize;
        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(ctx.gpu, self.pack_k, projected, packed, m, p as u32, stream)?;
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        if capture_verify_intermediates {
            ensure!(
                state.h_state_intermediates.len() + 1 >= tokens
                    && state.conv_state_intermediates.len() >= tokens,
                "GLM-5 KDA verify needs K-1 h and K conv intermediates (h={}, conv={}, K={tokens})",
                state.h_state_intermediates.len(),
                state.conv_state_intermediates.len(),
            );
            let packed_row_bytes = 3 * p * bf16;
            let gate_row_bytes = p * bf16;
            let beta_row_bytes = self.heads * bf16;
            let conv_inter_base = state.conv_state_intermediates[0];
            let h_inter_base = state.h_state_intermediates[0];
            let fused_conv_snapshots = tokens == 5
                && self.conv_prefill_tp_snap_k.0 != 0
                && verify_batched_conv_snapshot_enabled()
                && !conv_inter_base.is_null()
                && state.conv_state_intermediates[..tokens - 1]
                    .iter()
                    .enumerate()
                    .all(|(t, ptr)| {
                        ptr.0 == conv_inter_base.0 + (t * self.conv_state_bytes) as u64
                    });
            let fused_recurrent_snapshots = tokens == 5
                && self.recurrent_verify_snap_k.0 != 0
                && verify_batched_recurrent_snapshot_enabled()
                && !h_inter_base.is_null()
                && state.h_state_intermediates[..tokens - 1]
                    .iter()
                    .enumerate()
                    .all(|(t, ptr)| ptr.0 == h_inter_base.0 + (t * self.h_state_bytes) as u64);

            if fused_conv_snapshots || fused_recurrent_snapshots {
                if fused_conv_snapshots {
                    ops::conv1d_update_prefill_tp_snap(
                        ctx.gpu,
                        self.conv_prefill_tp_snap_k,
                        state.conv_state,
                        packed,
                        &self.weights.conv,
                        DevicePtr::NULL,
                        convolved,
                        conv_inter_base,
                        self.conv_state_bytes / size_of::<f32>(),
                        (3 * p) as u32,
                        self.conv_width as u32,
                        m,
                        (3 * p) as u32,
                        (3 * p) as u32,
                        stream,
                    )?;
                } else {
                    for t in 0..tokens {
                        ops::conv1d_update_prefill(
                            ctx.gpu,
                            self.conv_prefill_k,
                            self.conv_prefill_tp_k,
                            state.conv_state,
                            packed.offset(t * packed_row_bytes),
                            &self.weights.conv,
                            DevicePtr::NULL,
                            convolved.offset(t * packed_row_bytes),
                            (3 * p) as u32,
                            self.conv_width as u32,
                            1,
                            (3 * p) as u32,
                            (3 * p) as u32,
                            stream,
                        )?;
                        if t + 1 < tokens {
                            ctx.gpu.copy_d2d_async(
                                state.conv_state,
                                state.conv_state_intermediates[t],
                                self.conv_state_bytes,
                                stream,
                            )?;
                        }
                    }
                }

                if fused_recurrent_snapshots {
                    ops::kda_recurrent_verify_snap(
                        ctx.gpu,
                        self.recurrent_verify_snap_k,
                        convolved,
                        g1,
                        beta,
                        self.weights.a_log.weight,
                        self.weights.dt_bias.weight,
                        state.h_state,
                        core_out,
                        h_inter_base,
                        self.h_state_bytes / size_of::<f32>(),
                        m,
                        self.heads as u32,
                        self.dim as u32,
                        self.lower_bound,
                        stream,
                    )?;
                } else {
                    for t in 0..tokens {
                        self.run_recurrent(
                            convolved.offset(t * packed_row_bytes),
                            g1.offset(t * gate_row_bytes),
                            beta.offset(t * beta_row_bytes),
                            state.h_state,
                            core_out.offset(t * gate_row_bytes),
                            1,
                            true,
                            ctx,
                            stream,
                        )?;
                        if t + 1 < tokens {
                            ctx.gpu.copy_d2d_async(
                                state.h_state,
                                state.h_state_intermediates[t],
                                self.h_state_bytes,
                                stream,
                            )?;
                        }
                    }
                }
            } else {
                // Preserve the original interleaved path as the exact fallback.
                for t in 0..tokens {
                    ops::conv1d_update_prefill(
                        ctx.gpu,
                        self.conv_prefill_k,
                        self.conv_prefill_tp_k,
                        state.conv_state,
                        packed.offset(t * packed_row_bytes),
                        &self.weights.conv,
                        DevicePtr::NULL,
                        convolved.offset(t * packed_row_bytes),
                        (3 * p) as u32,
                        self.conv_width as u32,
                        1,
                        (3 * p) as u32,
                        (3 * p) as u32,
                        stream,
                    )?;
                    self.run_recurrent(
                        convolved.offset(t * packed_row_bytes),
                        g1.offset(t * gate_row_bytes),
                        beta.offset(t * beta_row_bytes),
                        state.h_state,
                        core_out.offset(t * gate_row_bytes),
                        1,
                        true,
                        ctx,
                        stream,
                    )?;
                    // A partial accept can select states after rows 0..K-2;
                    // the post-row K-1 state is already canonical on full accept.
                    if t + 1 < tokens {
                        ctx.gpu.copy_d2d_async(
                            state.h_state,
                            state.h_state_intermediates[t],
                            self.h_state_bytes,
                            stream,
                        )?;
                        ctx.gpu.copy_d2d_async(
                            state.conv_state,
                            state.conv_state_intermediates[t],
                            self.conv_state_bytes,
                            stream,
                        )?;
                    }
                }
            }
        } else {
            ops::conv1d_update_prefill(
                ctx.gpu,
                self.conv_prefill_k,
                self.conv_prefill_tp_k,
                state.conv_state,
                packed,
                &self.weights.conv,
                DevicePtr::NULL,
                convolved,
                (3 * p) as u32,
                self.conv_width as u32,
                m,
                (3 * p) as u32,
                (3 * p) as u32,
                stream,
            )?;
            if let Some(flash) = self
                .flash_prefill
                .as_ref()
                .filter(|_| flash_prefill::eligible(tokens, decode, ctx.graph_capture))
            {
                // Convolution has finished reading packed. Its storage and the
                // idle expert buffers can now be borrowed on this same stream.
                flash.forward(
                    convolved,
                    g1,
                    beta,
                    self.weights.a_log.weight,
                    self.weights.dt_bias.weight,
                    state.h_state,
                    core_out,
                    tokens,
                    ctx,
                    stream,
                )?;
            } else {
                self.run_recurrent(
                    convolved,
                    g1,
                    beta,
                    state.h_state,
                    core_out,
                    m,
                    decode,
                    ctx,
                    stream,
                )?;
            }
        }
        Ok(core_out)
    }
}
