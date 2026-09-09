// SPDX-License-Identifier: AGPL-3.0-only

//! Two unchanged temporal K5 attention passes with owner-preserved mHC tails.

use super::*;
use crate::layer::glm_pair_verify::{
    GlmPairFfn, GlmPairLayerInput, GlmPairWorkspace, OWNER_NORM_BYTES,
};
use anyhow::ensure;
use spark_runtime::kv_cache::KvCacheDtype;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn pair_mla_supported(&self) -> bool {
        self.hc.as_ref().is_some_and(|hc| hc.hc_mult == 4)
            && self
                .mla
                .as_ref()
                .is_some_and(|mla| mla.rope == 0 && mla.o_lora_rank == 0)
            && !self.ffn.is_none()
    }

    pub(in crate::layers::qwen3_attention) fn validate_pair_mla(
        &self,
        context: &ForwardContext,
        mode: GlmPairFfn,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.pair_mla_supported(),
            "GLM pair requires mHC/NoPE MLA with FFN"
        );
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16,
            "GLM pair MLA layer cache dtype"
        );
        let workspace = GlmPairWorkspace::new(context, mode)?;
        workspace.validate_context(context, stream)?;
        let positions = [0, 1, 2, 3, 4];
        let c = ctx::MultiSeqCtx::new(
            self,
            context,
            context.buffers.hidden_states(),
            context.buffers.residual(),
            5,
            &positions,
            16,
            stream,
        );
        let mla = self.mla.as_ref().unwrap();
        ensure!(
            self.glm_mla_multi_seq_eligible(&c, mla)?
                && !super::super::super::glm_multi_seq_sparse_enabled(&context.config.model_type),
            "GLM pair requires existing dense temporal K5 MLA path"
        );
        self.validate_glm_c4(&c)?;
        match mode {
            GlmPairFfn::TwoK5 => {
                self.ffn
                    .validate_pair_k5(context.buffers.norm_output(), context, stream)?
            }
            GlmPairFfn::Joint => {
                self.ffn
                    .validate_pair_verify(context.buffers.norm_output(), context, stream)?
            }
        }
        for kernel in [
            self.rms_norm_w_k,
            self.dense_gemv_batchm_k,
            self.w4a16_batchm.kernel(5),
            self.mla_batched_gemv_batch5_k,
            self.mla_cache_assemble_batched_k,
            self.reshape_cache_k,
            self.paged_decode_mla_k,
            self.hc_expand_k,
            self.hc_pre_k,
            self.hc_post_k,
        ] {
            ensure!(kernel.0 != 0, "GLM pair MLA required handle missing");
        }
        let hc = self.hc.as_ref().unwrap();
        ensure!(
            self.block_idx + 1 != context.config.num_hidden_layers
                || (hc.head.is_some() && self.hc_head_k.0 != 0),
            "GLM pair final MLA requires its actual mHC head"
        );
        ensure!(
            !mla.w_uk_t.weight.is_null()
                && !mla.w_uv.weight.is_null()
                && !mla.q_a_norm.weight.is_null()
                && !mla.kv_a_norm.weight.is_null(),
            "GLM pair MLA absorbed/norm weights missing"
        );
        let sizes = context.buffers.sizes();
        for (have, n, width) in [
            (sizes.ssm_ba, 5usize, mla.q_lora_rank),
            (sizes.ssm_deinterleaved, 5 * c.nq as usize, c.hd as usize),
            (sizes.expert_gate_out, 5, mla.kv_lora_rank),
            (sizes.expert_up_out, 5 * c.nq as usize, mla.kv_lora_rank),
            (sizes.attn_output, 5 * c.nq as usize, mla.kv_lora_rank),
            (sizes.qkv_output, 10, mla.kv_lora_rank),
            (sizes.ssm_qkvz, 5 * c.nq as usize, mla.v_dim),
        ] {
            let needed = n
                .checked_mul(width)
                .and_then(|x| x.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("GLM pair MLA scratch overflow"))?;
            ensure!(
                needed > 0 && have >= needed,
                "GLM pair MLA working scratch capacity"
            );
        }
        Ok(())
    }

    pub(in crate::layers::qwen3_attention) fn decode_pair_mla(
        &self,
        owners: [GlmPairLayerInput<'_>; 2],
        cache: &mut PagedKvCache,
        workspace: &mut GlmPairWorkspace,
        contexts: [&ForwardContext; 2],
        stream: u64,
    ) -> Result<()> {
        for context in contexts {
            self.validate_pair_mla(context, workspace.mode, stream)?;
        }
        workspace.begin_layer(self.block_idx, &owners, contexts, stream)?;
        ensure!(
            cache.block_size() == 16
                && cache.config().dtype == KvCacheDtype::Bf16
                && cache.config().cache_blocks_per_seq.is_none(),
            "GLM pair MLA requires BF16 dense16 cache"
        );
        for (owner, input) in owners.iter().enumerate() {
            ensure!(
                input.state.as_any().is::<crate::layer::EmptyLayerState>()
                    && input.block_table.len() > input.positions[4] / 16
                    && input
                        .block_table
                        .iter()
                        .all(|&block| (block as usize) < cache.num_blocks()),
                "GLM pair MLA actual state/block map invalid"
            );
            let meta = contexts[owner]
                .attn_metadata
                .ok_or_else(|| anyhow::anyhow!("GLM pair MLA metadata missing"))?;
            ensure!(
                meta.num_seqs == 5
                    && meta.max_blocks_per_seq as usize >= input.block_table.len()
                    && !meta.positions.is_null()
                    && !meta.slot.is_null()
                    && !meta.seq_len.is_null()
                    && !meta.block_table.is_null(),
                "GLM pair MLA K5 metadata invalid"
            );
        }
        for &p0 in owners[0].positions {
            for &p1 in owners[1].positions {
                ensure!(
                    owners[0].block_table[p0 / 16] != owners[1].block_table[p1 / 16]
                        || p0 % 16 != p1 % 16,
                    "GLM pair MLA writable cache slots alias"
                );
            }
        }
        let make_ctx = |owner: usize| {
            let mut c = ctx::MultiSeqCtx::new(
                self,
                contexts[owner],
                owners[owner].hidden,
                contexts[owner]
                    .buffers
                    .residual()
                    .offset(owner * OWNER_NORM_BYTES),
                5,
                owners[owner].positions,
                16,
                stream,
            );
            c.seq_slot = contexts[owner].attn_metadata.unwrap().seq_slot;
            c
        };
        let cs = [make_ctx(0), make_ctx(1)];
        let mut phases = [None, None];
        for owner in 0..2 {
            workspace.restore_highway(owner, stream)?;
            phases[owner] =
                self.ms_hc_attention_norm(&cs[owner], cache, contexts[owner], stream)?;
            ensure!(phases[owner].is_some(), "GLM pair MLA FFN phase missing");
            workspace.save_attention(owner, stream)?;
        }
        workspace.pack_norms(stream)?;
        let joint = if workspace.mode == GlmPairFfn::Joint {
            Some(self.ffn.forward_pair_verify(
                contexts[0].buffers.norm_output(),
                contexts[0],
                stream,
            )?)
        } else {
            None
        };
        for (owner, phase) in phases.into_iter().enumerate() {
            workspace.restore_ffn(owner, joint.is_none(), stream)?;
            let phase = phase.expect("both MLA attention phases completed");
            if let Some(output) = joint {
                self.ms_hc_supplied_post(
                    &cs[owner],
                    phase,
                    output.offset(owner * OWNER_NORM_BYTES),
                    contexts[owner],
                    stream,
                )?;
            } else {
                self.ms_hc_ffn_post(&cs[owner], phase, contexts[owner], stream)?;
            }
            workspace.save_highway(owner, stream)?;
        }
        workspace.finish_layer();
        Ok(())
    }
}
