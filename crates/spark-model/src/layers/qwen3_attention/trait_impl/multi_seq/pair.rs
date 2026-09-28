// SPDX-License-Identifier: AGPL-3.0-only

//! Unchanged temporal K5 attention passes with owner-preserved mHC tails.

use super::*;
use crate::layer::glm_pair_verify::{
    GlmPairFfn, GlmPairLayerInput, GlmPairWorkspace, OWNER_NORM_BYTES,
};
use crate::layer::{glm_verify_ffn::GlmVerifyFfn, glm_verify_scratch::GlmVerifyScratch};
use anyhow::ensure;
use spark_runtime::kv_cache::KvCacheDtype;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn pair_mla_supported(&self) -> bool {
        self.qsa.is_none()
            && self.hc.as_ref().is_some_and(|hc| {
                hc.hc_mult == 4
                    && hc.attn.lowrank.is_none()
                    && hc.ffn.lowrank.is_none()
                    && hc.head.as_ref().is_none_or(|head| head.lowrank.is_none())
            })
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
        self.validate_verify_mla(context, GlmVerifyFfn::Pair(mode), stream)
    }

    pub(in crate::layers::qwen3_attention) fn validate_verify_mla(
        &self,
        context: &ForwardContext,
        mode: GlmVerifyFfn,
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
        let workspace = GlmVerifyScratch::new(context, mode.owners())?;
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
        mode.validate(&self.ffn, context.buffers.norm_output(), context, stream)?;
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
        mut owners: [GlmPairLayerInput<'_>; 2],
        cache: &mut PagedKvCache,
        workspace: &mut GlmPairWorkspace,
        contexts: [&ForwardContext; 2],
        stream: u64,
    ) -> Result<()> {
        self.decode_verify_mla(
            &mut owners,
            cache,
            &mut workspace.scratch,
            &contexts,
            GlmVerifyFfn::Pair(workspace.mode),
            stream,
        )
    }

    pub(in crate::layers::qwen3_attention) fn decode_verify_mla(
        &self,
        owners: &mut [GlmPairLayerInput<'_>],
        cache: &mut PagedKvCache,
        workspace: &mut GlmVerifyScratch,
        contexts: &[&ForwardContext],
        mode: GlmVerifyFfn,
        stream: u64,
    ) -> Result<()> {
        let count = mode.owners();
        ensure!(
            owners.len() == count && contexts.len() == count && workspace.owner_count() == count,
            "GLM MLA owner/context/scratch/policy count mismatch"
        );
        for context in contexts {
            self.validate_verify_mla(context, mode, stream)?;
        }
        workspace.begin_layer(self.block_idx, owners, contexts, stream)?;
        ensure!(
            cache.block_size() == 16
                && cache.config().dtype == KvCacheDtype::Bf16
                && cache.config().cache_blocks_per_seq.is_none(),
            "GLM pair MLA requires BF16 dense16 cache"
        );
        for (owner, input) in owners.iter().enumerate() {
            // Upstream attention allocates this wrapper even when QSA is absent.
            // Accept only its truly stateless form (or the legacy empty state),
            // never QSA carry, recurrent/PLE state, or another layer's state.
            let stateless = input.state.as_any().is::<crate::layer::EmptyLayerState>()
                || input
                    .state
                    .as_any()
                    .downcast_ref::<crate::layer::AttnLayerState>()
                    .is_some_and(|state| state.qsa.is_none());
            ensure!(
                stateless
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
        // Complete every cross-owner writable K5 slot check before the first
        // attention writer, including nonadjacent owners in a wider batch.
        for (index, input) in owners.iter().enumerate() {
            for other in &owners[..index] {
                for &p0 in input.positions {
                    for &p1 in other.positions {
                        ensure!(
                            input.block_table[p0 / 16] != other.block_table[p1 / 16]
                                || p0 % 16 != p1 % 16,
                            "GLM pair MLA writable cache slots alias"
                        );
                    }
                }
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
        let cs: [Option<ctx::MultiSeqCtx<'_>>; 8] =
            std::array::from_fn(|owner| (owner < count).then(|| make_ctx(owner)));
        let mut phases = [None, None, None, None, None, None, None, None];
        for owner in 0..count {
            workspace.restore_highway(owner, stream)?;
            phases[owner] = self.ms_hc_attention_norm(
                cs[owner].as_ref().expect("bounded owner context"),
                cache,
                contexts[owner],
                stream,
            )?;
            ensure!(phases[owner].is_some(), "GLM pair MLA FFN phase missing");
            workspace.save_attention(owner, stream)?;
        }
        workspace.pack_norms(stream)?;
        let joint = mode.forward(
            &self.ffn,
            contexts[0].buffers.norm_output(),
            contexts[0],
            stream,
        )?;
        for (owner, phase) in phases.into_iter().enumerate().take(count) {
            workspace.restore_ffn(owner, joint.is_none(), stream)?;
            let phase = phase.expect("all MLA attention phases completed");
            let context = cs[owner].as_ref().expect("bounded owner context");
            if let Some(output) = joint {
                self.ms_hc_supplied_post(
                    context,
                    phase,
                    output.offset(owner * OWNER_NORM_BYTES),
                    contexts[owner],
                    stream,
                )?;
            } else {
                self.ms_hc_ffn_post(context, phase, contexts[owner], stream)?;
            }
            workspace.save_highway(owner, stream)?;
        }
        workspace.finish_layer();
        Ok(())
    }
}
