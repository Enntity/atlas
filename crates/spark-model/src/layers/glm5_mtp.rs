// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 Flash model-specific MTP draft proposer.
//!
//! GLM's predictor is a complete appended decoder layer, not the compact
//! Qwen-shaped head represented by [`crate::layers::MtpHead`]. Every draft is
//! still verified by the distributed target model, so this proposer executes
//! independently on rank 0 with a private one-layer MLA cache.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::glm5::Glm5MtpModule;
use crate::weight_map::DenseWeight;

pub struct Glm5MtpProposerState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub last_num_drafted: usize,
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for Glm5MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Glm5MtpHead {
    module: Glm5MtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    mtp_vocab_size: u32,
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl Glm5MtpHead {
    pub fn new(
        module: Glm5MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: config.kv_lora_rank + config.qk_rope_head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = max_seq_len / kv_config.block_size + 1;
        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            kv_cache: Mutex::new(PagedKvCache::new(kv_config, num_blocks, gpu)?),
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Glm5MtpProposerState> {
        Ok(Glm5MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size;
        let h_u32 = h as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        // Upstream GLM: eh_proj(cat(enorm(embed(token)), hnorm(target_hidden))).
        let embed = ctx.buffers.ssm_deinterleaved();
        ctx.gpu.copy_d2d_async(
            self.embed_tokens.weight.offset(token as usize * row_bytes),
            embed,
            row_bytes,
            stream,
        )?;
        let eh_input = ctx.buffers.ssm_qkvz();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed,
            &self.module.enorm,
            eh_input,
            1,
            h_u32,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.module.hnorm,
            eh_input.offset(row_bytes),
            1,
            h_u32,
            eps,
            stream,
        )?;
        let h_in = ctx.buffers.hidden_states();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            eh_input,
            &self.module.eh_proj,
            h_in,
            h_u32,
            (2 * h) as u32,
            stream,
        )?;

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = state.seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let block_idx = state.block_table[state.seq_len / bs];
        let slot = block_idx as i64 * bs as i64 + (state.seq_len % bs) as i64;
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut disk_ids = Vec::new();
        let mut disk_last = vec![0u32; 1];
        // The MTP decoder layer is constructed with `residual=None` upstream.
        // Atlas represents that as a zeroed residual row before its fused
        // add+norm path initializes the residual stream from `h_in`.
        let residual = ctx.buffers.residual();
        ctx.gpu.memset_async(residual, 0, row_bytes, stream)?;
        self.module.body.decode(
            h_in,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_ids,
            &mut disk_last,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        let h_out = ctx.buffers.hidden_states();
        let final_hidden = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            h_out,
            &self.module.norm,
            final_hidden,
            1,
            h_u32,
            eps,
            stream,
        )?;
        let vocab = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            final_hidden,
            &self.lm_head,
            logits,
            vocab,
            h_u32,
            stream,
        )?;
        let out = ctx.buffers.scratch();
        let draft = if let Some(mask) = grammar_bitmask {
            grammar_argmax(ctx.gpu, logits, vocab as usize, mask)?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h(out, &mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len += 1;
        Ok(draft)
    }
}

fn grammar_argmax(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    mask: &[i32],
) -> Result<u32> {
    let mut raw = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut raw)?;
    let mut best = (0u32, f32::NEG_INFINITY);
    for token in 0..vocab {
        if token / 32 >= mask.len() || mask[token / 32] & (1i32 << (token % 32)) == 0 {
            continue;
        }
        let bf16 = u16::from_le_bytes([raw[2 * token], raw[2 * token + 1]]);
        let value = f32::from_bits((bf16 as u32) << 16);
        if value > best.1 {
            best = (token as u32, value);
        }
    }
    Ok(best.0)
}

impl DraftProposer for Glm5MtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("invalid GLM-5 MTP proposer state"))?;
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        for i in 0..num_drafts {
            let draft = self.forward_one(
                token,
                hidden,
                position + i,
                state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            drafts.push(draft);
            token = draft;
            // GLM recycles the MTP layer's post-shared-norm hidden.
            hidden = ctx.buffers.norm_output();
        }
        state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("invalid GLM-5 MTP proposer state"))?;
        state.seq_len = state
            .seq_len
            .saturating_sub(state.last_num_drafted.saturating_sub(num_accepted));
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("invalid GLM-5 MTP proposer state"))?;
        if !state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&state.block_table);
            state.block_table.clear();
        }
        state.seq_len = 0;
        Ok(())
    }
}
