// SPDX-License-Identifier: AGPL-3.0-only
//! Shared constructor; legacy allocation/kernel ordering is unchanged.
use super::*;

impl Glm5MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        module: Glm5MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        Self::new_with_capacity(
            module,
            embed_tokens,
            lm_head,
            lm_head_nvfp4,
            config,
            gpu,
            mtp_vocab_size,
            max_seq_len,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_with_capacity(
        module: Glm5MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        paired_context: Option<usize>,
    ) -> Result<Self> {
        let hidden_trace_enabled = hidden_trace::configured()?;
        anyhow::ensure!(
            paired_context.is_none() || !hidden_trace_enabled,
            "paired handoff does not support the C1 hidden diagnostic"
        );
        let cache_shape = GlmMlaShape::new(config.kv_lora_rank, config.qk_rope_head_dim)?;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: cache_shape.num_kv_heads(),
            head_dim: cache_shape.head_dim(),
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let sparse_index = (config.index_kpool > 0 && config.index_head_dim > 0)
            .then(|| cache_shape.bf16_index(config.index_kpool, config.index_head_dim))
            .transpose()?;
        let cache_plan = GlmCachePlan::new(cache_shape, &kv_config, sparse_index)?;
        let num_blocks = if let Some(context) = paired_context {
            paired::blocks_per_slot(context)?
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("paired KV block count overflow"))?
        } else {
            max_seq_len / kv_config.block_size + 1
        };
        cache_plan.bytes_for_blocks(num_blocks)?;
        let mut kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;
        if let Some(index) = sparse_index {
            kv_cache.attach_sparse_index(index, gpu)?;
        }
        let mut result = Self {
            paired: None,
            hidden_trace_enabled,
            module,
            embed_tokens,
            lm_head,
            lm_head_nvfp4,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            fused_eh_norm_k: gpu
                .kernel("glm_mtp_eh_norm", "glm_mtp_eh_norm")
                .unwrap_or(KernelHandle(0)),
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            argmax_value_k: gpu.kernel("argmax", "argmax_bf16_value")?,
        };
        if let Some(context) = paired_context {
            let pool = paired::Pool::new(gpu, context, &result.kv_cache.lock(), config.ep_rank)?;
            result.paired = Some(Mutex::new(pool));
        }
        Ok(result)
    }
}
