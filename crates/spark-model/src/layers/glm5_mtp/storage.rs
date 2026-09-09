// SPDX-License-Identifier: AGPL-3.0-only
//! One checked private-cache layout for construction and preflight accounting.
use super::*;
use anyhow::Context;
use spark_runtime::kv_cache::SparseIndexCacheConfig;

pub(super) struct PrivateStoragePlan {
    pub(super) config: KvCacheConfig,
    pub(super) index: Option<SparseIndexCacheConfig>,
    pub(super) blocks: usize,
    bytes: usize,
}

impl PrivateStoragePlan {
    pub(super) fn new(
        config: &atlas_core::config::ModelConfig,
        max_seq_len: usize,
        paired: Option<(usize, paired::OwnerCapacity)>,
    ) -> Result<Self> {
        let shape = GlmMlaShape::new(config.kv_lora_rank, config.qk_rope_head_dim)?;
        let kv = KvCacheConfig {
            block_size: 16,
            num_kv_heads: shape.num_kv_heads(),
            head_dim: shape.head_dim(),
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let index = (config.index_kpool > 0 && config.index_head_dim > 0)
            .then(|| shape.bf16_index(config.index_kpool, config.index_head_dim))
            .transpose()?;
        let cache = GlmCachePlan::new(shape, &kv, index)?;
        let (blocks, slab) = if let Some((context, capacity)) = paired {
            (capacity.cache_blocks(context)?, capacity.slab_bytes())
        } else {
            (max_seq_len / kv.block_size + 1, 0)
        };
        let bytes = cache
            .bytes_for_blocks(blocks)?
            .checked_add(slab)
            .context("GLM private cache/slab reserve overflow")?;
        Ok(Self {
            config: kv,
            index,
            blocks,
            bytes,
        })
    }
}

impl Glm5MtpHead {
    /// Device payload only: private K/V, optional index/tails and owner slab.
    /// Shared weights, allocator overhead and target state remain separate.
    /// This immutable quote is not serving admission or an allocation receipt.
    pub fn paired_private_reserve_bytes(
        config: &atlas_core::config::ModelConfig,
        context: usize,
        owners: usize,
    ) -> Result<usize> {
        Self::validate_paired_shape(config)?;
        let capacity = paired::OwnerCapacity::new(owners)?;
        Ok(PrivateStoragePlan::new(config, context, Some((context, capacity)))?.bytes)
    }
}
