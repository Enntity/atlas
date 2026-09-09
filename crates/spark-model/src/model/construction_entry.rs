// SPDX-License-Identifier: AGPL-3.0-only
//! Ordinary constructor entry retains its existing ownership and call signature.
use super::TransformerModel;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use std::sync::Arc;

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: ModelConfig,
        embed_tokens: DenseWeight,
        final_norm: DenseWeight,
        lm_head_weight: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        lm_head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
        mtp_lm_head_nvfp4: Option<QuantizedWeight>,
        layers: Vec<Box<dyn TransformerLayer>>,
        buffers: BufferArena,
        kv_cache: PagedKvCache,
        mtp_weights: Vec<MtpWeights>,
        gpu: Box<dyn GpuBackend>,
        max_seq_len: usize,
        max_batch_size: usize,
        mtp_quant: crate::layers::MtpQuantization,
        use_speculative: bool,
        external_mtp_proposer: bool,
        prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
        mtp_vocab_size: u32,
        comm: Option<Arc<dyn spark_comm::CommBackend>>,
        self_speculative: bool,
        num_drafts: usize,
        vision_encoder: Option<crate::layers::VisionEncoder>,
        ssm_cache_slots: usize,
        ssm_checkpoint_interval: usize,
    ) -> Result<Self> {
        Self::new_with_cold_retention(
            config,
            embed_tokens,
            final_norm,
            lm_head_weight,
            lm_head_nvfp4,
            lm_head_fp8,
            mtp_lm_head_nvfp4,
            layers,
            buffers,
            kv_cache,
            mtp_weights,
            gpu,
            max_seq_len,
            max_batch_size,
            mtp_quant,
            use_speculative,
            external_mtp_proposer,
            prefix_cache,
            mtp_vocab_size,
            comm,
            self_speculative,
            num_drafts,
            vision_encoder,
            ssm_cache_slots,
            ssm_checkpoint_interval,
            false,
        )
    }
}
