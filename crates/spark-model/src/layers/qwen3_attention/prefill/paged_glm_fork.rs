// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_LAYER_FORK=index|1` (`layers/glm_layer_fork.rs`): a dense GLM
//! owner's semantic-index maintenance on the layer's side stream.
//!
//! `glm_chunk_attention` forks each owner after the joint KV cache write and
//! joins before its W_uk absorb. Inside the fork the index maintenance reads
//! `normed`, the slot mapping and the tail map, and writes its keys and gates
//! (`ssm_qkvz`, whose K/V entries the cache write has consumed) and the
//! layer's index tail and pool caches; the compute stream reads the latent KV
//! pool and `q_latent` and writes the BF16 latent view and `qkv_output` (q_b).
//! A dense owner selects nothing, so nothing on the compute stream reads the
//! index caches before the join. Owners with batched projections, sharded
//! latents (a peer exchange runs during the index update) and a q_b on
//! cuBLASLt keep the in-line order.

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::Qwen3AttentionLayer;
use super::{GlmChunkOwner, projection};
use crate::layer::ForwardContext;
use crate::layers::glm_layer_fork::{self, ForkLane};

impl Qwen3AttentionLayer {
    /// Fork `o`'s index maintenance (unless its projections are `batched`)
    /// onto the side stream, which the returned lane then names, when the
    /// switch, the owner and the q_b projection allow it.
    pub(super) fn glm_index_fork(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        o: &GlmChunkOwner,
        batched: bool,
        stream: u64,
    ) -> Result<Option<ForkLane>> {
        let Some(lane) = self.index_fork else {
            return Ok(None);
        };
        let wq_b = &self.mla.as_ref().expect("GLM index fork without MLA").wq_b;
        let ready = !batched
            && o.dense_is_exact(ctx.config.index_topk)
            && kv_cache.latent_shard().is_none()
            && glm_layer_fork::eager(ctx, stream)
            && (!projection::enabled(&ctx.config.model_type)?
                || self.mla_dense_is_custom(wq_b, o.rows as u32, ctx));
        if !ready {
            return Ok(None);
        }
        lane.fork(ctx.gpu, stream)?;
        Ok(Some(lane))
    }
}
