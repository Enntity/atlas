// SPDX-License-Identifier: AGPL-3.0-only

use super::ctx::MultiSeqCtx;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::model::glm_c4;
use anyhow::{Result, ensure};

impl Qwen3AttentionLayer {
    pub(super) fn validate_glm_c4(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let ctx = c.fwd;
        if c.n != 4 || ctx.config.model_type != "glm5_next" {
            return Ok(());
        }
        glm_c4::validate_runtime(
            ctx.config,
            ctx.comm.map_or(0, |comm| comm.world_size()),
            std::env::var("ATLAS_EP_PROTOCOL").as_deref() == Ok("v2"),
            true,
        )?;
        glm_c4::validate_positions(c.seq_lens.iter().copied(), 4)?;
        glm_c4::validate_projection_handles(
            self.w4a16_batchm.kernel(4).0,
            self.dense_gemv_batchm_k.0,
        )?;
        glm_c4::validate_scratch(ctx.buffers.sizes(), ctx.config.hc_mult)?;
        ensure!(
            self.mla
                .as_ref()
                .is_some_and(|mla| self.glm_mla_multi_seq_eligible(c, mla)),
            "GLM C4 requires the dedicated batched MLA path"
        );
        Ok(())
    }
}
