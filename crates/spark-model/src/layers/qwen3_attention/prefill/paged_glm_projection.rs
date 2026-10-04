// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit experiment for four BF16 projections in GLM's paged prefill.
use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::{layer::ForwardContext, layers::ops, weight_map::DenseWeight};

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_PAGED_PREFILL_BF16_GEMM must be 0 or 1, got {other:?}"),
    }
}

pub(super) fn enabled(model: &str) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    match std::env::var("ATLAS_GLM_PAGED_PREFILL_BF16_GEMM") {
        Ok(value) => parse(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(error) => Err(error.into()),
    }
}

impl Qwen3AttentionLayer {
    /// `ATLAS_GLM_L2_AHEAD`: what [`Self::paged_glm_projection`] reads of
    /// `weight` (`n` rows by `k`) for `m` rows, as `mla_prefill_dense` picks
    /// it: the NVFP4 or MXFP8 twin, else the BF16 weight.
    pub(crate) fn paged_glm_reads(
        &self,
        weight: &DenseWeight,
        [m, n, k]: [u32; 3],
        ctx: &ForwardContext,
    ) -> Result<Vec<ops::L2Region>> {
        if enabled(&ctx.config.model_type)? && m <= 32 {
            let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(m).0 != 0;
            if let Some((_, q4)) = self.mla_q4.iter().find(|(w, _)| tc && *w == weight.weight) {
                return Ok(ops::L2Region::nvfp4(q4, n, k, k / 2, k / 16).to_vec());
            }
            if let Some((.., mx)) = self.mla_mx.iter().find(|(w, ..)| *w == weight.weight) {
                return Ok(ops::L2Region::mxfp8(mx, n as usize, k as usize).to_vec());
            }
        }
        Ok(vec![ops::L2Region::whole(
            weight.weight,
            n as usize * k as usize * 2,
        )])
    }

    /// `ATLAS_GLM_L2_AHEAD`: the sparse-MLA projections a verify of `rows`
    /// rows reads first: q_a, kv_a, then q_b (`l2_ahead_lead`).
    pub(crate) fn glm_l2_ahead_lead(
        &self,
        rows: u32,
        ctx: &ForwardContext,
    ) -> Result<Vec<ops::L2Region>> {
        let Some(mla) = self.mla.as_ref().filter(|m| m.glm_indexer.is_some()) else {
            return Ok(Vec::new());
        };
        let h = ctx.config.hidden_size as u32;
        let (q_lora, kv_lora) = (mla.q_lora_rank as u32, mla.kv_lora_rank as u32);
        let mut lead = self.paged_glm_reads(&mla.wq_a, [rows, q_lora, h], ctx)?;
        lead.extend(self.paged_glm_reads(&mla.wkv_a, [rows, kv_lora, h], ctx)?);
        lead.extend(self.glm_q_b_reads(rows, ctx)?);
        Ok(lead)
    }

    /// What q_b reads for `rows` rows.
    pub(crate) fn glm_q_b_reads(
        &self,
        rows: u32,
        ctx: &ForwardContext,
    ) -> Result<Vec<ops::L2Region>> {
        let Some(mla) = self.mla.as_ref() else {
            return Ok(Vec::new());
        };
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let n = nq * mla.nope as u32;
        self.paged_glm_reads(&mla.wq_b, [rows, n, mla.q_lora_rank as u32], ctx)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paged_glm_projection(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
        accelerated: bool,
    ) -> Result<()> {
        // The existing helper selects BF16 cuBLAS when enabled, otherwise its
        // established TC/scalar fallback. No weight or activation conversion.
        if accelerated && ctx.config.model_type == "glm5_next" {
            return self.mla_prefill_dense(input, weight, output, m, n, k, ctx, stream);
        }
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            input,
            weight,
            output,
            m,
            n,
            k,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn paged_projection_flag_is_explicit() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("0")).unwrap());
        assert!(parse(Some("1")).unwrap());
        for value in ["", "true", "2", " 1"] {
            assert!(parse(Some(value)).is_err());
        }
    }
}
