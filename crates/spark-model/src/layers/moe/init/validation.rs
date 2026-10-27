// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;

pub(super) fn routing(config: &ModelConfig, num_experts: usize) -> Result<bool> {
    // Sanity-check the routing config: top-k that exceeds the
    // expert count would index OOB in the topk kernel and produce
    // silent NaN routing. Catch the misconfiguration at load time.
    let c2_compact_moe = crate::layers::moe::forward_c2::c2_compact_requested()?;
    anyhow::ensure!(
        config.num_experts_per_tok <= num_experts && num_experts > 0,
        "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
        config.num_experts_per_tok,
        num_experts,
    );
    // The check above bounds top-k by the expert count, which is the OOB
    // the topk kernel can READ. It says nothing about the OOB the kernel
    // can WRITE: the sigmoid routing kernels stage their top-K in a
    // fixed-size shared array, and `top_k` was passed to them unbounded.
    anyhow::ensure!(
        config.num_experts_per_tok <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
            && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        "MoE config exceeds the routing kernels' fixed shared-memory bounds: \
         num_experts_per_tok={} (max {}), num_experts={} (max {}). Raise \
         MAX_TOP_K / MAX_EXPERTS in kernels/gb10/common/moe_topk_sigmoid.cu \
         and their mirrors in layers::ops together.",
        config.num_experts_per_tok,
        crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
        num_experts,
        crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
    );
    Ok(c2_compact_moe)
}
