// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const ROUTER_CUTLASS_MIN_ROWS: u32 = 6144;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum RouterPrefillGemm {
    Cutlass,
    CublasLt,
}

/// Tensor-core BF16 GEMM (FP32 accumulation) for a GLM prefill router, in
/// place of the order-preserving BN32 kernel (~5 TFLOPS); routing arithmetic
/// then differs from decode's in summation order only. `cublas` is
/// `ATLAS_GLM_ROUTER_PREFILL_CUBLAS=1`; `cutlass` the opt-in
/// `ATLAS_GLM_ROUTER_PREFILL_CUTLASS=1` nested under it, which moves chunks of
/// [`ROUTER_CUTLASS_MIN_ROWS`] or more onto `ops::bf16_gemm`.
pub(super) fn glm_router_prefill_gemm(
    glm: bool,
    num_tokens: u32,
    graph_capture: bool,
    cublas: bool,
    cutlass: bool,
) -> Option<RouterPrefillGemm> {
    if !glm || num_tokens <= 64 || graph_capture || !cublas {
        None
    } else if cutlass && num_tokens >= ROUTER_CUTLASS_MIN_ROWS {
        Some(RouterPrefillGemm::Cutlass)
    } else {
        Some(RouterPrefillGemm::CublasLt)
    }
}

impl MoeLayer {
    /// Whether the prefill router GEMM of `n` rows runs `ops::bf16_gemm` on
    /// its CUTLASS tiles (`router_gate_gemm_dense`'s CUTLASS arm), as a block
    /// of `rows` of those rows does too: each then computes its rows' bytes of
    /// the whole call: those tiles' config depends on N and K alone and never
    /// splits K.
    pub(in crate::layers::moe) fn router_prefill_row_blocks(
        &self,
        n: u32,
        rows: u32,
        ctx: &ForwardContext,
    ) -> bool {
        // An arm `prefill_gate_gemm` takes before the dense router.
        let other_arm = crate::layers::glm_packed_scope::active()
            || self.gate_fp8.is_some()
            || self.gate_nvfp4.is_some()
            || (n <= crate::layers::canonical_verify::MAX_ROWS
                && crate::layers::canonical_verify::enabled())
            || self.independent_grouped(ctx, n)
            || self.glm_c3_grouped(ctx, n)
            || self.glm_c2_grouped(ctx, n)
            || self.glm_c4_grouped(ctx, n);
        !other_arm
            && glm_router_prefill_gemm(
                ctx.config.model_type == "glm5_next",
                n,
                ctx.graph_capture,
                glm_router_prefill_cublas(),
                glm_router_prefill_cutlass(),
            ) == Some(RouterPrefillGemm::Cutlass)
            && ops::bf16_gemm_cutlass_rows(n)
            && ops::bf16_gemm_cutlass_rows(rows)
    }
}

pub(super) fn glm_router_prefill_cublas() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_ROUTER_PREFILL_CUBLAS").as_deref() == Ok("1"))
}

pub(super) fn glm_router_prefill_cutlass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_ROUTER_PREFILL_CUTLASS").as_deref() == Ok("1"))
}
