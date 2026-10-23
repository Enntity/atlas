// SPDX-License-Identifier: AGPL-3.0-only

use super::k128w_kernels;
use spark_runtime::gpu::mock::MockGpuBackend;

fn handles(k128w: bool, silu: bool, persist: bool) -> ([u64; 4], u32) {
    let (down, gate_up, ctas) = k128w_kernels(&MockGpuBackend::new(), k128w, silu, persist);
    let ids = [down.grid, down.persist, gate_up.grid, gate_up.persist].map(|h| h.0);
    (ids.map(|h| u64::from(h != 0)), ctas)
}

#[test]
fn k128w_persist_adds_the_twins_and_keeps_the_grid_kernels() {
    // Flag off: the base grid kernels only.
    assert_eq!(handles(true, true, false), ([1, 0, 1, 0], 0));
    // Flag on: both sets, two CTAs per SM (the mock's 48 SMs).
    assert_eq!(handles(true, true, true), ([1, 1, 1, 1], 96));
    assert_eq!(handles(true, false, true), ([1, 1, 0, 0], 96));
    // K128W off: nothing, whatever the flag.
    assert_eq!(handles(false, true, true), ([0, 0, 0, 0], 0));
}
#[test]
fn unrelated_targets_do_not_resolve_glm_moe_kernels() {
    for model in [
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_6_moe",
        "qwen4_exp",
        "glm5_next",
    ] {
        let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = model.into();
        let k = super::OptionalKernels::resolve(&MockGpuBackend::new(), &config);
        let glm = model == "glm5_next";
        for handle in [
            k.moe_grouped_gemm_t_k64_m32,
            k.moe_w4a4_prequant_t_k64,
            k.moe_w4a4_prequant_t_k64_vecscale,
            k.moe_nvfp4_mmq_gate_up_k,
            k.moe_nvfp4_mmq_down_k,
            k.moe_nvfp4_mmq_quantize_k,
            k.moe_nvfp4_mmq_repack_k,
            k.moe_nvfp4_mmq_silu_scale2_k,
            k.moe_nvfp4_mmq_scale2_rows_k,
            k.moe_expert_gate_up_shared_mmq_k,
            k.moe_expert_silu_down_shared_mmq_k,
            k.silu_mul_quant_nvfp4_k,
        ] {
            assert_eq!(handle.0 != 0, glm, "{model}: GLM-only MoE kernel");
        }
        assert_ne!(k.moe_w8a8_grouped_gemm_k.0, 0, "shared path must remain");
    }
}
#[test]
fn unrelated_targets_do_not_resolve_glm_router_or_batch5() {
    for model in [
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_6_moe",
        "qwen4_exp",
        "glm5_next",
    ] {
        let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = model.into();
        config.router_logits_f32 = true;
        let layer = super::MoeLayer::new(
            super::MoeWeights::empty(config.num_experts),
            config.num_experts,
            None,
            &MockGpuBackend::new(),
            &config,
        )
        .unwrap();
        for k in [
            layer.dense_gemm_router_m5,
            layer.dense_gemm_router_rows,
            layer.w4a16_batch5_dual_k,
        ] {
            assert_eq!(k.0 != 0, model == "glm5_next", "{model}");
        }
    }
}
