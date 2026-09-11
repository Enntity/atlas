// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}
fn args(config: &ModelConfig) -> GlmSparsePrefillTc<'_> {
    GlmSparsePrefillTc {
        config,
        dtype: KvCacheDtype::Bf16,
        identical_kv_latent: true,
        query: DevicePtr(16),
        k_cache: DevicePtr(32),
        v_cache: DevicePtr(48),
        indices: DevicePtr(64),
        output: DevicePtr(80),
        block_table: DevicePtr(96),
        rows: 1024,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    }
}

#[test]
fn actual_tc_dispatch_uses_one_query_token_per_cta_and_is_opt_in() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut a = args(&c);
    assert!(!dispatch(&gpu, &a, 7, false, false).unwrap());
    assert_eq!(gpu.launches_snapshot().len(), 0);
    a.rows = 1;
    assert!(!dispatch(&gpu, &a, 7, true, false).unwrap());
    assert_eq!(gpu.launches_snapshot().len(), 0);
    a.rows = 1024;
    assert!(dispatch(&gpu, &a, 7, true, false).unwrap());
    let launch = &gpu.launches_snapshot()[0];
    assert_eq!(launch.grid, [1, 1024, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
}

#[test]
fn enabled_unsupported_shapes_fail_before_launch() {
    let gpu = MockGpuBackend::new();
    let c = config();
    for variant in 0..7 {
        let mut a = args(&c);
        match variant {
            0 => a.heads = 64,
            1 => a.index_width = 2048,
            2 => a.rows = 0,
            3 => a.rows = 65536,
            4 => a.head_dim = 256,
            5 => a.scale = 0.125,
            _ => a.query = DevicePtr::NULL,
        }
        assert!(dispatch(&gpu, &a, 0, true, false).is_err());
        assert!(dispatch(&gpu, &a, 0, true, true).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
    for name in [
        "ATLAS_GLM_SPARSE_PREFILL_TC",
        "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE",
    ] {
        assert!(parse("glm5_next", name, Some("1")).unwrap());
        assert!(!parse("qwen3_next", name, Some("1")).unwrap());
        for value in ["", "true", "2"] {
            assert!(parse("glm5_next", name, Some(value)).is_err());
        }
    }
}

#[test]
fn kv_reuse_dispatch_requires_tc_and_identical_latent_writer() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut a = args(&c);
    assert!(dispatch(&gpu, &a, 7, false, true).is_err());
    a.identical_kv_latent = false;
    assert!(dispatch(&gpu, &a, 7, true, true).is_err());
    assert!(gpu.launches_snapshot().is_empty());
    a.identical_kv_latent = true;
    a.rows = 1;
    assert!(!dispatch(&gpu, &a, 7, true, true).unwrap());
    a.rows = 2048;
    assert!(dispatch(&gpu, &a, 7, true, true).unwrap());
    let launch = &gpu.launches_snapshot()[0];
    assert_eq!(launch.grid, [1, 2048, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
    assert_eq!(
        kernel_spec(true),
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
            69376,
        )
    );
    assert_eq!(
        kernel_spec(false),
        (
            "glm_sparse_prefill_tc",
            "glm_sparse_mla_prefill_bf16_head32_tc",
            101120,
        )
    );
}
