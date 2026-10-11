// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.num_attention_heads = 32;
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}

fn policy(c: &ModelConfig) -> BuildPolicy<'_> {
    BuildPolicy {
        config: c,
        speculative: false,
        self_speculative: false,
        context: 32768,
        block_size: 16,
        kv_dtype: KvCacheDtype::Bf16,
        layer_dtypes: &[],
        dflash: true,
    }
}

#[test]
fn glm_sparse_decode_build_accepts_only_the_long_context_dflash_lane() {
    let c = config();
    assert!(policy(&c).validate(true, true).is_ok());
    let mut g128 = policy(&c);
    g128.kv_dtype = KvCacheDtype::Fp8G128;
    assert!(g128.validate(true, true).is_ok());
    for variant in 0..9 {
        let mut p = policy(&c);
        let fp8 = [KvCacheDtype::Fp8];
        match variant {
            0 => p.dflash = false,
            1 => p.self_speculative = true,
            2 => p.context = 2048,
            3 => p.block_size = 32,
            4 => p.kv_dtype = KvCacheDtype::Fp8,
            5 => p.layer_dtypes = &fp8,
            6 => {
                assert!(p.validate(false, true).is_err(), "lane disabled");
                continue;
            }
            7 => {
                assert!(p.validate(true, false).is_err(), "long context disabled");
                continue;
            }
            _ => p.context = crate::speculative::glm_repair_policy::max_long_context() + 1,
        }
        assert!(p.validate(true, true).is_err(), "variant {variant}");
    }
}

#[test]
fn glm_sparse_decode_build_rejects_wrong_rank_geometry() {
    for variant in 0..4 {
        let mut c = config();
        match variant {
            0 => c.model_type = "other".into(),
            1 => c.tp_world_size = 1,
            2 => c.ep_world_size = 1,
            _ => c.num_attention_heads = 64,
        }
        assert!(policy(&c).validate(true, true).is_err());
    }
}

#[test]
fn larger_served_context_is_capped_to_the_verifier_domain() {
    let max = crate::speculative::glm_repair_policy::MAX_LONG_CONTEXT;
    assert_eq!(repair_context(2048), 2048);
    assert_eq!(repair_context(max), max);
    assert_eq!(repair_context(max + 4096), max);
}

#[test]
fn fp8_serial_decode_admits_only_serial_tp2_ep1_inference() {
    let mut c = config();
    c.ep_world_size = 1;
    let fp8 = [KvCacheDtype::Fp8G128];
    let mut p = policy(&c);
    p.dflash = false;
    p.context = 4096;
    p.kv_dtype = KvCacheDtype::Fp8G128;
    p.layer_dtypes = &[]; // Empty means `kv_dtype` applies uniformly.
    assert!(p.validate_fp8_serial_reader().is_ok());
    p.layer_dtypes = &fp8;
    assert!(p.validate_fp8_serial_reader().is_ok());
    let wrong_config = ModelConfig::qwen3_next_80b_nvfp4();

    for variant in 0..5 {
        let mut rejected = policy(&c);
        rejected.dflash = false;
        rejected.context = 4096;
        rejected.kv_dtype = KvCacheDtype::Fp8G128;
        rejected.layer_dtypes = &fp8;
        match variant {
            0 => rejected.speculative = true,
            1 => rejected.self_speculative = true,
            2 => rejected.dflash = true,
            3 => rejected.config = &wrong_config,
            _ => rejected.layer_dtypes = &[KvCacheDtype::Bf16, KvCacheDtype::Fp8G128],
        }
        assert!(
            rejected.validate_fp8_serial_reader().is_err(),
            "variant {variant}"
        );
    }
}

#[test]
fn fp8_serial_startup_initializes_reader() {
    let mut c = config();
    c.ep_world_size = 1;
    let fp8 = [KvCacheDtype::Fp8G128];
    let p = BuildPolicy {
        config: &c,
        speculative: false,
        self_speculative: false,
        context: 4096,
        block_size: 16,
        kv_dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: &fp8,
        dflash: false,
    };
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();

    initialize_with(&gpu, p, false, false).unwrap();

    assert_eq!(
        gpu.kernel_requests_snapshot(),
        [(
            "glm_sparse_prefill_kv_reuse".into(),
            "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad".into()
        )]
    );
    assert_eq!(gpu.sync_count(), 1);
}

#[test]
fn fp8_serial_startup_rejects_speculation_before_kernel_initialization() {
    let mut c = config();
    c.ep_world_size = 1;
    let fp8 = [KvCacheDtype::Fp8G128];
    let p = BuildPolicy {
        config: &c,
        speculative: true,
        self_speculative: false,
        context: 4096,
        block_size: 16,
        kv_dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: &fp8,
        dflash: false,
    };
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();

    assert!(initialize_with(&gpu, p, false, false).is_err());
    assert!(gpu.kernel_requests_snapshot().is_empty());
}
