// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.num_attention_heads = 32;
    c
}

fn policy(c: &ModelConfig) -> BuildPolicy<'_> {
    BuildPolicy {
        config: c,
        mode: GlmMtpBuildMode::Legacy,
        speculative: true,
        self_speculative: false,
        drafts: 2,
        owners: 4,
        context: 32768,
        block_size: 16,
        kv_dtype: KvCacheDtype::Bf16,
        layer_dtypes: &[],
        alternate_owner: false,
        dflash: false,
    }
}

#[test]
fn glm_sparse_decode_build_accepts_only_repaired_long_mtp2() {
    let c = config();
    assert!(policy(&c).validate(true, true).is_ok());
    for variant in 0..13 {
        let mut p = policy(&c);
        let fp8 = [KvCacheDtype::Fp8];
        match variant {
            0 => p.mode = GlmMtpBuildMode::Paired,
            1 => p.speculative = false,
            2 => p.self_speculative = true,
            3 => p.drafts = 1,
            4 => p.owners = 1,
            5 => p.context = 32769,
            6 => p.context = 2048,
            7 => p.block_size = 32,
            8 => p.kv_dtype = KvCacheDtype::Fp8,
            9 => p.layer_dtypes = &fp8,
            10 => p.alternate_owner = true,
            11 => {
                assert!(p.validate(false, true).is_err());
                continue;
            }
            _ => {
                assert!(p.validate(true, false).is_err());
                continue;
            }
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
fn larger_served_context_is_capped_only_for_repair_startup() {
    assert_eq!(repair_context(2048), 2048);
    assert_eq!(repair_context(32768), 32768);
    assert_eq!(repair_context(36864), 32768);

    let c = config();
    let mut capped = policy(&c);
    capped.context = repair_context(36864);
    assert!(capped.validate(true, true).is_ok());

    // Keep the verifier's upper bound strict. The cap belongs at the factory
    // boundary; it must not turn an out-of-domain repair policy into a valid
    // policy when callers construct one directly.
    let mut uncapped = policy(&c);
    uncapped.context = 36864;
    assert!(uncapped.validate(true, true).is_err());
}
