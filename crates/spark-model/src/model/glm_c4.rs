// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded, opt-in independent C4 policy shared by server and model dispatch.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferSizes;

pub struct C4Policy<'a> {
    pub model_type: &'a str,
    pub enabled: bool,
    pub world: usize,
    pub tp: usize,
    pub ep: usize,
    pub ep_v2: bool,
    pub independent: bool,
    pub kda_multi: bool,
    pub mla_multi: bool,
    pub sparse: bool,
    pub sparse_graphs: bool,
    pub c4_sparse: bool,
    pub no_multiseq_graphs: bool,
    pub kv_overcommit_disabled: bool,
    pub fp32_state: bool,
}

impl C4Policy<'_> {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.enabled && self.model_type == "glm5_next",
            "independent GLM C4 requires ATLAS_GLM_C4_DECODE=1 and exact glm5_next"
        );
        ensure!(
            self.world == 2 && self.tp == 2 && self.ep == 2 && self.ep_v2,
            "GLM C4 requires world2/TP2/EP2 and ATLAS_EP_PROTOCOL=v2"
        );
        ensure!(
            self.independent
                && self.kda_multi
                && self.mla_multi
                && self.sparse == self.c4_sparse
                && !self.sparse_graphs
                && self.fp32_state,
            "GLM C4 requires non-speculative decode, KDA/MLA_MULTI_SEQ=1, FP32 KDA state, sparse graphs off and base sparse matching C4_SPARSE"
        );
        ensure!(
            !self.c4_sparse || (self.no_multiseq_graphs && self.kv_overcommit_disabled),
            "GLM C4_SPARSE requires ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1 and ATLAS_KV_OVERCOMMIT=0"
        );
        Ok(())
    }
}

pub fn enabled(model_type: &str) -> bool {
    model_type == "glm5_next" && std::env::var("ATLAS_GLM_C4_DECODE").as_deref() == Ok("1")
}

pub const SPARSE_CONTEXT_LIMIT: usize = 16384;

pub fn validate_prefill_budget(tokens: usize, c4_sparse: bool) -> Result<()> {
    ensure!(
        !c4_sparse || (1..=1024).contains(&tokens),
        "GLM C4_SPARSE requires a configured prefill budget in 1..1024"
    );
    Ok(())
}

fn context_limit(c4_sparse: bool) -> usize {
    if c4_sparse {
        SPARSE_CONTEXT_LIMIT
    } else {
        2048
    }
}

pub fn sparse_enabled(model_type: &str) -> bool {
    model_type == "glm5_next" && std::env::var("ATLAS_GLM_C4_SPARSE").as_deref() == Ok("1")
}

/// Direct-server boundary rejects misspelled flags and other architectures.
pub fn validate_sparse_flag(model_type: &str, value: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(value, None | Some("0" | "1")),
        "ATLAS_GLM_C4_SPARSE must be 0 or 1"
    );
    ensure!(
        value != Some("1") || model_type == "glm5_next",
        "ATLAS_GLM_C4_SPARSE requires glm5_next"
    );
    Ok(value == Some("1"))
}

pub fn policy_from_env(
    model_type: &str,
    world: usize,
    tp: usize,
    ep: usize,
    ep_v2: bool,
    independent: bool,
) -> C4Policy<'_> {
    C4Policy {
        model_type,
        enabled: enabled(model_type),
        world,
        tp,
        ep,
        ep_v2,
        independent,
        kda_multi: std::env::var("ATLAS_GLM_KDA_MULTI_SEQ").as_deref() == Ok("1"),
        mla_multi: std::env::var("ATLAS_GLM_MLA_MULTI_SEQ").as_deref() == Ok("1"),
        sparse: std::env::var("ATLAS_GLM_MULTI_SEQ_SPARSE").as_deref() == Ok("1"),
        sparse_graphs: std::env::var("ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS").as_deref() == Ok("1"),
        c4_sparse: sparse_enabled(model_type),
        no_multiseq_graphs: std::env::var("ATLAS_NO_DECODE_GRAPHS_MULTISEQ").as_deref() == Ok("1"),
        kv_overcommit_disabled: matches!(
            std::env::var("ATLAS_KV_OVERCOMMIT").as_deref(),
            Ok("0" | "false")
        ),
        fp32_state: !crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
    }
}

pub fn validate_launch_limits(
    active: usize,
    admitted: usize,
    context: usize,
    bf16: bool,
) -> Result<()> {
    validate_launch_limits_for_mode(active, admitted, context, bf16, sparse_enabled("glm5_next"))
}

pub fn validate_launch_limits_for_mode(
    active: usize,
    admitted: usize,
    context: usize,
    bf16: bool,
    c4_sparse: bool,
) -> Result<()> {
    let limit = context_limit(c4_sparse);
    ensure!(
        active == 4 && admitted == 4 && (1..=limit).contains(&context) && bf16,
        "GLM C4 requires active4/admitted4, context1..{limit} and BF16 KV"
    );
    Ok(())
}

pub fn validate_runtime(
    config: &ModelConfig,
    world: usize,
    ep_v2: bool,
    independent: bool,
) -> Result<()> {
    policy_from_env(
        &config.model_type,
        world,
        config.tp_world_size,
        config.ep_world_size,
        ep_v2,
        independent,
    )
    .validate()?;
    ensure!(
        config.hidden_size == 4096
            && config.num_attention_heads == 32
            && config.q_lora_rank == 1536
            && config.kv_lora_rank == 512
            && config.qk_nope_head_dim == 256
            && config.qk_rope_head_dim == 0
            && config.v_head_dim == 256
            && config.index_topk == 2048
            && config.hc_mult > 0,
        "GLM C4 requires the validated local TP2 MLA/mHC geometry"
    );
    Ok(())
}

pub fn batched_kda_rows(rows: usize, kda: bool, c4: bool) -> Result<bool> {
    if rows == 4 {
        ensure!(
            kda && c4,
            "GLM C4 must use opted-in true batched KDA, never scalar fallback"
        );
        return Ok(true);
    }
    Ok(kda && matches!(rows, 2 | 3))
}

pub fn validate_positions(positions: impl IntoIterator<Item = usize>, rows: usize) -> Result<()> {
    validate_positions_for_mode(positions, rows, sparse_enabled("glm5_next"))
}

pub fn validate_positions_for_mode(
    positions: impl IntoIterator<Item = usize>,
    rows: usize,
    c4_sparse: bool,
) -> Result<()> {
    ensure!(
        matches!(rows, 2..=4),
        "GLM C4 lane accepts only independent C2/C3/C4 batch widths"
    );
    let mut count = 0;
    let limit = context_limit(c4_sparse);
    for position in positions {
        ensure!(position < limit, "GLM C4 position must be below {limit}");
        count += 1;
    }
    ensure!(count == rows, "GLM C4 host position count mismatch");
    Ok(())
}

pub fn validate_projection_handles(nvfp4_m4: u64, bf16_batchm: u64) -> Result<()> {
    ensure!(
        nvfp4_m4 != 0 && bf16_batchm != 0,
        "GLM C4 requires nonzero exact-M4 NVFP4 and BF16 batch-M projections"
    );
    Ok(())
}

fn hc_bytes(mult: usize) -> Result<(usize, usize, usize)> {
    ensure!(mult > 0, "GLM C4 requires mHC streams");
    let post = mult
        .checked_mul(4 * 4)
        .ok_or_else(|| anyhow::anyhow!("C4 mHC size overflow"))?;
    let highway = post
        .checked_mul(4096)
        .ok_or_else(|| anyhow::anyhow!("C4 mHC size overflow"))?;
    let comb = post
        .checked_mul(mult)
        .ok_or_else(|| anyhow::anyhow!("C4 mHC size overflow"))?;
    Ok((highway, post, comb))
}

/// Fixed C4 lower bounds, checked before lookup and again before layer work.
pub fn validate_scratch(s: &BufferSizes, hc_mult: usize) -> Result<()> {
    let (highway, post, comb) = hc_bytes(hc_mult)?;
    for (name, available, required) in [
        ("hidden", s.hidden_states, 4 * 4096 * 2),
        ("norm", s.norm_output, 4 * 4096 * 2),
        ("QKV projections", s.qkv_output, 100608),
        ("packed QKV", s.ssm_qkvz, 98304),
        ("convolved QKV", s.ssm_conv_out_f32, 98304),
        ("gates/expanded Q", s.ssm_deinterleaved, 65536),
        ("Q latent", s.ssm_ba, 12288),
        ("absorbed attention", s.attn_output, 131072),
        ("expert gate", s.expert_gate_out, 131072),
        ("expert up/absorbed Q", s.expert_up_out, 131072),
        ("expert down", s.expert_down_out, 262144),
        ("sort metadata", s.gate_logits, 1540),
        ("MoE output", s.moe_output, 32768),
        ("mHC highway", s.hc_streams, highway),
        ("mHC post", s.hc_post, post),
        ("mHC comb", s.hc_comb, comb),
    ] {
        ensure!(
            available >= required,
            "GLM C4 {name} scratch needs {required} bytes, has {available}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> C4Policy<'static> {
        C4Policy {
            model_type: "glm5_next",
            enabled: true,
            world: 2,
            tp: 2,
            ep: 2,
            ep_v2: true,
            independent: true,
            kda_multi: true,
            mla_multi: true,
            sparse: false,
            sparse_graphs: false,
            c4_sparse: false,
            no_multiseq_graphs: false,
            kv_overcommit_disabled: false,
            fp32_state: true,
        }
    }

    #[test]
    fn c4_sparse_requires_explicit_eager_bounded_policy() {
        let long = || C4Policy {
            c4_sparse: true,
            sparse: true,
            no_multiseq_graphs: true,
            kv_overcommit_disabled: true,
            ..policy()
        };
        assert!(long().validate().is_ok());
        for bad in [
            C4Policy {
                enabled: false,
                ..long()
            },
            C4Policy {
                sparse: false,
                ..long()
            },
            C4Policy {
                sparse_graphs: true,
                ..long()
            },
            C4Policy {
                no_multiseq_graphs: false,
                ..long()
            },
            C4Policy {
                kv_overcommit_disabled: false,
                ..long()
            },
            C4Policy {
                independent: false,
                ..long()
            },
            C4Policy {
                fp32_state: false,
                ..long()
            },
            C4Policy {
                mla_multi: false,
                ..long()
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn c4_sparse_flag_and_context_boundaries_fail_closed() {
        assert!(validate_prefill_budget(1, true).is_ok());
        assert!(validate_prefill_budget(1024, true).is_ok());
        for invalid in [0, 1025, 6144, usize::MAX] {
            assert!(validate_prefill_budget(invalid, true).is_err());
            assert!(validate_prefill_budget(invalid, false).is_ok());
        }
        assert!(!validate_sparse_flag("glm5_next", None).unwrap());
        assert!(!validate_sparse_flag("other", Some("0")).unwrap());
        assert!(validate_sparse_flag("glm5_next", Some("1")).unwrap());
        assert!(validate_sparse_flag("other", Some("1")).is_err());
        assert!(validate_sparse_flag("glm5_next", Some("true")).is_err());
        for limit in [2048, SPARSE_CONTEXT_LIMIT] {
            let sparse = limit > 2048;
            assert!(validate_launch_limits_for_mode(4, 4, limit, true, sparse).is_ok());
            for (active, admitted, context, bf16) in [
                (4, 4, 0, true),
                (4, 4, limit + 1, true),
                (3, 4, limit, true),
                (4, 5, limit, true),
                (4, 4, limit, false),
            ] {
                assert!(
                    validate_launch_limits_for_mode(active, admitted, context, bf16, sparse)
                        .is_err()
                );
            }
            for rows in 2..=4 {
                assert!(validate_positions_for_mode(vec![limit - 1; rows], rows, sparse).is_ok());
                assert!(validate_positions_for_mode(vec![limit; rows], rows, sparse).is_err());
                assert!(validate_positions_for_mode(vec![usize::MAX; rows], rows, sparse).is_err());
                assert!(validate_positions_for_mode(vec![0; rows - 1], rows, sparse).is_err());
            }
        }
        assert!(validate_positions_for_mode([0], 1, true).is_err());
        assert!(validate_positions_for_mode([0; 5], 5, true).is_err());
    }

    #[test]
    fn exact_opt_in_topology_and_modes_are_mandatory() {
        assert!(policy().validate().is_ok());
        for invalid in [
            C4Policy {
                enabled: false,
                ..policy()
            },
            C4Policy {
                model_type: "qwen3_next",
                ..policy()
            },
            C4Policy {
                world: 1,
                ..policy()
            },
            C4Policy { tp: 1, ..policy() },
            C4Policy { ep: 1, ..policy() },
            C4Policy {
                ep_v2: false,
                ..policy()
            },
            C4Policy {
                independent: false,
                ..policy()
            },
            C4Policy {
                kda_multi: false,
                ..policy()
            },
            C4Policy {
                mla_multi: false,
                ..policy()
            },
            C4Policy {
                sparse: true,
                ..policy()
            },
            C4Policy {
                sparse_graphs: true,
                ..policy()
            },
            C4Policy {
                fp32_state: false,
                ..policy()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn initial_launch_caps_are_four_short_independent_sessions() {
        assert!(validate_launch_limits(4, 4, 1024, true).is_ok());
        assert!(validate_launch_limits(4, 4, 2048, true).is_ok());
        for (active, admitted, context, bf16) in [
            (3, 4, 1024, true),
            (4, 5, 1024, true),
            (5, 5, 1024, true),
            (4, 4, 2049, true),
            (4, 4, 0, true),
            (4, 4, 1024, false),
        ] {
            assert!(validate_launch_limits(active, admitted, context, bf16).is_err());
        }
    }

    #[test]
    fn c4_never_falls_back_to_scalar_kda_dispatch() {
        assert!(batched_kda_rows(4, true, true).unwrap());
        assert!(batched_kda_rows(4, true, false).is_err());
        assert!(batched_kda_rows(4, false, true).is_err());
        for n in [2, 3] {
            assert!(batched_kda_rows(n, true, false).unwrap());
            assert!(!batched_kda_rows(n, false, false).unwrap());
        }
        assert!(!batched_kda_rows(1, true, true).unwrap());
    }

    #[test]
    fn replay_positions_and_projection_handles_fail_closed() {
        assert!(validate_positions([0, 15, 1023, 2047], 4).is_ok());
        assert!(validate_positions([2048, 0, 1, 2], 4).is_err());
        assert!(validate_positions([usize::MAX, 0, 1, 2], 4).is_err());
        assert!(validate_positions([0, 1, 2], 4).is_err());
        assert!(validate_projection_handles(1, 2).is_ok());
        assert!(validate_projection_handles(0, 2).is_err());
        assert!(validate_projection_handles(1, 0).is_err());
        assert_eq!(hc_bytes(4).unwrap(), (262144, 64, 256));
        assert!(hc_bytes(usize::MAX).is_err());
        assert!(hc_bytes(0).is_err());
    }

    #[test]
    fn every_c4_arena_is_checked_against_its_live_dtype_and_rows() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = 4096;
        config.hc_mult = 4;
        let mut sizes = BufferSizes::from_config(&config, 1024, 2048, 16, 4);
        assert!(validate_scratch(&sizes, 4).is_ok());
        sizes.qkv_output = 4 * (3 * 4096 + 32 + 128 + 128) * 2;
        assert_eq!(sizes.qkv_output, 100608);
        assert!(validate_scratch(&sizes, 4).is_ok());
        sizes.qkv_output -= 1;
        assert!(validate_scratch(&sizes, 4).is_err());
        sizes.qkv_output += 1;
        sizes.hc_streams = 4 * 4 * 4096 * 2; // BF16-sized is insufficient for FP32 highway.
        assert!(validate_scratch(&sizes, 4).is_err());
        sizes.hc_streams *= 2;
        sizes.expert_up_out = 131071;
        assert!(validate_scratch(&sizes, 4).is_err());
    }
}
