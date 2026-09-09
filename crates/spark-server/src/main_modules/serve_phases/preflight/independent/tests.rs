// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use clap::Parser;

// Supply only the I/O boundary; all topology/admission/accounting is production.
fn prepare_reserve(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    free_mem: usize,
) -> Result<(Option<Topology>, ReservePreflight)> {
    let (_, _, topology, reserve) = super::prepare_reserve(args, config, || Ok(((), free_mem)))?;
    Ok((topology, reserve))
}

fn args(cap: usize, rank: usize) -> cli::ServeArgs {
    let cli::Command::Serve(mut args) = cli::Cli::try_parse_from([
        "spark",
        "serve",
        "unused",
        "--world-size",
        "2",
        "--tp-size",
        "2",
        "--ep-size",
        "2",
        "--kv-cache-dtype",
        "bf16",
        "--max-seq-len",
        "2048",
        "--max-prefill-tokens",
        "1024",
        "--swap-space-gb",
        "0",
    ])
    .unwrap()
    .command
    else {
        panic!("serve args")
    };
    args.max_batch_size = cap;
    args.max_num_seqs = cap;
    args.rank = rank;
    args.num_drafts = Some(4);
    args
}

fn config() -> ModelConfig {
    atlas_core::config::parse_config(&serde_json::json!({
        "model_type":"glm5_next", "text_config": {
            "model_type":"glm5_next_text", "hidden_size":4096,
            "num_hidden_layers":45, "num_nextn_predict_layers":1,
            "intermediate_size":12288, "vocab_size":154880,
            "max_position_embeddings":1048576, "rms_norm_eps":1e-5,
            "num_attention_heads":64, "num_key_value_heads":64,
            "kv_lora_rank":512, "q_lora_rank":1536,
            "qk_nope_head_dim":256, "qk_rope_head_dim":0, "v_head_dim":256,
            "linear_attn_config":{"num_heads":64,"head_dim":128,
                "short_conv_kernel_size":4,"gate_lower_bound":-5.0},
            "n_routed_experts":288, "num_experts_per_tok":8,
            "moe_intermediate_size":2048,"n_shared_experts":1,
            "norm_topk_prob":true,"scoring_func":"sigmoid","topk_method":"noaux_tc",
            "routed_scaling_factor":2.5,"first_k_dense_replace":3,
            "layer_types":(0..45).map(|i|if (i+1)%4==0 {"deepseek_sparse_attention"} else {"linear_attention"}).collect::<Vec<_>>(),
            "hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6,
            "index_n_heads":32,"index_head_dim":128,"index_topk":2048,
            "index_kpool":4,"index_kpool_compress":true,
            "index_kpool_always_select_tail":true,"indexer_types":vec!["full";45]
        }, "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}
    }).to_string()).unwrap()
}

fn isolated(name: &str, overrides: &[(&str, &str)]) -> bool {
    const CHILD: &str = "ATLAS_SERVER_INDEPENDENT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("ATLAS_") {
                child.env_remove(key);
            }
        }
        let result = child
            .arg("--exact")
            .arg(format!(
                "main_modules::serve_phases::preflight::independent::tests::{name}"
            ))
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .envs(overrides.iter().copied())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("1 passed"),
            "child did not run requested test"
        );
        return true;
    }
    false
}

fn assert_allocation(
    a: &cli::ServeArgs,
    cfg: &ModelConfig,
    reserve: &ReservePreflight,
    rows: usize,
) {
    assert_eq!(cfg.linear_num_key_heads, 32);
    assert_eq!(cfg.linear_num_value_heads, 32);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_ssm_layers(), 34);
    assert_eq!(reserve.max_batch_tokens_pre, rows);
    assert_eq!(
        reserve.resolved_prefill.as_ref().unwrap().max_batch_tokens,
        rows
    );
    assert_eq!(
        reserve.resolved_prefill.as_ref().unwrap().prefill_budget,
        1024
    );
    assert_eq!(
        reserve.buffer_arena_bytes,
        spark_runtime::buffers::BufferSizes::from_config(
            cfg,
            rows,
            a.max_seq_len,
            a.block_size,
            a.max_batch_size
        )
        .total_bytes()
    );
    let blob = cfg.num_ssm_layers() * (cfg.ssm_h_state_bytes() + cfg.ssm_conv_state_bytes());
    assert_eq!(blob, 77987840); // Actual TP-local FP32 H + conv across34 KDA layers.
    let ring =
        spark_model::ssm_reserve::decode_rollback_ring_slots(cfg.num_ssm_layers(), false).slots;
    assert_eq!(ring, 8);
    let gdn = rows.min(a.max_seq_len) * (12288 * 2 + 32 * 2 * 4 + 4096 * 2 * 2);
    assert_eq!(reserve.gdn_two_phase_bytes, gdn);
    assert_eq!(
        reserve.inference_reserve,
        (a.max_batch_size + 1 + a.ssm_cache_slots + ring * a.max_batch_size) * blob
            + gdn
            + (512 << 20)
    );
}

#[test]
fn selected_preparation_admits_c8_with_local_allocation_shape() {
    if isolated(
        "selected_preparation_admits_c8_with_local_allocation_shape",
        &[],
    ) {
        return;
    }
    for cap in 2..=8 {
        for rank in 0..2 {
            let a = args(cap, rank);
            let mut cfg = config();
            let (topology, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30)
                .unwrap_or_else(|e| panic!("actual C8 preparation refused: {e:#}"));
            assert_eq!(topology.unwrap().tp_rank, rank);
            assert_allocation(&a, &cfg, &reserve, 1024 + cap);
            let total = reserve.inference_reserve + reserve.buffer_arena_bytes;
            assert!(prepare_reserve(&a, &mut config(), total).is_ok());
            let error = prepare_reserve(&a, &mut config(), total - 1).err().unwrap();
            assert!(error.to_string().contains("Preflight failed"));
        }
    }
    // Auto-derived world is validated from the resolved topology, not raw argv.
    let mut a = args(8, 1);
    a.world_size = 1;
    let (topology, reserve) = prepare_reserve(&a, &mut config(), 128usize << 30).unwrap();
    assert_eq!(topology.unwrap().world_size, 2);
    assert_eq!(reserve.max_batch_tokens_pre, 1032);
}

#[test]
fn selected_override_is_reserved_and_carried_to_build() {
    if isolated(
        "selected_override_is_reserved_and_carried_to_build",
        &[("ATLAS_MAX_BATCH_TOKENS", "4096")],
    ) {
        return;
    }
    for cap in [4, 8] {
        for rank in 0..2 {
            let a = args(cap, rank);
            let mut cfg = config();
            let (_, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30).unwrap();
            assert_allocation(&a, &cfg, &reserve, 4096);
        }
    }
}

#[test]
fn selected_invalid_profiles_refuse_before_weights() {
    if isolated("selected_invalid_profiles_refuse_before_weights", &[]) {
        return;
    }
    for case in 0..18 {
        let mut a = args(8, 0);
        match case {
            0 => a.max_batch_size = 9,
            1 => a.max_num_seqs = 7,
            2 => a.speculative = true,
            3 => a.self_speculative = true,
            4 => a.ngram_speculative = true,
            5 => a.dflash = true,
            6 => a.kv_cache_dtype = Some("fp8".into()),
            7 => a.max_seq_len = 2049,
            8 => a.high_speed_swap = true,
            9 => a.swap_space_gb = 1,
            10 => a.lora_adapter.push(("x".into(), "unused".into())),
            11 => a
                .lora_stageable
                .push(("x".into(), "p".into(), "unused".into())),
            12 => a.lora_stageable_disk.push(("x".into(), "unused".into())),
            13 => a.tp_size = 1,
            14 => a.rank = 2,
            15 => a.block_size = 0,
            16 => a.max_prefill_tokens = usize::MAX,
            17 => a.ssm_cache_slots = usize::MAX,
            _ => unreachable!(),
        }
        assert!(
            prepare_reserve(&a, &mut config(), 128usize << 30).is_err(),
            "case {case}"
        );
    }
    let mut cfg = config();
    cfg.linear_num_key_heads = 63;
    assert!(prepare_reserve(&args(8, 0), &mut cfg, 128usize << 30).is_err());
    let mut invalid = args(9, 0);
    let mut initialized = false;
    assert!(
        super::prepare_reserve(&invalid, &mut config(), || {
            initialized = true;
            Ok(((), 128usize << 30))
        })
        .is_err()
    );
    assert!(
        !initialized,
        "invalid admission initialized the GPU backend"
    );
    invalid.max_batch_size = 8;
    invalid.max_num_seqs = 8;
    assert!(
        super::prepare_reserve(&invalid, &mut config(), || {
            initialized = true;
            Ok(((), 128usize << 30))
        })
        .is_ok()
    );
    assert!(initialized);
}

#[test]
fn selected_huge_arena_override_refuses_without_overflow() {
    if isolated(
        "selected_huge_arena_override_refuses_without_overflow",
        &[("ATLAS_MAX_BATCH_TOKENS", "18446744073709551615")],
    ) {
        return;
    }
    let error = prepare_reserve(&args(8, 0), &mut config(), 128usize << 30)
        .err()
        .unwrap();
    assert!(error.to_string().contains("arena rows exceed"));
}

#[test]
fn off_keeps_legacy_global_reserve_and_late_topology() {
    if isolated(
        "off_keeps_legacy_global_reserve_and_late_topology",
        &[
            ("ATLAS_GLM_INDEPENDENT_DECODE", "0"),
            ("ATLAS_GLM_C4_DECODE", "1"),
            ("ATLAS_GLM_KDA_MULTI_SEQ", "1"),
            ("ATLAS_GLM_MLA_MULTI_SEQ", "1"),
        ],
    ) {
        return;
    }
    for rank in 0..2 {
        let a = args(4, rank);
        let mut cfg = config();
        let (topology, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30).unwrap();
        assert!(topology.is_none());
        assert!(reserve.resolved_prefill.is_none());
        assert_eq!(cfg.linear_num_key_heads, 64);
        assert_eq!(reserve.max_batch_tokens_pre, 1024);
        assert_eq!(reserve.inference_reserve, 8732016640); // Historical8327.5MiB.
        assert_eq!(
            reserve.buffer_arena_bytes,
            spark_runtime::buffers::BufferSizes::from_config(&cfg, 1024, 2048, 16, 4).total_bytes()
        );
        let topology = resolve_topology(&a, &mut cfg).unwrap();
        assert_eq!(topology.tp_rank, rank);
        assert_eq!(cfg.linear_num_key_heads, 32);
    }
    assert!(prepare_reserve(&args(8, 0), &mut config(), 128usize << 30).is_err());
}
