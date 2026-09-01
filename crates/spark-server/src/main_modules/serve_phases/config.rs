// SPDX-License-Identifier: AGPL-3.0-only

//! Config / model-dir / vocab-cap helpers.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn merge_sidecar_quant_config(model_dir: &Path, config: &mut ModelConfig) {
    if config.quantization_config.is_some() {
        return;
    }
    let hf_quant_path = model_dir.join("hf_quant_config.json");
    if !hf_quant_path.exists() {
        return;
    }
    match std::fs::read_to_string(&hf_quant_path) {
        Ok(raw_hq) => {
            let wrapped = format!(r#"{{"quantization_config":{raw_hq}}}"#);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&wrapped) {
                config.quantization_config = atlas_core::config::parse_quantization_config(&v);
            }
        }
        Err(e) => tracing::warn!("Failed to read sibling hf_quant_config.json: {e}"),
    }
}

pub(crate) fn load_model_config(model_dir: &Path) -> Result<(ModelConfig, String)> {
    let config_path = model_dir.join("config.json");
    let params_path = model_dir.join("params.json");

    // Bare-GGUF directory (no config.json/params.json): synthesize the config
    // from the GGUF metadata block. Weight loading already routes to GgufLoader.
    if !config_path.exists()
        && !params_path.exists()
        && spark_runtime::weights::find_gguf(model_dir).is_some()
    {
        let config = spark_runtime::weights::config_from_gguf_dir(model_dir)
            .context("Failed to build ModelConfig from GGUF metadata")?;
        tracing::info!(
            "Built ModelConfig from GGUF metadata (model_type={}, layers={}, hidden={})",
            config.model_type,
            config.num_hidden_layers,
            config.hidden_size,
        );
        // No config.json string exists; the only downstream consumer
        // (resolve_model_name) falls back to the directory name.
        return Ok((config, String::new()));
    }

    let config_json = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else if params_path.exists() {
        std::fs::read_to_string(&params_path)
            .with_context(|| format!("Failed to read {}", params_path.display()))?
    } else {
        anyhow::bail!(
            "No config.json, params.json, or .gguf found in {}",
            model_dir.display()
        );
    };
    let config = if params_path.exists() && !config_path.exists() {
        atlas_core::config::parse_mistral_params(&config_json)
            .context("Failed to parse params.json (Mistral format)")?
    } else {
        atlas_core::config::parse_config(&config_json).context("Failed to parse config.json")?
    };
    Ok((config, config_json))
}

/// Fail-closed bring-up contract for the initial GLM-5.3 execution path.
/// Every rejected feature needs architecture state that the first native path
/// deliberately does not snapshot or migrate yet.
pub(crate) fn validate_glm53_runtime(args: &cli::ServeArgs, config: &ModelConfig) -> Result<()> {
    let ep_protocol_v2 = matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2"));
    validate_glm53_runtime_contract(args, config, ep_protocol_v2)
}

fn validate_glm53_runtime_contract(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    ep_protocol_v2: bool,
) -> Result<()> {
    if config.model_type != "glm5_next" {
        return Ok(());
    }
    let plan = atlas_core::glm5::Glm53FlashPlan::from_config(config)?;
    plan.validate_bootstrap_topology(args.ep_size, args.tp_size)?;
    if args.world_size != 2 {
        anyhow::bail!("GLM-5.3-Flash bootstrap requires --world-size 2");
    }
    if !ep_protocol_v2 {
        anyhow::bail!(
            "GLM-5.3-Flash concurrency requires ATLAS_EP_PROTOCOL=v2 on both ranks; \
             EP v1 silently forces max_batch_size=1"
        );
    }
    if args.scheduling_policy != "phase-interleave" {
        anyhow::bail!(
            "GLM-5.3-Flash requires --scheduling-policy phase-interleave until \
             same-forward KDA/DSA prefill+decode has passed the two-Spark latency gate"
        );
    }
    if args.max_prefill_tokens == 0 {
        anyhow::bail!(
            "GLM-5.3-Flash requires chunked prefill; set --max-prefill-tokens explicitly"
        );
    }
    if args.enable_prefix_caching {
        anyhow::bail!(
            "GLM-5.3-Flash prefix caching is disabled until KV, KDA, and semantic-index state restore atomically"
        );
    }
    if args.speculative || args.self_speculative || args.ngram_speculative {
        anyhow::bail!(
            "GLM-5.3-Flash supports only its checkpoint-matched DFlash2 proposer; use --dflash"
        );
    }
    if args.swap_space_gb > 0 || args.high_speed_swap {
        anyhow::bail!(
            "GLM-5.3-Flash state spilling is disabled until KV, KDA, and semantic-index state migrate atomically"
        );
    }
    Ok(())
}

pub(crate) fn resolve_model_dir(args: &cli::ServeArgs) -> Result<std::path::PathBuf> {
    use crate::model_resolver;
    if let Some(ref path) = args.model_from_path {
        model_resolver::resolve_model_dir(
            path.to_str().context("Invalid model path")?,
            args.cache_dir.as_deref(),
        )
    } else {
        let model_spec = args
            .model
            .as_deref()
            .context("Either MODEL or --model-from-path is required")?;
        model_resolver::resolve_model_dir(model_spec, args.cache_dir.as_deref())
    }
}

pub(crate) fn cap_vocab_size_to_tokenizer(model_dir: &Path, config: &mut ModelConfig) {
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists()
        && let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path)
    {
        let tok_vocab = tok.get_vocab_size(true);
        if tok_vocab > 0 && tok_vocab < config.vocab_size {
            tracing::info!(
                "Capping vocab_size from {} to {} (tokenizer)",
                config.vocab_size,
                tok_vocab,
            );
            config.vocab_size = tok_vocab;
        }
    }
}

/// Where the effective `num_drafts` came from. The CLI → MODEL.toml → engine
/// precedence is explicit (PCND); `apply_model_default_num_drafts` logs per
/// source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumDraftsSource {
    Cli,
    ModelDefault,
    EngineDefault,
}

/// Resolve the effective draft count. An explicitly passed `--num-drafts`
/// ALWAYS wins — including `--num-drafts 1`: the previous sentinel test
/// (`args.num_drafts == 1` against the clap default) could not tell an
/// explicit 1 from an omitted flag and silently served the MODEL.toml
/// default instead. An omitted flag falls back to MODEL.toml
/// `[behavior].default_num_drafts` when set (> 0), else the engine default.
pub(crate) fn resolve_num_drafts(
    cli_num_drafts: Option<usize>,
    model_default_num_drafts: u32,
) -> (usize, NumDraftsSource) {
    let model_default = (model_default_num_drafts > 0).then_some(model_default_num_drafts as usize);
    match (cli_num_drafts, model_default) {
        (Some(v), _) => (v, NumDraftsSource::Cli),
        (None, Some(md)) => (md, NumDraftsSource::ModelDefault),
        (None, None) => (cli::DEFAULT_NUM_DRAFTS, NumDraftsSource::EngineDefault),
    }
}

pub(crate) fn apply_model_default_num_drafts(
    args: &mut cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) {
    let (effective, source) =
        resolve_num_drafts(args.num_drafts, ptx_set.behavior.default_num_drafts);
    match source {
        NumDraftsSource::Cli => {
            let model_default = ptx_set.behavior.default_num_drafts as usize;
            if ptx_set.behavior.default_num_drafts > 0 && model_default != effective {
                tracing::info!(
                    "num_drafts: {} (K={}) from --num-drafts, overriding MODEL.toml default_num_drafts={}",
                    effective,
                    effective + 1,
                    model_default,
                );
            }
        }
        NumDraftsSource::ModelDefault => {
            tracing::info!(
                "num_drafts: using MODEL.toml default_num_drafts={} (K={}) — pass --num-drafts to override",
                effective,
                effective + 1,
            );
        }
        NumDraftsSource::EngineDefault => {}
    }
    args.num_drafts = Some(effective);
}

#[cfg(test)]
mod tests {
    use super::{NumDraftsSource, resolve_num_drafts, validate_glm53_runtime_contract};
    use atlas_core::config::{LayerType, ModelConfig};
    use clap::Parser;

    fn glm_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".to_string();
        config.num_hidden_layers = 45;
        config.layer_types = (0..45)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        config.linear_num_key_heads = 64;
        config.linear_num_value_heads = 64;
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_conv_kernel_dim = 4;
        config.kv_lora_rank = 512;
        config.index_head_dim = 128;
        config.index_topk = 2048;
        config.index_kpool = 4;
        config.mlp_only_layers = vec![0, 1, 2];
        config
    }

    fn glm_args(extra: &[&str]) -> crate::cli::ServeArgs {
        let mut argv = vec![
            "spark",
            "serve",
            "zai-org/GLM-5.3-Flash",
            "--world-size",
            "2",
            "--ep-size",
            "2",
            "--tp-size",
            "1",
            "--scheduling-policy",
            "phase-interleave",
            "--phase-decode-steps",
            "4",
            "--phase-prefill-steps",
            "1",
            "--swap-space-gb",
            "0",
        ];
        argv.extend_from_slice(extra);
        let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
        let crate::cli::Command::Serve(args) = cli.command else {
            unreachable!("test parses a serve command")
        };
        args
    }

    #[test]
    fn glm_bootstrap_contract_accepts_only_the_state_safe_lane() {
        let result = validate_glm53_runtime_contract(&glm_args(&[]), &glm_config(), true);
        assert!(result.is_ok(), "{result:?}");

        let prefix = validate_glm53_runtime_contract(
            &glm_args(&["--enable-prefix-caching"]),
            &glm_config(),
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(prefix.contains("prefix caching"));

        let protocol = validate_glm53_runtime_contract(&glm_args(&[]), &glm_config(), false)
            .unwrap_err()
            .to_string();
        assert!(protocol.contains("ATLAS_EP_PROTOCOL=v2"));
    }

    #[test]
    fn glm_bootstrap_contract_refuses_wrong_topology_and_scheduler() {
        let mut wrong_topology = glm_args(&[]);
        wrong_topology.ep_size = 1;
        wrong_topology.tp_size = 2;
        let topology = validate_glm53_runtime_contract(&wrong_topology, &glm_config(), true)
            .unwrap_err()
            .to_string();
        assert!(topology.contains("EP=2 and TP=1"));

        let mut fifo = glm_args(&[]);
        fifo.scheduling_policy = "fifo".to_string();
        let scheduler = validate_glm53_runtime_contract(&fifo, &glm_config(), true)
            .unwrap_err()
            .to_string();
        assert!(scheduler.contains("phase-interleave"));
    }

    /// The observed dgx2 bug: `--num-drafts 1` on a model with
    /// `default_num_drafts = 3` must serve 1 (K=2), not 3 (K=4).
    #[test]
    fn explicit_cli_value_equal_to_engine_default_beats_model_default() {
        assert_eq!(resolve_num_drafts(Some(1), 3), (1, NumDraftsSource::Cli));
    }

    #[test]
    fn explicit_cli_value_beats_model_default() {
        assert_eq!(resolve_num_drafts(Some(2), 3), (2, NumDraftsSource::Cli));
        assert_eq!(resolve_num_drafts(Some(3), 1), (3, NumDraftsSource::Cli));
    }

    #[test]
    fn omitted_flag_falls_back_to_model_default() {
        assert_eq!(
            resolve_num_drafts(None, 3),
            (3, NumDraftsSource::ModelDefault)
        );
    }

    #[test]
    fn omitted_flag_without_model_default_uses_engine_default() {
        assert_eq!(
            resolve_num_drafts(None, 0),
            (
                crate::cli::DEFAULT_NUM_DRAFTS,
                NumDraftsSource::EngineDefault
            )
        );
    }
}
