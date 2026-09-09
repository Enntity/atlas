// SPDX-License-Identifier: AGPL-3.0-only

//! Selected single-threaded ingress and resolved launch checks, before GPU work.
#[cfg(target_os = "linux")]
use crate::cli::ServeArgs;
use crate::cli::{Cli, Command};
use anyhow::{Result, ensure};

pub(crate) struct SelectedStartup {
    #[cfg(target_os = "linux")]
    pub(crate) received: super::inherited_startup::ReceivedStartup,
}

impl SelectedStartup {
    /// Called from synchronous main before building the Tokio runtime.
    pub(crate) fn receive(cli: &Cli) -> Result<Option<Self>> {
        let Command::Serve(args) = &cli.command else {
            ensure!(
                std::env::var_os("ATLAS_GLM_PAIR_FD").is_none(),
                "paired inherited authority is only valid for selected serving"
            );
            return Ok(None);
        };
        if !args.glm_paired_mtp {
            ensure!(
                std::env::var_os("ATLAS_GLM_PAIR_FD").is_none(),
                "inherited paired launch requires --glm-paired-mtp"
            );
            return Ok(None);
        }
        #[cfg(not(target_os = "linux"))]
        anyhow::bail!("supervised paired serving requires Linux");
        #[cfg(target_os = "linux")]
        {
            let received = unsafe { super::inherited_startup::receive(20000, 512 * 1024 * 1024) }?;
            validate_args(args, &received.recipe)?;
            Ok(Some(Self { received }))
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn validate_args(args: &ServeArgs, recipe: &atlas_glm_pair_wire::Recipe) -> Result<()> {
    recipe.profile.validate()?;
    let p = recipe.profile;
    ensure!(
        args.glm_paired_mtp
            && args.rank == usize::from(recipe.rank)
            && args.world_size == usize::from(recipe.world)
            && args.tp_size == usize::from(p.tp)
            && args.ep_size == usize::from(p.ep)
            && args.max_batch_size == usize::from(p.max_sequences)
            && args.max_num_seqs == usize::from(p.max_sequences)
            && args.max_seq_len == p.context as usize
            && args.max_prefill_tokens == p.prefill as usize
            && args.num_drafts == Some(usize::from(p.drafts)),
        "resolved CLI differs from fixed paired recipe"
    );
    ensure!(
        args.speculative
            && !args.dflash
            && !args.self_speculative
            && !args.ngram_speculative
            && args.mtp_quantization == "bf16"
            && args.mtp_vocab == 0
            && args.kv_cache_dtype.as_deref() == Some("bf16")
            && args.ssm_h_dtype.as_deref() == Some("f32")
            && args.ssm_rollback_mode == "snapshot"
            && args.block_size == 16
            && args.gpu_ordinal == 0,
        "paired serving requires full-vocabulary BF16 MTP4/BF16 KV/FP32 state"
    );
    ensure!(
        args.no_tui
            && args.no_auto_swap
            && !args.auto_swap
            && !args.enable_prefix_caching
            && args.warmup_prompt.is_none()
            && !args.high_speed_swap
            && args.swap_space_gb == 0
            && !args.adaptive_sampling
            && args.lora_adapter.is_empty()
            && args.lora_stageable.is_empty()
            && args.lora_stageable_disk.is_empty()
            && args.draft_model.is_none()
            && !args.check_kernels
            && !args.dangerously_allow_unresolved_kernel_lookups
            && !args.profile
            && args.disable_tool_grammar == Some(true),
        "paired launch excludes hot-swap, TUI, prefix, alternate owners, grammar and profiling"
    );
    ensure!(
        args.model.is_some() || args.model_from_path.is_some(),
        "paired launch requires a model"
    );
    ensure!(
        args.oom_guard_mb >= 4096
            && args.gpu_memory_utilization.is_finite()
            && args.gpu_memory_utilization > 0.0
            && args.gpu_memory_utilization <= 0.90,
        "paired launch requires explicit conservative memory headroom"
    );
    let value = |key: &str| {
        recipe
            .environment
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    ensure!(
        value("ATLAS_MAX_BATCH_TOKENS").is_none(),
        "paired launch uses the resolved bounded row budget, not an override"
    );
    ensure!(
        value("ATLAS_NO_MTP_EAGER_DRAFTER").is_none(),
        "paired launch requires eager drafter prefill; even =0 disables it"
    );
    for (key, expected) in [
        ("ATLAS_GLM_PAIR_FD", "3"),
        ("ATLAS_EP_PROTOCOL", "v2"),
        ("ATLAS_GLM_MTP_DISTRIBUTED", "1"),
        ("ATLAS_GLM_INDEPENDENT_DECODE", "0"),
        ("ATLAS_DFLASH_DEBUG_NO_GRAPH", "1"),
        ("ATLAS_GLM_MTP_REPAIR", "0"),
        ("ATLAS_DFLASH_ADAPTIVE", "0"),
        ("ATLAS_KV_OVERCOMMIT", "0"),
        ("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1"),
    ] {
        ensure!(
            value(key) == Some(expected),
            "paired launch requires {key}={expected}"
        );
    }
    for key in [
        "ATLAS_EP_GRAPHS",
        "ATLAS_GLM_TP_VERIFY_GRAPH",
        "ATLAS_GLM_MTP_HIDDEN_TRACE",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_SINGLE_DEPTH_ADAPT",
        "ATLAS_DFLASH_RESUME_GUARD",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_MTP_REFEED_ACCEPTED",
        "ATLAS_DRAFT_CONF_TAU",
    ] {
        ensure!(
            matches!(value(key), None | Some("0")),
            "paired launch forbids {key}"
        );
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
#[path = "startup_tests.rs"]
mod tests;
