// SPDX-License-Identifier: AGPL-3.0-only

//! Model factory: builds the right model from config + weights.
//!
//! Weight loader selection is registry-driven — add new models by implementing
//! [`ModelWeightLoader`] and registering in [`loader_for_config`]. No other
//! code changes needed.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;
use std::ffi::OsStr;

use crate::mistral_loader::MistralWeightLoader;
use crate::weight_loader::LongcatWeightLoader;
use crate::weight_loader::{
    DeepSeekV4WeightLoader, DflashConfig, Gemma4WeightLoader, Glm5WeightLoader, LagunaWeightLoader,
    MinimaxM2WeightLoader, ModelWeightLoader, NemotronHWeightLoader, NllbWeightLoader,
    Qwen3VLWeightLoader, Qwen3WeightLoader, Qwen4ExpWeightLoader, Qwen35DenseWeightLoader,
    Qwen35WeightLoader, Step3p7WeightLoader,
};

mod dspark_admission;
mod exl3_preflight;
#[allow(unused_imports)]
pub(crate) use dspark_admission::{
    admit_lightning_dspark, admit_lightning_dspark_build, admit_lightning_dspark_product_build,
};
pub use exl3_preflight::preflight_glm53_tp2_packed;
#[cfg(test)]
mod dspark_admission_tests;
#[cfg(test)]
mod dspark_product_admission_tests;

/// DFlash speculative-decoding build arguments. `None` for non-DFlash runs;
/// `Some(...)` carries the drafter's separate [`WeightStore`], parsed
/// `config.json`, and CLI overrides for γ and the sliding-window size.
///
/// Construction order: the caller (`spark-server::main`) loads the drafter
/// checkpoint into a fresh [`WeightStore`] via the same `WeightStore::load`
/// path used for the target, then parses `config.json` via
/// [`crate::weight_loader::dflash_loader::parse_dflash_config`]. Both inputs
/// flow through to [`build_model`] which validates dimensions against the
/// target before constructing [`crate::layers::BlockDiffusionDraftHead`].
pub struct DflashBuildArgs<'a> {
    pub drafter_store: &'a WeightStore,
    pub drafter_config: DflashConfig,
    pub gamma: Option<usize>,
    pub window_size: Option<usize>,
}

/// LoRA adapter build arguments (`--lora-adapter NAME=PATH`). `None` for
/// base-only runs; `Some(...)` carries the adapter's separate on-device
/// [`WeightStore`] (loaded via
/// `spark_runtime::weights::adapter::load_adapter_safetensors`), the parsed
/// `adapter_config.json`, and the pool-shape CLI knobs.
///
/// Unlike DFlash (loaded post-construction), the LoRA pool is allocated at
/// the TOP of `build_model` — before the buffer arena and the free-memory
/// snapshot — so its bytes are automatically debited from the KV budget.
pub struct LoraBuildArgs<'a> {
    /// One or more adapters to pack (repeated `--lora-adapter NAME=PATH`),
    /// each carrying its NAME, its on-device `WeightStore`, and its parsed
    /// `adapter_config.json`. Slot k = `adapters[k]`. A single element is
    /// byte-identical to the pre-multi-adapter path.
    pub adapters: Vec<crate::lora::LoraAdapterInput<'a>>,
    pub max_lora_rank: usize,
    pub max_loras: usize,
}

// ── Loader registry ─────────────────────────────────────────────────────────
// Adding a new model: implement ModelWeightLoader and add a match arm below.
// Everything else (KV cache, buffers, TransformerModel) is model-agnostic.

/// Select the weight loader for a given model config.
///
/// This is the ONLY place model_type strings are matched. All downstream
/// code is model-agnostic.
pub fn loader_for_config(config: &ModelConfig) -> Result<Box<dyn ModelWeightLoader>> {
    let normalized = config.model_type.to_lowercase().replace(['-', '.'], "_");
    match normalized.as_str() {
        // Qwen3 family: sub-dispatch by config predicates
        "qwen3_next" => Ok(Box::new(Qwen3WeightLoader)),
        "qwen3_vl_moe" => Ok(Box::new(Qwen3VLWeightLoader)),
        "qwen3_5_moe" | "qwen3_5" | "qwen35_moe" | "qwen35" => {
            // Dense check has to come first. Qwen3.6-27B-FP8 is the dense text
            // sibling of the Qwen3.6 VL family — its config declares the same
            // `vision_config` block as the MoE-VL siblings (so `is_qwen3_vl()`
            // returns true), but the checkpoint ships no vision tower and no
            // MoE router, so the VL loader panics on a missing `mlp.gate`.
            // `is_qwen35_dense()` requires `num_experts == 0`, which only the
            // dense text models satisfy — VL-MoE always has experts.
            if config.is_qwen35_dense() {
                Ok(Box::new(Qwen35DenseWeightLoader))
            } else if config.is_qwen3_vl() {
                Ok(Box::new(Qwen3VLWeightLoader))
            } else {
                Ok(Box::new(Qwen35WeightLoader))
            }
        }
        // Qwen3.6: identical architecture to Qwen3.5 MoE at the weight level
        // (GDN + full-attention + MoE hybrid, same expert layout, same MTP).
        // Only difference is MRoPE-interleaved layout + attn_output_gate on
        // full-attention layers — both handled at forward-pass layer time,
        // not during weight loading.
        "qwen3_6_moe" | "holo3_1_moe" => Ok(Box::new(Qwen35WeightLoader)),
        // Nemotron-H family (Mamba-2 + MoE + Attention), including Puzzle
        // (heterogeneous per-block MoE intermediate / top-k).
        "nemotron_h" | "nemotron_h_puzzle" => Ok(Box::new(NemotronHWeightLoader)),
        // NLLB / M2M-100 encoder-decoder translation family.
        "m2m_100" | "nllb" => Ok(Box::new(NllbWeightLoader)),
        // Gemma-4 family (pure attention, GeGLU, sliding + full attention)
        "gemma4" | "gemma_4" => Ok(Box::new(Gemma4WeightLoader)),
        // Mistral family (MLA + MoE, GQA fallback for initial bring-up)
        "mistral" => Ok(Box::new(MistralWeightLoader)),
        // LongCat-Flash(-Lite): MLA dual-sublayer blocks + shortcut MoE with
        // zero-computation experts (+ n-gram input embeddings).
        "longcat_flash_ngram" | "longcat_flash" => Ok(Box::new(LongcatWeightLoader)),
        // Qwen3.8-Flash-Next. `dispatch.rs` normalizes the older
        // `qwen3_8_flash_next` naming onto `qwen4_exp`, so one arm covers both
        // published quantizations.
        "qwen4_exp" => Ok(Box::new(Qwen4ExpWeightLoader)),
        // MiniMax M2 family (M2.1 / M2.7) — full attention + 256-expert
        // sigmoid-routed MoE + 3-module MTP.
        "minimax_m2" => Ok(Box::new(MinimaxM2WeightLoader)),
        // Step 3.7 Flash — 288-expert sigmoid-routed MoE + shared expert +
        // mixed full/sliding attention + attention gate + 3 MTP modules.
        "step3p7" => Ok(Box::new(Step3p7WeightLoader)),
        "laguna" => Ok(Box::new(LagunaWeightLoader)),
        // DeepSeek-V4 family (Flash) — MLA + MoE + CSA/HCA hybrid attention + mHC.
        "deepseek_v4" => Ok(Box::new(DeepSeekV4WeightLoader)),
        "glm5_next" => Ok(Box::new(Glm5WeightLoader)),
        _ => bail!(
            "Unsupported model type: '{}' (normalized: '{}'). \
             Supported: qwen3_next, qwen3_5_moe, qwen3_5, qwen3_6_moe, holo3_1_moe, qwen3_vl_moe, qwen4_exp, nemotron_h, nemotron_h_puzzle, gemma4, mistral, minimax_m2, step3p7, laguna, deepseek_v4, m2m_100",
            config.model_type,
            normalized,
        ),
    }
}

/// Review finding m3 ("factory refusal now"): the C2 norm-substitution knob
/// (`ATLAS_EXL3_NORM_SUBSTITUTE_DIR`) engages only where
/// `NormSubstitution::from_env` runs — the GLM-5/KDA loaders
/// (`weight_loader/glm5/{exl3_layers,layers}.rs`). Every other model would
/// silently ignore a set knob, so abort the load instead (PCND), mirroring
/// the `Exl3GatePerturbation` refusal in `factory/build.rs`. Called at the
/// TOP of [`build_model`] so every model type pays it — including the NLLB
/// path, which returns before [`loader_for_config`] is ever consulted.
pub(crate) fn refuse_unengaged_norm_substitution(
    config: &ModelConfig,
    store: &WeightStore,
) -> Result<()> {
    // The native EXL3 branch of `build_model` calls `exl3_layers::load_layers`
    // (which engages the knob) whatever the loader selection says, so an EXL3
    // checkpoint engages too. An unsupported model type engages nothing and
    // is refused further below by `loader_for_config` itself — the clearer
    // first error here names the env var.
    let engages = store.exl3_checkpoint().is_some()
        || loader_for_config(config)
            .map(|loader| loader.engages_norm_substitution())
            .unwrap_or(false);
    ensure_norm_substitution_engaged(
        std::env::var_os(spark_runtime::weights::exl3::norm_substitution::ENV_SUBSTITUTE_DIR)
            .as_deref(),
        &config.model_type,
        engages,
    )
}

/// Env-free core (unit-testable without process-global `set_var` races — see
/// `model/ssm_tier/capability.rs`): `dir` is the raw
/// `ATLAS_EXL3_NORM_SUBSTITUTE_DIR` value; unset or empty means the knob is
/// off and no model is refused.
fn ensure_norm_substitution_engaged(
    dir: Option<&OsStr>,
    model_type: &str,
    engages: bool,
) -> Result<()> {
    if engages || !dir.is_some_and(|dir| !dir.is_empty()) {
        return Ok(());
    }
    bail!(
        "{} is only honoured by the GLM-5/KDA load path (weight_loader/glm5); \
         model type {:?} would silently ignore it",
        spark_runtime::weights::exl3::norm_substitution::ENV_SUBSTITUTE_DIR,
        model_type,
    )
}

mod build;
pub(crate) mod glm_dense_cache;
mod glm_hc_prewarm;
mod glm_shared_cache;
mod glm_sparse_decode;
mod lm_head_setup;
mod m2_setup;

pub use build::build_model;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::mtp_head::MtpQuantization;
    use spark_runtime::kv_cache::KvCacheDtype;
    use spark_runtime::prefix_cache::PrefixCache;
    use spark_runtime::weights::exl3::norm_substitution::ENV_SUBSTITUTE_DIR;

    // Process-wide env guard for the substitution-refusal tests (repo
    // ENV_LOCK pattern: `main_modules/refusal.rs`,
    // `tool_parser/tests/streaming_frag_env.rs`). Every test whose code path
    // reads ATLAS_EXL3_NORM_SUBSTITUTE_DIR holds it — including
    // `test_unsupported_model_type`, since `build_model` now pays the m3
    // refusal check before loader dispatch.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct SubstituteDirGuard {
        prev: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for SubstituteDirGuard {
        fn drop(&mut self) {
            // SAFETY: serialised by ENV_LOCK; no other thread touches this
            // var without holding the same lock.
            unsafe {
                match &self.prev {
                    Some(prev) => std::env::set_var(ENV_SUBSTITUTE_DIR, prev),
                    None => std::env::remove_var(ENV_SUBSTITUTE_DIR),
                }
            }
        }
    }

    fn set_substitute_dir(value: Option<&str>) -> SubstituteDirGuard {
        let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var_os(ENV_SUBSTITUTE_DIR);
        // SAFETY: see Drop.
        unsafe {
            match value {
                Some(value) => std::env::set_var(ENV_SUBSTITUTE_DIR, value),
                None => std::env::remove_var(ENV_SUBSTITUTE_DIR),
            }
        }
        SubstituteDirGuard { prev, _lock: lock }
    }

    /// `build_model` with the mock GPU and an empty store — safe only for
    /// callers whose expected error fires at the very top of the build.
    fn build_for_test(config: ModelConfig) -> Result<()> {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let prefix_cache: Box<dyn PrefixCache> =
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching);
        build_model(
            config,
            WeightStore::empty(),
            Box::new(gpu),
            1,
            16,
            4096,
            8,
            MtpQuantization::Nvfp4,
            false,
            prefix_cache,
            0,
            None,
            false,
            1,
            KvCacheDtype::Fp8,
            1024 * 1024 * 1024,
            0.90,
            0,
            vec![],
            0,
            None,
            None, // dflash_args
            None, // lora_args
            None, // nllb_lang
            None, // nllb_lora_dir
        )
        .map(|_| ())
    }

    // ── C2 norm-substitution factory refusal (review finding m3) ──────────

    /// Every non-KDA model load — including NLLB, which returns before
    /// `loader_for_config` — must abort on a set knob, naming the env var and
    /// the model type instead of silently ignoring it.
    #[test]
    fn substitute_dir_set_refuses_non_kda_model_load_naming_var_and_type() {
        let _env_guard = set_substitute_dir(Some("/nonexistent-fixtures"));
        for model_type in ["qwen3_next", "nllb", "m2m_100"] {
            let mut config = ModelConfig::qwen3_next_80b_nvfp4();
            config.model_type = model_type.to_string();
            let err = build_for_test(config)
                .expect_err("a non-KDA model must not load with the knob set");
            let msg = format!("{err:#}");
            assert!(msg.contains(ENV_SUBSTITUTE_DIR), "{model_type}: {msg}");
            assert!(msg.contains(model_type), "{model_type}: {msg}");
        }
    }

    /// The GLM-5/KDA path is not refused, and its load proceeds into
    /// `NormSubstitution::from_env` exactly as today (the D1 graph interlock
    /// rejects the bare test env before any manifest work — proof the parse
    /// site ran).
    #[test]
    fn substitute_dir_set_allows_glm5_kda_and_proceeds_to_from_env() {
        let _env_guard = set_substitute_dir(Some("/nonexistent-fixtures"));
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".to_string();
        config.max_position_embeddings = 4096;
        config.index_topk = 2048; // GLM-5 Atlas bring-up gate in `load_all`
        let store = WeightStore::empty();
        refuse_unengaged_norm_substitution(&config, &store)
            .expect("GLM-5/KDA engages the substitution knob");
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let err = Glm5WeightLoader
            .load_layers(&store, &config, &gpu, &[])
            .map(|_| ())
            .expect_err("the from_env graph interlock must reject this test env");
        assert!(
            err.to_string().contains("ATLAS_NO_DECODE_GRAPHS"),
            "{err:#}"
        );
    }

    /// Unset or empty is off for every model — no load is ever refused.
    #[test]
    fn substitute_dir_unset_or_empty_never_refuses_any_model() {
        for value in [None, Some("")] {
            let _env_guard = set_substitute_dir(value);
            for model_type in ["qwen3_next", "nllb", "glm5_next"] {
                let mut config = ModelConfig::qwen3_next_80b_nvfp4();
                config.model_type = model_type.to_string();
                refuse_unengaged_norm_substitution(&config, &WeightStore::empty())
                    .unwrap_or_else(|err| panic!("{model_type}: {err:#}"));
            }
        }
    }

    #[test]
    fn test_unsupported_model_type() {
        // `build_model` now reads ATLAS_EXL3_NORM_SUBSTITUTE_DIR (review
        // finding m3); pin it off under the env lock so this test cannot
        // observe a transient engaged state from the refusal tests below.
        let _env_guard = set_substitute_dir(None);
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "llama".to_string();

        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let store = WeightStore::empty();

        let prefix_cache: Box<dyn PrefixCache> =
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching);
        let result = build_model(
            config,
            store,
            Box::new(gpu),
            1,
            16,
            4096,
            8,
            MtpQuantization::Nvfp4,
            false,
            prefix_cache,
            0,
            None,
            false,
            1,
            KvCacheDtype::Fp8,
            1024 * 1024 * 1024,
            0.90,
            0,
            vec![],
            0,
            None,
            None, // dflash_args
            None, // lora_args
            None, // nllb_lang
            None, // nllb_lora_dir
        );
        match result {
            Err(e) => assert!(e.to_string().contains("Unsupported model type: 'llama'")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn test_loader_selection() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen3_next".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "nemotron_h".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "holo3_1_moe".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "m2m_100".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "unsupported_model".to_string();
        assert!(loader_for_config(&config).is_err());
    }

    // The generic `ModelWeightLoader` for NLLB must fail fast: real NLLB serving
    // goes through the dedicated `NllbGpuModel` encoder-decoder runtime, which
    // `build_model` selects before this loader is ever consulted. Reaching this
    // loader is a routing bug, so every generic entry point bails.
    #[test]
    fn test_nllb_generic_loader_fails_fast_dedicated_runtime_serves() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "nllb".to_string();
        let loader = loader_for_config(&config).unwrap();
        let store = WeightStore::empty();
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();

        let err = loader.load_embedding(&store, &config, &gpu).unwrap_err();
        assert!(
            err.to_string()
                .contains("dedicated GPU encoder-decoder runtime"),
            "{err}"
        );
    }
}
