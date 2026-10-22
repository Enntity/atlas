// SPDX-License-Identifier: AGPL-3.0-only

//! Model-directory construction for [`ChatTokenizer`].

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

use super::super::{ChatEncoding, ChatTokenizer};

impl ChatTokenizer {
    pub fn from_model_dir(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        // MODEL.toml `[behavior].jinja_template`; empty = derive from
        // `model_type`.
        template_override: &str,
        repo_root: Option<&Path>,
        disable_template_overrides: bool,
        // `tojson` spaced like Python's `json.dumps`, as transformers renders
        // it (MODEL.toml `[behavior].verbatim_prompt`); otherwise the
        // process-wide `ATLAS_USE_HF_REF_JSON_DUMPS` choice.
        hf_tool_json: bool,
    ) -> Result<Self> {
        Self::from_model_dir_with(
            model_dir,
            eos_token_id,
            supports_thinking,
            model_type,
            template_override,
            repo_root,
            disable_template_overrides,
            if hf_tool_json {
                super::super::jinja_helpers::ToolJsonStyle::HfSpaced
            } else {
                super::super::jinja_helpers::ToolJsonStyle::from_env()
            },
        )
    }

    /// [`from_model_dir`](Self::from_model_dir) with an explicit tool-JSON
    /// style instead of the process environment's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_model_dir_with(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        template_override: &str,
        repo_root: Option<&Path>,
        disable_template_overrides: bool,
        tool_json_style: super::super::jinja_helpers::ToolJsonStyle,
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // Template-source priority.
        //
        // Conceptually the default is now MODEL-FIRST: render off the
        // model's OWN `chat_template.jinja` / `tokenizer_config.json`.
        // Atlas's cross-cutting behaviors (autoclose-think,
        // think-control, F76 arg-parse) are applied in Rust
        // message-preprocessing (see `preprocess_for_render`), so a model
        // no longer needs a bespoke `jinja-templates/{model_type}.jinja`
        // override that is otherwise a byte-copy of its own template.
        // This is what makes `holo3_1_moe.jinja` REDUNDANT: Holo renders
        // correctly off its own template + Rust behaviors. (The override
        // file itself is still present for now only because
        // `tokenizer/tests.rs::render_holo_template_*` reads it directly;
        // it goes away together with those tests.)
        //
        // A `jinja-templates/{model_type}.jinja` override is OPT-IN by
        // FILE PRESENCE: dropping the file in is the explicit signal that
        // this model genuinely needs a template fix the Rust preprocessing
        // can't express (MiniMax's `_args.items()`, Gemma-4's
        // `strip_thinking`, etc.). We deliberately do NOT prefer the
        // model's own template when such a file exists — that would
        // silently undo those fixes. Instead, the operator opts OUT of all
        // overrides with `--disable-template-overrides`, which forces
        // every model onto its own template (relying purely on the Rust
        // behaviors).
        //
        // Priority (high → low):
        //   1. jinja-templates/<template> override, where <template> is
        //      MODEL.toml [behavior].jinja_template when set and
        //      {model_type}.jinja otherwise
        //      (opt-in: file present AND overrides not disabled)
        //   2. tokenizer_config.json / chat_template.jinja (the MODEL's own)
        //   3. Default ChatML fallback
        let override_tmpl = if disable_template_overrides {
            None
        } else {
            super::super::jinja_helpers::load_override_template(
                model_type,
                template_override,
                repo_root,
            )
        };
        let chat_template = if let Some(override_tmpl) = override_tmpl {
            override_tmpl
        } else if let Some(config_tmpl) =
            super::super::jinja_helpers::load_config_template(model_dir)?
        {
            config_tmpl
        } else {
            tracing::warn!("No chat template found — using default ChatML");
            super::super::jinja_helpers::default_chatml_template(supports_thinking)
        };

        let jinja_env =
            super::super::jinja_helpers::build_jinja_env_with(&chat_template, tool_json_style)?;

        // Load OpenAI-variant template if it exists (jinja-templates/openai/{model_type}.jinja).
        // This variant gates historical <think> wrappers on enable_thinking, preventing
        // spontaneous thinking during tool-use when thinking is disabled.
        let openai_jinja_env = super::super::jinja_helpers::load_openai_template(
            model_type, repo_root,
        )
        .and_then(|tmpl| {
            tracing::info!("Loaded OpenAI-variant Jinja template for {model_type}");
            super::super::jinja_helpers::build_jinja_env_with(&tmpl, tool_json_style).ok()
        });
        let chat_encoding = if model_type == "deepseek_v4" {
            tracing::info!("Using checkpoint-native DeepSeek-V4 message encoding");
            ChatEncoding::DeepseekV4
        } else {
            ChatEncoding::Jinja
        };

        tracing::info!("Loaded tokenizer from {}", tokenizer_path.display());
        Ok(Self {
            tokenizer,
            large_encode_gate: super::super::large_encode::configured(
                std::env::var("ATLAS_LARGE_ENCODE_CONCURRENCY")
                    .ok()
                    .as_deref(),
            )
            .map_err(anyhow::Error::msg)?,
            ids_only: super::super::ids_only::configured(
                std::env::var("ATLAS_TOKENIZER_IDS_ONLY").ok().as_deref(),
            )
            .map_err(anyhow::Error::msg)?,
            eos_token_id,
            supports_thinking,
            // GLM-5 and Qwen4_exp register their tool-envelope XML as added
            // tokens. Keep the scope explicit: other Qwen variants may use
            // different marker semantics and must retain their prompt bytes.
            glm_tool_markup_as_text: matches!(model_type, "glm5_next" | "qwen4_exp"),
            chat_encoding,
            chat_template,
            jinja_env,
            openai_jinja_env,
        })
    }
}
