// SPDX-License-Identifier: AGPL-3.0-only

//! Inspect an official GLM-5.3-Flash config without loading weights or CUDA.

use std::path::PathBuf;

use anyhow::{Context, Result};
use atlas_core::config::parse_config;
use atlas_core::glm5::Glm53FlashPlan;

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: glm53_contract /path/to/config.json [context_tokens]")?;
    let context_tokens = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("context_tokens must be an unsigned integer")?
        .unwrap_or(32_768);
    let json = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let config = parse_config(&json)?;
    let plan = Glm53FlashPlan::from_config(&config)?;
    let recurrent = plan.kda_recurrent_bytes_per_sequence()?;
    let convolution = plan.kda_conv_bytes_per_sequence()?;
    let dsa = plan.dsa_cache_bytes_for_context(context_tokens, 2)?;
    let total = plan.persistent_bytes_per_sequence(context_tokens, 2)?;

    println!("model_type={}", config.model_type);
    println!(
        "layers={} kda={} dsa={} dense={} moe={}",
        config.num_hidden_layers,
        plan.kda_layers.len(),
        plan.dsa_layers.len(),
        plan.dense_mlp_layers.len(),
        plan.moe_layers.len()
    );
    println!("eos={:?}", config.eos_token_ids);
    println!("kda_recurrent_mib={:.2}", mib(recurrent));
    println!("kda_conv_mib={:.2}", mib(convolution));
    println!("dsa_cache_mib@{context_tokens}={:.2}", mib(dsa));
    println!("persistent_mib_per_sequence={:.2}", mib(total));
    println!(
        "prefix_cache_kv_only_safe={}",
        config.kv_only_prefix_cache_is_safe()
    );
    println!("native_execution_implemented=true");
    println!("end_to_end_validated=false");
    Ok(())
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
