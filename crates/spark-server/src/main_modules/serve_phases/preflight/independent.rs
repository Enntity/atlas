// SPDX-License-Identifier: AGPL-3.0-only
//! Preparation seam: selected topology must precede allocation accounting.
use super::super::{Topology, resolve_topology};
use super::*;

pub(crate) fn prepare_reserve<B>(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    init_backend: impl FnOnce() -> Result<(B, usize)>,
) -> Result<(B, usize, Option<Topology>, ReservePreflight)> {
    // Selected profile errors must be reported before CUDA initialization.
    // OFF retains the original backend -> reserve -> late topology order.
    let topology = prepare_topology(args, config)?;
    let (backend, free_mem) = init_backend()?;
    let reserve = preflight_reserve(args, config, free_mem)?;
    Ok((backend, free_mem, topology, reserve))
}

fn prepare_topology(args: &cli::ServeArgs, config: &mut ModelConfig) -> Result<Option<Topology>> {
    if !spark_model::model::glm_independent::enabled(&config.model_type)? {
        return Ok(None);
    }
    // This is the actual resolver and the only division of global heads. The
    // returned topology is retained through the later weight/build handoff.
    let topology = resolve_topology(args, config)?;
    let independent =
        !(args.speculative || args.self_speculative || args.ngram_speculative || args.dflash);
    let ep_v2 = std::env::var("ATLAS_EP_PROTOCOL").as_deref() == Ok("v2");
    spark_model::model::glm_independent::Launch {
        model_type: &config.model_type,
        world: topology.world_size,
        tp: topology.tp_size,
        ep: topology.ep_size,
        ep_v2,
        active: args.max_batch_size,
        admitted: args.max_num_seqs,
        context: args.max_seq_len,
        bf16: args.kv_cache_dtype.as_deref() == Some("bf16"),
        independent,
        lora: !args.lora_adapter.is_empty()
            || !args.lora_stageable.is_empty()
            || !args.lora_stageable_disk.is_empty(),
        hss: args.high_speed_swap,
        swap: args.swap_space_gb != 0,
    }
    .validate()?;
    anyhow::ensure!(
        args.rank < topology.world_size,
        "independent rank exceeds world"
    );
    anyhow::ensure!(
        (1..=2048).contains(&args.max_prefill_tokens) && args.block_size > 0,
        "independent decode requires bounded prefill1..2048 and a nonzero block size"
    );
    spark_model::model::glm_independent::validate_runtime(
        config,
        topology.world_size,
        ep_v2,
        independent,
    )?;
    Ok(Some(topology))
}

#[cfg(test)]
mod tests;
