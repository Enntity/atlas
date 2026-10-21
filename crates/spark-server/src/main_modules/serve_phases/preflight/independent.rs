// SPDX-License-Identifier: AGPL-3.0-only
//! Preparation seam: the long-context GLM lane resolves its topology before
//! allocation accounting, so every reserve uses the local shapes it allocates.
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
    if spark_model::speculative::glm_repair_policy::long_context_enabled() {
        anyhow::ensure!(
            ((args.dflash && spark_model::speculative::glm_repair_policy::dflash_enabled())
                || (args.speculative
                    && !args.dflash
                    && spark_model::layers::glm_native::mode::strict_native_mtp()))
                && config.model_type == "glm5_next",
            "long-context GLM preparation requires the GLM DFlash lane or explicit strict native MTP"
        );
        let topology = resolve_topology(args, config)?;
        anyhow::ensure!(
            matches!(topology.world_size, 2 | 4)
                && topology.tp_size == topology.world_size
                && topology.ep_size == topology.world_size
                && config.tp_rank == config.ep_rank
                && config.ep_rank == args.rank
                && args.rank < topology.world_size,
            "resolved long-context GLM topology mismatch before GPU initialization \
             (the long-context lane runs on the TP2/EP2 pair or the TP4/EP4 chain)"
        );
        // The serve handoff consumes Some(topology), so global heads are divided
        // exactly once and every reserve uses the same local shapes as allocation.
        return Ok(Some(topology));
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
