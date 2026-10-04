// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_LAYER_FORK` (default off): run an independent sub-chain of a GLM
//! verify layer on a side stream instead of in line on the compute stream.
//!
//! A verify step runs the target model on one stream, so chains that share
//! no data still wait for each other (2026-10-03 e30 nsys, C1 prose, marginal
//! time per step: shared expert 0.96 ms beside a router -> top-k -> sort ->
//! worklist -> quantize chain of 1.47 ms; semantic-index maintenance 0.57 ms
//! beside the latent dequant and q_b, 0.72 ms). Values:
//!
//! * `moe`: the TP-split shared expert (`ATLAS_GLM_SHARED_TP_SPLIT=1`) of a
//!   grouped routed FFN forks after the router GEMV and joins at the
//!   unpermute that blends it, so it overlaps top-k, sort, worklist,
//!   quantization and the routed GEMMs. It runs on the layer's existing
//!   auxiliary stream and events (`MoeLayer::prefill_stream`, `event_a/_b`).
//! * `index`: a dense (`seq_len <= index_topk`) MLA owner's semantic-index
//!   maintenance (key / gate projections, layernorm, tail write, pool
//!   finalize) forks after the KV cache write and joins before the W_uk
//!   absorb, so it overlaps the latent dequant and q_b. Only where q_b runs a
//!   custom kernel: the index projections use cuBLASLt, whose workspace is
//!   process-global, so no other cuBLASLt call may run beside them.
//! * `1`: both.
//!
//! Exactness: the same kernels run with the same arguments in the same
//! order on each stream; only the interleaving of the two independent chains
//! changes, and no buffer one chain writes is read or written by the other
//! inside the fork (`moe/shared_fork.rs`, `qwen3_attention/prefill/
//! paged_glm_fork.rs` list them), so every output byte is unchanged. Both
//! ranks fork at the same points; the collectives stay on the compute stream
//! after the joins. Eager forwards only (verify, short prefill chunks): graph
//! capture and profiling keep the in-line order.
//! `scripts/dev/glm_layer_fork_bench.cu` times the fork against the in-line
//! order on emulated layers and checks the output bits.
//!
//! Prior art (docs/glm-prior-art.md): SGLang's dual-stream DeepSeek MoE
//! (`alt_stream`: shared experts on one stream, router and routed experts on
//! the other) and vLLM's shared-experts stream, both Apache-2.0. No code
//! copied; the index fork is the same technique applied here.

use anyhow::{Result, anyhow, bail};
use spark_runtime::gpu::GpuBackend;
use std::sync::OnceLock;

use crate::layer::ForwardContext;

/// Which chains fork.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mode {
    pub moe: bool,
    pub index: bool,
}

fn parse(value: Option<&str>) -> Result<Mode> {
    let (moe, index) = match value {
        None | Some("0") => (false, false),
        Some("1") => (true, true),
        Some("moe") => (true, false),
        Some("index") => (false, true),
        Some(other) => bail!("ATLAS_GLM_LAYER_FORK must be 0, 1, moe or index, got {other:?}"),
    };
    Ok(Mode { moe, index })
}

/// The switch, read once (an invalid value is an error at every call; the
/// attention layer build surfaces it at boot).
pub(crate) fn mode() -> Result<Mode> {
    static MODE: OnceLock<std::result::Result<Mode, String>> = OnceLock::new();
    let mode = MODE.get_or_init(|| {
        let mode = parse(std::env::var("ATLAS_GLM_LAYER_FORK").ok().as_deref());
        if let Ok(m) = &mode
            && (m.moe || m.index)
        {
            tracing::info!(
                "ATLAS_GLM_LAYER_FORK: side stream for the shared expert {} / semantic index {}",
                m.moe,
                m.index
            );
        }
        mode.map_err(|error| error.to_string())
    });
    mode.clone().map_err(|error| anyhow!(error))
}

/// The index lane of an attention layer with a GLM semantic indexer
/// (`indexed`): a new stream and two events under `index`, else nothing. Its
/// build surfaces an invalid switch at boot.
pub(crate) fn index_lane(gpu: &dyn GpuBackend, indexed: bool) -> Result<Option<ForkLane>> {
    if !indexed {
        return Ok(None);
    }
    mode()?.index.then(|| ForkLane::new(gpu)).transpose()
}

/// Whether `stream` runs eagerly and unprofiled: a fork keeps no meaning in a
/// captured graph built for one stream, and profiling times the chain in line.
pub(crate) fn eager(ctx: &ForwardContext, stream: u64) -> bool {
    !ctx.graph_capture && !ctx.profile && !ctx.gpu.stream_is_capturing(stream)
}

/// A side stream and the two events that fork it from, and join it back to,
/// the compute stream. `cuStreamWaitEvent` waits on the record made before
/// it, so one lane serves any number of fork / join pairs in sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ForkLane {
    pub side: u64,
    pub fork: u64,
    pub join: u64,
}

impl ForkLane {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            side: gpu.create_stream()?,
            fork: gpu.create_event()?,
            join: gpu.create_event()?,
        })
    }

    /// The side stream, ordered after everything enqueued on `main` so far.
    pub fn fork(&self, gpu: &dyn GpuBackend, main: u64) -> Result<u64> {
        gpu.record_event(self.fork, main)?;
        gpu.stream_wait_event(self.side, self.fork)?;
        Ok(self.side)
    }

    /// Order everything enqueued on `main` from here after the side stream.
    pub fn join(&self, gpu: &dyn GpuBackend, main: u64) -> Result<()> {
        gpu.record_event(self.join, self.side)?;
        gpu.stream_wait_event(main, self.join)
    }
}

#[cfg(test)]
#[path = "glm_layer_fork_tests.rs"]
mod tests;
