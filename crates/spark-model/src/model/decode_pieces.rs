// SPDX-License-Identifier: AGPL-3.0-only

//! Piecewise CUDA-graph capture of the qwen4_exp (Qwen3.8-Flash-Next)
//! decode and MTP verify steps (`ATLAS_QWEN4EXP_DECODE_GRAPH=1`, default
//! off). The GLM verify mechanism (`verify_pieces.rs`,
//! `docs/glm-verify-graphs.md`) applied to this model's layer stack.
//!
//! A whole-step graph is vetoed for this model: the QSA indexer on every
//! full-attention layer keeps host state (its ingest counter) and launches
//! with position-dependent geometry (`select_geometry`'s `n_sel`), and the
//! PLE layer hashes n-grams and faults rows in on the host and gathers into
//! a per-SEQUENCE conv carry. The device radix top-k (now the default) took
//! the host sort out of QSA, but not the counter or the geometry. The other
//! 35 GDN layers, each an mHC-bracketed gated-delta mixer plus an mHC-
//! bracketed 512-expert MoE, launch nothing that depends on the step. So
//! each maximal run of GDN layers that does not veto graphs
//! (`TransformerLayer::decode_graph_unsupported`, true for the PLE layer)
//! goes through [`super::verify_pieces::Pieces`]: eager warm-up on the first
//! visit, capture on the second, replay after. Embed, the metadata upload,
//! the PLE layer, the 12 attention layers and the head stay eager. Runs on
//! the 48-layer `[GDN, GDN, GDN, attention] x 12` stack with PLE at layer 1:
//! `[0]`, `[2]`, then `[4i..4i+3)` for i in 1..12 -- 13 runs, 35 layers.
//!
//! Keys: `(step, SSM pool slot per row, rows, first layer)`. A run bakes the
//! fixed arenas (hidden, highway streams, scratch, MoE buffers, the decode
//! metadata at its fixed address) and the SSM pool addresses of its rows'
//! slots (h/conv state, verify intermediates, deferred-commit staging), all
//! a function of the slot; the step (decode / batched decode / verify) and
//! the row count pick the kernels. Nothing per-sequence (the PLE carry, QSA
//! keys) is in a run, so runs survive `free_sequence` like the GLM ones.
//!
//! Collectives. Default: every collective splits the capture and replays
//! eagerly between graphs (`Recorder`), so both TP ranks issue the same
//! collective calls in the same order whatever either rank captured.
//! `ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES=1` instead records the RDMA
//! one-shot all-reduce inside the graph when the communicator offers it
//! (`ATLAS_RDMA_ONESHOT=1`). That stays rank-symmetric: (1) the eager
//! `all_reduce_async` already takes the one-shot first under the same
//! size-only eligibility, so a captured reduce is the very op the eager
//! pass would have enqueued, with the same arguments; (2) the one-shot
//! protocol advances its sequence on the device and the proxy counts
//! staged ops as they EXECUTE, so a graph replay, an eager call and a mix
//! across ranks are indistinguishable to the peer; (3) a capture pass
//! executes nothing, and a refused or failed capture re-runs eagerly, so
//! each rank executes each step's ops exactly once and in program order;
//! (4) a size the one-shot refuses splits as in the default. A rank-local
//! difference (diagnostics, budget) costs speed, never pairing.
//!
//! Lossless: `ctx.graph_capture` stays false inside a run, so every layer
//! takes the eager kernels and collective calls; the stream order is the
//! eager order.

use anyhow::{Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::kv_cache::PagedKvCache;

use super::types::TransformerModel;
use super::verify_pieces::Pieces;
use crate::layer::ForwardContext;
use crate::traits::SequenceState;

/// `ATLAS_QWEN4EXP_DECODE_GRAPH=1`; read once, from the profile both ranks
/// share (`startup_parity`).
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES=1`: record capturable one-shot
/// all-reduces inside the graphs (module docs). Only read under
/// [`requested`].
pub(crate) fn collectives_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES").as_deref() == Ok("1")
    })
}

/// Which step a run belongs to: a GDN layer launches different kernels for
/// a single decode row (`decode`), a batch of sequences
/// (`decode_multi_seq`) and K rows of one sequence (`decode_batched`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PieceStep {
    Decode,
    Batch,
    Verify,
}

/// `(step, SSM pool slot per row owner, rows, first layer of the run)`.
pub(crate) type DecodePieceKey = (PieceStep, Vec<u32>, usize, usize);

/// The model's cache. Budget: `ATLAS_QWEN4EXP_DECODE_GRAPH_MAX_GRAPHS`,
/// default 600 (~100-150 MiB on GB10, see `VerifyPieces`). Split at
/// collectives, a TP2 key is 13 runs + one graph per reduce: about 83 for decode
/// and the K=2/3 verify or C2/C3 (35 GDN layers x 2 reduces), 188 for C4
/// (the per-row MoE loop reduces 4 times a layer). 13 with the collectives
/// captured, and at TP1. A warm key over the budget stays eager.
pub(crate) fn new_cache() -> Pieces<DecodePieceKey> {
    let budget = std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH_MAX_GRAPHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    Pieces::new(
        "decode",
        "(step, ssm slots, rows, layer)",
        budget,
        requested() && collectives_requested(),
    )
}

/// Admission. `eager_only` folds every reason the step must stay eager or
/// already runs as a whole graph.
pub(crate) fn admitted(requested: bool, model_type: &str, eager_only: bool) -> bool {
    requested && model_type == "qwen4_exp" && !eager_only
}

/// Diagnostics that sync or read back inside a GDN run. Under a capture
/// they would refuse every key, so they keep the step eager. Rank-local is
/// safe: a rank that stays eager issues the same collective calls.
fn diagnostics_sync(k4_diag: bool) -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        [
            "ATLAS_QWEN4EXP_DECODE_PROF",
            "ATLAS_QWEN4EXP_VERIFY_PROF",
            "ATLAS_SSM_DETAIL_PROFILE",
            "ATLAS_SSM_MS_PROFILE",
            "ATLAS_CONC_HSD",
            "ATLAS_DFLASH_CAPTURE_TRACE",
        ]
        .iter()
        .any(|k| std::env::var(k).is_ok_and(|v| v == "1" || v == "true"))
    });
    env || k4_diag
        || super::graph_flags::k2_diag()
        || super::graph_flags::ms_profile()
        || super::graph_flags::ssm_save_dump()
        || super::graph_flags::verify_layer_trace()
        // Arms the MoE and SSM readbacks keyed on `ctx.graph_capture`,
        // which pieces leave false.
        || tracing::level_filters::LevelFilter::current() >= tracing::Level::DEBUG
}

/// Where layer `i` sits among the maximal runs of `capturable` layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunPos {
    /// Not capturable: the caller runs it eagerly.
    Eager,
    /// First layer of the run `i..end`.
    Start(usize),
    /// Inside a run its first layer already handled.
    Inside,
}

pub(crate) fn run_pos(capturable: impl Fn(usize) -> bool, i: usize, layers: usize) -> RunPos {
    if !capturable(i) {
        return RunPos::Eager;
    }
    if i > 0 && capturable(i - 1) {
        return RunPos::Inside;
    }
    RunPos::Start((i..layers).find(|&j| !capturable(j)).unwrap_or(layers))
}

impl TransformerModel {
    /// Whether this step takes the pieces. `use_graphs`: the step already
    /// runs as one graph. Every input is shared configuration except the
    /// diagnostics, which only cost speed when they differ (module docs).
    pub(super) fn decode_pieces_admitted(
        &self,
        use_graphs: bool,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
    ) -> bool {
        // Switch and model first: every other model pays nothing per step.
        admitted(requested(), &self.config.model_type, false)
            && !(use_graphs
                || self.profile
                || ctx.profile
                || ctx.ssm_batch.is_some()
                || ctx.midchunk_capture.is_some()
                // NVMe-backed KV does host I/O per step.
                || kv_cache.config().cache_blocks_per_seq.is_some()
                // As for the GLM pieces: adapters stay eager.
                || self.lora.is_some()
                || diagnostics_sync(self.levers.k4_diag))
    }

    /// Whether layer `i` may sit inside a captured run.
    fn piece_capturable(&self, i: usize) -> bool {
        self.config.layer_type(i) == LayerType::LinearAttention
            && !self.layers[i].decode_graph_unsupported()
    }

    /// Run the maximal run of capturable GDN layers starting at `layer_idx`
    /// through the piecewise cache, calling `layer(li, ctx)` for each of its
    /// layers with the context the run must use. `Ok(false)`: `layer_idx` is
    /// not capturable, the caller runs it. `Ok(true)`: handled now, or with
    /// its run's first layer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_piece_run(
        &self,
        layer_idx: usize,
        step: PieceStep,
        slots: &[u32],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
        mut layer: impl FnMut(usize, &ForwardContext) -> Result<()>,
    ) -> Result<bool> {
        let end = match run_pos(|i| self.piece_capturable(i), layer_idx, self.layers.len()) {
            RunPos::Eager => return Ok(false),
            RunPos::Inside => return Ok(true),
            RunPos::Start(end) => end,
        };
        ensure!(
            ctx.midchunk_capture.is_none(),
            "piecewise decode graph: decode has no mid-chunk capture"
        );
        let key = (step, slots.to_vec(), rows, layer_idx);
        self.decode_pieces
            .run_with(key, self.gpu.as_ref(), ctx.comm, stream, |comm| {
                // The decode contexts carry no mid-chunk capture (the one
                // field that is not `Copy`).
                let ctx = ForwardContext {
                    comm,
                    midchunk_capture: None,
                    ..*ctx
                };
                (layer_idx..end).try_for_each(|li| layer(li, &ctx))
            })?;
        Ok(true)
    }

    /// [`Self::gdn_piece_run`] for the single-row decode
    /// (`decode_forward_body`): each layer runs `decode` and the DFlash
    /// capture of row 0, as the eager loop does. A sequence without an SSM
    /// pool slot runs eagerly.
    pub(super) fn decode_gdn_piece_run(
        &self,
        layer_idx: usize,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(slot) = seq.ssm_slot_idx() else {
            return Ok(false);
        };
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        self.gdn_piece_run(
            layer_idx,
            PieceStep::Decode,
            &[slot as u32],
            1,
            ctx,
            stream,
            |li, ctx| {
                self.layers[li].decode(
                    hidden,
                    residual,
                    seq.layer_states[li].as_mut(),
                    kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    ctx,
                    stream,
                )?;
                self.try_dflash_capture(li, 0, stream)
            },
        )
    }

    /// [`Self::gdn_piece_run`] for a K-row MTP verify of one sequence
    /// (`verify_b` K=2, `verify_c` K=3): each layer runs `decode_batched`
    /// and the DFlash capture of the last row, as the eager loops do. A
    /// sequence without an SSM pool slot runs eagerly.
    pub(super) fn verify_gdn_piece_run(
        &self,
        layer_idx: usize,
        k: usize,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(slot) = seq.ssm_slot_idx() else {
            return Ok(false);
        };
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        self.gdn_piece_run(
            layer_idx,
            PieceStep::Verify,
            &[slot as u32],
            k,
            ctx,
            stream,
            |li, ctx| {
                self.layers[li].decode_batched(
                    hidden,
                    residual,
                    k,
                    seq.layer_states[li].as_mut(),
                    kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    ctx,
                    stream,
                )?;
                self.try_dflash_capture(li, k - 1, stream)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_qwen4exp_only() {
        assert!(admitted(true, "qwen4_exp", false));
        assert!(!admitted(false, "qwen4_exp", false));
        assert!(!admitted(true, "qwen4_exp", true));
        for model in ["glm5_next", "qwen3_next", ""] {
            assert!(!admitted(true, model, false));
        }
    }

    /// The Flash-Next stack: `[GDN, GDN, GDN, attention] x 12`, PLE (a
    /// graph veto) on layer 1.
    fn flash_next(i: usize) -> bool {
        i % 4 != 3 && i != 1
    }

    #[test]
    fn flash_next_runs() {
        let starts: Vec<(usize, usize)> = (0..48)
            .filter_map(|i| match run_pos(flash_next, i, 48) {
                RunPos::Start(end) => Some((i, end)),
                _ => None,
            })
            .collect();
        let mut want = vec![(0, 1), (2, 3)];
        want.extend((1..12).map(|b| (4 * b, 4 * b + 3)));
        assert_eq!(starts, want);
        let captured: usize = starts.iter().map(|(s, e)| e - s).sum();
        assert_eq!(captured, 35, "every GDN layer but the PLE one");
        for i in [1, 3, 47] {
            assert_eq!(run_pos(flash_next, i, 48), RunPos::Eager);
        }
        for i in [5, 6, 46] {
            assert_eq!(run_pos(flash_next, i, 48), RunPos::Inside);
        }
    }

    #[test]
    fn a_run_reaching_the_last_layer_ends_there() {
        assert_eq!(run_pos(|i| i >= 2, 2, 5), RunPos::Start(5));
        assert_eq!(run_pos(|_| true, 0, 3), RunPos::Start(3));
        assert_eq!(run_pos(|_| false, 0, 3), RunPos::Eager);
    }
}
