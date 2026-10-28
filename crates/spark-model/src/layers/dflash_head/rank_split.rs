// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split propose (`ATLAS_GLM_DRAFT_TP`, default off).
//!
//! The drafter runs on the head rank while the workers wait for the next
//! verify. A propose is the read of the drafter's NVFP4 weights, so with this
//! switch every rank reads its share of the largest ones: each of the
//! `ranks` ranks (the TP2 pair or the TP4 chain) computes a contiguous share
//! of the OUTPUT rows of a layer's gate/up and down projections and of the
//! vocabulary head, with the unchanged tensor-core GEMV over a row range of
//! the same twin, and the shares are swapped over the pair exchange or
//! gathered over the chain (the head's rows go out the same way, or by the
//! chain's broadcast). A GEMV output row depends only on its weight row and
//! the input, so every value is the one the unsplit launch writes and the
//! drafts are bit-identical.
//!
//! Every rank walks `Geometry::steps` of its own rank. The swaps are the
//! same list in the same order with the same sizes on every rank
//! (`Geometry::swaps`); only the local pieces between them differ. Pieces
//! are stream work with no host sync, so the head captures each run of pieces
//! between two swaps as one CUDA graph and the worker enqueues its whole walk
//! at once.
//!
//! `ATLAS_GLM_DRAFT_TP=1` splits the MLP and the head; `mlp`, `head` or
//! `attn` (comma separated) selects parts. `attn` (with `mlp`) also shares
//! each layer's q/k/v and o projections the same way: the head swaps the
//! normed rows out, every rank projects its share of the output rows, the
//! head joins them and runs the rest of attention alone, swaps the attention
//! rows out for o_proj, and joins that (four more swaps a layer for most of
//! the head's remaining weight reads). Both ranks must run the same value
//! (`model::startup_parity`).
//!
//! `ATLAS_GLM_DRAFT_TP_BATCH=1` (with the switch above, default off) splits
//! the batched B×gamma propose of an owner-batched step the same way
//! (`rank_split_batch`): the plan's rows are B×gamma, the halves are the
//! same tensor-core GEMV over a row range at the same row count the unsplit
//! batched launch uses, and the swaps carry the batch buffers. Without it the
//! worker idles through every batched propose (about 10 ms each in a C4
//! prose nsys profile, 2026-10-03).
//!
//! `ATLAS_GLM_DRAFT_TP_CTX=1` (with the switch above, default off) splits a
//! split propose's context append too (`rank_split_ctx`): the `fc` and fused
//! K/V projections of the context rows the propose appends, the largest
//! drafter reads the head still ran alone (about 0.5 ms a propose in a C1
//! prose nsys profile, 2026-10-03). The head announces those rows with the
//! propose and walks them as the first swaps of the plan.
//!
//! `ATLAS_GLM_DRAFT_TP_TOPK=1` (with the head split, default off) swaps
//! each rank's candidate lists instead of its logits (`Swap::Candidates`):
//! each rank reduces its share of the vocabulary to every row's exact top-16
//! on the device (`dflash2_selector_topk`, one launch a sequence), the
//! workers send their lists (1 KiB a rank at gamma 8, where the logits half
//! is 1.2 MiB and crosses the pair through the host), and the head merges
//! every rank's lists of each row and walks each sequence's selector chain
//! (`dflash2_selector_walk`, `_ranks` on the chain). Each list is exact in
//! the selector's total order, so the drafts are the unsplit selector's. A
//! batched propose (`ATLAS_GLM_DRAFT_TP_BATCH`) swaps its B sequences' lists
//! the same way, and its selector walks them in place of the joined logits.
//!
//! Prior art: splitting the drafter across both ranks follows MiaAI-Lab's
//! `DFLASH_DRAFT_TP=2` (<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks>
//! `.env.example`, `start.sh`; vLLM `draft_tensor_parallel_size`). Idea only,
//! no code. The output-row share on 16-row CTA boundaries, swapped over the
//! RDMA pair with bit-identical drafts, is ours (docs/glm-prior-art.md).

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

#[path = "rank_split_exec.rs"]
mod exec;
pub use exec::RankSplit;
pub(crate) use exec::{CtxSeq, CtxTurn, TopkSelect};

#[path = "rank_split_ctx.rs"]
mod ctx;
pub(crate) use ctx::CtxRows;
use ctx::ctx_steps;
pub use ctx::{announce_extra, announce_word, ctx_requested, extra_follows};

/// What the worker shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parts {
    /// Each layer's gate/up and down projections.
    pub mlp: bool,
    /// The vocabulary head.
    pub head: bool,
    /// Each layer's q/k/v and o projections (needs `mlp`).
    pub attn: bool,
}

/// Bytes per row of a `Swap::Candidates` payload: one row's top-K list, f32
/// values and u32 ids (`ops::dflash2_selector_lists_bytes`).
const CANDIDATE_ROW_BYTES: usize = 8 * crate::layers::ops::DFLASH2_SELECTOR_MAX_TOP_K;

impl Parts {
    /// `1` = the MLP and the head, else a comma list of `mlp`, `head` and
    /// `attn`; unset, empty
    /// or `0` = off. Anything else is refused: a typo must not run unsplit
    /// on one rank.
    pub(crate) fn parse(value: Option<&str>) -> Result<Option<Self>> {
        let all = Self {
            mlp: true,
            head: true,
            attn: false,
        };
        match value.map(str::trim) {
            None | Some("") | Some("0") => Ok(None),
            Some("1") => Ok(Some(all)),
            Some(list) => {
                let mut parts = Self {
                    mlp: false,
                    head: false,
                    attn: false,
                };
                for part in list.split(',') {
                    match part.trim() {
                        "mlp" => parts.mlp = true,
                        "head" => parts.head = true,
                        "attn" => parts.attn = true,
                        other => bail!("ATLAS_GLM_DRAFT_TP: unknown part '{other}' in '{list}'"),
                    }
                }
                Ok(Some(parts))
            }
        }
    }

    /// The value both ranks must agree on (0 = off).
    pub(crate) fn word(parts: Option<Self>) -> u64 {
        parts.map_or(0, |p| {
            p.mlp as u64 | (p.head as u64) << 1 | (p.attn as u64) << 5
        })
    }
}

/// `ATLAS_GLM_DRAFT_TP`, from the profile both ranks share.
pub fn requested() -> Result<Option<Parts>> {
    Parts::parse(std::env::var("ATLAS_GLM_DRAFT_TP").ok().as_deref())
}

/// `ATLAS_GLM_DRAFT_TP_BATCH`: `1` splits batched proposes too; unset, empty
/// or `0` = off; anything else is refused.
pub(crate) fn parse_batch(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_DRAFT_TP_BATCH must be 0 or 1, got '{other}'"),
    }
}

/// `ATLAS_GLM_DRAFT_TP_BATCH`, from the profile both ranks share.
pub fn batch_requested() -> Result<bool> {
    parse_batch(std::env::var("ATLAS_GLM_DRAFT_TP_BATCH").ok().as_deref())
}

/// `ATLAS_GLM_DRAFT_TP_TOPK`: `1` swaps candidate lists instead of logits;
/// unset, empty or `0` = off; anything else is refused.
pub(crate) fn parse_topk(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_DRAFT_TP_TOPK must be 0 or 1, got '{other}'"),
    }
}

/// `ATLAS_GLM_DRAFT_TP_TOPK`, from the profile both ranks share.
pub fn topk_requested() -> Result<bool> {
    parse_topk(std::env::var("ATLAS_GLM_DRAFT_TP_TOPK").ok().as_deref())
}

/// The value both ranks must agree on: [`Parts::word`], plus bit 2 for the
/// batched split, bit 3 for the context split and bit 4 for the candidate
/// swap (so the word is unchanged with them off).
pub(crate) fn parity_word(parts: Option<Parts>, batch: bool, ctx: bool, topk: bool) -> u64 {
    Parts::word(parts) | (batch as u64) << 2 | (ctx as u64) << 3 | (topk as u64) << 4
}

/// Where a propose's whole rows live on a rank: the MLP input and final-norm
/// rows (`norm`), the gated activation (`inter`), the down projection the
/// head's residual adds (`acc`) and the logits (`logits`). A single-sequence
/// propose uses the block scratch ([`Frame::serial`]), a batched one the
/// head's B×gamma buffers (`BlockDiffusionDraftHead::batch_frame`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub norm: DevicePtr,
    pub inter: DevicePtr,
    pub acc: DevicePtr,
    pub logits: DevicePtr,
    /// A context append's input rows (the head's accumulator rows, the
    /// worker's landing buffer), its `fc` output (normed in place) and its
    /// fused K/V output (`ATLAS_GLM_DRAFT_TP_CTX`).
    pub ctx_in: DevicePtr,
    pub ctx_fc: DevicePtr,
    pub ctx_kv: DevicePtr,
    /// The `attn` part: where the head's q/k/v projections land, the
    /// attention rows o_proj reads, and o_proj's output.
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub attn: DevicePtr,
    pub attn_proj: DevicePtr,
}

impl Frame {
    pub(crate) fn serial(scratch: &super::DflashScratch) -> Self {
        Self {
            norm: scratch.norm_buf,
            inter: scratch.mlp_intermediate,
            acc: scratch.stream_acc,
            logits: scratch.logits,
            ctx_in: DevicePtr::NULL,
            ctx_fc: scratch.fc_proj,
            ctx_kv: scratch.fused_kv_out,
            q: scratch.q_buf,
            k: scratch.k_buf,
            v: scratch.v_buf,
            attn: scratch.attn_out,
            attn_proj: scratch.stream_acc,
        }
    }
}

/// One exchange of the walk. The payload is `[gamma, width]` BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Swap {
    /// Layer input of the MLP (the post-attention norm), head to worker.
    Input(usize),
    /// Each rank's share of the SiLU-gated activation.
    Activation(usize),
    /// Each rank's share of the down projection; the head keeps them all.
    Output(usize),
    /// The final-norm rows, head to worker.
    Hidden,
    /// Each rank's share of the vocabulary logits; the head keeps them all.
    Logits,
    /// The workers' top-K candidate lists of their shares, to the head.
    Candidates,
    /// Context append `i`'s input rows, head to worker.
    CtxInput(usize),
    /// Each rank's share of its `fc` rows; every rank keeps them all.
    CtxHidden(usize),
    /// Each rank's share of its fused K/V rows; the head keeps them all.
    CtxKv(usize),
    /// `attn`: the layer's normed input rows, head to worker.
    AttnIn(usize),
    /// `attn`: each rank's shares of q, k and v; the head keeps them all.
    Qkv(usize),
    /// `attn`: the attention rows o_proj reads, head to worker.
    AttnOut(usize),
    /// `attn`: each rank's share of o_proj; the head keeps them all.
    OProj(usize),
}

/// Stream work between two swaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Layer input norm through attention (head only).
    Attention(usize),
    /// The whole unsplit post-attention half (head only).
    Post(usize),
    /// o_proj, residual and the MLP input norm (head only).
    Project(usize),
    /// This rank's share of gate and up, SiLU-gated.
    GateUp(usize),
    /// This rank's share of the down projection.
    Down(usize),
    /// The MLP residual (head only).
    Residual(usize),
    /// The whole unsplit tail (head only).
    Tail,
    /// The final norm (head only).
    Norm,
    /// This rank's share of the vocabulary projection.
    Vocab,
    /// Every row's top-K candidates of this rank's share (the head also
    /// projects the selector's hidden rows).
    TopK,
    /// Draft selection over the joined logits, or over both ranks'
    /// candidate lists (head only).
    Select,
    /// This rank's share of context append `i`'s `fc` projection.
    CtxFc(usize),
    /// `hidden_norm` over the joined `fc` rows (every rank).
    CtxNorm(usize),
    /// This rank's share of context append `i`'s fused K/V projection.
    CtxKv(usize),
    /// `attn`: the input norm and attention conv (head only).
    AttnPre(usize),
    /// `attn`: this rank's shares of the q, k and v projections.
    Qkv(usize),
    /// `attn`: q/k norms, RoPE, the cache write and attention (head only).
    AttnCore(usize),
    /// `attn`: this rank's share of o_proj.
    OProj(usize),
    /// `attn`: o_proj's residual and the MLP input norm (head only).
    ProjectRest(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Run(Piece),
    Swap(Swap),
}

/// The shapes a split is planned from; identical on every rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub gamma: usize,
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    pub layers: usize,
    pub parts: Parts,
    /// `fc` input width and fused K/V output rows of a context append; zero
    /// without `ATLAS_GLM_DRAFT_TP_CTX`.
    pub ctx_in: usize,
    pub ctx_kv: usize,
    /// The context rows of the current propose's appends.
    pub ctx: CtxRows,
    /// `ATLAS_GLM_DRAFT_TP_TOPK`: the head split swaps candidate lists, not
    /// logits (single-sequence proposes only).
    pub topk: bool,
    /// The ranks sharing the propose: 2 (the pair) or 4 (the chain).
    pub ranks: usize,
    /// The attention projections' widths (`attn`).
    pub q_dim: usize,
    pub kv_dim: usize,
}

/// Output rows a tensor-core GEMV CTA owns. A share starts on a CTA boundary,
/// so its launch runs exactly the CTAs the unsplit launch runs for its rows.
const CTA_ROWS: usize = 16;

/// Rank `rank`'s contiguous share of `n` output rows of `ranks`: `(first,
/// rows)`. Rank `r` starts at the largest CTA boundary at or under `r * n /
/// ranks` and the last takes the rest (on the pair: rank 0 the rows below
/// the boundary at or under `n / 2`, rank 1 the rest).
pub(crate) fn share(n: usize, rank: usize, ranks: usize) -> (usize, usize) {
    let cut = |r: usize| {
        if r == ranks {
            n
        } else {
            n * r / ranks / CTA_ROWS * CTA_ROWS
        }
    };
    (cut(rank), cut(rank + 1) - cut(rank))
}

/// The widest of the `ranks` shares of `n` rows.
pub(crate) fn widest(n: usize, ranks: usize) -> usize {
    (0..ranks).map(|r| share(n, r, ranks).1).max().unwrap_or(n)
}

#[path = "rank_split_geometry.rs"]
mod geometry;

pub(crate) fn swaps_of(steps: &[Step]) -> Vec<Swap> {
    steps
        .iter()
        .filter_map(|s| match s {
            Step::Swap(x) => Some(*x),
            Step::Run(_) => None,
        })
        .collect()
}

/// What a walk does at each step.
pub(crate) trait SplitOps {
    fn piece(&self, piece: Piece) -> Result<()>;
    fn swap(&self, swap: Swap) -> Result<()>;
}

/// How the runs of pieces between swaps execute.
pub(crate) enum Graphs<'a> {
    /// Launch every piece.
    Eager,
    /// Capture each run as one graph (pushed here) and launch it.
    Capture(&'a mut Vec<GraphHandle>),
    /// Launch the graph captured for each run (a zero handle runs eagerly).
    Replay(&'a [GraphHandle]),
}

/// Graphs a walk of `steps` captures: one per swap, plus one.
pub(crate) fn run_count(steps: &[Step]) -> usize {
    swaps_of(steps).len() + 1
}

/// Walk `steps` on `stream`: each run of pieces, then the swap after it.
/// Swaps are never captured (the pair exchange refuses a capturing stream).
pub(crate) fn walk(
    steps: &[Step],
    mut graphs: Graphs<'_>,
    gpu: &dyn GpuBackend,
    stream: u64,
    ops: &dyn SplitOps,
) -> Result<()> {
    let eager = |run: &[Step]| -> Result<()> {
        run.iter().try_for_each(|s| match s {
            Step::Run(piece) => ops.piece(*piece),
            Step::Swap(_) => unreachable!("a run holds pieces only"),
        })
    };
    // `split` drops the separators: swap `index - 1` precedes run `index`.
    let swaps = swaps_of(steps);
    for (index, run) in steps.split(|s| matches!(s, Step::Swap(_))).enumerate() {
        if index > 0 {
            ops.swap(swaps[index - 1])?;
        }
        match &mut graphs {
            Graphs::Eager => eager(run)?,
            Graphs::Replay(handles) => match handles.get(index) {
                Some(g) if g.0 != 0 => gpu.launch_graph(*g, stream)?,
                _ => eager(run)?,
            },
            Graphs::Capture(out) => {
                gpu.begin_capture(stream)?;
                if let Err(e) = eager(run) {
                    gpu.abort_capture_if_active(stream);
                    return Err(e);
                }
                let graph = gpu.end_capture(stream)?;
                out.push(graph);
                // An empty capture replays eagerly, as the unsplit path does.
                if graph.0 != 0 {
                    gpu.launch_graph(graph, stream)?;
                } else {
                    eager(run)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "rank_split_tests.rs"]
mod tests;
