// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split propose (`ATLAS_GLM_DRAFT_TP`, default off).
//!
//! The drafter runs on the head rank while the worker waits for the next
//! verify. A propose is the read of the drafter's NVFP4 weights, so with this
//! switch the worker reads half of the largest ones: each rank computes a
//! contiguous half of the OUTPUT rows of a layer's gate/up and down
//! projections and of the vocabulary head, with the unchanged tensor-core
//! GEMV over a row range of the same twin, and the halves are swapped over
//! the pair exchange. A GEMV output row depends only on its weight row and
//! the input, so every value is the one the unsplit launch writes and the
//! drafts are bit-identical.
//!
//! Both ranks walk `Geometry::steps` of their own rank. The swaps are the
//! same list in the same order with the same sizes on both
//! ([`Geometry::swaps`]); only the local pieces between them differ. Pieces
//! are stream work with no host sync, so the head captures each run of pieces
//! between two swaps as one CUDA graph and the worker enqueues its whole walk
//! at once.
//!
//! `ATLAS_GLM_DRAFT_TP=1` splits every part; `mlp` or `head` (comma separated)
//! selects parts for an A/B. Both ranks must run the same value
//! (`model::startup_parity`).

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{GpuBackend, GraphHandle};

#[path = "rank_split_exec.rs"]
mod exec;
pub use exec::RankSplit;

/// What the worker shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parts {
    /// Each layer's gate/up and down projections.
    pub mlp: bool,
    /// The vocabulary head.
    pub head: bool,
}

impl Parts {
    /// `1` = every part, else a comma list of `mlp` and `head`; unset, empty
    /// or `0` = off. Anything else is refused: a typo must not run unsplit
    /// on one rank.
    pub(crate) fn parse(value: Option<&str>) -> Result<Option<Self>> {
        let all = Self {
            mlp: true,
            head: true,
        };
        match value.map(str::trim) {
            None | Some("") | Some("0") => Ok(None),
            Some("1") => Ok(Some(all)),
            Some(list) => {
                let mut parts = Self {
                    mlp: false,
                    head: false,
                };
                for part in list.split(',') {
                    match part.trim() {
                        "mlp" => parts.mlp = true,
                        "head" => parts.head = true,
                        other => bail!("ATLAS_GLM_DRAFT_TP: unknown part '{other}' in '{list}'"),
                    }
                }
                Ok(Some(parts))
            }
        }
    }

    /// The value both ranks must agree on (0 = off).
    pub(crate) fn word(parts: Option<Self>) -> u64 {
        parts.map_or(0, |p| p.mlp as u64 | (p.head as u64) << 1)
    }
}

/// `ATLAS_GLM_DRAFT_TP`, from the profile both ranks share.
pub fn requested() -> Result<Option<Parts>> {
    Parts::parse(std::env::var("ATLAS_GLM_DRAFT_TP").ok().as_deref())
}

/// One exchange of the walk. The payload is `[gamma, width]` BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Swap {
    /// Layer input of the MLP (the post-attention norm), head to worker.
    Input(usize),
    /// Each rank's half of the SiLU-gated activation.
    Activation(usize),
    /// Each rank's half of the down projection; the head keeps both.
    Output(usize),
    /// The final-norm rows, head to worker.
    Hidden,
    /// Each rank's half of the vocabulary logits; the head keeps both.
    Logits,
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
    /// This rank's half of gate and up, SiLU-gated.
    GateUp(usize),
    /// This rank's half of the down projection.
    Down(usize),
    /// The MLP residual (head only).
    Residual(usize),
    /// The whole unsplit tail (head only).
    Tail,
    /// The final norm (head only).
    Norm,
    /// This rank's half of the vocabulary projection.
    Vocab,
    /// Draft selection over the joined logits (head only).
    Select,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Run(Piece),
    Swap(Swap),
}

/// The shapes a split is planned from; identical on both ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub gamma: usize,
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    pub layers: usize,
    pub parts: Parts,
}

/// Output rows a tensor-core GEMV CTA owns. A half starts on a CTA boundary,
/// so its launch runs exactly the CTAs the unsplit launch runs for its rows.
const CTA_ROWS: usize = 16;

/// Rank `rank`'s contiguous share of `n` output rows: `(first, rows)`. Rank 0
/// takes the rows below the largest CTA boundary at or under `n / 2`, rank 1
/// the rest.
pub(crate) fn half(n: usize, rank: usize) -> (usize, usize) {
    let cut = n / 2 / CTA_ROWS * CTA_ROWS;
    if rank == 0 { (0, cut) } else { (cut, n - cut) }
}

impl Geometry {
    /// Every row range is non-empty and starts on a 16-byte boundary of the
    /// packed twin (its K is a multiple of 32).
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.gamma > 0 && self.layers > 0 && (self.parts.mlp || self.parts.head),
            "ATLAS_GLM_DRAFT_TP: nothing to split ({self:?})"
        );
        ensure!(
            self.hidden.is_multiple_of(32) && self.inter.is_multiple_of(32),
            "ATLAS_GLM_DRAFT_TP needs hidden and intermediate sizes in multiples of 32 ({self:?})"
        );
        ensure!(
            self.vocab >= 2 * CTA_ROWS,
            "ATLAS_GLM_DRAFT_TP: drafter vocabulary too small to split ({})",
            self.vocab
        );
        Ok(())
    }

    /// BF16 values per row of a swap's payload: both ranks send the larger
    /// share's width (the smaller share's tail is unused).
    fn width(&self, swap: Swap) -> usize {
        match swap {
            Swap::Input(_) | Swap::Hidden => self.hidden,
            Swap::Activation(_) => half(self.inter, 1).1,
            Swap::Output(_) => half(self.hidden, 1).1,
            Swap::Logits => half(self.vocab, 1).1,
        }
    }

    /// Bytes each rank sends and receives in `swap`.
    pub(crate) fn bytes(&self, swap: Swap) -> usize {
        self.gamma * self.width(swap) * 2
    }

    /// Rank `rank`'s walk. Rank 0 runs the whole propose forward; rank 1
    /// only its halves.
    pub(crate) fn steps(&self, rank: usize) -> Vec<Step> {
        use Piece::*;
        let head = rank == 0;
        let mut steps = Vec::new();
        // Pieces only the head runs, pieces both run, and swaps.
        let own = |steps: &mut Vec<Step>, piece| {
            if head {
                steps.push(Step::Run(piece));
            }
        };
        let both = |steps: &mut Vec<Step>, piece| steps.push(Step::Run(piece));
        let swap = |steps: &mut Vec<Step>, swap| steps.push(Step::Swap(swap));
        for l in 0..self.layers {
            own(&mut steps, Attention(l));
            if !self.parts.mlp {
                own(&mut steps, Post(l));
                continue;
            }
            own(&mut steps, Project(l));
            swap(&mut steps, Swap::Input(l));
            both(&mut steps, GateUp(l));
            swap(&mut steps, Swap::Activation(l));
            both(&mut steps, Down(l));
            swap(&mut steps, Swap::Output(l));
            own(&mut steps, Residual(l));
        }
        if !self.parts.head {
            own(&mut steps, Tail);
            return steps;
        }
        own(&mut steps, Norm);
        swap(&mut steps, Swap::Hidden);
        both(&mut steps, Vocab);
        swap(&mut steps, Swap::Logits);
        own(&mut steps, Select);
        steps
    }

    /// The swaps of a propose, in order: the same list on both ranks.
    pub(crate) fn swaps(&self) -> Vec<Swap> {
        swaps_of(&self.steps(1))
    }
}

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
