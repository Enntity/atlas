// SPDX-License-Identifier: AGPL-3.0-only

//! What each of the unrolled visits does, derived once from the config.
//!
//! Mirrors `Inkling2LoopTextModel.forward` (transformers PR #51050): a block
//! stack B of saved layer inputs grows at every block-boundary layer; the
//! attention input reads B, the MLP input reads B plus the boundary push. The
//! loop core runs `num_passes` times; each later pass restarts from the
//! prelude's output and stack, and its layers read messages from the previous
//! pass (that pass's blocks after the prelude's and the pass's own first push,
//! plus its final hidden state). The last visit applies the final block
//! residual before the shared final norm.

use atlas_core::config::Inkling2Config;

/// Per-visit control flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Visit {
    pub physical: usize,
    pub full: bool,
    /// 0 outside the core and on the first pass.
    pub pass: usize,
    /// Index within the core (loop layer) for core visits.
    pub loop_layer: Option<usize>,
    /// The layer pushes its input onto B (and restarts the residual).
    pub boundary: bool,
    /// Blocks on B when the attention input is formed.
    pub blocks_before: usize,
    /// Visit 0: the embedding norm runs first.
    pub first: bool,
    /// Last visit: the final block residual runs after it.
    pub last: bool,
    /// Last prelude visit: save its output hidden and B for the later passes.
    pub save_prelude: bool,
    /// First visit of a later pass: stash the previous pass (its B after the
    /// prelude blocks + 1, and the hidden entering this visit), then restore
    /// the prelude's hidden and B.
    pub restart: bool,
    /// Blocks of the previous pass the messages read (later passes only).
    pub message_blocks: usize,
    /// LoopCD: this visit's role in the reference coda pass.
    pub loopcd: LoopCdRole,
}

/// What a visit does for LoopCD's reference coda pass (`inkling2_loopcd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LoopCdRole {
    #[default]
    None,
    /// The second pass's restart: first stash the first pass (its output
    /// and the blocks it pushed, `message_blocks + 1` from `blocks_before`).
    SaveFirstPass,
    /// A reference coda visit; the first sets the final output aside and
    /// restores the first pass, the last writes the reference rows and puts
    /// the final output back.
    Reference { enter: bool, leave: bool },
}

pub(crate) fn schedule(ink: &Inkling2Config) -> Vec<Visit> {
    let block = ink.attn_res_block_size;
    let visits = ink.model_visits();
    let mut out = Vec::with_capacity(visits);
    let mut blocks = 0usize;
    let mut prelude_blocks = 0usize;
    let mut message_blocks = 0usize;
    for v in 0..visits {
        let physical = ink.visit_physical(v);
        let pass = ink.visit_pass(v);
        let core_visits =
            ink.core_start..ink.core_start + ink.num_passes * (ink.core_end - ink.core_start);
        let in_core = core_visits.contains(&v);
        let restart = in_core && pass > 0 && physical == ink.core_start;
        if restart {
            // previous pass's blocks past the prelude's and the pass's first push
            message_blocks = blocks - prelude_blocks - 1;
            blocks = prelude_blocks;
        }
        let boundary = physical.is_multiple_of(block);
        out.push(Visit {
            physical,
            full: !ink.local_physical.contains(&physical),
            pass,
            loop_layer: in_core.then(|| physical - ink.core_start),
            boundary,
            blocks_before: blocks,
            first: v == 0,
            last: v + 1 == visits,
            save_prelude: v + 1 == ink.core_start,
            restart,
            message_blocks: if pass > 0 { message_blocks } else { 0 },
            loopcd: if restart && pass == 1 && ink.reference_visits() > 0 {
                LoopCdRole::SaveFirstPass
            } else {
                LoopCdRole::None
            },
        });
        if boundary {
            blocks += 1;
        }
        if v + 1 == ink.core_start {
            prelude_blocks = blocks;
        }
    }
    reference_visits(ink, &mut out);
    out
}

/// LoopCD's reference coda: the model's coda visits again, from the first
/// pass's state, which holds as many blocks as the last pass's.
fn reference_visits(ink: &atlas_core::config::Inkling2Config, out: &mut Vec<Visit>) {
    let n = ink.reference_visits();
    let coda = out.len() - (ink.physical_layers - ink.core_end);
    for i in 0..n {
        let mut v = out[coda + i];
        v.last = false;
        v.loopcd = LoopCdRole::Reference {
            enter: i == 0,
            leave: i + 1 == n,
        };
        out.push(v);
    }
}

/// Where LoopCD's first-pass stash sits on the block stack: `(first block,
/// blocks)`, `(0, 0)` with LoopCD off.
pub(crate) fn loopcd_stash(visits: &[Visit]) -> (usize, usize) {
    visits
        .iter()
        .find(|v| v.loopcd == LoopCdRole::SaveFirstPass)
        .map_or((0, 0), |v| (v.blocks_before, v.message_blocks + 1))
}

/// Most blocks B ever holds (sizes the shared stack).
pub(crate) fn max_blocks(visits: &[Visit]) -> usize {
    visits
        .iter()
        .map(|v| v.blocks_before + usize::from(v.boundary))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
