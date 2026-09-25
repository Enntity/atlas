// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched repaired long-context K3 verify: 2..=4 owners x 3 temporal
//! rows in one target traversal instead of one traversal per owner.
//!
//! Rows are flat and owner-major: owner `o` holds rows `[3o, 3o + 3)`. Every
//! per-owner state (KDA recurrence and its rollback snapshots, MLA cache
//! writes and causal semantic index) advances only against that owner's own
//! rows, in the same order as the single-owner verifier. Stateless work
//! (projections, mHC, norms, the FFN and the LM head) runs once over all rows,
//! so each weight is read once per step rather than once per owner.

use super::{AttnMetadataDev, LayerState};
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Verify rows per owner: the served `--num-drafts=2` long-context lane.
pub const ROWS: usize = 3;
/// The repaired long-context lane admits exactly four owners.
pub const MAX_OWNERS: usize = 4;
pub const MAX_ROWS: usize = ROWS * MAX_OWNERS;

/// `ATLAS_GLM_LONG_BATCH_VERIFY=1` opts in; anything but 0/1 is refused.
pub fn enabled() -> Result<bool> {
    let raw = std::env::var("ATLAS_GLM_LONG_BATCH_VERIFY").ok();
    ensure!(
        matches!(raw.as_deref(), None | Some("0") | Some("1")),
        "ATLAS_GLM_LONG_BATCH_VERIFY must be 0 or 1"
    );
    Ok(raw.as_deref() == Some("1"))
}

/// One owner's inputs for one layer.
pub struct GlmLongOwner<'a> {
    pub state: &'a mut (dyn LayerState + 'static),
    /// Absolute positions of this owner's three rows (consecutive).
    pub positions: [usize; ROWS],
    /// This owner's own three-row attention metadata.
    pub meta: AttnMetadataDev,
}

/// Row geometry shared by the stage copies.
#[derive(Clone, Copy, Debug)]
pub struct RowBytes {
    pub hidden: usize,
    pub highway: usize,
    pub post: usize,
    pub comb: usize,
    pub logits: usize,
}

impl RowBytes {
    pub fn new(hidden_size: usize, hc_mult: usize, vocab: usize) -> Self {
        Self {
            hidden: hidden_size * 2,
            highway: hc_mult * hidden_size * 4,
            post: hc_mult * 4,
            comb: hc_mult * hc_mult * 4,
            logits: vocab * 2,
        }
    }
    fn total(&self) -> usize {
        MAX_ROWS * (3 * self.hidden + self.highway + self.post + self.comb + self.logits)
    }
}

/// Dedicated device staging, disjoint from the arena, for up to `MAX_ROWS`.
///
/// During the traversal MLA layers verify each owner in place at arena rows
/// `[0, 3)`; `highway`/`post`/`comb`/`norm` hold every owner's rows between
/// that per-owner attention and the joint FFN. After the traversal the same
/// storage (plus `hidden`/`logits`) holds each owner's final rows, which
/// `restore` copies back to rows `[0, 3)` so the unchanged single-owner
/// verdict, repair and propose tail sees exactly a single-owner verify.
#[derive(Clone, Copy, Debug)]
pub struct GlmLongStage {
    pub rows: RowBytes,
    pub hidden: DevicePtr,
    pub norm: DevicePtr,
    pub highway: DevicePtr,
    pub post: DevicePtr,
    pub comb: DevicePtr,
    pub logits: DevicePtr,
    /// Joint FFN output, one owner's three rows at a time.
    pub ffn: DevicePtr,
}

impl GlmLongStage {
    pub fn alloc(gpu: &dyn GpuBackend, rows: RowBytes) -> Result<Self> {
        let base = gpu.alloc(rows.total())?;
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let ptr = base.offset(at);
            at += MAX_ROWS * bytes;
            ptr
        };
        Ok(Self {
            rows,
            hidden: take(rows.hidden),
            norm: take(rows.hidden),
            highway: take(rows.highway),
            post: take(rows.post),
            comb: take(rows.comb),
            logits: take(rows.logits),
            ffn: take(rows.hidden),
        })
    }

    /// Copy `count` rows of each `(arena, stage, row_bytes)` span between
    /// arena row `arena_row` and stage row `stage_row`.
    pub fn copy(
        &self,
        gpu: &dyn GpuBackend,
        spans: &[(DevicePtr, DevicePtr, usize)],
        arena_row: usize,
        stage_row: usize,
        count: usize,
        to_stage: bool,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            stage_row + count <= MAX_ROWS && arena_row + count <= MAX_ROWS,
            "GLM long owner stage rows out of range"
        );
        for &(arena, stage, bytes) in spans {
            let (a, s) = (
                arena.offset(arena_row * bytes),
                stage.offset(stage_row * bytes),
            );
            let (src, dst) = if to_stage { (a, s) } else { (s, a) };
            gpu.copy_d2d_async(src, dst, count * bytes, stream)?;
        }
        Ok(())
    }
}

/// The FFN of every owner with exactly the single-owner K=3 verifier's
/// arithmetic. Routed MoE runs once over all rows through the served C3
/// grouped path; other FFNs run per owner from arena rows [0, 3). Returns the
/// `[owners * 3, H]` output.
pub fn ffn_per_owner(
    ffn: &crate::layers::FfnComponent,
    owners: usize,
    stage: &GlmLongStage,
    ctx: &super::ForwardContext,
    stream: u64,
) -> Result<DevicePtr> {
    let b = ctx.buffers;
    let rows = owners * ROWS;
    match (ffn_mode(), ffn) {
        (FfnMode::Grouped, crate::layers::FfnComponent::Moe(_)) => {
            // The served K3 MoE is the C3 grouped path; run it once over every
            // owner's rows with the same row-for-row arithmetic.
            crate::layers::moe::with_owner_rows(rows as u32, || {
                ffn.forward_prefill(b.norm_output(), rows, ctx, stream)
            })?;
            return Ok(b.moe_output());
        }
        (FfnMode::Prefill, _) => {
            ffn.forward_prefill(b.norm_output(), rows, ctx, stream)?;
            return Ok(b.moe_output());
        }
        _ => {}
    }
    let bytes = stage.rows.hidden;
    let input = [(b.norm_output(), stage.norm, bytes)];
    let output = [(b.moe_output(), stage.ffn, bytes)];
    stage.copy(ctx.gpu, &input, 0, 0, owners * ROWS, true, stream)?;
    for owner in 0..owners {
        stage.copy(ctx.gpu, &input, 0, owner * ROWS, ROWS, false, stream)?;
        ffn.forward_k3(b.norm_output(), ctx, stream)?;
        stage.copy(ctx.gpu, &output, 0, owner * ROWS, ROWS, true, stream)?;
    }
    Ok(stage.ffn)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FfnMode {
    /// Default: routed MoE jointly through the exact C3 grouped arithmetic;
    /// dense FFN layers per owner.
    Grouped,
    /// Every FFN per owner through `forward_k3` (diagnostic).
    Owner,
    /// Generic prefill FFN over all rows. Measurably NOT the verifier's
    /// arithmetic (diagnostic only).
    Prefill,
}

/// `ATLAS_GLM_LONG_BATCH_FFN=grouped|owner|prefill`.
fn ffn_mode() -> FfnMode {
    static MODE: std::sync::OnceLock<FfnMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("ATLAS_GLM_LONG_BATCH_FFN").as_deref() {
            Ok("owner") => FfnMode::Owner,
            Ok("prefill") => FfnMode::Prefill,
            _ => FfnMode::Grouped,
        },
    )
}
