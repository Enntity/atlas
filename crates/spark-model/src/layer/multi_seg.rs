// SPDX-License-Identifier: AGPL-3.0-only

//! One sequence's rows of a multi-sequence prefill pass
//! (`ATLAS_QWEN4EXP_PREFILL_MULTI`, `TransformerLayer::prefill_multi`).

use super::{ForwardContext, LayerState};

/// A sequence's rows `[row0, row0 + rows)` of the pass, at sequence positions
/// `[start, start + rows)`, with the per-sequence state a layer's recurrent and
/// attention parts need. `ctx` is the pass's context with THIS sequence's
/// attention metadata and token ids; the row-wise parts take the pass's own.
pub struct MultiSeg<'s, 'c> {
    pub row0: usize,
    pub rows: usize,
    pub start: usize,
    pub kv_write_start: usize,
    pub state: &'s mut dyn LayerState,
    pub block_table: &'s mut Vec<u32>,
    pub disk_block_ids: &'s mut Vec<u32>,
    pub disk_last_offloaded_per_layer: &'s mut Vec<u32>,
    pub ctx: &'s ForwardContext<'c>,
}
