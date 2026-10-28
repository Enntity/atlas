// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeLayout {
    /// Peer heads' absorbed queries `[rows, 32, 512]` BF16.
    pub q_peer: usize,
    /// Localized IDs `[rows, 2051]` i32.
    pub ids: usize,
    /// Own heads: `splits + 1` FP32 partials (the peer's last) and LSEs.
    pub own_o: usize,
    pub own_lse: usize,
    /// Merged LSE the BF16 merge also writes `[rows, 32]`.
    pub out_lse: usize,
    /// Peer heads over this rank's tokens: `splits` partials and LSEs.
    pub peer_o: usize,
    pub peer_lse: usize,
    /// One merged FP32 partial `[rows, 32, 512]` then its LSE `[rows, 32]`.
    pub send: usize,
    pub recv: usize,
    pub total: usize,
}

impl MergeLayout {
    pub fn new(rows: u32, splits: u32) -> Self {
        let (rows, splits) = (rows as usize, splits as usize);
        let heads = HEADS as usize;
        let part = rows * heads * LATENT as usize * 4;
        let lse = rows * heads * 4;
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        let q_peer = take(rows * heads * LATENT as usize * 2);
        let ids = take(rows * WIDTH as usize * 4);
        let own_o = take((splits + 1) * part);
        let own_lse = take((splits + 1) * lse);
        let out_lse = take(lse);
        let peer_o = take(splits * part);
        let peer_lse = take(splits * lse);
        let send = take(part + lse);
        let recv = take(part + lse);
        Self {
            q_peer,
            ids,
            own_o,
            own_lse,
            out_lse,
            peer_o,
            peer_lse,
            send,
            recv,
            total: at,
        }
    }

    /// Bytes of one exchanged partial: FP32 output then LSE.
    pub fn partial_bytes(rows: u32) -> usize {
        rows as usize * HEADS as usize * (LATENT as usize + 1) * 4
    }

    /// Per-row owned-ID counts `u32[rows]` of the compact form, which merges
    /// the peer's partial in place and so leaves the last own partition free.
    pub fn counts(&self, rows: u32, splits: u32) -> usize {
        self.own_o + splits as usize * rows as usize * (HEADS * LATENT) as usize * 4
    }
}

/// Offsets (from the work region) of one merge-form owner's buffers beyond
/// the pair (`ChainMerge`): every rank attends all [`TOTAL_HEADS`] over the
/// tokens it stores, as two [`HEADS`]-head groups in one paired launch, and
/// the ranks all-gather queries and merged partials instead of swapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainLayout {
    /// Gathered queries: rank `o`'s `[rows, local heads, 512]` BF16 at
    /// `q_all + o * q_part`.
    pub q_all: usize,
    pub q_part: usize,
    /// The queries regrouped `[2][rows, 32, 512]` BF16 (heads in order).
    pub q_groups: usize,
    /// Localized IDs `[rows, 2051]` i32 and their per-row counts `u32[rows]`.
    pub ids: usize,
    pub counts: usize,
    /// Both groups' `splits` FP32 partials `[2][splits][rows, 32, 512]` and
    /// LSEs `[2][splits][rows, 32]`.
    pub part_o: usize,
    pub part_lse: usize,
    /// Gathered merged partials: rank `o`'s at `gathered + o * part`, each
    /// group's `[rows, 32, 512]` FP32 then its `[rows, 32]` LSE
    /// ([`Self::group_o`], [`Self::group_lse`]).
    pub gathered: usize,
    pub part: usize,
    /// Bytes of one rank's merged partials (the gathered part's payload).
    pub part_bytes: usize,
    /// This rank's heads' partial from each rank `[world][rows, local, 512]`
    /// FP32 and LSE `[world][rows, local]`, the final merge's partitions.
    pub mine_o: usize,
    pub mine_lse: usize,
    /// The final merge's LSE `[rows, local]`.
    pub out_lse: usize,
    pub total: usize,
}

impl ChainLayout {
    pub fn new(rows: u32, world: usize) -> Self {
        let rows = rows as usize;
        let local = local_heads(world) as usize;
        let latent = LATENT as usize;
        let splits = MERGE_SPLITS as usize;
        let group_o = rows * HEADS as usize * latent * 4;
        let group_lse = rows * HEADS as usize * 4;
        let q_part = align(rows * local * latent * 2);
        let part_bytes = align(group_o) + align(group_lse) + align(group_o) + group_lse;
        let part = align(part_bytes);
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        let q_all = take(world * q_part);
        let q_groups = take(2 * rows * HEADS as usize * latent * 2);
        let ids = take(rows * WIDTH as usize * 4);
        let counts = take(rows * 4);
        let part_o = take(2 * splits * group_o);
        let part_lse = take(2 * splits * group_lse);
        let gathered = take(world * part);
        let mine_o = take(world * rows * local * latent * 4);
        let mine_lse = take(world * rows * local * 4);
        let out_lse = take(rows * local * 4);
        Self {
            q_all,
            q_part,
            q_groups,
            ids,
            counts,
            part_o,
            part_lse,
            gathered,
            part,
            part_bytes,
            mine_o,
            mine_lse,
            out_lse,
            total: at,
        }
    }

    /// Offset within a rank's gathered part of group `g`'s FP32 partial.
    pub fn group_o(rows: u32, g: usize) -> usize {
        let group_o = rows as usize * (HEADS * LATENT) as usize * 4;
        let group_lse = rows as usize * HEADS as usize * 4;
        g * (align(group_o) + align(group_lse))
    }

    /// Offset within a rank's gathered part of group `g`'s LSEs.
    pub fn group_lse(rows: u32, g: usize) -> usize {
        Self::group_o(rows, g) + align(rows as usize * (HEADS * LATENT) as usize * 4)
    }
}
