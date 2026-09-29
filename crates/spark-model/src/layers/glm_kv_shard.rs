// SPDX-License-Identifier: AGPL-3.0-only
//! Token-sharded GLM MLA latents over the TP pair (`ATLAS_GLM_KV_SHARD=1`).
//!
//! Each physical KV block's latents live on one rank
//! (`spark_runtime::kv_cache::LatentShard`); the semantic-index keys stay
//! replicated, so both ranks still select the same top-k tokens locally with
//! no exchange. Attention then runs in one of two exact forms
//! (docs/glm-kv-shard.md):
//!
//! - **merge** (at most [`MERGE_MAX_ROWS`] rows: decode, DFlash verify,
//!   short prefill tails): the ranks swap their heads' absorbed queries, each
//!   attends ALL heads over the selected tokens it stores, the ranks swap the
//!   FP32 partial outputs + LSEs of each other's heads, and each merges its
//!   own heads' two partials (flash-decoding LSE merge);
//! - **view** (larger prefill chunks): each rank assembles the sequence's
//!   whole latent history — its own blocks plus the peer's, exchanged — in a
//!   scratch view the unchanged kernels read through an identity table.
//!
//! This module holds the policy, the scratch layout both forms carve from
//! the shard's one allocation, and the pair exchange.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheConfig, LatentShard, LatentShardSpec};

/// Heads per rank (GLM-5.3: 64 heads over TP2) = one sparse-kernel launch.
pub const HEADS: u32 = 32;
/// Latent width (`kv_lora_rank`).
pub const LATENT: u32 = 512;
/// Selected IDs per row (`index_topk + index_kpool - 1`).
pub const WIDTH: u32 = 2051;
/// Owners up to this many rows take the merge form; larger ones the view.
pub const MERGE_MAX_ROWS: usize = 64;
/// At most 15 local splits, so the final merge (plus the peer's partial)
/// stays within `glm_sparse_decode_split_merge`'s 16 partitions.
pub const MAX_SPLITS: u32 = 15;
/// Blocks per view-exchange round (the send and receive pieces).
pub const PIECE_BLOCKS: usize = 2048;
/// Tokens past `max_seq_len` a view must still hold (verify rows).
const VIEW_SLACK_TOKENS: usize = 64;

fn parse(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be 0 or 1, got {other:?}"),
    }
}

fn flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(name, None),
        Err(error) => Err(error.into()),
    }
}

/// `ATLAS_GLM_KV_SHARD=1`: store each block's latents on one rank.
pub fn requested() -> Result<bool> {
    flag("ATLAS_GLM_KV_SHARD")
}

/// `ATLAS_GLM_KV_SHARD_CHECK=1`: before each sharded attention, confirm
/// both ranks hold the same block table (one host sync + exchange each).
pub fn check_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_GLM_KV_SHARD_CHECK").unwrap_or(false))
}

fn align(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

/// CTAs (rows x partitions) past which more partitions only cost scratch:
/// four waves of GB10's 48 SMs.
const MAX_PARTIAL_CTAS: u32 = 192;

/// Local sparse partitions of a merge-form owner of `rows` rows: the
/// verify split count, capped so the scratch stays bounded.
pub fn merge_splits(rows: u32) -> u32 {
    crate::layers::ops::sparse_split_count(rows, HEADS, WIDTH)
        .min(MAX_SPLITS)
        .min((MAX_PARTIAL_CTAS / rows.max(1)).max(1))
}

/// Offsets (from the work region) of one merge-form owner's buffers.
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
}

/// The shard scratch carved by both attention forms and the cache write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    /// Assembled view: `view_blocks` blocks of `block_bytes`.
    pub view: usize,
    /// `u32[view_blocks]` lists: own local slots, their logical positions,
    /// and the peer's logical positions.
    pub mine_slot: usize,
    pub mine_dst: usize,
    pub peer_dst: usize,
    /// Table-hash exchange words (`ATLAS_GLM_KV_SHARD_CHECK`).
    pub check: usize,
    /// Local write slots `i64[write_rows]`.
    pub slots: usize,
    /// View send/receive pieces, or one merge-form owner's buffers.
    pub work: usize,
    /// Bytes of one view piece.
    pub piece_bytes: usize,
    pub block_bytes: usize,
    pub total: usize,
}

impl ScratchLayout {
    pub fn new(view_blocks: usize, write_rows: usize, block_bytes: usize) -> Self {
        let piece_bytes = PIECE_BLOCKS * block_bytes;
        let merge = (1..=MERGE_MAX_ROWS as u32)
            .map(|rows| MergeLayout::new(rows, merge_splits(rows)).total)
            .max()
            .unwrap_or(0);
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        let view = take(view_blocks * block_bytes);
        let mine_slot = take(view_blocks * 4);
        let mine_dst = take(view_blocks * 4);
        let peer_dst = take(view_blocks * 4);
        let check = take(16);
        let slots = take(write_rows * 8);
        let work = take((2 * align(piece_bytes)).max(merge));
        Self {
            view,
            mine_slot,
            mine_dst,
            peer_dst,
            check,
            slots,
            work,
            piece_bytes,
            block_bytes,
            total: at,
        }
    }

    /// The layout of `shard`'s scratch over `config`'s widest latent block.
    pub fn of(shard: &LatentShard, config: &KvCacheConfig) -> Result<Self> {
        let layout = Self::new(
            shard.spec.view_blocks,
            shard.spec.write_rows,
            latent_block_bytes(config),
        );
        ensure!(
            layout.total == shard.spec.scratch_bytes && !shard.scratch.is_null(),
            "GLM KV shard scratch disagrees with its layout"
        );
        Ok(layout)
    }
}

/// The widest per-layer latent (K) block.
pub fn latent_block_bytes(config: &KvCacheConfig) -> usize {
    (0..config.num_layers)
        .map(|layer| config.k_block_bytes_for_layer(layer))
        .max()
        .unwrap_or(0)
}

/// The shard spec of `rank` for sequences up to `max_seq_len` tokens and
/// cache writes of up to `write_rows` rows.
pub fn spec(
    rank: usize,
    config: &KvCacheConfig,
    max_seq_len: usize,
    write_rows: usize,
) -> LatentShardSpec {
    let view_blocks = (max_seq_len + VIEW_SLACK_TOKENS).div_ceil(config.block_size);
    let layout = ScratchLayout::new(view_blocks, write_rows, latent_block_bytes(config));
    LatentShardSpec {
        rank,
        world: 2,
        scratch_bytes: layout.total,
        view_blocks,
        write_rows,
    }
}

/// Exchange `bytes` with the other rank on `stream`: send from `send`,
/// receive the peer's equally sized payload into `recv`. Both ranks must
/// call it in the same order with the same `bytes`. The RDMA pair serves it
/// when up (and not capturing); NCCL point-to-point otherwise.
pub fn pair_exchange(
    comm: &dyn spark_comm::CommBackend,
    send: DevicePtr,
    recv: DevicePtr,
    bytes: usize,
    stream: u64,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    if comm.exchange_async(send.0, recv.0, bytes, false, stream)? {
        return Ok(());
    }
    let peer = 1 - comm.rank();
    comm.group_start()?;
    comm.send_to(send.0, bytes, peer, stream)?;
    comm.recv_from(recv.0, bytes, peer, stream)?;
    comm.group_end()
}

/// FNV-1a of a block table, the `ATLAS_GLM_KV_SHARD_CHECK` fingerprint.
pub fn table_hash(table: &[u32]) -> u64 {
    table
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64 ^ table.len() as u64, |h, byte| {
            (h ^ byte as u64).wrapping_mul(0x0100_0000_01b3)
        })
}

#[cfg(test)]
#[path = "glm_kv_shard_tests.rs"]
mod tests;
