// SPDX-License-Identifier: AGPL-3.0-only

//! GLM MLA attention over token-sharded latents (`ATLAS_GLM_KV_SHARD=1`, see
//! `layers::glm_kv_shard`): the merge form for few-row owners, the view form
//! for prefill chunks, and the owner-only cache-write slots.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, LatentShard, PagedKvCache};

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::glm_kv_shard::{
    self as shard, HEADS, LATENT, MergeLayout, ScratchLayout, WIDTH,
};
use crate::layers::ops;

/// One merge-form owner: `rows` query rows of one sequence.
#[derive(Clone, Copy)]
pub(in crate::layers::qwen3_attention) struct ShardRows {
    /// This rank's heads' absorbed queries `[rows, 32, 512]` BF16.
    pub query: DevicePtr,
    /// Selected IDs `[rows, 2051]`; `None` attends causally, row `r` to
    /// tokens `[0, causal_start + r + 1)`.
    pub selected: Option<DevicePtr>,
    pub causal_start: u32,
    /// The sequence's block table (logical → physical).
    pub block_table: DevicePtr,
    pub rows: u32,
    /// The last row's causal extent when the host knows it (table check).
    pub end: Option<usize>,
}

fn owner_rank(s: &LatentShard) -> ops::ShardRank {
    ops::ShardRank {
        rank: s.spec.rank as u32,
        world: s.spec.world as u32,
    }
}

/// The shard, the pair communicator of this rank and the scratch layout.
fn pair<'a>(
    kv_cache: &PagedKvCache,
    ctx: &ForwardContext<'a>,
) -> Result<(LatentShard, &'a dyn CommBackend, ScratchLayout)> {
    let s = kv_cache
        .latent_shard()
        .context("GLM KV shard attention on an unsharded cache")?;
    let comm = ctx
        .comm
        .filter(|c| c.world_size() == 2 && c.rank() == s.spec.rank)
        .context("GLM KV shard attention needs this rank's TP-pair communicator")?;
    Ok((s, comm, ScratchLayout::of(&s, kv_cache.config())?))
}

/// Logical blocks `[0, n)` of a device block table, read on `stream`.
fn read_table(gpu: &dyn GpuBackend, table: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h_on_stream(table, &mut raw, stream)?;
    Ok(raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Qwen3AttentionLayer {
    fn glm_shard_geometry(&self, kv_cache: &PagedKvCache) -> Result<()> {
        ensure!(
            matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
                && kv_cache.block_size() == 16
                && self.mla.as_ref().is_some_and(|m| {
                    m.kv_lora_rank == LATENT as usize && m.rope == 0 && m.glm_indexer.is_some()
                }),
            "GLM KV shard needs the NoPE-512 semantic-index MLA on a 16-token BF16/fp8_g128 cache"
        );
        Ok(())
    }

    /// `ATLAS_GLM_KV_SHARD_CHECK=1`: this rank's table must place every
    /// logical block on the rank its index names, and both ranks must see
    /// the same number of blocks (their physical ids legitimately differ).
    fn glm_shard_check(
        &self,
        s: &LatentShard,
        comm: &dyn CommBackend,
        gpu: &dyn GpuBackend,
        words: DevicePtr,
        table: &[u32],
        stream: u64,
    ) -> Result<()> {
        s.check_table(table)
            .with_context(|| format!("attention layer {}", self.attn_layer_idx))?;
        let blocks = table.len() as u64;
        gpu.copy_h2d_async(&blocks.to_le_bytes(), words, stream)?;
        shard::pair_exchange(comm, words, words.offset(8), 8, stream)?;
        let mut peer = [0u8; 8];
        gpu.copy_d2h_on_stream(words.offset(8), &mut peer, stream)?;
        let peer = u64::from_le_bytes(peer);
        ensure!(
            peer == blocks,
            "GLM KV shard: rank {} attends {blocks} blocks, its peer {peer}, at attention layer {}",
            s.spec.rank,
            self.attn_layer_idx
        );
        Ok(())
    }

    /// Merge form: this rank's heads' attention over the selected tokens of
    /// BOTH ranks into `output` (`[rows, 32, 512]` BF16). Each rank attends
    /// every head over the tokens it stores; the peer's heads' partial is
    /// merged to one FP32 partial and swapped, then joins this rank's own
    /// partitions in one exact LSE merge.
    pub(in crate::layers::qwen3_attention) fn glm_shard_merge_attention(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        a: ShardRows,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.glm_shard_geometry(kv_cache)?;
        let (s, comm, layout) = pair(kv_cache, ctx)?;
        ensure!(
            (1..=shard::MERGE_MAX_ROWS as u32).contains(&a.rows),
            "GLM KV shard merge form takes 1..={} rows, got {}",
            shard::MERGE_MAX_ROWS,
            a.rows
        );
        let gpu = ctx.gpu;
        if shard::check_requested()
            && let Some(end) = a.end
        {
            let table = read_table(gpu, a.block_table, end.div_ceil(16), stream)?;
            let words = s.scratch.offset(layout.check);
            self.glm_shard_check(&s, comm, gpu, words, &table, stream)?;
        }
        let splits = shard::merge_splits(a.rows);
        let m = MergeLayout::new(a.rows, splits);
        ensure!(
            layout.work + m.total <= layout.total,
            "GLM KV shard merge scratch overflow"
        );
        let at = |offset: usize| s.scratch.offset(layout.work + offset);
        let rows = a.rows as usize;
        let part = rows * (HEADS * LATENT) as usize * 4;
        let lse = rows * HEADS as usize * 4;

        // 1. Swap queries: the peer's heads attend over this rank's tokens too.
        let q_bytes = rows * (HEADS * LATENT) as usize * 2;
        shard::pair_exchange(comm, a.query, at(m.q_peer), q_bytes, stream)?;
        // 2. Selected IDs -> local token IDs; the peer's tokens become -1.
        ops::glm_kv_shard_localize(
            gpu,
            a.selected,
            at(m.ids),
            a.block_table,
            a.rows,
            WIDTH,
            16,
            owner_rank(&s),
            a.causal_start,
            stream,
        )?;
        let pool = kv_cache.latent_pool_ptr(self.attn_layer_idx);
        let hd = self.mla.as_ref().map_or(0, |m| m.nope) as u32;
        let tc = |query: DevicePtr| ops::GlmSparsePrefillTc {
            config: ctx.config,
            dtype: self.kv_dtype,
            identical_kv_latent: true,
            query,
            k_cache: pool,
            v_cache: pool,
            indices: at(m.ids),
            output,
            block_table: s.identity,
            rows: a.rows,
            heads: HEADS,
            head_dim: LATENT,
            index_width: WIDTH,
            block_size: 16,
            scale: self.effective_attn_scale(hd),
        };
        // 3. The peer's heads over this rank's tokens: one FP32 partial + LSE.
        let send = at(m.send);
        if splits == 1 {
            ops::launch_sparse_partials(
                gpu,
                &tc(at(m.q_peer)),
                1,
                send,
                send.offset(part),
                stream,
            )?;
        } else {
            let peer_heads = tc(at(m.q_peer));
            let (po, pl) = (at(m.peer_o), at(m.peer_lse));
            ops::launch_sparse_partials(gpu, &peer_heads, splits, po, pl, stream)?;
            ops::launch_merge_f32(gpu, po, pl, send, send.offset(part), a.rows, splits, stream)?;
        }
        let recv = at(m.recv);
        shard::pair_exchange(comm, send, recv, MergeLayout::partial_bytes(a.rows), stream)?;
        // 4. This rank's heads: its own partitions, then the peer's partial
        //    as partition `splits`, in one LSE merge to BF16.
        let (own_o, own_lse) = (at(m.own_o), at(m.own_lse));
        ops::launch_sparse_partials(gpu, &tc(a.query), splits, own_o, own_lse, stream)?;
        let tail = splits as usize;
        gpu.copy_d2d_async(recv, own_o.offset(tail * part), part, stream)?;
        gpu.copy_d2d_async(recv.offset(part), own_lse.offset(tail * lse), lse, stream)?;
        ops::launch_merge(
            gpu,
            ops::merge_kernel(gpu)?,
            own_o,
            own_lse,
            output,
            at(m.out_lse),
            a.rows,
            a.rows * HEADS,
            splits + 1,
            stream,
        )
    }

    /// [`Self::glm_shard_merge_attention`] for one few-row prefill or verify
    /// owner: `selected` is its `(IDs, width)` selection, `None` when the
    /// whole causal history is selected.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_shard_merge_owner(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        o: &super::GlmChunkOwner,
        query: DevicePtr,
        selected: Option<(DevicePtr, u32)>,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if let Some((_, width)) = selected {
            ensure!(
                width == WIDTH,
                "GLM KV shard expects {WIDTH} selected IDs, got {width}"
            );
        }
        let rows = ShardRows {
            query,
            selected: selected.map(|(ids, _)| ids),
            causal_start: o.seq_len_start as u32,
            block_table: o.meta.block_table,
            rows: o.rows as u32,
            end: Some(o.seq_len_start + o.rows),
        };
        self.glm_shard_merge_attention(kv_cache, ctx, rows, output, stream)
    }

    /// View form: assemble the sequence's latents for logical tokens
    /// `[0, end)` — this rank's blocks copied locally, the peer's exchanged
    /// in pieces — and return `(view, identity table)` for the unchanged
    /// kernels, which read the view in this layer's cache dtype.
    pub(in crate::layers::qwen3_attention) fn glm_shard_assemble_view(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        block_table: DevicePtr,
        end: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        self.glm_shard_geometry(kv_cache)?;
        let (s, comm, layout) = pair(kv_cache, ctx)?;
        let blocks = end.div_ceil(kv_cache.block_size());
        ensure!(
            blocks > 0 && blocks <= s.spec.view_blocks,
            "GLM KV shard view holds {} blocks, the owner needs {blocks}",
            s.spec.view_blocks
        );
        let gpu = ctx.gpu;
        let table = read_table(gpu, block_table, blocks, stream)?;
        if shard::check_requested() {
            let words = s.scratch.offset(layout.check);
            self.glm_shard_check(&s, comm, gpu, words, &table, stream)?;
        }
        // Validates the residue invariant: a violation would pair the wrong
        // blocks across the ranks.
        let plan = s.plan(&table)?;
        let at = |offset: usize| s.scratch.offset(offset);
        let (mine_slot, mine_dst, peer_dst) = (
            at(layout.mine_slot),
            at(layout.mine_dst),
            at(layout.peer_dst),
        );
        gpu.copy_h2d_async(&u32_bytes(&plan.mine_slot), mine_slot, stream)?;
        gpu.copy_h2d_async(&u32_bytes(&plan.mine_logical), mine_dst, stream)?;
        gpu.copy_h2d_async(&u32_bytes(&plan.peer_logical), peer_dst, stream)?;
        let pool = kv_cache.latent_pool_ptr(self.attn_layer_idx);
        let bytes = kv_cache
            .config()
            .k_block_bytes_for_layer(self.attn_layer_idx);
        ensure!(bytes <= layout.block_bytes, "GLM KV shard view stride");
        let view = at(layout.view);
        let (mine, peer) = (plan.mine_slot.len(), plan.peer_logical.len());
        ops::glm_kv_shard_copy_blocks(
            gpu,
            pool,
            Some(mine_slot),
            view,
            Some(mine_dst),
            mine,
            bytes,
            stream,
        )?;
        // Both ranks derive the same rounds from the same table.
        let rounds = mine.max(peer);
        let send = at(layout.work);
        let recv = send.offset(layout.piece_bytes.next_multiple_of(256));
        let mut start = 0usize;
        while start < rounds {
            let n = shard::PIECE_BLOCKS.min(rounds - start);
            let out_n = mine.saturating_sub(start).min(n);
            let in_n = peer.saturating_sub(start).min(n);
            let from = Some(mine_slot.offset(start * 4));
            ops::glm_kv_shard_copy_blocks(gpu, pool, from, send, None, out_n, bytes, stream)?;
            shard::pair_exchange(comm, send, recv, n * bytes, stream)?;
            let to = Some(peer_dst.offset(start * 4));
            ops::glm_kv_shard_copy_blocks(gpu, recv, None, view, to, in_n, bytes, stream)?;
            start += n;
        }
        Ok((view, s.identity))
    }

    /// This rank's local write slots for `num_tokens` global `slot`s (`-1`
    /// where the peer stores the block).
    pub(in crate::layers::qwen3_attention) fn glm_shard_local_slots(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
        slot: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let s = kv_cache
            .latent_shard()
            .context("GLM KV shard write on an unsharded cache")?;
        let layout = ScratchLayout::of(&s, kv_cache.config())?;
        ensure!(
            num_tokens as usize <= s.spec.write_rows,
            "GLM KV shard write of {num_tokens} rows exceeds {}",
            s.spec.write_rows
        );
        let local = s.scratch.offset(layout.slots);
        let bs = kv_cache.block_size() as u32;
        ops::glm_kv_shard_map_slots(gpu, slot, local, num_tokens, bs, owner_rank(&s), stream)?;
        Ok(local)
    }
}
