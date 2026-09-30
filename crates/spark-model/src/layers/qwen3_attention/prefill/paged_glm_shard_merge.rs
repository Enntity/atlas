// SPDX-License-Identifier: AGPL-3.0-only

//! The launches and exchanges of one merge-form owner over token-sharded
//! latents (`layers::glm_kv_shard`), free of the layer so their order and
//! arguments are testable.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, LatentShard};

use super::{ShardRows, owner_rank};
use crate::layers::glm_kv_shard::{self as shard, HEADS, LATENT, MergeLayout, MergeTuning, WIDTH};
use crate::layers::ops;

/// One layer's merge form on this rank.
pub(super) struct ShardMerge<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub comm: &'a dyn CommBackend,
    pub config: &'a ModelConfig,
    pub shard: LatentShard,
    /// The merge work region of the shard scratch: offset and bytes.
    pub work: usize,
    pub work_bytes: usize,
    pub dtype: KvCacheDtype,
    /// This rank's latent pool of the layer.
    pub pool: DevicePtr,
    pub scale: f32,
    /// `overlap` already excludes graph capture.
    pub tuning: MergeTuning,
}

impl ShardMerge<'_> {
    /// This rank's heads' attention over the selected tokens of both ranks
    /// into `output` (`[rows, 32, 512]` BF16).
    pub(super) fn run(&self, a: ShardRows, output: DevicePtr, stream: u64) -> Result<()> {
        let (gpu, comm, s) = (self.gpu, self.comm, &self.shard);
        let MergeTuning { compact, overlap } = self.tuning;
        let splits = shard::merge_splits(a.rows);
        let m = MergeLayout::new(a.rows, splits);
        ensure!(
            m.total <= self.work_bytes,
            "GLM KV shard merge scratch overflow"
        );
        let at = |offset: usize| s.scratch.offset(self.work + offset);
        let rows = a.rows as usize;
        let part = rows * (HEADS * LATENT) as usize * 4;
        let lse = rows * HEADS as usize * 4;
        // Compact: each row's owned IDs first, their number in `counts`.
        let counts = compact.then(|| at(m.counts(a.rows, splits)));
        // Selected IDs -> local token IDs; the peer's tokens become -1.
        let localize = || {
            ops::glm_kv_shard_localize(
                gpu,
                a.selected,
                at(m.ids),
                counts,
                a.block_table,
                a.rows,
                WIDTH,
                16,
                owner_rank(s),
                a.causal_start,
                stream,
            )
        };

        // 1. Swap queries: the peer's heads attend over this rank's tokens too.
        // 2. Localize the selection (beside the swap when overlapping).
        let queries = (a.query, at(m.q_peer), rows * (HEADS * LATENT) as usize * 2);
        if a.queries_swapped {
            localize()?;
        } else if overlap {
            shard::overlapped_exchange(gpu, comm, s.lane, queries, stream, localize)?;
        } else {
            shard::pair_exchange(comm, queries.0, queries.1, queries.2, stream)?;
            localize()?;
        }
        let tc = |query: DevicePtr| ops::GlmSparsePrefillTc {
            config: self.config,
            dtype: self.dtype,
            identical_kv_latent: true,
            query,
            k_cache: self.pool,
            v_cache: self.pool,
            indices: at(m.ids),
            output,
            block_table: s.identity,
            rows: a.rows,
            heads: HEADS,
            head_dim: LATENT,
            index_width: WIDTH,
            block_size: 16,
            scale: self.scale,
        };
        // 3. The peer's heads over this rank's tokens: one FP32 partial + LSE.
        let send = at(m.send);
        let peer_heads = tc(at(m.q_peer));
        if splits == 1 {
            let send_lse = send.offset(part);
            ops::launch_sparse_partials(gpu, &peer_heads, 1, counts, send, send_lse, stream)?;
        } else {
            let (po, pl) = (at(m.peer_o), at(m.peer_lse));
            ops::launch_sparse_partials(gpu, &peer_heads, splits, counts, po, pl, stream)?;
            ops::launch_merge_f32(gpu, po, pl, send, send.offset(part), a.rows, splits, stream)?;
        }
        // 4. Swap the partials; this rank's heads' own partitions (beside the
        //    swap when overlapping) need nothing from the peer.
        let recv = at(m.recv);
        let partials = (send, recv, MergeLayout::partial_bytes(a.rows));
        let (own_o, own_lse) = (at(m.own_o), at(m.own_lse));
        let own = || {
            ops::launch_sparse_partials(gpu, &tc(a.query), splits, counts, own_o, own_lse, stream)
        };
        if overlap {
            shard::overlapped_exchange(gpu, comm, s.lane, partials, stream, own)?;
        } else {
            shard::pair_exchange(comm, partials.0, partials.1, partials.2, stream)?;
            own()?;
        }
        // 5. One LSE merge to BF16 with the peer's partial as partition
        //    `splits`: where it landed (compact), else copied behind the own.
        let out_lse = at(m.out_lse);
        if compact {
            return ops::launch_merge_extra(
                gpu, own_o, own_lse, output, out_lse, a.rows, splits, recv, stream,
            );
        }
        let tail = splits as usize;
        gpu.copy_d2d_async(recv, own_o.offset(tail * part), part, stream)?;
        gpu.copy_d2d_async(recv.offset(part), own_lse.offset(tail * lse), lse, stream)?;
        ops::launch_merge(
            gpu,
            ops::merge_kernel(gpu)?,
            own_o,
            own_lse,
            output,
            out_lse,
            a.rows,
            a.rows * HEADS,
            splits + 1,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "paged_glm_shard_merge_tests.rs"]
mod tests;
