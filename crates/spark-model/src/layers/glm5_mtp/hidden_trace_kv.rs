// SPDX-License-Identifier: AGPL-3.0-only
//! First-attempt bounded private-KV observation; never changes cache storage.

use super::*;

const SIDE_ROW: usize = 1024;
const SIDE_BLOCK: usize = 16 * SIDE_ROW;

struct Owner {
    pools: [DevicePtr; 2],
    capacity: usize,
    blocks: Vec<u32>,
    rows: usize,
    stream: u64,
}

impl Owner {
    fn validate(
        cache: &PagedKvCache,
        blocks: &[u32],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<([DevicePtr; 2], usize)> {
        ensure!(
            (1..=2043).contains(&rows),
            "first KV prefix outside admitted profile"
        );
        let c = cache.config();
        ensure!(
            c.num_layers == 1
                && c.block_size == 16
                && c.num_kv_heads == 1
                && c.head_dim == 512
                && c.dtype == KvCacheDtype::Bf16
                && c.layer_dims.is_empty()
                && c.cache_blocks_per_seq.is_none()
                && (c.layer_dtypes.is_empty() || c.layer_dtypes == [KvCacheDtype::Bf16]),
            "first KV pool geometry mismatch"
        );
        ensure!(
            ctx.config.model_type == "glm5_next"
                && ctx.config.hidden_size == 4096
                && ctx.config.kv_lora_rank == 512
                && ctx.config.qk_rope_head_dim == 0
                && rows + 1 <= ctx.config.index_topk,
            "first KV requires admitted dense NoPE profile"
        );
        let capacity = cache.num_blocks();
        ensure!(
            (1..=128).contains(&capacity)
                && rows + 1 <= capacity * 16
                && blocks.len() <= 128
                && blocks.len() >= (rows + 1).div_ceil(16),
            "first KV capacity mismatch"
        );
        ensure!(
            cache.dtype_for_layer(0) == KvCacheDtype::Bf16
                && cache.k_block_stride_bytes_for_layer(0) == SIDE_BLOCK
                && cache.v_block_stride_bytes_for_layer(0) == SIDE_BLOCK,
            "first KV actual stride mismatch"
        );
        for (i, block) in blocks.iter().enumerate() {
            ensure!(
                (*block as usize) < capacity && !blocks[..i].contains(block),
                "first KV invalid block map"
            );
            ensure!(
                cache.ref_count(*block) == 1,
                "first KV block must be exclusively owned"
            );
        }
        let pools = [cache.k_pool_ptr(0), cache.v_pool_ptr(0)];
        let bytes = (capacity * SIDE_BLOCK) as u64;
        let mut ends = [0; 2];
        for (i, p) in pools.iter().enumerate() {
            ensure!(p.0 != 0 && p.0 % 2 == 0, "first KV null or unaligned pool");
            ends[i] =
                p.0.checked_add(bytes)
                    .ok_or_else(|| anyhow::anyhow!("first KV pool span overflow"))?;
        }
        ensure!(
            ends[0] <= pools[1].0 || ends[1] <= pools[0].0,
            "first KV pool owners alias"
        );
        ensure!(
            !ctx.graph_capture && !ctx.gpu.stream_is_capturing(stream),
            "first KV requires eager stream"
        );
        Ok((pools, capacity))
    }

    fn pointer(&self, side: usize, row: usize, bytes: usize) -> Result<DevicePtr> {
        let offset = self.blocks[row / 16] as usize * SIDE_BLOCK + row % 16 * SIDE_ROW;
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.capacity * SIDE_BLOCK),
            "first KV read exceeds owner"
        );
        Ok(self.pools[side].offset(offset))
    }
}

/// Exists only for admitted attempt1/step0. The same bounded host allocation
/// serves prefix and appended-row copies; no prefix-sized host or GPU buffer.
pub(super) struct Probe {
    pub prefix: [u8; 32],
    pub appended: Option<[u8; 32]>,
    pub block_map: [u8; 32],
    owner: Owner,
    scratch: Box<[u8]>,
    post_spent: bool,
}

impl Probe {
    pub fn before(
        cache: &PagedKvCache,
        blocks: &[u32],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Self> {
        let (pools, capacity) = Owner::validate(cache, blocks, rows, ctx, stream)?;
        let owner = Owner {
            pools,
            capacity,
            blocks: blocks.to_vec(),
            rows,
            stream,
        };
        let mut scratch = vec![0; 2 * SIDE_BLOCK].into_boxed_slice();
        let mut hash = Self::hash(b"atlas/glm53/mtp-kv/prefix/v1\0", rows);
        for row in (0..rows).step_by(16) {
            let valid = (rows - row).min(16);
            for side in 0..2 {
                ctx.gpu.copy_d2h_on_stream(
                    owner.pointer(side, row, valid * SIDE_ROW)?,
                    &mut scratch[side * SIDE_BLOCK..side * SIDE_BLOCK + valid * SIDE_ROW],
                    stream,
                )?;
            }
            for i in 0..valid {
                for side in 0..2 {
                    let start = side * SIDE_BLOCK + i * SIDE_ROW;
                    hash.update(&scratch[start..start + SIDE_ROW]);
                }
            }
        }
        let mut map = Sha256::new();
        map.update(b"atlas/glm53/mtp-kv/block-map/v1\0");
        map.update((blocks.len() as u32).to_le_bytes());
        for block in blocks {
            map.update(block.to_le_bytes());
        }
        Ok(Self {
            prefix: hash.finalize().into(),
            appended: None,
            block_map: map.finalize().into(),
            owner,
            scratch,
            post_spent: false,
        })
    }

    pub fn after(
        &mut self,
        cache: &PagedKvCache,
        blocks: &[u32],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(!self.post_spent, "first KV appended hook already spent");
        self.post_spent = true;
        let (pools, capacity) = Owner::validate(cache, blocks, rows, ctx, stream)?;
        ensure!(
            self.owner.pools == pools
                && self.owner.capacity == capacity
                && self.owner.blocks == blocks
                && self.owner.rows == rows
                && self.owner.stream == stream,
            "first KV owner changed across body"
        );
        let mut hash = Self::hash(b"atlas/glm53/mtp-kv/appended/v1\0", rows);
        for side in 0..2 {
            let start = side * SIDE_BLOCK;
            let bytes = &mut self.scratch[start..start + SIDE_ROW];
            ctx.gpu
                .copy_d2h_on_stream(self.owner.pointer(side, rows, SIDE_ROW)?, bytes, stream)?;
            hash.update(bytes);
        }
        self.appended = Some(hash.finalize().into());
        Ok(())
    }

    fn hash(domain: &[u8], rows: usize) -> Sha256 {
        let mut hash = Sha256::new();
        hash.update(domain);
        hash.update((rows as u64).to_le_bytes());
        hash.update(512u32.to_le_bytes());
        hash.update(2u32.to_le_bytes());
        hash
    }
}

#[cfg(test)]
#[path = "hidden_trace_kv_tests.rs"]
mod tests;
