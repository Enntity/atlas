// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ verify phase 1c: upload the K-entry attention metadata block and
//! build its [`AttnMetadataDev`], split out of `verify_d.rs`.

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::layer::AttnMetadataDev;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn kgamma_upload_meta(
        &self,
        seq: &SequenceState,
        k: usize,
        bs: usize,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        // 1c. Upload K-entry attention metadata. Layout in scratch (after
        // mtp metadata reservation): positions[K*4] | slots[K*8] | seq_lens[K*4]
        // | block_table[K*max_blocks*4]. Need K*16 + K*max_blocks*4 bytes per
        // call — at K=17 max_blocks=512 that's ~36 KB which fits comfortably
        // in the scratch arena (offset 32768).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        // SAFETY: `positions` is built one line above by `(0..k).map(..)
        // .collect()`, so `positions.len() == k` exactly (collect on a
        // `Range` yields one element per step) — `k * 4 == size_of_val(&
        // positions[..])`. Every element is written by the collect, so no
        // uninitialised spare capacity is read. `u32` is POD.
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 256-byte gap mirrors K=4 layout for ABI compatibility with
        // attention kernels that index meta_base + fixed offsets.
        // SAFETY: `slots` is `vec![0i64; k]`, so its LEN (not merely its
        // capacity) is `k` and every element is zero-initialised before the
        // `for t in 0..k` loop overwrites it — `k * 8 == size_of_val(&
        // slots[..])`, with no read past `len` into spare capacity.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
        // SAFETY: `seq_lens` is `(0..k).map(..).collect()` on the line above,
        // so `seq_lens.len() == k` and `k * 4 == size_of_val(&seq_lens[..])`;
        // all `k` elements are initialised by the collect. `i32` is POD.
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf = vec![0i32; needed];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        // SAFETY: `bt_buf` is `vec![0i32; needed]` on the line above, so its
        // LEN is `needed` and `needed * 4 == size_of_val(&bt_buf[..])` — the
        // read stops at `len`, never in the `Vec`'s spare capacity. The
        // zero-init at construction covers the tail the `for row in 0..k`
        // fill leaves untouched when `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // Upload uniform LoRA slots before capture; +128 gap holds K<=32.
        debug_assert!(k <= 32, "γ verify seq_slot +128 gap holds K ≤ 32");
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, k, meta_base.offset(128), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
            seq_slot,
            moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
        };
        Ok(metadata)
    }
}
