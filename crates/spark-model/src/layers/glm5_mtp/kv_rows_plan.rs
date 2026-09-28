// SPDX-License-Identifier: AGPL-3.0-only

//! Checked address/launch planning for the existing BF16 KV-only chain.

use crate::layers::mtp_meta::MTP_META_OFFSET;
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Caller-owned storage: the byte extent is explicit, not inferred from a pointer.
#[derive(Clone, Copy, Debug)]
pub(super) struct DeviceSpan {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

impl DeviceSpan {
    pub fn end(self) -> Result<u64> {
        self.ptr
            .0
            .checked_add(self.bytes as u64)
            .context("GLM KV span overflow")
    }

    pub fn overlaps(self, other: Self) -> Result<bool> {
        Ok(self.bytes > 0
            && other.bytes > 0
            && self.ptr.0 < other.end()?
            && other.ptr.0 < self.end()?)
    }
}

pub(super) fn use_cublas(enabled: bool, rows: usize) -> bool {
    enabled && rows > 1
}

#[derive(Debug)]
pub(super) struct KvRowsPlan {
    pub rows: usize,
    pub row_bytes: usize,
    pub chunk_rows: usize,
    pub embedding_offsets: Vec<usize>,
    pub slots: Vec<i64>,
}

impl KvRowsPlan {
    /// Validate inputs before the primer allocates blocks. All tokens, including
    /// those in later chunks, are checked before the first embedding copy.
    #[allow(clippy::too_many_arguments)]
    pub fn inputs(
        tokens: &[u32],
        source: DeviceSpan,
        hidden: usize,
        vocab: usize,
        chunk_rows: usize,
        scratch_bytes: usize,
        forbidden: &[DeviceSpan],
    ) -> Result<usize> {
        ensure!(
            !tokens.is_empty() && hidden > 0 && chunk_rows > 0,
            "empty GLM KV row request"
        );
        ensure!(
            hidden <= u32::MAX as usize / 2 && chunk_rows <= u32::MAX as usize,
            "GLM KV projection dimensions exceed ABI"
        );
        let row_bytes = hidden
            .checked_mul(2)
            .context("GLM KV hidden width overflow")?;
        let needed = tokens
            .len()
            .checked_mul(row_bytes)
            .context("GLM KV hidden span overflow")?;
        ensure!(
            source.ptr.0 != 0 && source.ptr.0.is_multiple_of(2) && source.bytes >= needed,
            "GLM KV hidden source is absent, unaligned, or too small"
        );
        source.end()?;
        for &span in forbidden {
            ensure!(
                !source.overlaps(span)?,
                "GLM KV hidden source aliases mutable storage"
            );
        }
        vocab
            .checked_mul(row_bytes)
            .context("GLM KV embedding extent overflow")?;
        ensure!(
            tokens.iter().all(|&t| (t as usize) < vocab),
            "GLM KV token outside vocabulary"
        );
        let meta_bytes = tokens
            .len()
            .min(chunk_rows)
            .checked_mul(8)
            .and_then(|n| n.checked_add(MTP_META_OFFSET))
            .context("GLM KV metadata overflow")?;
        ensure!(
            meta_bytes <= scratch_bytes,
            "GLM KV slot metadata exceeds scratch"
        );
        Ok(row_bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: &[u32],
        source: DeviceSpan,
        hidden: usize,
        vocab: usize,
        row_base: usize,
        block_size: usize,
        blocks: &[u32],
        num_blocks: usize,
        chunk_rows: usize,
        scratch_bytes: usize,
        forbidden: &[DeviceSpan],
    ) -> Result<Self> {
        let row_bytes = Self::inputs(
            tokens,
            source,
            hidden,
            vocab,
            chunk_rows,
            scratch_bytes,
            forbidden,
        )?;
        ensure!(
            block_size > 0 && num_blocks <= u32::MAX as usize,
            "invalid GLM KV block geometry"
        );
        let end = row_base
            .checked_add(tokens.len())
            .context("GLM KV destination overflow")?;
        let capacity = blocks
            .len()
            .checked_mul(block_size)
            .context("GLM KV table capacity overflow")?;
        ensure!(
            end <= capacity,
            "GLM KV destination exceeds allocated block table"
        );
        let mut seen = std::collections::HashSet::new();
        for &block in blocks {
            ensure!(
                (block as usize) < num_blocks && seen.insert(block),
                "GLM KV destination block is out of range or repeated"
            );
        }
        let slots = (row_base..end)
            .map(|row| {
                (blocks[row / block_size] as usize)
                    .checked_mul(block_size)
                    .and_then(|base| base.checked_add(row % block_size))
                    .and_then(|slot| i64::try_from(slot).ok())
                    .context("GLM KV physical slot overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rows: tokens.len(),
            row_bytes,
            chunk_rows,
            embedding_offsets: tokens.iter().map(|&t| t as usize * row_bytes).collect(),
            slots,
        })
    }

    fn chunk(&self, start: usize, rows: usize) -> Result<std::ops::Range<usize>> {
        let end = start.checked_add(rows).context("GLM KV chunk overflow")?;
        ensure!(
            rows > 0 && rows <= self.chunk_rows && end <= self.rows,
            "invalid GLM KV chunk"
        );
        Ok(start..end)
    }

    pub fn copy_embeddings(
        &self,
        gpu: &dyn GpuBackend,
        weight: DevicePtr,
        output: DevicePtr,
        start: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let range = self.chunk(start, rows)?;
        for (row, index) in range.enumerate() {
            gpu.copy_d2d_async(
                weight.offset(self.embedding_offsets[index]),
                output.offset(row * self.row_bytes),
                self.row_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    pub fn upload_slots(
        &self,
        gpu: &dyn GpuBackend,
        output: DevicePtr,
        start: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let range = self.chunk(start, rows)?;
        // The plan remains alive through the caller's per-chunk synchronize.
        let slots = &self.slots[range];
        let bytes =
            unsafe { std::slice::from_raw_parts(slots.as_ptr().cast::<u8>(), slots.len() * 8) };
        gpu.copy_h2d_async(bytes, output, stream)
    }
}

#[cfg(test)]
#[path = "kv_rows_tests.rs"]
mod tests;
