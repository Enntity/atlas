// SPDX-License-Identifier: AGPL-3.0-only
//! Checked owner slabs, disjoint from the actual MTP metadata/chunk writers.
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use crate::layers::mtp_meta::{MTP_META_HEADER_BYTES, MTP_META_OFFSET};
use anyhow::{Context, Result, ensure};

pub(super) struct GlmOwnerMetadata {
    owners: usize,
    base: usize,
    pub(super) stride: usize,
}

impl GlmOwnerMetadata {
    pub(super) fn new(
        shape: GlmOwnerBatchShape,
        max_blocks: usize,
        max_batch_tokens: usize,
        scratch_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=128).contains(&max_blocks) && max_batch_tokens >= shape.rows() * 2,
            "owner metadata block/row capacity"
        );
        let stride = (768 + 5 * max_blocks * 4).next_multiple_of(256);
        let base = if shape.owners() <= 4 {
            // Preserve the qualified E7 layout byte for byte.
            32768usize
        } else {
            // The paired private pool admits context+4 <=2048, block size16:
            // at most128 private blocks, even when the target table is smaller.
            // KV-only repair also stages i64 slots here, in actual arena-sized
            // chunks. Reserve the larger writer, not a guessed prefill limit.
            let slots = max_batch_tokens
                .checked_mul(8)
                .context("owner MTP slot staging overflow")?;
            let reserved = slots.max(MTP_META_HEADER_BYTES + 128 * 4);
            MTP_META_OFFSET
                .checked_add(reserved)
                .and_then(|end| end.checked_next_multiple_of(256))
                .context("owner MTP reserved end overflow")?
        };
        let end = base
            .checked_add(shape.owners() * stride)
            .context("owner metadata end overflow")?;
        ensure!(
            (shape.owners() > 4 || end <= MTP_META_OFFSET) && end <= scratch_bytes,
            "owner metadata scratch capacity"
        );
        Ok(Self {
            owners: shape.owners(),
            base,
            stride,
        })
    }

    pub(super) fn offset(&self, owner: usize) -> Result<usize> {
        ensure!(owner < self.owners, "owner metadata ordinal outside cohort");
        Ok(self.base + owner * self.stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_offsets_remain_exact() {
        for owners in 3..=4 {
            let shape = GlmOwnerBatchShape::new(owners).unwrap();
            let layout = GlmOwnerMetadata::new(shape, 128, owners * 10, 49152).unwrap();
            assert_eq!(layout.stride, 3328);
            for owner in 0..owners {
                assert_eq!(layout.offset(owner).unwrap(), 32768 + owner * 3328);
            }
            assert!(layout.offset(owners).is_err());
        }
    }

    #[test]
    fn wide_metadata_follows_actual_mtp_chunk_envelope() {
        for (rows, base) in [(80, 49920), (1024, 57344), (1032, 57600), (2048, 65536)] {
            for owners in 5..=8 {
                let shape = GlmOwnerBatchShape::new(owners).unwrap();
                let end = base + owners * 3328;
                let layout = GlmOwnerMetadata::new(shape, 128, rows, end).unwrap();
                let mtp_end = MTP_META_OFFSET + (MTP_META_HEADER_BYTES + 128 * 4).max(rows * 8);
                assert!(layout.offset(0).unwrap() >= mtp_end);
                for owner in 0..owners {
                    assert_eq!(layout.offset(owner).unwrap(), base + owner * 3328);
                }
                assert!(GlmOwnerMetadata::new(shape, 128, rows, end - 1).is_err());
                assert!(layout.offset(owners).is_err());
            }
        }
    }

    #[test]
    fn actual_selected_arena_quote_covers_eight_owner_metadata() {
        let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        config.hidden_size = 4096;
        config.num_experts_per_tok = 8;
        config.mrope_interleaved = false;
        // The selected prefill budget adds eight decode slots to1024 rows.
        let sizes = spark_runtime::buffers::BufferSizes::from_config(&config, 1032, 2044, 16, 8);
        assert_eq!(sizes.scratch, 111464);
        let shape = GlmOwnerBatchShape::new(8).unwrap();
        let layout = GlmOwnerMetadata::new(shape, 128, 1032, sizes.scratch).unwrap();
        assert_eq!(layout.offset(7).unwrap() + layout.stride, 84224);
        for (blocks, rows) in [(0, 1024), (129, 1024), (128, 79), (128, usize::MAX)] {
            assert!(GlmOwnerMetadata::new(shape, blocks, rows, usize::MAX).is_err());
        }
    }
}
