// SPDX-License-Identifier: AGPL-3.0-only
//! Dense, hidden-preserving byte sentinels, not MLA/KDA/CUDA numerical oracles.
use super::*;

pub(super) struct Body {
    pub(super) record: Arc<Recorder>,
    pub(super) target: bool,
}

impl Body {
    fn target_rows(&self, h: DevicePtr, n: usize, pos: usize, s: u64) -> Result<()> {
        self.record.event(Event::Target(n, pos, s))?;
        for row in 0..n {
            let dst = h.offset(row * ROW_BYTES);
            let source = self.record.read_span(dst, 1)[0];
            let sentinel = source.wrapping_add(0x20).wrapping_add((pos + row) as u8);
            self.record
                .inner
                .copy_h2d(&vec![sentinel; ROW_BYTES], dst)?;
        }
        Ok(())
    }

    fn write_row(&self, h: DevicePtr, cache: &PagedKvCache, slot: i64) -> Result<()> {
        ensure!(slot >= 0, "negative actual KV slot");
        let block = slot as usize / cache.block_size();
        ensure!(block < cache.num_blocks(), "actual KV block outside pool");
        let offset = slot as usize % cache.block_size() * 1024;
        let bytes = self.record.read_span(h, 1024);
        for ptr in [
            cache.k_cache_ptr(0, block as u32),
            cache.v_cache_ptr(0, block as u32),
        ] {
            self.record.inner.copy_h2d(&bytes, ptr.offset(offset))?;
        }
        Ok(())
    }
}

impl TransformerLayer for Body {
    fn supports_glm_pair_verify(&self) -> bool {
        self.target
    }
    fn validate_glm_pair_verify(
        &self,
        ctx: &ForwardContext,
        mode: crate::layer::glm_pair_verify::GlmPairFfn,
        stream: u64,
    ) -> Result<()> {
        ensure!(self.target, "private fixture body is not a paired target");
        let workspace = crate::layer::glm_pair_verify::GlmPairWorkspace::new(ctx, mode)?;
        workspace.validate_context(ctx, stream)
    }
    fn decode_glm_pair_verify(
        &self,
        owners: [crate::layer::glm_pair_verify::GlmPairLayerInput<'_>; 2],
        _: &mut PagedKvCache,
        workspace: &mut crate::layer::glm_pair_verify::GlmPairWorkspace<'_>,
        contexts: [&ForwardContext; 2],
        stream: u64,
    ) -> Result<()> {
        // Same byte-sentinel target as the existing scalar/K5 tests, now using
        // two actual owner descriptors. This is NOT a KDA/MLA numerical oracle.
        ensure!(self.target, "private fixture body is not a paired target");
        workspace.begin_layer(0, &owners, contexts, stream)?;
        for owner in owners {
            self.target_rows(owner.hidden, 5, owner.positions[0], stream)?;
        }
        workspace.finish_layer();
        Ok(())
    }
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        if self.record.record_state_allocations.load(Ordering::Relaxed) {
            self.record.event(Event::AllocState(self.target))?;
        }
        Ok(Box::new(EmptyLayerState))
    }
    fn supports_mla_kv_only(&self) -> bool {
        !self.target
    }
    fn prefill(
        &self,
        h: DevicePtr,
        _: DevicePtr,
        rows: usize,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        position: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: usize,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(self.target, "private body must use real KV-only entry");
        self.target_rows(h, rows, position, stream)
    }
    fn decode(
        &self,
        h: DevicePtr,
        _: DevicePtr,
        _: &mut dyn LayerState,
        cache: &mut PagedKvCache,
        position: usize,
        blocks: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.target {
            return self.target_rows(h, 1, position, stream);
        }
        self.record.event(Event::Body(position, stream))?;
        let meta = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("missing private metadata"))?;
        ensure!(
            meta.num_seqs == 1 && meta.max_blocks_per_seq as usize == blocks.len(),
            "private metadata owner extent mismatch"
        );
        let slot = i64::from_ne_bytes(self.record.read_span(meta.slot, 8).try_into().unwrap());
        let rows = i32::from_ne_bytes(self.record.read_span(meta.seq_len, 4).try_into().unwrap());
        ensure!(
            rows > 0 && rows as usize == position + 1,
            "private cursor mismatch"
        );
        let map: Vec<_> = self
            .record
            .read_span(meta.block_table, blocks.len() * 4)
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(map == *blocks, "private uploaded block ownership mismatch");
        let block = *blocks
            .get(position / cache.block_size())
            .ok_or_else(|| anyhow::anyhow!("private logical row not mapped"))?;
        let canonical = block as usize * cache.block_size() + position % cache.block_size();
        ensure!(
            slot >= 0 && slot as usize == canonical,
            "private slot differs from live map"
        );
        self.write_row(h, cache, slot)
    }
    fn prefill_mla_kv_only(
        &self,
        h: DevicePtr,
        rows: usize,
        cache: &mut PagedKvCache,
        slots: DevicePtr,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        ensure!(!self.target, "target cannot act as private KV body");
        let slots: Vec<_> = self
            .record
            .read_span(slots, rows * 8)
            .chunks_exact(8)
            .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        self.record.event(Event::Kv(slots.clone(), stream))?;
        for (row, slot) in slots.into_iter().enumerate() {
            self.write_row(h.offset(row * ROW_BYTES), cache, slot)?;
        }
        Ok(true)
    }
}
