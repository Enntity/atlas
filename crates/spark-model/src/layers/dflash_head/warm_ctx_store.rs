// SPDX-License-Identifier: AGPL-3.0-only

//! The `ATLAS_DFLASH_WARM_CTX=restore` store (`warm_ctx` module docs): drafter
//! windows saved with prefill checkpoints, keyed by prefix content, and the
//! sequence-side save/restore.

use anyhow::Result;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::DflashProposerState;
use super::warm_ctx::{CapturePlan, CtxRun, store_slots_from_env};

/// Content key of a store entry: the token prefix `[0, end)` a checkpoint
/// covers (the prefix cache's key tokens) and the adapter. A collision-free
/// digest, so equal keys mean equal prefixes.
pub(crate) type StoreKey = [u8; 32];

pub(crate) fn store_key(adapter_id: u64, key_tokens: &[u32]) -> StoreKey {
    let mut h = Sha256::new();
    h.update(b"atlas/dflash-warm-ctx/v1");
    h.update(adapter_id.to_le_bytes());
    h.update((key_tokens.len() as u64).to_le_bytes());
    let mut buf = [0u8; 4096];
    for chunk in key_tokens.chunks(buf.len() / 4) {
        for (dst, t) in buf.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&t.to_le_bytes());
        }
        h.update(&buf[..chunk.len() * 4]);
    }
    h.finalize().into()
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    key: StoreKey,
    /// Positions `[lo, hi)`; row 0 of the entry's window is `lo`.
    lo: usize,
    hi: usize,
    used: u64,
}

#[derive(Default)]
struct StoreInner {
    slab: Option<DevicePtr>,
    /// Bytes of one row and rows of one window, fixed by the first save.
    row_bytes: usize,
    window_rows: usize,
    entries: Vec<Option<Entry>>,
    tick: u64,
    /// The slab allocation failed: the store stays off.
    failed: bool,
}

/// Device-resident drafter windows keyed by prefix content (module docs).
pub struct WarmCtxStore {
    capacity: usize,
    inner: Mutex<StoreInner>,
}

impl WarmCtxStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            inner: Mutex::new(StoreInner::default()),
        }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(store_slots_from_env()?))
    }

    /// Entries currently held (tests, logs).
    pub fn len(&self) -> usize {
        self.inner.lock().entries.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Release the slab (model teardown).
    pub fn free(&self, gpu: &dyn GpuBackend) {
        let mut inner = self.inner.lock();
        if let Some(slab) = inner.slab.take()
            && let Err(error) = gpu.free(slab)
        {
            tracing::warn!("DFlash warm ctx: freeing the store slab failed ({error})");
        }
        inner.entries.clear();
    }

    /// Store rows `[lo, hi)` (at most a window) read from `src` under `key`.
    /// Replaces an entry with the same key, else takes a free or the least
    /// recently used window. `Ok(false)` = not stored (store off, or a shape
    /// the slab was not sized for).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn save(
        &self,
        gpu: &dyn GpuBackend,
        key: StoreKey,
        src: DevicePtr,
        lo: usize,
        hi: usize,
        row_bytes: usize,
        window_rows: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.capacity == 0 || lo >= hi || hi - lo > window_rows || row_bytes == 0 {
            return Ok(false);
        }
        let mut inner = self.inner.lock();
        if inner.failed {
            return Ok(false);
        }
        let slab = match inner.slab {
            Some(slab) => {
                if (inner.row_bytes, inner.window_rows) != (row_bytes, window_rows) {
                    return Ok(false);
                }
                slab
            }
            None => {
                let bytes = self.capacity * window_rows * row_bytes;
                match gpu.alloc(bytes) {
                    Ok(slab) => {
                        tracing::info!(
                            "DFlash warm ctx: store of {} windows x {window_rows} rows x \
                             {row_bytes} B = {} MiB",
                            self.capacity,
                            bytes >> 20
                        );
                        inner.slab = Some(slab);
                        inner.row_bytes = row_bytes;
                        inner.window_rows = window_rows;
                        inner.entries = vec![None; self.capacity];
                        slab
                    }
                    Err(error) => {
                        tracing::warn!(
                            "DFlash warm ctx: allocating the {} MiB store failed ({error}); \
                             warm turns fall back to the seeded window",
                            bytes >> 20
                        );
                        inner.failed = true;
                        return Ok(false);
                    }
                }
            }
        };
        inner.tick += 1;
        let tick = inner.tick;
        let slot = inner
            .entries
            .iter()
            .position(|e| e.is_some_and(|e| e.key == key))
            .or_else(|| inner.entries.iter().position(Option::is_none))
            .unwrap_or_else(|| {
                (0..inner.entries.len())
                    .min_by_key(|&i| inner.entries[i].map_or(0, |e| e.used))
                    .unwrap_or(0)
            });
        let window_bytes = window_rows * row_bytes;
        gpu.copy_d2d_async(
            src,
            slab.offset(slot * window_bytes),
            (hi - lo) * row_bytes,
            stream,
        )?;
        inner.entries[slot] = Some(Entry {
            key,
            lo,
            hi,
            used: tick,
        });
        Self::order_slab_reuse(gpu, stream)?;
        Ok(true)
    }

    /// Copy the stored rows `[max(lo, want_lo), end)` of the entry for `key`
    /// that ends at `end` to `dst`. Returns that first position (`end` when
    /// the entry holds nothing at or above `want_lo`), or `None` on a miss.
    pub(crate) fn restore(
        &self,
        gpu: &dyn GpuBackend,
        key: StoreKey,
        end: usize,
        want_lo: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<Option<usize>> {
        let mut inner = self.inner.lock();
        let Some(slab) = inner.slab else {
            return Ok(None);
        };
        let Some(slot) = inner
            .entries
            .iter()
            .position(|e| e.is_some_and(|e| e.key == key && e.hi == end))
        else {
            return Ok(None);
        };
        inner.tick += 1;
        let tick = inner.tick;
        let (row_bytes, window_bytes) = (inner.row_bytes, inner.window_rows * inner.row_bytes);
        let entry = inner.entries[slot].as_mut().expect("found above");
        entry.used = tick;
        let entry_lo = entry.lo;
        let lo = entry_lo.max(want_lo).min(end);
        if lo < end {
            let src = slab.offset(slot * window_bytes + (lo - entry_lo) * row_bytes);
            gpu.copy_d2d_async(src, dst, (end - lo) * row_bytes, stream)?;
            Self::order_slab_reuse(gpu, stream)?;
        }
        Ok(Some(lo))
    }

    /// A slab window may be refilled from another prefill stream next time;
    /// copies off the default stream finish before the lock is released.
    /// (Multi-rank worlds prefill on the default stream: no sync there.)
    fn order_slab_reuse(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if stream != gpu.default_stream() {
            gpu.synchronize(stream)?;
        }
        Ok(())
    }
}

impl DflashProposerState {
    /// Record a prefill capture of positions `[plan.first, plan.end)`.
    pub(crate) fn note_capture(&mut self, plan: &CapturePlan) {
        CtxRun::note(&mut self.ctx_run, plan.window_start, plan.first, plan.end);
    }

    /// Chunk 0 of a prefill, after the prefix lookup: the target's state was
    /// restored at `skip` (0 = computes from the start) of a
    /// `prompt_len`-token prompt. Sets where the window may start and, under
    /// `restore`, copies the stored rows under `skip` into the accumulator.
    /// `key_tokens` are the prompt's prefix-cache key tokens (`None`: the
    /// prompt has no key, so nothing can have been stored for it).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_warm_ctx(
        &mut self,
        store: &WarmCtxStore,
        adapter_id: u64,
        key_tokens: Option<&[u32]>,
        skip: usize,
        prompt_len: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.ctx_floor = 0;
        self.ctx_run = None;
        if !self.warm_ctx.seeds() || skip == 0 {
            return Ok(());
        }
        self.ctx_floor = skip;
        let want_lo = prompt_len.max(skip).saturating_sub(self.max_ctx_len);
        if !self.warm_ctx.restores() || want_lo >= skip || self.ctx_hidden_acc.0 == 0 {
            return Ok(());
        }
        let restored = match key_tokens.and_then(|t| t.get(..skip)) {
            Some(prefix) => {
                let key = store_key(adapter_id, prefix);
                store.restore(gpu, key, skip, want_lo, self.ctx_hidden_acc, stream)?
            }
            None => None,
        };
        match restored {
            Some(lo) => {
                self.ctx_floor = lo;
                // The window now starts at `lo` (>= want_lo), slot 0.
                CtxRun::note(&mut self.ctx_run, lo, lo, skip);
                tracing::info!(
                    "DFlash warm ctx: restored {} rows [{lo}, {skip}) under a {prompt_len}-token \
                     prompt ({} wanted from {want_lo})",
                    skip - lo,
                    skip - want_lo,
                );
            }
            None => tracing::info!(
                "DFlash warm ctx: no stored rows at {skip}; window seeded from there \
                 ({} rows short)",
                skip - want_lo
            ),
        }
        Ok(())
    }

    /// A prefill checkpoint was saved at the end of `prefix` (its
    /// prefix-cache key tokens): store the accumulator's real rows that end
    /// there. `Ok(false)` when there are none or the store declines.
    pub(crate) fn save_warm_ctx(
        &self,
        store: &WarmCtxStore,
        adapter_id: u64,
        prefix: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let end = prefix.len();
        if !self.warm_ctx.restores() || self.ctx_hidden_acc.0 == 0 {
            return Ok(false);
        }
        let Some(run) = CtxRun::rows_to(self.ctx_run, end) else {
            return Ok(false);
        };
        let key = store_key(adapter_id, prefix);
        let src = self
            .ctx_hidden_acc
            .offset((run.lo - run.window_start) * self.ctx_slot_bytes);
        store.save(
            gpu,
            key,
            src,
            run.lo,
            end,
            self.ctx_slot_bytes,
            self.max_ctx_len,
            stream,
        )
    }
}
