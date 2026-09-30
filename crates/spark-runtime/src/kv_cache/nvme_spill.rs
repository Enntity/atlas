// SPDX-License-Identifier: AGPL-3.0-only

//! Byte movement for the prefix cache's NVMe spill tier (`ATLAS_KV_NVME_DIR`;
//! bookkeeping lives in the radix tree — see `crate::prefix_cache::nvme`).
//!
//! One record = one physical block's bytes on THIS rank, gathered from every
//! per-block device region the cache owns for that block:
//!
//! ```text
//! [layer0 K][layer0 V if not aliased][layer0 index values][layer0 index scales]
//! [layer1 …] … [zero pad] [trailer: magic u64 | tag u64 | checksum u64]
//! ```
//!
//! padded to a 4 KiB multiple (O_DIRECT). The raw sparse-index TAILS are not
//! part of a record: they only hold the raw keys of a pool still being
//! assembled, and only FULL blocks (every pool finalised) are ever spilled.
//! The trailer binds the record to the node that spilled it (the tree's tag)
//! and to its bytes (checksum); any mismatch fails the restore — the caller
//! then recomputes, so a torn, stale or misdirected record is never served.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail, ensure};

use super::PagedKvCache;
use crate::gpu::{DevicePtr, GpuBackend};
use crate::prefix_cache::{DiskRef, SpillOrder};

/// Spill writes that failed (process-wide), for throttled logging.
static WRITE_FAILURES: AtomicU64 = AtomicU64::new(0);

const MAGIC: u64 = u64::from_le_bytes(*b"ATLKVNV1");
const TRAILER: usize = 3 * std::mem::size_of::<u64>();
const RECORD_ALIGN: usize = 4096;
/// Records staged per gather/scatter round (one stream sync per round).
const STAGING_RECORDS: usize = 32;

/// One per-block device region: block `b` lives at `base + b·stride`.
#[derive(Clone, Copy, Debug)]
struct Segment {
    base: DevicePtr,
    stride: usize,
}

pub(super) struct NvmeSpill {
    store: Box<dyn atlas_tier::SwapStore>,
    segments: Vec<Segment>,
    payload: usize,
    record: usize,
    staging: *mut u8,
    staging_bytes: usize,
}

// SAFETY: `staging` is a uniquely owned host allocation; the cache (and so
// this struct) is only ever used behind the model's `Mutex<PagedKvCache>`.
unsafe impl Send for NvmeSpill {}
unsafe impl Sync for NvmeSpill {}

/// Fast 64-bit checksum over the payload: four independent multiply-fold
/// lanes over 32-byte strides (so the multiply latency chains overlap — a
/// ~104 KB GLM record costs a few µs), folded together with the length.
/// Detects torn / partial / misdirected records; not a cryptographic MAC.
fn checksum(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let word = |w: &[u8]| u64::from_le_bytes(w.try_into().expect("8-byte word"));
    let mut lanes = [
        0x243f_6a88_85a3_08d3,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    ];
    let mut strides = bytes.chunks_exact(32);
    for st in &mut strides {
        for (l, w) in lanes.iter_mut().zip(st.chunks_exact(8)) {
            *l = (*l ^ word(w)).wrapping_mul(K).rotate_left(27);
        }
    }
    let mut tail = [0u8; 32];
    tail[..strides.remainder().len()].copy_from_slice(strides.remainder());
    let mut h = bytes.len() as u64;
    for (l, w) in lanes.iter().zip(tail.chunks_exact(8)) {
        h = (h ^ l ^ word(w)).wrapping_mul(K).rotate_left(31);
    }
    h
}

fn stamp(rec: &mut [u8], payload: usize, tag: u64) {
    let sum = checksum(&rec[..payload]);
    let t = rec.len() - TRAILER;
    rec[t..t + 8].copy_from_slice(&MAGIC.to_le_bytes());
    rec[t + 8..t + 16].copy_from_slice(&tag.to_le_bytes());
    rec[t + 16..].copy_from_slice(&sum.to_le_bytes());
}

fn verify(rec: &[u8], payload: usize, tag: u64) -> bool {
    let t = rec.len() - TRAILER;
    let word = |o: usize| u64::from_le_bytes(rec[t + o..t + o + 8].try_into().expect("8 bytes"));
    word(0) == MAGIC && word(8) == tag && word(16) == checksum(&rec[..payload])
}

#[cfg(test)]
pub(super) fn stamp_for_test(rec: &mut [u8], payload: usize, tag: u64) {
    stamp(rec, payload, tag)
}

#[cfg(test)]
pub(super) use run_layout as run_layout_for_test;

#[cfg(test)]
pub(super) fn verify_for_test(rec: &[u8], payload: usize, tag: u64) -> bool {
    verify(rec, payload, tag)
}

impl NvmeSpill {
    pub(super) fn free_staging(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let ptr = std::mem::replace(&mut self.staging, std::ptr::null_mut());
        gpu.free_host_pinned(ptr, self.staging_bytes)
    }
}

impl PagedKvCache {
    fn nvme_segments(&self) -> Vec<Segment> {
        let mut segs = Vec::new();
        for l in &self.layers {
            let mut push = |base: DevicePtr, stride: usize| {
                if !base.is_null() && stride > 0 {
                    segs.push(Segment { base, stride });
                }
            };
            push(l.k_pool, l.k_block_stride);
            push(l.owned_v_pool(), l.v_block_stride);
            push(l.sparse_index_values, l.sparse_index_values_block_stride);
            push(l.sparse_index_scales, l.sparse_index_scales_block_stride);
        }
        segs
    }

    /// Bytes of one spill record for this cache's current layout (attach the
    /// sparse index FIRST). Size the store's records with this.
    pub fn nvme_record_bytes(&self) -> usize {
        let payload: usize = self.nvme_segments().iter().map(|s| s.stride).sum();
        (payload + TRAILER).next_multiple_of(RECORD_ALIGN)
    }

    /// Attach the spill store (records of [`Self::nvme_record_bytes`]).
    pub fn attach_nvme_spill(
        &mut self,
        store: Box<dyn atlas_tier::SwapStore>,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        ensure!(self.nvme.is_none(), "NVMe spill store already attached");
        ensure!(
            self.config.cache_blocks_per_seq.is_none(),
            "the NVMe prefix spill tier cannot be combined with --high-speed-swap"
        );
        let segments = self.nvme_segments();
        let payload: usize = segments.iter().map(|s| s.stride).sum();
        let record = self.nvme_record_bytes();
        ensure!(payload > 0, "KV cache has no per-block regions to spill");
        if store.record_bytes() != record {
            bail!(
                "NVMe spill store record size {} != KV record size {record}",
                store.record_bytes()
            );
        }
        let staging_bytes = record * STAGING_RECORDS;
        let staging = gpu.alloc_host_pinned(staging_bytes)?;
        self.nvme = Some(NvmeSpill {
            store,
            segments,
            payload,
            record,
            staging,
            staging_bytes,
        });
        Ok(())
    }

    pub fn nvme_attached(&self) -> bool {
        self.nvme.is_some()
    }

    /// Prefix-cache blocks to evict per allocation miss: one without the
    /// tier (unchanged); a staging batch with it, so a spill pays its two
    /// stream syncs once per batch instead of once per block.
    pub fn evict_batch(&self) -> usize {
        if self.nvme.is_some() {
            STAGING_RECORDS
        } else {
            1
        }
    }

    /// Write each order's block to its record. Must run BEFORE the blocks go
    /// back to the free list. Returns the orders that did NOT reach disk (the
    /// tree must drop those nodes). Work in flight on EVERY stream is drained
    /// first (a device-wide sync); cached blocks are complete and never
    /// rewritten after that.
    pub fn nvme_write(
        &mut self,
        orders: &[SpillOrder],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Vec<SpillOrder> {
        let Some(mut spill) = self.nvme.take() else {
            return orders.to_vec();
        };
        let mut failed = Vec::new();
        for batch in orders.chunks(STAGING_RECORDS) {
            if let Err(e) = write_batch(&mut spill, batch, gpu, stream, &mut failed) {
                // Nothing of this batch reached disk (writes follow the gather).
                tracing::warn!("NVMe KV spill: gather failed ({e:#}); dropping batch");
                failed.extend_from_slice(batch);
            }
        }
        self.nvme = Some(spill);
        failed
    }

    /// Read `disk[i]` into freshly allocated `blocks[i]` (in order), verifying
    /// every trailer. Returns how many leading blocks hold verified bytes and
    /// whether the next one FAILED (I/O error or bad record) — as opposed to
    /// simply not being attempted. Scatter completes before returning.
    pub fn nvme_read(
        &mut self,
        disk: &[DiskRef],
        blocks: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> (usize, bool) {
        let Some(mut spill) = self.nvme.take() else {
            return (0, false);
        };
        let n = disk.len().min(blocks.len());
        let mut done = 0;
        let mut failed = false;
        while done < n && !failed {
            let end = (done + STAGING_RECORDS).min(n);
            match read_batch(
                &mut spill,
                &disk[done..end],
                &blocks[done..end],
                gpu,
                stream,
            ) {
                Ok(k) => {
                    failed = k < end - done;
                    done += k;
                }
                Err(e) => {
                    tracing::warn!("NVMe KV restore: scatter failed ({e:#})");
                    failed = true;
                }
            }
        }
        self.nvme = Some(spill);
        (done, failed)
    }
}

fn write_batch(
    spill: &mut NvmeSpill,
    batch: &[SpillOrder],
    gpu: &dyn GpuBackend,
    stream: u64,
    failed: &mut Vec<SpillOrder>,
) -> Result<()> {
    let (record, payload) = (spill.record, spill.payload);
    // SAFETY: the staging buffer is owned by `spill` and disjoint from
    // `spill.store` / `spill.segments`, which are borrowed alongside it.
    let staging = unsafe { std::slice::from_raw_parts_mut(spill.staging, spill.staging_bytes) };
    // Device-wide, not just `stream`: the victims' last writes may still be in
    // flight on ANOTHER stream (the scheduler's prefill stream, MoE/secondary
    // streams — all non-blocking), and a record gathered before they land
    // would checksum and later restore torn KV. Paid once per staging batch.
    gpu.synchronize_device()?;
    let mut enqueue = || -> Result<()> {
        for (i, o) in batch.iter().enumerate() {
            let mut off = i * record;
            for s in &spill.segments {
                let src = s.base.offset(o.block as usize * s.stride);
                gpu.copy_d2h_async(src, &mut staging[off..off + s.stride], stream)?;
                off += s.stride;
            }
        }
        Ok(())
    };
    let enq = enqueue();
    // Drain even after a failed enqueue: earlier chunks still DMA into staging.
    gpu.synchronize(stream)?;
    enq?;
    for (i, o) in batch.iter().enumerate() {
        let rec = &mut staging[i * record..(i + 1) * record];
        rec[payload..].fill(0);
        stamp(rec, payload, o.tag);
        if let Err(e) = spill.store.write_record(o.slot as usize, rec) {
            // A full / failing disk fails EVERY spill: log at 1, 2, 4, 8, …
            let n = WRITE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::warn!(
                    "NVMe KV spill: write of slot {} failed ({e:#}) — {n} failed spills so far; \
                     those blocks are evicted as without the tier (recompute on reuse)",
                    o.slot
                );
            }
            failed.push(*o);
        }
    }
    Ok(())
}

/// Staging position of each of `slots` after [`read_runs`]: maximal runs of
/// consecutive slots (either direction — a chain spills leaf-first, so its
/// path order is usually DEScending) are read with one `read_records` each and
/// land in slot order within the run's window. Pure, for testing.
pub(super) fn run_layout(slots: &[u32]) -> Vec<(usize, usize, u32)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let step = slots
            .get(i + 1)
            .map_or(0, |&n| i64::from(n) - i64::from(slots[i]));
        let mut j = i;
        if step == 1 || step == -1 {
            while j + 1 < slots.len() && i64::from(slots[j + 1]) - i64::from(slots[j]) == step {
                j += 1;
            }
        }
        let first = slots[i..=j].iter().copied().min().expect("non-empty run");
        runs.push((i, j + 1 - i, first));
        i = j + 1;
    }
    runs
}

/// Read `disk` into staging; returns each entry's staging record index for
/// the leading entries that were read (stops at the first unreadable one).
fn read_runs(spill: &NvmeSpill, disk: &[DiskRef], staging: &mut [u8]) -> Vec<usize> {
    let record = spill.record;
    let slots: Vec<u32> = disk.iter().map(|d| d.slot).collect();
    let mut pos = Vec::with_capacity(disk.len());
    for (start, len, first) in run_layout(&slots) {
        let window = &mut staging[start * record..(start + len) * record];
        if len > 1 && spill.store.read_records(first as usize, window).is_ok() {
            pos.extend((start..start + len).map(|k| start + (slots[k] - first) as usize));
            continue;
        }
        // Single record, or the ranged read failed: find the exact failure.
        for k in start..start + len {
            let rec = &mut staging[k * record..(k + 1) * record];
            if let Err(e) = spill.store.read_record(slots[k] as usize, rec) {
                tracing::warn!("NVMe KV restore: read of slot {} failed ({e:#})", slots[k]);
                return pos;
            }
            pos.push(k);
        }
    }
    pos
}

fn read_batch(
    spill: &mut NvmeSpill,
    disk: &[DiskRef],
    blocks: &[u32],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<usize> {
    let (record, payload) = (spill.record, spill.payload);
    // SAFETY: as in `write_batch`.
    let staging = unsafe { std::slice::from_raw_parts_mut(spill.staging, spill.staging_bytes) };
    let pos = read_runs(spill, disk, staging);
    let mut ok = 0;
    for (d, &p) in disk.iter().zip(&pos) {
        if !verify(&staging[p * record..(p + 1) * record], payload, d.tag) {
            tracing::warn!(
                "NVMe KV restore: slot {} failed verification (tag/checksum) — recomputing",
                d.slot
            );
            break;
        }
        ok += 1;
    }
    let enqueue = || -> Result<()> {
        for (&b, &p) in blocks[..ok].iter().zip(&pos) {
            let mut off = p * record;
            for s in &spill.segments {
                let dst = s.base.offset(b as usize * s.stride);
                gpu.copy_h2d_async_retained(&staging[off..off + s.stride], dst, stream)?;
                off += s.stride;
            }
        }
        Ok(())
    };
    let enq = enqueue();
    gpu.synchronize(stream)?;
    enq?;
    Ok(ok)
}
