// SPDX-License-Identifier: AGPL-3.0-only

//! The two tier seams — [`SlotArena`] (hot) and [`SwapStore`] (cold) — plus
//! [`SwapStats`], the counters [`crate::Residency`] keeps over them.

use anyhow::Result;

/// The hot tier: a RAM arena as a set of `num_slots` fixed-size slots. The
/// cache peer implements this over its `mmap`'d MR (page-aligned →
/// O_DIRECT-safe); in-process consumers use [`crate::VecSlotArena`].
pub trait SlotArena: Send {
    fn slot_bytes(&self) -> usize;
    fn num_slots(&self) -> usize;
    /// Copy arena slot → `out` (for spilling a victim to disk). `out.len()`
    /// MUST equal `slot_bytes()`.
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()>;
    /// Copy `bytes` → arena slot (for faulting a record back in). `bytes.len()`
    /// MUST equal `slot_bytes()`.
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()>;
}

/// The cold tier: an unbounded fixed-stride record store addressed by a
/// monotonic `disk_slot` index. The peer implements this over an O_DIRECT NVMe
/// file ([`crate::DirectSwapFile`]); [`crate::MemSwapStore`] is the host-RAM
/// variant.
pub trait SwapStore: Send {
    fn record_bytes(&self) -> usize;
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()>;
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()>;
    /// Read the `out.len() / record_bytes()` CONSECUTIVE records starting at
    /// `first_slot` into `out`, in slot order. Default: one `read_record` per
    /// record; a store with a cheaper contiguous path (one large O_DIRECT
    /// `pread`) overrides it. `out.len()` must be a record multiple.
    fn read_records(&self, first_slot: usize, out: &mut [u8]) -> Result<()> {
        let rb = self.record_bytes();
        if rb == 0 || !out.len().is_multiple_of(rb) {
            anyhow::bail!(
                "read_records: {} bytes is not a multiple of {rb}",
                out.len()
            );
        }
        for (i, rec) in out.chunks_exact_mut(rb).enumerate() {
            self.read_record(first_slot + i, rec)?;
        }
        Ok(())
    }
    /// Optional: reclaim disk space for a freed slot (default no-op; a hole in
    /// a preallocated file is fine — the free-list reuses the index).
    fn discard_record(&mut self, _disk_slot: usize) {}
}

// Boxed trait objects compose (lets a consumer pick arena/swap impls at
// runtime: `Residency<Box<dyn SlotArena>, Box<dyn SwapStore>>`).
impl<T: SlotArena + ?Sized> SlotArena for Box<T> {
    fn slot_bytes(&self) -> usize {
        (**self).slot_bytes()
    }
    fn num_slots(&self) -> usize {
        (**self).num_slots()
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_slot(slot, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_slot(slot, bytes)
    }
}

impl<T: SwapStore + ?Sized> SwapStore for Box<T> {
    fn record_bytes(&self) -> usize {
        (**self).record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_record(disk_slot, out)
    }
    fn read_records(&self, first_slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_records(first_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        (**self).discard_record(disk_slot)
    }
}

#[derive(Default, Debug, Clone)]
pub struct SwapStats {
    pub puts: u64,
    pub gets: u64,
    pub get_miss: u64,
    pub spills_to_disk: u64,
    pub faults_from_disk: u64,
    pub resident_hits: u64,
    /// Cold on-disk snapshots dropped because the disk cap was hit (a later GET
    /// for one cleanly misses → recompute).
    pub disk_evictions: u64,
}
