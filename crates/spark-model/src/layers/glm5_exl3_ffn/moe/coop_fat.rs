// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1`: grouped multi-row routed MoE for
//! prefill launches (`rows > thin_capacity`), where the per-slot
//! `exl3_moe_mul1_matmul_precision_mma` streams one packed expert matrix per
//! `(row, expert)` slot. Slots are grouped by expert into M-tile units of at
//! most 16 rows (`exl3_moe_group_units`); each unit streams its expert's
//! weights once for all its rows, with the accumulation profile
//! `exl3_moe_mul1_matmul_precision_mma` selects per expert count:
//!
//!   - `counts[e] <= thin_capacity`: tile (32, 256), 8 split-K slices, running
//!     fold rounded to F16 after every slice
//!     (`exl3_coop_moe8_multirow_b4_k32_n256`, the
//!     `ATLAS_EXL3_COOP_MOE_MULTIROW` kernel);
//!   - `counts[e] > thin_capacity`: tile (16, 128), 1 slice, unrounded F32
//!     (`exl3_coop_moe_fat_multirow_b4_k16_n128`).
//!
//! MMA rows are independent, so every row equals the per-slot kernel's value
//! bit for bit. The router, input Hadamards, activation, gate/up output
//! Hadamards, weighted down output Hadamard and fixed-order combine keep their
//! existing launches; only the two routed matmuls change.
//!
//! Thin launches (`rows <= thin_capacity`) keep the existing paths: this module
//! returns `Ok(false)` for them, and nothing here is even resolved unless the
//! flag is on (default off: no kernel loaded, no scratch allocated).
use super::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1`.
pub(super) fn coop_moe_fat_multirow() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE_FAT_MULTIROW").as_deref() == Ok("1"))
}

/// `exl3_coop_moe8_multirow_b4_k32_n256` dynamic shared memory: 6 stages x
/// (A 16x32 F16 = 1024 B + B 2x16 4-bit trellis = 4096 B), mirroring
/// `EXL3_COOP_MOE8_SHAPE`. TILEBLOCKS_K == 2, so the reduce buffer is unused.
const COOP_FAT_THIN_SMEM_BYTES: u32 = 6 * (1024 + 4096);
/// `exl3_coop_moe_fat_multirow_b4_k16_n128` dynamic shared memory:
/// `EXL3_COOP_MOE_FAT_SH` (6) stages x (A 16x16 F16 = 512 B + B 1x8 4-bit
/// trellis = 1024 B), mirroring `EXL3_COOP_MOE_FAT_SMEM`. TILEBLOCKS_K == 1,
/// so the reduce buffer is unused.
const COOP_FAT_SMEM_BYTES: u32 = 6 * (512 + 1024);
const COOP_FAT_BLOCK: u32 = 256;
/// The vendored inner's TILESIZE_M: every exported unit is at most this wide.
const COOP_FAT_M_TILE: u32 = 16;

fn warn_fat(reason: &str) {
    static SEEN: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if seen.insert(reason.to_owned()) {
        tracing::warn!(
            "ATLAS_EXL3_COOP_MOE_FAT_MULTIROW: {reason}; using the existing routed EXL3 matmul"
        );
    }
}

/// Grouped multi-row resources: the two grouped kernels, the unit split, the
/// F16 staging pair and the device scratch they share.
pub(in crate::layers::glm5_exl3_ffn) struct CoopFat {
    /// Thin-profile grouped kernel (same kernel as `CoopMultirow`).
    thin: KernelHandle,
    /// Fat-profile grouped kernel.
    fat: KernelHandle,
    /// `exl3_moe_group_units`.
    group: KernelHandle,
    /// `exl3_moe_input_hadamard_f16h` (per-slot F16 transformed rows).
    transform: KernelHandle,
    /// `exl3_f32_to_f16` (exact F16 of the down activation scratch).
    convert: KernelHandle,
    /// Two `1 + 4 * max_slots` int regions: thin units, then fat units.
    groups: DevicePtr,
    /// One row of `max(hidden, intermediate) / 128` ints per
    /// `(projection, unit)`, zeroed at load (self-resetting protocol).
    locks: DevicePtr,
    thin_grid: u32,
    fat_grid: u32,
    max_slots: usize,
    /// A rejected cooperative launch disables the grouped route for this layer
    /// (the existing stages then reproduce the launch), like `CoopMoe`.
    disabled: AtomicBool,
}

impl CoopFat {
    /// `None` keeps the existing kernels. Never fails the layer: a missing
    /// kernel, unsupported profile or allocation failure only logs.
    pub(super) fn resolve(
        gpu: &dyn GpuBackend,
        tables: &[ExpertTable],
        hidden: usize,
        intermediate: usize,
        capacity: usize,
        topk: usize,
        policy: RoutedPrecisionPolicy,
    ) -> Option<Self> {
        if (
            policy.tile_k,
            policy.tile_n,
            policy.slices,
            policy.half_partial,
        ) != (32, 256, 8, true)
        {
            warn_fat("thin profile is not 32/256/8/half_partial");
            return None;
        }
        if (
            policy.fat_tile_k,
            policy.fat_tile_n,
            policy.fat_slices,
            policy.fat_half_partial,
        ) != (16, 128, 1, false)
        {
            warn_fat("fat profile is not 16/128/1/unrounded");
            return None;
        }
        // Both tile widths divide the projections, so no unit straddles a tile.
        if !hidden.is_multiple_of(256) || !intermediate.is_multiple_of(256) {
            warn_fat("projection widths are not multiples of tile_n 256");
            return None;
        }
        if !tables
            .iter()
            .all(|t| t.shards.iter().all(|s| s.plan().bits() == 4))
        {
            warn_fat("routed experts are not all 4-bit");
            return None;
        }
        let sm_count = match gpu.sm_count() {
            Ok(count) if count >= 1 => count,
            _ => {
                warn_fat("SM count unavailable");
                return None;
            }
        };
        let kernel = |name: &str| {
            gpu.kernel("exl3_coop_gemm", name)
                .map_err(|e| warn_fat(&format!("{name} unavailable ({e:#})")))
                .ok()
        };
        let thin = kernel("exl3_coop_moe8_multirow_b4_k32_n256")?;
        let fat = kernel("exl3_coop_moe_fat_multirow_b4_k16_n128")?;
        let group = kernel("exl3_moe_group_units")?;
        let transform = kernel("exl3_moe_input_hadamard_f16h")?;
        let convert = kernel("exl3_f32_to_f16")?;
        let blocks = |handle: KernelHandle, smem: u32| -> Option<u32> {
            match gpu.cooperative_blocks_per_sm(handle, COOP_FAT_BLOCK, smem) {
                Ok(count) if count >= 1 => Some(count),
                _ => None,
            }
        };
        let (Some(thin_blocks), Some(fat_blocks)) = (
            blocks(thin, COOP_FAT_THIN_SMEM_BYTES),
            blocks(fat, COOP_FAT_SMEM_BYTES),
        ) else {
            warn_fat("grouped multi-row occupancy unavailable");
            return None;
        };
        let Some(max_slots) = capacity.checked_mul(topk).filter(|&n| n > 0) else {
            warn_fat("grouped multi-row slot count invalid");
            return None;
        };
        // Units are bounded by the slot count (every unit has at least one row),
        // so one `1 + 4 * slots` region per class is enough.
        let groups = match gpu.alloc(2 * (1 + 4 * max_slots) * 4) {
            Ok(ptr) => ptr,
            Err(e) => {
                warn_fat(&format!("grouped multi-row unit scratch failed ({e:#})"));
                return None;
            }
        };
        // `projections` is at most 2, so at most `2 * units` lock rows.
        let lock_bytes = 2 * max_slots * (hidden.max(intermediate) / 128) * 4;
        let locks = match gpu.alloc(lock_bytes) {
            Ok(ptr) => ptr,
            Err(e) => {
                let _ = gpu.free(groups);
                warn_fat(&format!("grouped multi-row lock scratch failed ({e:#})"));
                return None;
            }
        };
        if let Err(e) = gpu.memset(locks, 0, lock_bytes) {
            let _ = gpu.free(locks);
            let _ = gpu.free(groups);
            warn_fat(&format!(
                "grouped multi-row lock initialization failed ({e:#})"
            ));
            return None;
        }
        Some(Self {
            thin,
            fat,
            group,
            transform,
            convert,
            groups,
            locks,
            thin_grid: thin_blocks * sm_count,
            fat_grid: fat_blocks * sm_count,
            max_slots,
            disabled: AtomicBool::new(false),
        })
    }

    pub(super) fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut error = None;
        retain_error(&mut error, gpu.free(self.groups));
        retain_error(&mut error, gpu.free(self.locks));
        error.map_or(Ok(()), Err)
    }
}

impl NativeMoeFfn {
    /// The whole routed chain for a grouped-serve (prefill) launch: gate|up
    /// grouped over the expert units, the unchanged activation, the down
    /// projection grouped the same way, and the unchanged output Hadamards.
    /// `Ok(false)`: nothing enqueued, the caller runs the existing stages.
    pub(super) fn coop_routed_fat(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
    ) -> Result<bool> {
        let Some(fat) = self.coop_fat.as_ref() else {
            return Ok(false);
        };
        if fat.disabled.load(Ordering::Relaxed) {
            return Ok(false);
        }
        // Thin launches keep their own paths; the fat route only serves the
        // launches the per-slot kernel would run at M=1 per slot.
        if rows <= self.precision.thin_capacity as usize {
            return Ok(false);
        }
        let slots = rows * self.topk;
        ensure!(
            slots <= fat.max_slots,
            "EXL3 grouped MoE slots {} exceed scratch capacity {}",
            slots,
            fat.max_slots
        );
        let (h, i) = (self.hidden, self.intermediate);
        let (rows32, topk, experts) = (rows as u32, self.topk as u32, self.experts as u32);
        // 1. Split this launch's slots into M-tile units per accumulation class.
        KernelLaunch::new(gpu, fat.group)
            .grid([1, 1, 1])
            .block([COOP_FAT_BLOCK, 1, 1])
            .arg_ptr(self.buffers[1])
            .arg_ptr(fat.groups)
            .arg_u32(slots as u32)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(COOP_FAT_M_TILE)
            .launch(self.stream)?;
        // 2. Per-slot F16 transformed gate then up inputs, both in the F32
        // matmul-input scratch (2 x F16 fits; up's input uses the down-output
        // scratch, free until the down projection).
        let xh16 = [self.buffers[7], self.buffers[7].offset(slots * h * 2)];
        for (index, xh) in xh16.into_iter().enumerate() {
            KernelLaunch::new(gpu, fat.transform)
                .grid([h as u32 / 128, slots as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(input)
                .arg_ptr(self.buffers[1])
                .arg_ptr(self.tables[index].ptr)
                .arg_ptr(xh)
                .arg_u32(rows32)
                .arg_u32(h as u32)
                .arg_u32(topk)
                .arg_u32(experts)
                .launch(self.stream)?;
        }
        // 3. Grouped gate|up: thin units, then fat units.
        let yh = [self.buffers[8], self.buffers[8].offset(slots * i * 4)];
        if !self.coop_fat_grouped(gpu, fat, 2, xh16, [0, 1], yh, slots, h, i)? {
            return Ok(false);
        }
        self.coop_gate_up_outputs(gpu, yh, rows32, i, topk, experts)?;
        // 4. Unchanged activation and down input Hadamard (F32 scratch).
        self.activate_down(gpu, rows)?;
        // 5. Down: exact F32 -> F16 of the activation scratch into the
        // down-output scratch (free until the weighted output), then the
        // grouped down projection.
        let count = (slots * i) as u64;
        KernelLaunch::new(gpu, fat.convert)
            .grid([
                u32::try_from(count.div_ceil(256)).context("convert grid")?,
                1,
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(self.buffers[7])
            .arg_ptr(self.buffers[5])
            .arg_u64(count)
            .launch(self.stream)?;
        if !self.coop_fat_grouped(
            gpu,
            fat,
            1,
            [self.buffers[5], self.buffers[5]],
            [2, 2],
            [self.buffers[8], self.buffers[8]],
            slots,
            i,
            h,
        )? {
            return Ok(false);
        }
        // 6. Unchanged weighted down output transform.
        self.output_hadamard(gpu, 2, self.buffers[5], rows)?;
        Ok(true)
    }

    /// Both grouped launches over the one unit split: the thin units
    /// (`exl3_coop_moe8_multirow_b4_k32_n256`) and the fat units
    /// (`exl3_coop_moe_fat_multirow_b4_k16_n128`). The fat kernel reads its own
    /// region of the unit scratch (`exl3_moe_group_units` writes the thin units
    /// at offset 0 and the fat units at offset `1 + 4 * slots` ints). Returns
    /// `Ok(false)` if a cooperative launch was rejected (nothing was enqueued
    /// for it), after disabling this route: the caller's existing stages then
    /// recompute the launch, so a partly enqueued grouped attempt is harmless.
    #[allow(clippy::too_many_arguments)]
    fn coop_fat_grouped(
        &self,
        gpu: &dyn GpuBackend,
        fat: &CoopFat,
        projections: usize,
        xh: [DevicePtr; 2],
        tables: [usize; 2],
        yh: [DevicePtr; 2],
        slots: usize,
        k: usize,
        n: usize,
    ) -> Result<bool> {
        let launched = KernelLaunch::new(gpu, fat.thin)
            .grid([fat.thin_grid, 1, 1])
            .block([COOP_FAT_BLOCK, 1, 1])
            .shared_mem(COOP_FAT_THIN_SMEM_BYTES)
            .arg_ptr(xh[0])
            .arg_ptr(xh[1])
            .arg_ptr(fat.groups)
            .arg_ptr(self.tables[tables[0]].ptr)
            .arg_ptr(self.tables[tables[1]].ptr)
            .arg_ptr(yh[0])
            .arg_ptr(yh[1])
            .arg_ptr(self.buffers[9])
            .arg_ptr(fat.locks)
            .arg_u32(projections as u32)
            .arg_u32(slots as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .arg_u32(self.experts as u32)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(self.precision.slices)
            .launch_cooperative(self.stream);
        if let Err(e) = launched {
            fat.disabled.store(true, Ordering::Relaxed);
            warn_fat(&format!("grouped thin launch failed ({e:#})"));
            return Ok(false);
        }
        let launched = KernelLaunch::new(gpu, fat.fat)
            .grid([fat.fat_grid, 1, 1])
            .block([COOP_FAT_BLOCK, 1, 1])
            .shared_mem(COOP_FAT_SMEM_BYTES)
            .arg_ptr(xh[0])
            .arg_ptr(xh[1])
            .arg_ptr(fat.groups.offset((1 + 4 * slots) * 4))
            .arg_ptr(self.tables[tables[0]].ptr)
            .arg_ptr(self.tables[tables[1]].ptr)
            .arg_ptr(yh[0])
            .arg_ptr(yh[1])
            .arg_ptr(fat.locks)
            .arg_u32(projections as u32)
            .arg_u32(slots as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .arg_u32(self.experts as u32)
            .arg_u32(self.precision.fat_slices)
            .launch_cooperative(self.stream);
        if let Err(e) = launched {
            fat.disabled.store(true, Ordering::Relaxed);
            warn_fat(&format!("grouped fat launch failed ({e:#})"));
            return Ok(false);
        }
        Ok(true)
    }
}
