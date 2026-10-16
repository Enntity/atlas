// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_EXL3_COOP_MOE=1`: thin routed-expert projections (every route thin,
//! i.e. rows <= thin_capacity; always true at decode) run the vendored
//! exllamav3 cooperative GEMM core (`exl3_coop_moe_thin_b4_k32_n256` in
//! kernels/gb10/common/exl3_coop_gemm.cu) in place of
//! `exl3_moe_mul1_matmul_precision_mma[_split]`. Same scratch in and out, same
//! per-slot arithmetic (thin profile 32/256/8, F16-rounded running fold), so
//! the existing output Hadamard and fixed-order combine are unchanged.
//! Fat launches (rows > thin_capacity) keep the existing kernel (logged once).
//! See docs/design/exl3-coop-gemm-port.md.
use super::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Reference `SMEM_MAX`.
const COOP_SMEM_BYTES: u32 = 90 * 1024;
/// Reference block for tile_k 32 (`256 * 32 / 16`).
const COOP_BLOCK: u32 = 512;

pub(super) fn coop_moe() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE").as_deref() == Ok("1"))
}

/// `ATLAS_EXL3_COOP_MOE_V2=1`: v2 scheduling (gate+up in one launch,
/// per-(slice, column) work items). Implies the v1 resolution; v1 stays
/// selectable on its own.
pub(super) fn coop_moe_v2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE_V2").as_deref() == Ok("1"))
}

/// `ATLAS_EXL3_COOP_MOE_V3=1`: v2 scheduling with the 256-thread both-sub
/// kernel (`exl3_coop_moe3_thin_b4_k32_n256`, two CTAs per SM). Implies v2;
/// falls back to the v2 kernel if v3 cannot be resolved.
pub(super) fn coop_moe_v3() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE_V3").as_deref() == Ok("1"))
}

/// v3 dynamic shared memory: stage buffers only, SH_STAGES (6) x (A 16x32
/// F16 = 1024 B + B 2x16 trellis tiles at 4 bits = 4096 B). Must match
/// `EXL3_COOP_MOE3_SHAPE` in exl3_coop_gemm.cu.
const COOP_V3_SMEM_BYTES: u32 = 6 * (1024 + 4096);
const COOP_V3_BLOCK: u32 = 256;

/// `ATLAS_EXL3_COOP_MOE_V6=1`: v3 with the per-item A row staged once in
/// shared memory (`exl3_coop_moe6_thin_b4_k32_n256`); removes a global-load
/// stall per k-tile. Implies v2 scheduling; falls back to v3/v2.
/// `ATLAS_EXL3_COOP_MOE_MULTIROW=1`: launches with rows >= the threshold
/// (`ATLAS_EXL3_COOP_MOE_MULTIROW_MIN_ROWS`, default 4; speculative verify)
/// group slots by expert and stream each selected expert once for all its
/// rows (`exl3_coop_moe8_multirow_b4_k32_n256`). Per-row values are those of
/// the per-slot kernels (row-independent MMA, same partition).
/// The grouped kernel's M tile holds 16 rows per expert group and poisons
/// larger groups; at most 16 launch rows guarantees at most 16 routes per
/// expert (top-k experts are distinct per row). Prefill chunks above this
/// keep the per-slot path.
const MULTIROW_MAX_ROWS: usize = 16;

pub(super) fn coop_moe_multirow_min_rows() -> Option<usize> {
    static ON: OnceLock<Option<usize>> = OnceLock::new();
    *ON.get_or_init(|| {
        (std::env::var("ATLAS_EXL3_COOP_MOE_MULTIROW").as_deref() == Ok("1")).then(|| {
            std::env::var("ATLAS_EXL3_COOP_MOE_MULTIROW_MIN_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4)
                .max(2)
        })
    })
}

/// v8 dynamic smem: v3's stage buffers (A comes from F16 scratch via cp.async).
const COOP_V8_SMEM_BYTES: u32 = COOP_V3_SMEM_BYTES;

/// Multi-row resources: kernels, the device-side grouping scratch
/// (1 + 4 * max_slots ints) and the fixed co-resident grid.
pub(in crate::layers::glm5_exl3_ffn) struct CoopMultirow {
    kernel: KernelHandle,
    group: KernelHandle,
    transform: KernelHandle,
    convert: KernelHandle,
    pub(super) groups: DevicePtr,
    grid: u32,
    min_rows: usize,
}

/// `ATLAS_EXL3_COOP_MOE_V10=1`: thin gate|up and down on the streaming
/// persistent kernel (`exl3_coop_moe10_thin_b4_k32_n256`: one cp.async
/// pipeline across each CTA's items, 3 CTAs/SM) from F16 per-slot inputs
/// (`exl3_moe_input_hadamard_f16h`; down: exact F32 -> F16 of the
/// activation scratch). Same per-item arithmetic as v2-v7 (bit-exact).
pub(super) fn coop_moe_v10() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE_V10").as_deref() == Ok("1"))
}

/// v10 dynamic smem: 4 stages x (A 16x32 F16 + B 2x16 4-bit tiles).
const COOP_V10_SMEM_BYTES: u32 = 4 * (1024 + 4096);

/// `ATLAS_EXL3_COOP_MOE_V9=1` (implies V7): gate|up from the raw input, then
/// `exl3_coop_moe9_down_b4_k32_n256` computes the gate/up output transforms,
/// the thin activation and the down input transform in its staging (inner
/// header note 16): 3 fewer launches per layer, identical values.
pub(super) fn coop_moe_v9() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE_V9").as_deref() == Ok("1"))
}

/// `ATLAS_EXL3_COOP_MOE_V7=1` (also implied by V9): gate|up with the input
/// transform fused into the A staging (`exl3_coop_moe7_thin_b4_k32_n256`, no
/// separate exl3_moe_input_hadamard_f16 launches). Implies v6 for down.
pub(super) fn coop_moe_v7() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        coop_moe_v9() || std::env::var("ATLAS_EXL3_COOP_MOE_V7").as_deref() == Ok("1")
    })
}

pub(super) fn coop_moe_v6() -> bool {
    if coop_moe_v7() {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_MOE_V6").as_deref() == Ok("1"))
}

/// One v2-scheduled kernel: launch shape and its co-resident grid cap.
#[derive(Clone, Copy)]
struct ScheduledKernel {
    kernel: KernelHandle,
    block: u32,
    smem: u32,
    grid_cap: u32,
}

fn warn_once(reason: &str) {
    static SEEN: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if seen.insert(reason.to_owned()) {
        tracing::warn!("ATLAS_EXL3_COOP_MOE: {reason}; using the existing routed EXL3 matmul");
    }
}

pub(in crate::layers::glm5_exl3_ffn) struct CoopMoe {
    kernel: KernelHandle,
    /// `exl3_coop_moe2_thin_b4_k32_n256` under `ATLAS_EXL3_COOP_MOE_V2=1`, or
    /// `exl3_coop_moe3_thin_b4_k32_n256` under `ATLAS_EXL3_COOP_MOE_V3=1`.
    kernel_v2: Option<ScheduledKernel>,
    /// v7 gate|up kernel (fused input transform), same launch shape as v6.
    kernel_v7: Option<ScheduledKernel>,
    /// v9 down kernel (fused activation chain), same launch shape as v6.
    kernel_v9: Option<ScheduledKernel>,
    /// v10 streaming kernel plus its F16 input transform and F32->F16 convert.
    v10: Option<(ScheduledKernel, KernelHandle, KernelHandle)>,
    /// `ATLAS_EXL3_COOP_MOE_MULTIROW=1` grouped path (rows >= min_rows).
    pub(super) multirow: Option<CoopMultirow>,
    /// One row of `max(hidden, intermediate) / 128` zeroed ints per slot
    /// (self-resetting lock protocol), for up to `thin rows * topk` slots.
    pub(super) locks: DevicePtr,
    max_slots: usize,
    sm_count: u32,
    disabled: AtomicBool,
}

impl CoopMoe {
    /// `None` keeps the existing kernels. Never fails the layer: an
    /// unsupported checkpoint or allocation failure only logs.
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
            warn_once("thin profile is not 32/256/8/half_partial");
            return None;
        }
        if !hidden.is_multiple_of(256) || !intermediate.is_multiple_of(256) {
            warn_once("projection widths are not multiples of tile_n 256");
            return None;
        }
        if !tables
            .iter()
            .all(|t| t.shards.iter().all(|s| s.plan().bits() == 4))
        {
            warn_once("routed experts are not all 4-bit (only bits 4 is instantiated)");
            return None;
        }
        let sm_count = match gpu.sm_count() {
            Ok(count) if count >= 1 => count,
            _ => {
                warn_once("SM count unavailable");
                return None;
            }
        };
        let kernel = match gpu.kernel("exl3_coop_gemm", "exl3_coop_moe_thin_b4_k32_n256") {
            Ok(kernel) => kernel,
            Err(e) => {
                warn_once(&format!("kernel unavailable ({e:#})"));
                return None;
            }
        };
        // v2 keeps up's F32 output after gate's in the [slots, max(h, i)]
        // matmul scratch and up's input in the [slots, hidden] down-output
        // scratch (free until the down projection), with two lock rows per
        // slot: needs 2 * intermediate <= max(hidden, intermediate).
        let kernel_v2 = if !coop_moe_v2() && !coop_moe_v3() && !coop_moe_v6() {
            None
        } else if 2 * intermediate > hidden.max(intermediate) {
            warn_once("v2 needs 2 * intermediate <= hidden");
            None
        } else {
            // v6: v3's stage buffers plus the staged F16 A row (size_k <= 4096).
            let v6_smem = COOP_V3_SMEM_BYTES + 2 * hidden.max(intermediate) as u32;
            let v6 = if !coop_moe_v6() {
                None
            } else if hidden.max(intermediate) > 4096 {
                warn_once("v6 stages at most 4096 inputs; using v3");
                None
            } else {
                match gpu.kernel("exl3_coop_gemm", "exl3_coop_moe6_thin_b4_k32_n256") {
                    Ok(kernel) => {
                        match gpu.cooperative_blocks_per_sm(kernel, COOP_V3_BLOCK, v6_smem) {
                            Ok(blocks) if blocks >= 1 => Some(ScheduledKernel {
                                kernel,
                                block: COOP_V3_BLOCK,
                                smem: v6_smem,
                                grid_cap: blocks * sm_count,
                            }),
                            Ok(_) => {
                                warn_once("v6 kernel has zero occupancy; using v3");
                                None
                            }
                            Err(e) => {
                                warn_once(&format!("v6 occupancy query failed ({e:#}); using v3"));
                                None
                            }
                        }
                    }
                    Err(e) => {
                        warn_once(&format!("v6 kernel unavailable ({e:#}); using v3"));
                        None
                    }
                }
            };
            let v3 = if v6.is_some() {
                v6
            } else if coop_moe_v3() || coop_moe_v6() {
                match gpu.kernel("exl3_coop_gemm", "exl3_coop_moe3_thin_b4_k32_n256") {
                    Ok(kernel) => match gpu.cooperative_blocks_per_sm(
                        kernel,
                        COOP_V3_BLOCK,
                        COOP_V3_SMEM_BYTES,
                    ) {
                        Ok(blocks) if blocks >= 1 => Some(ScheduledKernel {
                            kernel,
                            block: COOP_V3_BLOCK,
                            smem: COOP_V3_SMEM_BYTES,
                            grid_cap: blocks * sm_count,
                        }),
                        Ok(_) => {
                            warn_once("v3 kernel has zero occupancy; using v2");
                            None
                        }
                        Err(e) => {
                            warn_once(&format!("v3 occupancy query failed ({e:#}); using v2"));
                            None
                        }
                    },
                    Err(e) => {
                        warn_once(&format!("v3 kernel unavailable ({e:#}); using v2"));
                        None
                    }
                }
            } else {
                None
            };
            v3.or_else(
                || match gpu.kernel("exl3_coop_gemm", "exl3_coop_moe2_thin_b4_k32_n256") {
                    Ok(kernel) => Some(ScheduledKernel {
                        kernel,
                        block: COOP_BLOCK,
                        smem: COOP_SMEM_BYTES,
                        grid_cap: sm_count,
                    }),
                    Err(e) => {
                        warn_once(&format!("v2 kernel unavailable ({e:#})"));
                        None
                    }
                },
            )
        };
        // v7 shares v6's shape (and only runs where v6 resolved).
        let kernel_v7 = match kernel_v2 {
            Some(v6) if coop_moe_v7() && v6.smem > COOP_V3_SMEM_BYTES => {
                match gpu.kernel("exl3_coop_gemm", "exl3_coop_moe7_thin_b4_k32_n256") {
                    Ok(kernel) => match gpu.cooperative_blocks_per_sm(kernel, v6.block, v6.smem) {
                        Ok(blocks) if blocks >= 1 => Some(ScheduledKernel {
                            kernel,
                            block: v6.block,
                            smem: v6.smem,
                            grid_cap: blocks * sm_count,
                        }),
                        _ => {
                            warn_once("v7 occupancy unavailable; using v6");
                            None
                        }
                    },
                    Err(e) => {
                        warn_once(&format!("v7 kernel unavailable ({e:#}); using v6"));
                        None
                    }
                }
            }
            _ => None,
        };
        let kernel_v9 = match (kernel_v7, coop_moe_v9()) {
            (Some(v7), true) => {
                match gpu.kernel("exl3_coop_gemm", "exl3_coop_moe9_down_b4_k32_n256") {
                    Ok(kernel) => match gpu.cooperative_blocks_per_sm(kernel, v7.block, v7.smem) {
                        Ok(blocks) if blocks >= 1 => Some(ScheduledKernel {
                            kernel,
                            block: v7.block,
                            smem: v7.smem,
                            grid_cap: blocks * sm_count,
                        }),
                        _ => {
                            warn_once("v9 occupancy unavailable; using v7");
                            None
                        }
                    },
                    Err(e) => {
                        warn_once(&format!("v9 kernel unavailable ({e:#}); using v7"));
                        None
                    }
                }
            }
            _ => None,
        };
        let v10 = (coop_moe_v10() && kernel_v2.is_some())
            .then(|| {
                let k = |name: &str| {
                    gpu.kernel("exl3_coop_gemm", name)
                        .map_err(|e| warn_once(&format!("v10: {name} unavailable ({e:#})")))
                        .ok()
                };
                let kernel = k("exl3_coop_moe10_thin_b4_k32_n256")?;
                let blocks = gpu
                    .cooperative_blocks_per_sm(kernel, COOP_V3_BLOCK, COOP_V10_SMEM_BYTES)
                    .ok()
                    .filter(|&b| b >= 1)?;
                Some((
                    ScheduledKernel {
                        kernel,
                        block: COOP_V3_BLOCK,
                        smem: COOP_V10_SMEM_BYTES,
                        grid_cap: blocks * sm_count,
                    },
                    k("exl3_moe_input_hadamard_f16h")?,
                    k("exl3_f32_to_f16")?,
                ))
            })
            .flatten();
        if coop_moe_v10() && v10.is_none() {
            warn_once("v10 unavailable (needs the v2+ path and its kernels)");
        }
        let max_slots = capacity.min(policy.thin_capacity as usize) * topk;
        let lock_bytes = max_slots * (hidden.max(intermediate) / 128) * 4;
        let locks = match gpu.alloc(lock_bytes) {
            Ok(ptr) => ptr,
            Err(e) => {
                warn_once(&format!("lock allocation failed ({e:#})"));
                return None;
            }
        };
        if let Err(e) = gpu.memset(locks, 0, lock_bytes) {
            let _ = gpu.free(locks);
            warn_once(&format!("lock initialization failed ({e:#})"));
            return None;
        }
        let multirow = coop_moe_multirow_min_rows().and_then(|min_rows| {
            // The grouped gate|up rides the v2 gate|up hook; down reuses its
            // groups, so it must only run where that hook runs.
            if kernel_v2.is_none() {
                warn_once("multirow needs the v2+ gate|up path (ATLAS_EXL3_COOP_MOE_V2/V3/V6)");
                return None;
            }
            if 2 * intermediate > hidden.max(intermediate) {
                warn_once("multirow needs 2 * intermediate <= hidden");
                return None;
            }
            let k = |name: &str| {
                gpu.kernel("exl3_coop_gemm", name)
                    .map_err(|e| warn_once(&format!("multirow: {name} unavailable ({e:#})")))
                    .ok()
            };
            let (kernel, group, transform, convert) = (
                k("exl3_coop_moe8_multirow_b4_k32_n256")?,
                k("exl3_moe_group_slots")?,
                k("exl3_moe_input_hadamard_f16h")?,
                k("exl3_f32_to_f16")?,
            );
            let blocks =
                match gpu.cooperative_blocks_per_sm(kernel, COOP_V3_BLOCK, COOP_V8_SMEM_BYTES) {
                    Ok(b) if b >= 1 => b,
                    _ => {
                        warn_once("multirow occupancy unavailable");
                        return None;
                    }
                };
            let groups = gpu
                .alloc((1 + 4 * max_slots) * 4)
                .map_err(|e| warn_once(&format!("multirow group scratch ({e:#})")))
                .ok()?;
            Some(CoopMultirow {
                kernel,
                group,
                transform,
                convert,
                groups,
                grid: blocks * sm_count,
                min_rows,
            })
        });
        Some(Self {
            kernel,
            kernel_v2,
            kernel_v7,
            kernel_v9,
            v10,
            multirow,
            locks,
            max_slots,
            sm_count,
            disabled: AtomicBool::new(false),
        })
    }
}

impl NativeMoeFfn {
    /// Launch the cooperative thin matmul for projection `index` if this
    /// launch is entirely thin. `Ok(false)` means the caller must run the
    /// existing kernel (nothing was enqueued).
    pub(super) fn coop_matmul(
        &self,
        gpu: &dyn GpuBackend,
        index: usize,
        rows: usize,
    ) -> Result<bool> {
        let Some(coop) = self.coop.as_ref() else {
            return Ok(false);
        };
        if coop.disabled.load(Ordering::Relaxed) {
            return Ok(false);
        }
        // Each row routes to distinct experts, so counts[e] <= rows: rows <=
        // thin_capacity guarantees every route is thin (the kernel also
        // poisons any slot whose count says otherwise).
        if rows > self.precision.thin_capacity as usize {
            warn_once("fat (prefill) routed launch");
            return Ok(false);
        }
        let (k, n) = if index == 2 {
            (self.intermediate, self.hidden)
        } else {
            (self.hidden, self.intermediate)
        };
        let slots = rows * self.topk;
        ensure!(
            slots <= coop.max_slots,
            "EXL3 cooperative MoE slots exceed lock capacity"
        );
        if index == 2
            && let Some(mr) = coop
                .multirow
                .as_ref()
                .filter(|m| rows >= m.min_rows && rows <= MULTIROW_MAX_ROWS)
        {
            // Down: the activation's F32 scratch (F16-representable) -> F16
            // in the down-output scratch (free until the down output
            // transform), then the grouped launch on this layer's groups.
            let count = (slots * k) as u64;
            KernelLaunch::new(gpu, mr.convert)
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
            let yh = self.buffers[8];
            return self.coop_launch_multirow(
                gpu,
                mr,
                [self.buffers[5], self.buffers[5]],
                [2, 2],
                [yh, yh],
                1,
                slots,
                k,
                n,
            );
        }
        if index == 2
            && let Some((v10, _, convert)) = coop.v10
        {
            let count = (slots * k) as u64;
            KernelLaunch::new(gpu, convert)
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
            let yh = self.buffers[8];
            return self.coop_launch_v2(
                gpu,
                v10,
                [self.buffers[5], self.buffers[5]],
                [2, 2],
                [yh, yh],
                1,
                slots,
                k,
                n,
            );
        }
        if let Some(kernel) = coop.kernel_v2 {
            let yh = self.buffers[8];
            return self.coop_launch_v2(
                gpu,
                kernel,
                [self.buffers[7], self.buffers[7]],
                [index, index],
                [yh, yh],
                1,
                slots,
                k,
                n,
            );
        }
        let items = slots * self.precision.slices as usize;
        let grid = u32::try_from(items.min(coop.sm_count as usize))
            .context("EXL3 cooperative MoE grid overflow")?;
        let launched = KernelLaunch::new(gpu, coop.kernel)
            .grid([grid, 1, 1])
            .block([COOP_BLOCK, 1, 1])
            .shared_mem(COOP_SMEM_BYTES)
            .arg_ptr(self.buffers[7])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[index].ptr)
            .arg_ptr(self.buffers[8])
            .arg_ptr(self.buffers[9])
            .arg_ptr(coop.locks)
            .arg_u32(slots as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .arg_u32(self.experts as u32)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(self.precision.slices)
            .launch_cooperative(self.stream);
        match launched {
            Ok(()) => Ok(true),
            // A rejected cooperative launch enqueues nothing.
            Err(e) => {
                coop.disabled.store(true, Ordering::Relaxed);
                warn_once(&format!("cooperative launch failed ({e:#})"));
                Ok(false)
            }
        }
    }
}

impl NativeMoeFfn {
    /// Chunks per slot for the v2 item count (mirrors the kernel).
    fn coop_chunks(k: usize, n: usize, slices: usize) -> usize {
        let (tiles_k, total) = (k / 32, (k / 32) * (n / 256));
        (0..slices)
            .filter_map(|s| {
                let (beg, end) = (total * s / slices, total * (s + 1) / slices);
                (end > beg).then(|| (end - 1) / tiles_k - beg / tiles_k + 1)
            })
            .sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn coop_launch_v2(
        &self,
        gpu: &dyn GpuBackend,
        kernel: ScheduledKernel,
        xh: [DevicePtr; 2],
        tables: [usize; 2],
        yh: [DevicePtr; 2],
        projections: usize,
        slots: usize,
        k: usize,
        n: usize,
    ) -> Result<bool> {
        let Some(coop) = self.coop.as_ref() else {
            return Ok(false);
        };
        let slices = self.precision.slices as usize;
        let items = projections * slots * Self::coop_chunks(k, n, slices);
        // Co-resident grid: SMs x occupancy (1 for v2, 2 expected for v3).
        let grid = u32::try_from(items.min(kernel.grid_cap as usize))
            .context("EXL3 cooperative MoE grid overflow")?;
        let launched = KernelLaunch::new(gpu, kernel.kernel)
            .grid([grid, 1, 1])
            .block([kernel.block, 1, 1])
            .shared_mem(kernel.smem)
            .arg_ptr(xh[0])
            .arg_ptr(xh[1])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[tables[0]].ptr)
            .arg_ptr(self.tables[tables[1]].ptr)
            .arg_ptr(yh[0])
            .arg_ptr(yh[1])
            .arg_ptr(self.buffers[9])
            .arg_ptr(coop.locks)
            .arg_u32(projections as u32)
            .arg_u32(slots as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .arg_u32(self.experts as u32)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(self.precision.slices)
            .launch_cooperative(self.stream);
        match launched {
            Ok(()) => Ok(true),
            Err(e) => {
                coop.disabled.store(true, Ordering::Relaxed);
                warn_once(&format!("cooperative launch failed ({e:#})"));
                Ok(false)
            }
        }
    }

    /// v2: gate and up in ONE cooperative launch. Writes exactly what
    /// `project(0, .., buffers[3])` + `project(1, .., buffers[4])` write
    /// (same input transforms, per-slot arithmetic and output Hadamards);
    /// up's input transform uses the down-output scratch (buffers[5], free
    /// until the down projection) and up's F32 matmul output follows gate's
    /// in buffers[8]. `Ok(false)`: the caller runs the two projections.
    /// The two unchanged gate/up output Hadamards (yh -> buffers[3]/[4]).
    pub(super) fn coop_gate_up_outputs(
        &self,
        gpu: &dyn GpuBackend,
        yh: [DevicePtr; 2],
        rows32: u32,
        n: usize,
        topk: u32,
        experts: u32,
    ) -> Result<()> {
        let slots = rows32 as usize * topk as usize;
        for (index, (yh, output)) in yh
            .into_iter()
            .zip([self.buffers[3], self.buffers[4]])
            .enumerate()
        {
            KernelLaunch::new(gpu, self.kernels[4])
                .grid([n as u32 / 128, slots as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(yh)
                .arg_ptr(self.buffers[1])
                .arg_ptr(self.tables[index].ptr)
                .arg_ptr(output)
                .arg_ptr(self.buffers[9])
                .arg_u32(rows32)
                .arg_u32(n as u32)
                .arg_u32(topk)
                .arg_u32(experts)
                .arg_u32(self.precision.thin_capacity)
                .launch(self.stream)?;
        }
        Ok(())
    }

    pub(super) fn coop_gate_up(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
    ) -> Result<bool> {
        let Some(coop) = self.coop.as_ref() else {
            return Ok(false);
        };
        let Some(kernel) = coop.kernel_v2 else {
            return Ok(false);
        };
        if coop.disabled.load(Ordering::Relaxed) || rows > self.precision.thin_capacity as usize {
            return Ok(false);
        }
        let slots = rows * self.topk;
        ensure!(
            slots <= coop.max_slots,
            "EXL3 cooperative MoE slots exceed lock capacity"
        );
        let (k, n) = (self.hidden, self.intermediate);
        let (rows32, topk, experts) = (rows as u32, self.topk as u32, self.experts as u32);
        let xh = [self.buffers[7], self.buffers[5]];
        let yh = [self.buffers[8], self.buffers[8].offset(slots * n * 4)];
        if let Some(mr) = coop
            .multirow
            .as_ref()
            .filter(|m| rows >= m.min_rows && rows <= MULTIROW_MAX_ROWS)
        {
            // Group this layer's slots by expert (reused by the down launch).
            KernelLaunch::new(gpu, mr.group)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buffers[1])
                .arg_ptr(mr.groups)
                .arg_u32(slots as u32)
                .arg_u32(experts)
                .launch(self.stream)?;
            // Per-slot F16 transformed rows: gate then up, both in the
            // [slots, max(h, i)] F32 matmul-input scratch (2 x F16 fits).
            let xh16 = [self.buffers[7], self.buffers[7].offset(slots * k * 2)];
            for (index, xh) in xh16.into_iter().enumerate() {
                KernelLaunch::new(gpu, mr.transform)
                    .grid([k as u32 / 128, slots as u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(input)
                    .arg_ptr(self.buffers[1])
                    .arg_ptr(self.tables[index].ptr)
                    .arg_ptr(xh)
                    .arg_u32(rows32)
                    .arg_u32(k as u32)
                    .arg_u32(topk)
                    .arg_u32(experts)
                    .launch(self.stream)?;
            }
            if !self.coop_launch_multirow(gpu, mr, xh16, [0, 1], yh, 2, slots, k, n)? {
                return Ok(false);
            }
            self.coop_gate_up_outputs(gpu, yh, rows32, n, topk, experts)?;
            return Ok(true);
        }
        if let Some((v10, transform, _)) = coop.v10 {
            // F16 per-slot transformed rows: gate then up, both in the F32
            // matmul-input scratch (2 x F16 fits).
            let xh16 = [self.buffers[7], self.buffers[7].offset(slots * k * 2)];
            for (index, xh) in xh16.into_iter().enumerate() {
                KernelLaunch::new(gpu, transform)
                    .grid([k as u32 / 128, slots as u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(input)
                    .arg_ptr(self.buffers[1])
                    .arg_ptr(self.tables[index].ptr)
                    .arg_ptr(xh)
                    .arg_u32(rows32)
                    .arg_u32(k as u32)
                    .arg_u32(topk)
                    .arg_u32(experts)
                    .launch(self.stream)?;
            }
            if !self.coop_launch_v2(gpu, v10, xh16, [0, 1], yh, 2, slots, k, n)? {
                return Ok(false);
            }
            self.coop_gate_up_outputs(gpu, yh, rows32, n, topk, experts)?;
            return Ok(true);
        }
        if let Some(v7) = coop.kernel_v7 {
            // Fused input transform: A is the raw layer input (row slot/topk).
            let items = 2 * slots * Self::coop_chunks(k, n, self.precision.slices as usize);
            let grid = u32::try_from(items.min(v7.grid_cap as usize))
                .context("EXL3 cooperative MoE grid overflow")?;
            let launched = KernelLaunch::new(gpu, v7.kernel)
                .grid([grid, 1, 1])
                .block([v7.block, 1, 1])
                .shared_mem(v7.smem)
                .arg_ptr(input)
                .arg_u32(topk)
                .arg_ptr(self.buffers[1])
                .arg_ptr(self.tables[0].ptr)
                .arg_ptr(self.tables[1].ptr)
                .arg_ptr(yh[0])
                .arg_ptr(yh[1])
                .arg_ptr(self.buffers[9])
                .arg_ptr(coop.locks)
                .arg_u32(2)
                .arg_u32(slots as u32)
                .arg_u32(k as u32)
                .arg_u32(n as u32)
                .arg_u32(experts)
                .arg_u32(self.precision.thin_capacity)
                .arg_u32(self.precision.slices)
                .launch_cooperative(self.stream);
            if let Err(e) = launched {
                coop.disabled.store(true, Ordering::Relaxed);
                warn_once(&format!("cooperative launch failed ({e:#})"));
                return Ok(false);
            }
            self.coop_gate_up_outputs(gpu, yh, rows32, n, topk, experts)?;
            return Ok(true);
        }
        for (index, xh) in xh.into_iter().enumerate() {
            KernelLaunch::new(gpu, self.kernels[2])
                .grid([k as u32 / 128, slots as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(input)
                .arg_ptr(self.buffers[1])
                .arg_ptr(self.tables[index].ptr)
                .arg_ptr(xh)
                .arg_u32(rows32)
                .arg_u32(k as u32)
                .arg_u32(topk)
                .arg_u32(0)
                .arg_u32(experts)
                .launch(self.stream)?;
        }
        if !self.coop_launch_v2(gpu, kernel, xh, [0, 1], yh, 2, slots, k, n)? {
            return Ok(false);
        }
        self.coop_gate_up_outputs(gpu, yh, rows32, n, topk, experts)?;
        Ok(true)
    }
}

impl NativeMoeFfn {
    /// Grouped multi-row launch (`exl3_coop_moe8_*`): fixed co-resident grid,
    /// the item count (projections x groups x chunks) is read on the device.
    #[allow(clippy::too_many_arguments)]
    fn coop_launch_multirow(
        &self,
        gpu: &dyn GpuBackend,
        mr: &CoopMultirow,
        xh: [DevicePtr; 2],
        tables: [usize; 2],
        yh: [DevicePtr; 2],
        projections: usize,
        slots: usize,
        k: usize,
        n: usize,
    ) -> Result<bool> {
        let Some(coop) = self.coop.as_ref() else {
            return Ok(false);
        };
        let launched = KernelLaunch::new(gpu, mr.kernel)
            .grid([mr.grid, 1, 1])
            .block([COOP_V3_BLOCK, 1, 1])
            .shared_mem(COOP_V8_SMEM_BYTES)
            .arg_ptr(xh[0])
            .arg_ptr(xh[1])
            .arg_ptr(mr.groups)
            .arg_ptr(self.tables[tables[0]].ptr)
            .arg_ptr(self.tables[tables[1]].ptr)
            .arg_ptr(yh[0])
            .arg_ptr(yh[1])
            .arg_ptr(self.buffers[9])
            .arg_ptr(coop.locks)
            .arg_u32(projections as u32)
            .arg_u32(slots as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .arg_u32(self.experts as u32)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(self.precision.slices)
            .launch_cooperative(self.stream);
        match launched {
            Ok(()) => Ok(true),
            Err(e) => {
                coop.disabled.store(true, Ordering::Relaxed);
                warn_once(&format!("cooperative launch failed ({e:#})"));
                Ok(false)
            }
        }
    }
}

impl NativeMoeFfn {
    /// `ATLAS_EXL3_COOP_MOE_V9`: the whole thin routed chain in three
    /// launches: gate|up (v7, from the raw input; F32 sums in buffers[8]),
    /// down (v9, staging computes gate/up output transforms + activation +
    /// down input transform; F32 sums in buffers[7], free in this flow), and
    /// the unchanged weighted down output transform into buffers[5].
    /// `Ok(false)`: nothing enqueued, the caller runs the separate stages.
    pub(super) fn coop_routed_v9(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
    ) -> Result<bool> {
        if !coop_moe_v9() || coop_moe_v10() {
            return Ok(false);
        }
        let Some(coop) = self.coop.as_ref() else {
            return Ok(false);
        };
        let (Some(v7), Some(v9)) = (coop.kernel_v7, coop.kernel_v9) else {
            return Ok(false);
        };
        if coop.disabled.load(Ordering::Relaxed)
            || rows > self.precision.thin_capacity as usize
            || coop.multirow.as_ref().is_some_and(|m| rows >= m.min_rows)
        {
            return Ok(false);
        }
        let slots = rows * self.topk;
        ensure!(
            slots <= coop.max_slots,
            "EXL3 cooperative MoE slots exceed lock capacity"
        );
        let (h, inter) = (self.hidden, self.intermediate);
        let (rows32, topk, experts) = (rows as u32, self.topk as u32, self.experts as u32);
        let slices = self.precision.slices as usize;
        let yh = [self.buffers[8], self.buffers[8].offset(slots * inter * 4)];
        let launch = |kernel: ScheduledKernel, items: usize| -> Result<KernelLaunch<'_>> {
            let grid = u32::try_from(items.min(kernel.grid_cap as usize))
                .context("EXL3 cooperative MoE grid overflow")?;
            Ok(KernelLaunch::new(gpu, kernel.kernel)
                .grid([grid, 1, 1])
                .block([kernel.block, 1, 1])
                .shared_mem(kernel.smem))
        };
        let gate_up = launch(v7, 2 * slots * Self::coop_chunks(h, inter, slices))?
            .arg_ptr(input)
            .arg_u32(topk)
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[0].ptr)
            .arg_ptr(self.tables[1].ptr)
            .arg_ptr(yh[0])
            .arg_ptr(yh[1])
            .arg_ptr(self.buffers[9])
            .arg_ptr(coop.locks)
            .arg_u32(2)
            .arg_u32(slots as u32)
            .arg_u32(h as u32)
            .arg_u32(inter as u32)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(self.precision.slices)
            .launch_cooperative(self.stream);
        if let Err(e) = gate_up {
            coop.disabled.store(true, Ordering::Relaxed);
            warn_once(&format!("cooperative launch failed ({e:#})"));
            return Ok(false);
        }
        let (limit, clamp) = match self.activation {
            Exl3FfnActivation::SiLu => (0.0, 0),
            Exl3FfnActivation::ClampedSiLu { limit } => (limit, 1),
        };
        // Once gate|up is enqueued this layer must finish on this path: a
        // failed down launch is an error (the separate stages would need the
        // gate/up output transforms that were skipped).
        launch(v9, slots * Self::coop_chunks(inter, h, slices))?
            .arg_ptr(yh[0])
            .arg_ptr(yh[1])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[0].ptr)
            .arg_ptr(self.tables[1].ptr)
            .arg_ptr(self.tables[2].ptr)
            .arg_ptr(self.buffers[7])
            .arg_ptr(self.buffers[9])
            .arg_ptr(coop.locks)
            .arg_u32(slots as u32)
            .arg_u32(inter as u32)
            .arg_u32(h as u32)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity)
            .arg_u32(self.precision.slices)
            .arg_f32(limit)
            .arg_u32(clamp)
            .launch_cooperative(self.stream)?;
        KernelLaunch::new(gpu, self.kernels[8])
            .grid([h as u32 / 128, slots as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(self.buffers[7])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[2].ptr)
            .arg_ptr(self.buffers[2])
            .arg_ptr(self.buffers[5])
            .arg_ptr(self.buffers[9])
            .arg_u32(rows32)
            .arg_u32(h as u32)
            .arg_u32(topk)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity)
            .launch(self.stream)?;
        Ok(true)
    }
}
