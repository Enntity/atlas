// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_EXL3_COOP_MOE=1`: thin routed-expert projections (every route thin,
//! i.e. rows <= thin_capacity; always true at decode) run the vendored
//! exllamav3 cooperative GEMM core (`exl3_coop_moe_thin_b4_k32_n256` in
//! kernels/gb10/glm-5.3-flash/exl3/exl3_coop_gemm.cu) in place of
//! `exl3_moe_mul1_matmul_precision_mma[_split]`. Same scratch in and out, same
//! per-slot arithmetic (thin profile 32/256/8, F16-rounded running fold), so
//! the existing output Hadamard and fixed-order combine are unchanged.
//! Fat launches (rows > thin_capacity) keep the existing kernel (logged once).
//! See docs/design/exl3-coop-gemm-port.md.
use super::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

mod routed;
mod thin;

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
