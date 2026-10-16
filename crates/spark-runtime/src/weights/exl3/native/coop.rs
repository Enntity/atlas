// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_EXL3_COOP_GEMM=1`: packed linears with an explicit numerics profile
//! run the vendored exllamav3 cooperative GEMM (kernels/gb10/common/
//! exl3_coop_gemm.cu) in place of `exl3_mul1_matmul_mma[_split]`.
//!
//! Same outer contract and stream as the existing path: an F16-output twin of
//! `exl3_input_hadamard_f16` (identical F16 values, stored as `__half` in the
//! F32 input scratch), the cooperative GEMM writing unrounded F32 into the
//! same output scratch, then the unchanged Atlas output Hadamard. The grid,
//! block and shared memory are the reference's for the profile's
//! (tile_k, tile_n, slices); see docs/design/exl3-coop-gemm-port.md.
//! Any profile/shape the vendored instantiations do not cover keeps the
//! existing kernels (logged once).
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

use super::Exl3LinearNumerics;
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use crate::kernel_args::KernelLaunch;

/// Reference `SMEM_MAX` (exl3_gemm_inner.cuh): every cooperative GEMM launch
/// requests this much dynamic shared memory regardless of shape.
pub(super) const COOP_SMEM_BYTES: u32 = 90 * 1024;
/// Reference `EXL3_GEMM_BASE_THREADS`.
const BASE_THREADS: u32 = 256;
/// Rows per GEMM pass (reference `TILESIZE_M`); one lock row per pass.
const CHUNK_ROWS: usize = 16;
/// Lock-row stride in ints: one lock per 128-column tile, enough for any tile_n.
const LOCK_TILE_N: usize = 128;

/// `ATLAS_EXL3_COOP_FUSED_INPUT=1`: coop GEMMs with the input transform fused
/// into the A staging (`exl3_coop_gemmh_*`, `exl3_coop_gemm3h_*`; inner header
/// note 13) when the staged rows fit; no separate input-transform launch.
pub fn coop_fused_input() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_FUSED_INPUT").as_deref() == Ok("1"))
}

/// `ATLAS_EXL3_COOP_FUSED_OUTPUT=1` (with fused input): BF16-output coop
/// GEMMs also apply the output transform in the last chunk of each column
/// tile (`exl3_coop_gemmf_*`, inner header note 14); no output launch.
pub fn coop_fused_output() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_FUSED_OUTPUT").as_deref() == Ok("1"))
}

/// Largest row count whose staged F16 A rows (rows x k) fit in `SMEM_MAX`
/// after the stage buffers and the reduce buffer of the (tile_k, tile_n)
/// shape (SH_STAGES 4, mirrors the inner's layout), capped at one 16-row pass.
fn fused_input_rows(tile_k: usize, tile_n: usize, bits: usize, k: usize) -> usize {
    let stages = 4 * (2 * 16 * tile_k + 2 * (tile_k / 16) * (tile_n / 16) * 16 * bits);
    let frags = 2 * (tile_n / 16) / 8;
    let reduce = 4 * (4 * BASE_THREADS as usize * frags);
    let avail = (COOP_SMEM_BYTES as usize).saturating_sub(stages + reduce);
    (avail / (2 * k)).min(CHUNK_ROWS)
}

pub fn coop_gemm() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_GEMM").as_deref() == Ok("1"))
}

/// One row bucket's resolved launch.
#[derive(Debug, Clone, Copy)]
pub(super) struct CoopPlan {
    pub kernel: KernelHandle,
    pub block: u32,
    /// CTAs launched: only slices whose reference split-K range intersects
    /// this component's column window (the others would return immediately).
    pub grid: u32,
    pub slice_base: u32,
    pub slices: u32,
    pub logical_tiles_n: u32,
    pub col_tile_offset: u32,
    /// Fused-input twin and the most rows it can stage (0: unavailable).
    pub fused: Option<KernelHandle>,
    pub fused_rows: usize,
    /// Fused input + output twin (BF16 output path), same row limit.
    pub fused_out: Option<KernelHandle>,
}

/// Resolved cooperative path for one workspace. `locks` is owned (released
/// with the workspace scratch); `disabled` latches a fallback after a failed
/// cooperative launch.
pub(super) struct CoopGemm {
    pub input_kernel: KernelHandle,
    pub plans: [Option<CoopPlan>; super::numerics::EXL3_PROFILE_SLOTS],
    pub locks: DevicePtr,
    pub disabled: AtomicBool,
}

fn warn_once(reason: String) {
    static SEEN: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(reason.clone()) {
        tracing::warn!("ATLAS_EXL3_COOP_GEMM: {reason}; using the existing EXL3 matmul kernels");
    }
}

pub(super) fn fallback_note(reason: String) {
    warn_once(reason)
}

/// Contiguous range of reference slices whose flattened (k-tile, n-tile)
/// range intersects the physical window [p0, p1). Mirrors the kernel's
/// `logical_tiles * s / num_slices` arithmetic exactly.
fn intersecting_slices(total: u64, slices: u64, p0: u64, p1: u64) -> Option<(u32, u32)> {
    let mut first = None;
    let mut last = 0;
    for s in 0..slices {
        let lb = total * s / slices;
        let le = total * (s + 1) / slices;
        if lb.max(p0) < le.min(p1) {
            first.get_or_insert(s);
            last = s;
        }
    }
    first.map(|f| (f as u32, (last - f + 1) as u32))
}

fn plan_for(
    gpu: &dyn GpuBackend,
    numerics: Exl3LinearNumerics,
    k: usize,
    n: usize,
    bucket: usize,
    sm_count: u32,
) -> Result<Option<CoopPlan>, String> {
    let Some(profile) = numerics.profile_at(bucket) else {
        return Ok(None);
    };
    let (tile_k, tile_n) = (profile.tile_k as usize, profile.tile_n as usize);
    let tag = format!(
        "K{k}/N{n}/logicalN{}/offset{}/bits{} tile ({tile_k},{tile_n}) slices {}",
        numerics.logical_n, numerics.logical_offset, numerics.bits, profile.slices
    );
    if !matches!((tile_k, tile_n), (32, 128) | (32, 256) | (16, 512)) {
        return Err(format!("{tag}: tile shape not instantiated"));
    }
    if !matches!(numerics.bits, 5 | 6) {
        return Err(format!("{tag}: bitrate not instantiated"));
    }
    if profile.half_partial {
        return Err(format!("{tag}: F16 partials not supported"));
    }
    if !k.is_multiple_of(tile_k)
        || !n.is_multiple_of(tile_n)
        || !numerics.logical_n.is_multiple_of(tile_n)
        || !numerics.logical_offset.is_multiple_of(tile_n)
    {
        return Err(format!("{tag}: component window not aligned to tile_n"));
    }
    if profile.slices == 0 || profile.slices > sm_count {
        return Err(format!(
            "{tag}: slices exceed {sm_count} SMs (cooperative residency)"
        ));
    }
    let tiles_k = (k / tile_k) as u64;
    let logical_tiles_n = numerics.logical_n / tile_n;
    let col_tile_offset = numerics.logical_offset / tile_n;
    let total = tiles_k * logical_tiles_n as u64;
    let p0 = col_tile_offset as u64 * tiles_k;
    let p1 = p0 + (n / tile_n) as u64 * tiles_k;
    if total > i32::MAX as u64 || (k * CHUNK_ROWS) > i32::MAX as usize {
        return Err(format!("{tag}: exceeds the kernel's int indexing"));
    }
    let (slice_base, grid) = intersecting_slices(total, profile.slices as u64, p0, p1)
        .ok_or_else(|| format!("{tag}: no slice intersects the component"))?;
    let name = format!("exl3_coop_gemm_b{}_k{tile_k}_n{tile_n}", numerics.bits);
    let kernel = gpu
        .kernel("exl3_coop_gemm", &name)
        .map_err(|e| format!("{tag}: {e:#}"))?;
    let fused_rows = fused_input_rows(tile_k, tile_n, numerics.bits as usize, k);
    let fused = (coop_fused_input() && fused_rows > 0)
        .then(|| {
            let name = format!("exl3_coop_gemmh_b{}_k{tile_k}_n{tile_n}", numerics.bits);
            gpu.kernel("exl3_coop_gemm", &name)
                .map_err(|e| warn_once(format!("{tag}: fused input unavailable ({e:#})")))
                .ok()
        })
        .flatten();
    let fused_out = (fused.is_some() && coop_fused_output())
        .then(|| {
            let name = format!("exl3_coop_gemmf_b{}_k{tile_k}_n{tile_n}", numerics.bits);
            gpu.kernel("exl3_coop_gemm", &name)
                .map_err(|e| warn_once(format!("{tag}: fused output unavailable ({e:#})")))
                .ok()
        })
        .flatten();
    Ok(Some(CoopPlan {
        kernel,
        block: BASE_THREADS * profile.tile_k / 16,
        grid,
        slice_base,
        slices: profile.slices,
        logical_tiles_n: logical_tiles_n as u32,
        col_tile_offset: col_tile_offset as u32,
        fused,
        fused_rows,
        fused_out,
    }))
}

/// Resolve the cooperative path for a workspace. `None` (with a one-time
/// warning) keeps the existing kernels. Allocates only the lock buffer.
pub(super) fn resolve(
    gpu: &dyn GpuBackend,
    numerics: Exl3LinearNumerics,
    k: usize,
    n: usize,
    capacity: usize,
) -> Result<Option<CoopGemm>> {
    let input_kernel = match gpu.kernel("exl3_coop_gemm", "exl3_coop_input_hadamard_f16h") {
        Ok(kernel) => kernel,
        Err(e) => {
            warn_once(format!("kernel module unavailable ({e:#})"));
            return Ok(None);
        }
    };
    let sm_count = gpu.sm_count().unwrap_or(0);
    let mut plans = [None; super::numerics::EXL3_PROFILE_SLOTS];
    for (bucket, plan) in plans.iter_mut().enumerate() {
        match plan_for(gpu, numerics, k, n, bucket, sm_count) {
            Ok(p) => *plan = p,
            Err(reason) => warn_once(reason),
        }
    }
    if plans.iter().all(Option::is_none) {
        return Ok(None);
    }
    // Reference protocol: zero-initialized ints that every launch leaves zero.
    // One lock row (n/128 ints) per 16-row pass, so passes need no grid barrier.
    let lock_bytes = capacity
        .div_ceil(CHUNK_ROWS)
        .checked_mul(n / LOCK_TILE_N)
        .and_then(|ints| ints.checked_mul(4))
        .context("EXL3 cooperative lock buffer size overflow")?;
    let locks = gpu.alloc(lock_bytes)?;
    if let Err(e) = gpu.memset(locks, 0, lock_bytes) {
        let _ = gpu.free(locks);
        return Err(e);
    }
    Ok(Some(CoopGemm {
        input_kernel,
        plans,
        locks,
        disabled: AtomicBool::new(false),
    }))
}

impl CoopGemm {
    pub(super) fn plan(&self, bucket: usize) -> Option<CoopPlan> {
        if self.disabled.load(Ordering::Relaxed) {
            return None;
        }
        self.plans[bucket]
    }

    pub(super) fn disable(&self) {
        self.disabled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::intersecting_slices;

    #[test]
    fn slice_window_matches_reference_partition() {
        // Whole matrix: every slice.
        assert_eq!(intersecting_slices(1024, 48, 0, 1024), Some((0, 48)));
        // KDA fused QKV, tile (16,512): tiles_k 256, logical 24 n-tiles, the
        // middle component covers n-tiles 8..16.
        let (total, p0, p1) = (256 * 24, 256 * 8, 256 * 16);
        let (base, grid) = intersecting_slices(total, 48, p0, p1).unwrap();
        assert_eq!((base, grid), (16, 16));
        // A slice straddling the window edge is included.
        assert_eq!(intersecting_slices(10, 3, 4, 10), Some((1, 2)));
    }
}

/// `ATLAS_EXL3_COOP_FUSED_QKV=1`: the three output components of one fused
/// reference matrix (GLM KDA q|k|v) run as ONE cooperative launch over the
/// fused width (`exl3_coop_gemm3_*`), each component keeping its own trellis,
/// scratch and output Hadamard. Requires the per-component workspaces to be
/// on the coop numerics path's geometry (see `Exl3FusedCoop::resolve`).
pub fn coop_fused_qkv() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_COOP_FUSED_QKV").as_deref() == Ok("1"))
}

fn warn_fused_once(reason: String) {
    static SEEN: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(reason.clone()) {
        tracing::warn!("EXL3 fused coop launch: {reason}; using per-component EXL3 launches");
    }
}

#[derive(Debug, Clone, Copy)]
struct FusedPlan {
    kernel: KernelHandle,
    block: u32,
    slices: u32,
    /// gemm3 fused-input twin and its staging row limit (shared input only).
    fused_input: Option<KernelHandle>,
    fused_rows: usize,
}

/// One fused launch for three equal-width components of one reference matrix.
pub struct Exl3FusedCoop {
    input_kernel: KernelHandle,
    plans: [Option<FusedPlan>; super::numerics::EXL3_PROFILE_SLOTS],
    locks: DevicePtr,
    numerics: Exl3LinearNumerics,
    /// Components (2: gemm2 with per-component inputs, 3: gemm3).
    count: usize,
    /// One input transform for all components (gemm3).
    shared_input: bool,
    k: u32,
    comp_n: u32,
    capacity: u32,
    stream: u64,
    disabled: AtomicBool,
}

impl Exl3FusedCoop {
    /// `None` (logged once per reason) keeps the per-component launches.
    /// Performs one synchronous D2H comparison of the three input scales and
    /// allocates only the lock rows.
    pub fn resolve(
        gpu: &dyn GpuBackend,
        parts: &[(
            &super::Exl3LinearWorkspace,
            &crate::weights::exl3::tp::Exl3DeviceShard,
        )],
    ) -> Option<Self> {
        match Self::try_resolve(gpu, parts) {
            Ok(fused) => Some(fused),
            Err(reason) => {
                warn_fused_once(reason);
                None
            }
        }
    }

    fn try_resolve(
        gpu: &dyn GpuBackend,
        parts: &[(
            &super::Exl3LinearWorkspace,
            &crate::weights::exl3::tp::Exl3DeviceShard,
        )],
    ) -> Result<Self, String> {
        let count = parts.len();
        if !matches!(count, 2 | 3) {
            return Err(format!("{count} components (2 or 3 supported)"));
        }
        let first = parts[0].0;
        let numerics = first
            .numerics
            .ok_or("component without a numerics profile")?;
        let (k, n) = (first.k as usize, first.n as usize);
        for (index, (ws, shard)) in parts.iter().enumerate() {
            let m = ws.numerics.ok_or("component without a numerics profile")?;
            if (ws.k, ws.n, ws.capacity, ws.stream)
                != (first.k, first.n, first.capacity, first.stream)
            {
                return Err("component workspaces differ in geometry/capacity/stream".into());
            }
            if (
                m.logical_n,
                m.bits,
                m.row_profiles,
                m.verify_rows_5_8,
                m.reference_min_rows,
                m.reference_pad,
            ) != (
                numerics.logical_n,
                numerics.bits,
                numerics.row_profiles,
                numerics.verify_rows_5_8,
                numerics.reference_min_rows,
                numerics.reference_pad,
            ) || m.logical_offset != index * n
            {
                return Err("components do not tile one fused reference matrix".into());
            }
            let plan = shard.plan();
            if (plan.in_features(), plan.out_features(), plan.bits()) != (k, n, numerics.bits) {
                return Err("component shard does not match its workspace".into());
            }
        }
        if numerics.logical_n != count * n {
            return Err(format!(
                "logical width {} is not {count} x {n}",
                numerics.logical_n
            ));
        }
        // gemm3 (KDA q|k|v): bits 6, one shared input transform.
        // gemm2 (shared expert gate|up): bits 5/6, per-component inputs.
        let shared_input = count == 3;
        if (count == 3 && numerics.bits != 6) || (count == 2 && !matches!(numerics.bits, 5 | 6)) {
            return Err(format!(
                "bits {} not instantiated (fused: 6)",
                numerics.bits
            ));
        }
        // One reference matrix has one input transform: the three q|k|v
        // components must carry the same input scales (slices of one tensor).
        if shared_input {
            let mut scales = Vec::new();
            for (_, shard) in parts {
                let suh = shard.buffers().map_err(|e| format!("{e:#}"))?[1];
                let mut bytes = vec![0u8; k * 2];
                gpu.copy_d2h(suh, &mut bytes)
                    .map_err(|e| format!("{e:#}"))?;
                scales.push(bytes);
            }
            if scales.iter().any(|s| *s != scales[0]) {
                return Err("component input scales differ".into());
            }
        }
        let input_kernel = gpu
            .kernel("exl3_coop_gemm", "exl3_coop_input_hadamard_f16h")
            .map_err(|e| format!("{e:#}"))?;
        let sm_count = gpu.sm_count().unwrap_or(0);
        let mut plans = [None; super::numerics::EXL3_PROFILE_SLOTS];
        for (bucket, slot) in plans.iter_mut().enumerate() {
            let Some(profile) = numerics.profile_at(bucket) else {
                continue;
            };
            let (tk, tn) = (profile.tile_k as usize, profile.tile_n as usize);
            let tag = format!(
                "K{k}/N{count}x{n} tile ({tk},{tn}) slices {}",
                profile.slices
            );
            let instantiated = match count {
                3 => matches!((tk, tn), (32, 128) | (32, 256) | (16, 512)),
                _ => matches!((tk, tn), (32, 128) | (32, 256)),
            };
            if !instantiated || profile.half_partial {
                warn_fused_once(format!("{tag}: profile not instantiated"));
                continue;
            }
            if !n.is_multiple_of(tn) || !k.is_multiple_of(tk) {
                warn_fused_once(format!("{tag}: component width not a tile multiple"));
                continue;
            }
            if profile.slices == 0 || profile.slices > sm_count {
                warn_fused_once(format!("{tag}: slices exceed {sm_count} SMs"));
                continue;
            }
            let name = format!("exl3_coop_gemm{count}_b{}_k{tk}_n{tn}", numerics.bits);
            match gpu.kernel("exl3_coop_gemm", &name) {
                Ok(kernel) => {
                    let fused_rows = fused_input_rows(tk, tn, numerics.bits as usize, k);
                    let fused_input = (shared_input && coop_fused_input() && fused_rows > 0)
                        .then(|| {
                            let name = format!("exl3_coop_gemm3h_b{}_k{tk}_n{tn}", numerics.bits);
                            gpu.kernel("exl3_coop_gemm", &name)
                                .map_err(|e| {
                                    warn_fused_once(format!(
                                        "{tag}: fused input unavailable ({e:#})"
                                    ))
                                })
                                .ok()
                        })
                        .flatten();
                    *slot = Some(FusedPlan {
                        kernel,
                        block: BASE_THREADS * profile.tile_k / 16,
                        slices: profile.slices,
                        fused_input,
                        fused_rows,
                    })
                }
                Err(e) => warn_fused_once(format!("{tag}: {e:#}")),
            }
        }
        if plans.iter().all(Option::is_none) {
            return Err("no row bucket has a fused plan".into());
        }
        let lock_bytes =
            (first.capacity as usize).div_ceil(CHUNK_ROWS) * (count * n / LOCK_TILE_N) * 4;
        let locks = gpu.alloc(lock_bytes).map_err(|e| format!("{e:#}"))?;
        if let Err(e) = gpu.memset(locks, 0, lock_bytes) {
            let _ = gpu.free(locks);
            return Err(format!("{e:#}"));
        }
        Ok(Self {
            input_kernel,
            plans,
            locks,
            numerics,
            count,
            shared_input,
            k: first.k,
            comp_n: first.n,
            capacity: first.capacity,
            stream: first.stream,
            disabled: AtomicBool::new(false),
        })
    }

    /// Launch q|k|v as one cooperative GEMM, then each component's unchanged
    /// output Hadamard into its own BF16 output. `Ok(false)`: nothing usable
    /// was enqueued and the caller must run the per-component launches.
    pub fn launch_bf16(
        &self,
        gpu: &dyn GpuBackend,
        parts: &[(
            &super::Exl3LinearWorkspace,
            &crate::weights::exl3::tp::Exl3DeviceShard,
            DevicePtr,
        )],
        input: DevicePtr,
        rows: usize,
    ) -> Result<bool> {
        if rows == 0 || self.disabled.load(Ordering::Relaxed) || self.locks.is_null() {
            return Ok(false);
        }
        anyhow::ensure!(
            parts.len() == self.count,
            "EXL3 fused launch component count mismatch"
        );
        anyhow::ensure!(
            rows <= self.capacity as usize,
            "EXL3 fused batch exceeds workspace capacity"
        );
        let Some(plan) = self.plans[self.numerics.profile_slot(rows)?] else {
            return Ok(false);
        };
        let mut trellis = Vec::with_capacity(self.count);
        let mut in_scales = Vec::with_capacity(self.count);
        let mut out_scales = Vec::with_capacity(self.count);
        for (ws, shard, output) in parts {
            anyhow::ensure!(
                (ws.k, ws.n, ws.stream) == (self.k, self.comp_n, self.stream)
                    && ws.buffers.iter().all(|p| !p.is_null()),
                "EXL3 fused component workspace mismatch or released"
            );
            anyhow::ensure!(
                !input.is_null()
                    && !output.is_null()
                    && parts.iter().all(|(other, _, _)| {
                        !other.buffers.contains(&input) && !other.buffers.contains(output)
                    }),
                "EXL3 fused activations must be non-null and not alias scratch"
            );
            let [t, i, o] = shard.buffers()?;
            trellis.push(t);
            in_scales.push(i);
            out_scales.push(o);
        }
        let rows32 = rows as u32;
        let fused_input = plan.fused_input.filter(|_| rows <= plan.fused_rows);
        let transforms = if fused_input.is_some() {
            0
        } else if self.shared_input {
            1
        } else {
            self.count
        };
        for index in 0..transforms {
            KernelLaunch::new(gpu, self.input_kernel)
                .grid([self.k / 128, rows32, 1])
                .block([128, 1, 1])
                .arg_ptr(input)
                .arg_ptr(in_scales[index])
                .arg_ptr(parts[index].0.buffers[0])
                .arg_u32(rows32)
                .arg_u32(self.k)
                .launch(self.stream)?;
        }
        let mut launch = KernelLaunch::new(gpu, fused_input.unwrap_or(plan.kernel))
            .grid([plan.slices, 1, 1])
            .block([plan.block, 1, 1])
            .shared_mem(COOP_SMEM_BYTES);
        if fused_input.is_some() {
            // A is the raw BF16 input; the transform runs in the prologue.
            launch = launch.arg_ptr(input);
        } else if self.shared_input {
            launch = launch.arg_ptr(parts[0].0.buffers[0]);
        } else {
            for (ws, _, _) in parts {
                launch = launch.arg_ptr(ws.buffers[0]);
            }
        }
        for &t in &trellis {
            launch = launch.arg_ptr(t);
        }
        for (ws, _, _) in parts {
            launch = launch.arg_ptr(ws.buffers[1]);
        }
        let launched = launch
            .arg_u32(rows32)
            .arg_u32(self.k)
            .arg_u32(self.comp_n)
            .arg_ptr(self.locks)
            .arg_u32(plan.slices);
        let launched = if fused_input.is_some() {
            launched.arg_ptr(in_scales[0])
        } else {
            launched
        }
        .launch_cooperative(self.stream);
        if let Err(e) = launched {
            self.disabled.store(true, Ordering::Relaxed);
            warn_fused_once(format!("cooperative launch failed ({e:#})"));
            return Ok(false);
        }
        for (index, (ws, _, output)) in parts.iter().enumerate() {
            KernelLaunch::new(gpu, ws.kernels[2])
                .grid([self.comp_n / 128, rows32, 1])
                .block([128, 1, 1])
                .arg_ptr(ws.buffers[1])
                .arg_ptr(out_scales[index])
                .arg_ptr(*output)
                .arg_u32(rows32)
                .arg_u32(self.comp_n)
                .launch(self.stream)?;
        }
        Ok(true)
    }

    pub fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !self.locks.is_null() {
            gpu.free(self.locks)?;
            self.locks = DevicePtr::NULL;
        }
        Ok(())
    }
}
