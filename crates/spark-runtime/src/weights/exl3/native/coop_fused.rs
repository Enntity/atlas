// SPDX-License-Identifier: AGPL-3.0-only
//! One cooperative launch over the components of a fused reference matrix
//! (`exl3_coop_gemm2_*` / `exl3_coop_gemm3_*`).
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use super::Exl3LinearNumerics;
use super::coop::{
    BASE_THREADS, CHUNK_ROWS, COOP_SMEM_BYTES, LOCK_TILE_N, coop_fused_input, fused_input_rows,
};
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use crate::kernel_args::KernelLaunch;

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
