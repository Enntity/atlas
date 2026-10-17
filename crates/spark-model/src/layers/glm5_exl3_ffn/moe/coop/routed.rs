// SPDX-License-Identifier: AGPL-3.0-only
//! Grouped multi-row launch and the three-launch v9 routed chain.

use super::*;

impl NativeMoeFfn {
    /// Grouped multi-row launch (`exl3_coop_moe8_*`): fixed co-resident grid,
    /// the item count (projections x groups x chunks) is read on the device.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn coop_launch_multirow(
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
    pub(in crate::layers::glm5_exl3_ffn::moe) fn coop_routed_v9(
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
