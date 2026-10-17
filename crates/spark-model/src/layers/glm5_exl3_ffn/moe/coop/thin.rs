// SPDX-License-Identifier: AGPL-3.0-only
//! Thin routed projections on the cooperative GEMM: the per-projection
//! launch and the one-launch gate|up pair.

use super::*;

impl NativeMoeFfn {
    /// Launch the cooperative thin matmul for projection `index` if this
    /// launch is entirely thin. `Ok(false)` means the caller must run the
    /// existing kernel (nothing was enqueued).
    pub(in crate::layers::glm5_exl3_ffn::moe) fn coop_matmul(
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
    pub(super) fn coop_chunks(k: usize, n: usize, slices: usize) -> usize {
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
    pub(in crate::layers::glm5_exl3_ffn::moe) fn coop_gate_up_outputs(
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

    pub(in crate::layers::glm5_exl3_ffn::moe) fn coop_gate_up(
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
