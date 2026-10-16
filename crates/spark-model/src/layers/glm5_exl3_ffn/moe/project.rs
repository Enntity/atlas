// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

impl NativeMoeFfn {
    fn dimensions(&self, index: usize, rows: usize) -> (u32, u32, u32, u32, u32, u32) {
        let (k, n) = if index == 2 {
            (self.intermediate, self.hidden)
        } else {
            (self.hidden, self.intermediate)
        };
        (
            rows as u32,
            k as u32,
            n as u32,
            self.topk as u32,
            self.experts as u32,
            (rows * self.topk) as u32,
        )
    }
    pub(super) fn project(
        &self,
        gpu: &dyn GpuBackend,
        index: usize,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        let (rows32, k, _, topk, experts, slots) = self.dimensions(index, rows);
        KernelLaunch::new(gpu, self.kernels[2])
            .grid([k / 128, slots, 1])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[index].ptr)
            .arg_ptr(self.buffers[7])
            .arg_u32(rows32)
            .arg_u32(k)
            .arg_u32(topk)
            .arg_u32(0)
            .arg_u32(experts)
            .launch(self.stream)?;
        self.matmul(gpu, index, rows)?;
        self.output_hadamard(gpu, index, output, rows)
    }
    pub(super) fn matmul(&self, gpu: &dyn GpuBackend, index: usize, rows: usize) -> Result<()> {
        if self.coop_matmul(gpu, index, rows)? {
            return Ok(());
        }
        let (rows, k, n, topk, experts, slots) = self.dimensions(index, rows);
        let columns_per_block = if spark_runtime::weights::exl3::native::split_mma() {
            16
        } else {
            128
        };
        let mut launch = KernelLaunch::new(gpu, self.kernels[3])
            .grid([n / columns_per_block, slots, 1])
            .block([128, 1, 1])
            .arg_ptr(self.buffers[7])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[index].ptr)
            .arg_ptr(self.buffers[8])
            .arg_ptr(self.buffers[9])
            .arg_u32(rows)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(topk)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity);
        for value in self.precision.matmul_args() {
            launch = launch.arg_u32(value);
        }
        launch.launch(self.stream)
    }
    pub(super) fn output_hadamard(
        &self,
        gpu: &dyn GpuBackend,
        index: usize,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        let (rows, _, n, topk, experts, slots) = self.dimensions(index, rows);
        let mut launch = KernelLaunch::new(gpu, self.kernels[if index == 2 { 8 } else { 4 }])
            .grid([n / 128, slots, 1])
            .block([128, 1, 1])
            .arg_ptr(self.buffers[8])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[index].ptr);
        if index == 2 {
            launch = launch.arg_ptr(self.buffers[2]);
        }
        launch
            .arg_ptr(output)
            .arg_ptr(self.buffers[9])
            .arg_u32(rows)
            .arg_u32(n)
            .arg_u32(topk)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity)
            .launch(self.stream)
    }
    pub(super) fn activate_down(&self, gpu: &dyn GpuBackend, rows: usize) -> Result<()> {
        let (rows, k, _, topk, experts, slots) = self.dimensions(2, rows);
        let (limit, clamp) = match self.activation {
            Exl3FfnActivation::SiLu => (0.0, 0),
            Exl3FfnActivation::ClampedSiLu { limit } => (limit, 1),
        };
        KernelLaunch::new(gpu, self.kernels[6])
            .grid([k / 128, slots, 1])
            .block([128, 1, 1])
            .arg_ptr(self.buffers[3])
            .arg_ptr(self.buffers[4])
            .arg_ptr(self.buffers[1])
            .arg_ptr(self.tables[2].ptr)
            .arg_ptr(self.buffers[7])
            .arg_ptr(self.buffers[9])
            .arg_u32(rows)
            .arg_u32(k)
            .arg_u32(topk)
            .arg_u32(experts)
            .arg_u32(self.precision.thin_capacity)
            .arg_f32(limit)
            .arg_u32(clamp)
            .launch(self.stream)
    }
}
