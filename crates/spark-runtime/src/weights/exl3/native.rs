// SPDX-License-Identifier: AGPL-3.0-only

//! Native packed EXL3 launch boundary.
//!
//! Correctness-first BF16 linear: input scale/H128 → packed matmul →
//! output H128/scale. Keeps checkpoint bytes packed for both decode and prefill.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;

use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use crate::kernel_args::KernelLaunch;

use super::{HADAMARD_WIDTH, tp::Exl3DeviceShard};
mod numerics;
pub use numerics::{Exl3CooperativeProfile, Exl3LinearNumerics};

/// Explicit rounding at the packed linear's input-transform boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3InputPrecision {
    /// F32 input-scale product and F32 normalized Hadamard output.
    F32,
    /// F16 input-scale product, F32 butterflies, then F16 normalized output.
    /// Scratch remains F32-owned and stores exactly representable F16 values.
    F16,
}

impl Exl3InputPrecision {
    fn kernel_name(self) -> &'static str {
        match self {
            Self::F32 => "exl3_input_hadamard",
            Self::F16 => "exl3_input_hadamard_f16",
        }
    }
}

/// Reusable activation scratch and resolved kernels, bound to one compute
/// stream. All launches on this workspace use that stream, so asynchronous
/// stages cannot overwrite another stream's pending intermediate values.
/// Callers must use the backend/device that owns the shard and workspace.
pub struct Exl3LinearWorkspace {
    k: u32,
    n: u32,
    capacity: u32,
    stream: u64,
    buffers: [DevicePtr; 2],
    kernels: [KernelHandle; 3],
    numerics: Option<Exl3LinearNumerics>,
}

impl Exl3LinearWorkspace {
    /// Allocate only F32 activation scratch, never a dense decoded matrix.
    /// Resolve the complete kernel family before allocating any buffers.
    pub fn new(
        gpu: &dyn GpuBackend,
        k: usize,
        n: usize,
        capacity: usize,
        stream: u64,
    ) -> Result<Self> {
        Self::new_with_input_precision(gpu, k, n, capacity, stream, Exl3InputPrecision::F32)
    }

    /// Allocate scratch with an explicitly selected input rounding contract.
    /// Required kernels resolve before allocations; no fallback changes arithmetic.
    pub fn new_with_input_precision(
        gpu: &dyn GpuBackend,
        k: usize,
        n: usize,
        capacity: usize,
        stream: u64,
        input_precision: Exl3InputPrecision,
    ) -> Result<Self> {
        Self::build(gpu, k, n, capacity, stream, input_precision, None)
    }

    /// Explicit cooperative arithmetic; unsupported geometry fails before GPU
    /// allocation. Legacy constructors keep their scalar accumulation policy.
    pub fn new_with_numerics(
        gpu: &dyn GpuBackend,
        k: usize,
        n: usize,
        capacity: usize,
        stream: u64,
        numerics: Exl3LinearNumerics,
    ) -> Result<Self> {
        numerics.validate(k, n, capacity)?;
        Self::build(
            gpu,
            k,
            n,
            capacity,
            stream,
            Exl3InputPrecision::F16,
            Some(numerics),
        )
    }

    /// Resolve the complete arithmetic family before a checkpoint loader
    /// uploads weights. Performs no allocation or GPU execution.
    pub fn preflight_numerics(
        gpu: &dyn GpuBackend,
        k: usize,
        n: usize,
        capacity: usize,
        numerics: Exl3LinearNumerics,
    ) -> Result<()> {
        numerics.validate(k, n, capacity)?;
        Self::dimensions(k, n, capacity)?;
        Self::resolve_kernels(gpu, Exl3InputPrecision::F16, true).map(|_| ())
    }

    fn resolve_kernels(
        gpu: &dyn GpuBackend,
        input_precision: Exl3InputPrecision,
        cooperative: bool,
    ) -> Result<[KernelHandle; 3]> {
        Ok([
            gpu.kernel("exl3", input_precision.kernel_name())?,
            if cooperative {
                gpu.kernel("exl3_linear_mma", "exl3_mul1_matmul_mma")?
            } else {
                gpu.kernel("exl3", "exl3_mul1_matmul")?
            },
            gpu.kernel("exl3", "exl3_output_hadamard")?,
        ])
    }

    fn dimensions(k: usize, n: usize, capacity: usize) -> Result<[u32; 2]> {
        ensure!(
            k > 0 && n > 0 && k.is_multiple_of(HADAMARD_WIDTH) && n.is_multiple_of(HADAMARD_WIDTH),
            "EXL3 native dimensions must be positive H128 multiples"
        );
        ensure!(
            (1..=65535).contains(&capacity),
            "EXL3 batch capacity must fit CUDA grid.y (1..=65535)"
        );
        Ok([
            u32::try_from(k).context("EXL3 input width exceeds kernel ABI")?,
            u32::try_from(n).context("EXL3 output width exceeds kernel ABI")?,
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: &dyn GpuBackend,
        k: usize,
        n: usize,
        capacity: usize,
        stream: u64,
        input_precision: Exl3InputPrecision,
        numerics: Option<Exl3LinearNumerics>,
    ) -> Result<Self> {
        let [k_arg, n_arg] = Self::dimensions(k, n, capacity)?;
        let scratch_size = |width: usize| {
            capacity
                .checked_mul(width)
                .and_then(|elements| elements.checked_mul(4))
                .context("EXL3 activation scratch size overflow")
        };
        let sizes = [scratch_size(k)?, scratch_size(n)?];
        let kernels = Self::resolve_kernels(gpu, input_precision, numerics.is_some())?;
        let mut result = Self {
            k: k_arg,
            n: n_arg,
            capacity: capacity as u32,
            stream,
            buffers: [DevicePtr::NULL; 2],
            kernels,
            numerics,
        };
        for (slot, size) in sizes.into_iter().enumerate() {
            match gpu.alloc(size) {
                Ok(ptr) => result.buffers[slot] = ptr,
                Err(error) => {
                    let cleanup = result.release(gpu);
                    return Err(match cleanup {
                        Ok(()) => error,
                        Err(cleanup) => {
                            error.context(format!("EXL3 scratch cleanup also failed: {cleanup:#}"))
                        }
                    });
                }
            }
        }
        Ok(result)
    }

    /// Apply this rank's linear to `[rows,k]` BF16 activations, returning
    /// `[rows,n]` BF16 outputs. Does not perform a collective: the caller
    /// sums row-parallel partials or retains column-parallel local outputs.
    pub fn launch_bf16(
        &self,
        gpu: &dyn GpuBackend,
        shard: &Exl3DeviceShard,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        self.launch_output(gpu, shard, input, output, rows, self.kernels[2])
    }

    /// Native F32 vocabulary logits without intermediate BF16 rounding.
    pub fn launch_f32(
        &self,
        gpu: &dyn GpuBackend,
        shard: &Exl3DeviceShard,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        let kernel = gpu.kernel("exl3", "exl3_output_hadamard_f32")?;
        self.launch_output(gpu, shard, input, output, rows, kernel)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_output(
        &self,
        gpu: &dyn GpuBackend,
        shard: &Exl3DeviceShard,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        output_kernel: KernelHandle,
    ) -> Result<()> {
        ensure!(
            rows <= self.capacity as usize,
            "EXL3 batch exceeds workspace capacity"
        );
        ensure!(
            shard.plan().in_features() == self.k as usize
                && shard.plan().out_features() == self.n as usize,
            "EXL3 shard dimensions do not match workspace"
        );
        ensure!(
            self.buffers.iter().all(|p| !p.is_null()),
            "EXL3 workspace has been released"
        );
        let [trellis, input_scale, output_scale] = shard.buffers()?;
        if let Some(numerics) = self.numerics {
            ensure!(
                shard.plan().bits() == numerics.bits,
                "EXL3 checkpoint bits differ from explicit numerical profile"
            );
        }
        if rows == 0 {
            return Ok(());
        }
        let matmul_args = self
            .numerics
            .map(|numerics| numerics.kernel_args(rows))
            .transpose()?;
        ensure!(
            !input.is_null() && !output.is_null(),
            "EXL3 activation buffers must be non-null"
        );
        ensure!(
            !self.buffers.contains(&input) && !self.buffers.contains(&output),
            "EXL3 activation buffers must not alias workspace scratch"
        );
        let rows = rows as u32;
        KernelLaunch::new(gpu, self.kernels[0])
            .grid([self.k / 128, rows, 1])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(input_scale)
            .arg_ptr(self.buffers[0])
            .arg_u32(rows)
            .arg_u32(self.k)
            .launch(self.stream)?;
        let mut matmul = KernelLaunch::new(gpu, self.kernels[1])
            .grid([self.n.div_ceil(128), rows, 1])
            .block([128, 1, 1])
            .arg_ptr(self.buffers[0])
            .arg_ptr(trellis)
            .arg_ptr(self.buffers[1])
            .arg_u32(rows)
            .arg_u32(self.n)
            .arg_u32(self.k)
            .arg_u32(u32::from(shard.plan().bits()));
        if let Some(arguments) = matmul_args {
            for value in arguments {
                matmul = matmul.arg_u32(value);
            }
        }
        matmul.launch(self.stream)?;
        KernelLaunch::new(gpu, output_kernel)
            .grid([self.n / 128, rows, 1])
            .block([128, 1, 1])
            .arg_ptr(self.buffers[1])
            .arg_ptr(output_scale)
            .arg_ptr(output)
            .arg_u32(rows)
            .arg_u32(self.n)
            .launch(self.stream)
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Exl3LinearWorkspace {
    fn label(&self) -> &'static str {
        "EXL3 linear scratch"
    }

    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut error = None;
        for buffer in self.buffers.iter_mut().rev() {
            if buffer.is_null() {
                continue;
            }
            match gpu.free(*buffer) {
                Ok(()) => *buffer = DevicePtr::NULL,
                Err(e) => {
                    if error.is_none() {
                        error = Some(e);
                    }
                }
            }
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests;
