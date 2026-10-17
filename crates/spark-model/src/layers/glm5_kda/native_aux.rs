// SPDX-License-Identifier: AGPL-3.0-only

//! Merged EXL3 KDA auxiliary projection: one classic math-0 BF16 GEMM over
//! the concatenated resident `[B; FA; GA]` rows, then row-strided scatter into
//! the existing beta/FA/GA buffers. The merged weight is borrowed; the dense
//! companion bundle owns it. Only the padded input/output scratch is owned.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::{Bf16Gemm, Bf16GemmNumerics, DevicePtr, GpuBackend};

/// Operator row minimum. M=1 pads to 3 (component-proven exact against the
/// reference); M=2 also pads to 3 but is unqualified.
const MIN_OPERATOR_ROWS: usize = 3;
const NUMERICS: Bf16GemmNumerics = Bf16GemmNumerics::Glm53AuxiliaryClassic;

pub(crate) struct NativeKdaAux {
    weight: DevicePtr,
    inner: usize,
    widths: [usize; 3],
    capacity: usize,
    stream: u64,
    /// `[max(capacity, 3), inner]` BF16; only rows below 3 are staged here.
    input: DevicePtr,
    /// `[max(capacity, 3), sum(widths)]` BF16 merged GEMM output.
    output: DevicePtr,
}

impl NativeKdaAux {
    /// `weight` is `[heads + 2 * dim, inner]` BF16, rows ordered B, FA, GA.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        weight: DevicePtr,
        inner: usize,
        heads: usize,
        dim: usize,
        capacity: usize,
        stream: u64,
    ) -> Result<Self> {
        ensure!(!weight.is_null(), "KDA merged auxiliary weight is not live");
        ensure!(
            inner > 0 && heads > 0 && dim > 0 && capacity > 0,
            "KDA merged auxiliary geometry must be positive"
        );
        let widths = [heads, dim, dim];
        let columns = heads
            .checked_add(dim * 2)
            .context("KDA merged auxiliary width overflow")?;
        let rows = capacity.max(MIN_OPERATOR_ROWS);
        for n in [rows, columns, inner] {
            ensure!(
                u32::try_from(n).is_ok_and(|n| n <= i32::MAX as u32),
                "KDA merged auxiliary dimension exceeds GEMM ABI"
            );
        }
        let bytes = |n: usize| {
            rows.checked_mul(n)
                .and_then(|v| v.checked_mul(2))
                .context("KDA merged auxiliary scratch overflow")
        };
        let (input_bytes, output_bytes) = (bytes(inner)?, bytes(columns)?);
        gpu.require_bf16_gemm_numerics(NUMERICS)?;
        let mut aux = Self {
            weight,
            inner,
            widths,
            capacity,
            stream,
            input: DevicePtr::NULL,
            output: DevicePtr::NULL,
        };
        let allocated = (|| -> Result<()> {
            aux.input = gpu.alloc(input_bytes)?;
            aux.output = gpu.alloc(output_bytes)?;
            Ok(())
        })();
        if let Err(error) = allocated {
            return Err(match aux.release(gpu) {
                Ok(()) => error,
                Err(cleanup) => error.context(format!(
                    "KDA merged auxiliary rollback also failed: {cleanup:#}"
                )),
            });
        }
        Ok(aux)
    }

    fn columns(&self) -> usize {
        self.widths.iter().sum()
    }

    #[cfg(test)]
    pub(crate) fn output(&self) -> DevicePtr {
        self.output
    }

    /// Validate everything before enqueueing any work.
    fn validate(
        &self,
        input: DevicePtr,
        outputs: [DevicePtr; 3],
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            !self.input.is_null() && !self.output.is_null(),
            "KDA merged auxiliary scratch has been released"
        );
        ensure!(
            stream == self.stream,
            "KDA merged auxiliary stream mismatch"
        );
        ensure!(
            rows > 0 && rows <= self.capacity,
            "KDA merged auxiliary rows {rows} exceed capacity {}",
            self.capacity
        );
        ensure!(
            !input.is_null() && outputs.iter().all(|p| !p.is_null()),
            "KDA merged auxiliary requires live buffers"
        );
        ensure!(
            outputs[0] != outputs[1] && outputs[0] != outputs[2] && outputs[1] != outputs[2],
            "KDA merged auxiliary outputs must be distinct"
        );
        Ok(())
    }

    /// `input[rows, inner]` → beta `[rows, heads]`, FA/GA `[rows, dim]`.
    pub(crate) fn forward(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        outputs: [DevicePtr; 3],
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.validate(input, outputs, rows, stream)?;
        let row_bytes = self.inner * 2;
        let operator_rows = rows.max(MIN_OPERATOR_ROWS);
        let gemm_input = if rows < MIN_OPERATOR_ROWS {
            gpu.copy_d2d_async(input, self.input, rows * row_bytes, stream)?;
            gpu.memset_async(
                self.input.offset(rows * row_bytes),
                0,
                (MIN_OPERATOR_ROWS - rows) * row_bytes,
                stream,
            )?;
            self.input
        } else {
            input
        };
        let columns = self.columns();
        gpu.bf16_gemm_act_weight_t(Bf16Gemm {
            input: gemm_input,
            weight: self.weight,
            output: self.output,
            rows: operator_rows as u32,
            columns: columns as u32,
            inner: self.inner as u32,
            stream,
            numerics: NUMERICS,
        })?;
        // Only physical rows leave the scratch; padded rows are discarded.
        let pitch = columns * 2;
        let mut offset = 0;
        for (destination, width) in outputs.into_iter().zip(self.widths) {
            gpu.copy_d2d_2d_async(
                self.output.offset(offset * 2),
                pitch,
                destination,
                width * 2,
                width * 2,
                rows,
                stream,
            )?;
            offset += width;
        }
        Ok(())
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for NativeKdaAux {
    fn label(&self) -> &'static str {
        "GLM EXL3 KDA merged auxiliary scratch"
    }
    /// Failed frees keep their pointer so a later release retries them.
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut first = None;
        for ptr in [&mut self.output, &mut self.input] {
            if ptr.is_null() {
                continue;
            }
            match gpu.free(*ptr) {
                Ok(()) => *ptr = DevicePtr::NULL,
                Err(error) => {
                    first.get_or_insert(error);
                }
            }
        }
        first.map_or(Ok(()), Err)
    }
}
