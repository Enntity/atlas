// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
impl NativeMoeFfn {
    pub fn forward_local(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        self.forward_local_captured(gpu, input, output, rows)
            .map(|_| ())
    }
    pub(in crate::layers::glm5_exl3_ffn) fn forward_local_captured(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<Option<usize>> {
        self.forward_local_profiled(gpu, input, output, rows, false)
    }
    pub(in crate::layers::glm5_exl3_ffn) fn forward_local_profiled(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        enabled: bool,
    ) -> Result<Option<usize>> {
        self.validate(rows, self.stream, self.rank, self.world)?;
        if rows == 0 {
            return Ok(None);
        }
        let _scratch_lease = self.scratch.acquire()?;
        let timer = profile::FfnProfile::new(
            gpu,
            self.stream,
            enabled,
            profile::ProfileMetadata {
                layer: self.layer,
                rows,
                rank: self.rank,
                world: self.world,
                routes: rows * self.topk,
                role: "routed",
                precision: Some(self.precision.matmul_args()),
                thin_capacity: Some(self.precision.thin_capacity),
            },
        );
        ensure!(
            !input.is_null() && !output.is_null(),
            "EXL3 MoE activations missing"
        );
        let invocation = self
            .capture
            .as_ref()
            .map(diagnostics::Exl3FfnRouter::begin)
            .transpose()?
            .flatten();
        let hshape = [rows, self.hidden];
        let routes = [rows, self.topk];
        let midshape = [rows, self.topk, self.intermediate];
        let downshape = [rows, self.topk, self.hidden];
        use diagnostics::DiagnosticDtype::{Bf16, F32, I32};
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "input",
            &[("input", Bf16, input, &hshape)],
            self.stream,
        )?;
        let n = u32::try_from(rows).context("EXL3 MoE rows ABI overflow")?;
        let h = u32::try_from(self.hidden).context("EXL3 MoE hidden ABI overflow")?;
        let e = u32::try_from(self.experts).context("EXL3 MoE experts ABI overflow")?;
        let topk = u32::try_from(self.topk).context("EXL3 MoE topk ABI overflow")?;
        timer.measure("router_gemm_transport", || {
            gpu.bf16_gemm_act_weight_t(spark_runtime::gpu::Bf16Gemm {
                numerics: spark_runtime::gpu::Bf16GemmNumerics::CublasLt,
                input,
                weight: self.companions[0].ptr()?,
                output: self.buffers[10],
                rows: n,
                columns: e,
                inner: h,
                stream: self.stream,
            })?;
            let router_values = n
                .checked_mul(e)
                .context("EXL3 router conversion size overflow")?;
            KernelLaunch::new(gpu, self.kernels[0])
                .grid([router_values.div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buffers[10])
                .arg_ptr(self.buffers[0])
                .arg_u32(router_values)
                .launch(self.stream)?;
            Ok(())
        })?;
        timer.measure("topk", || {
            KernelLaunch::new(gpu, self.kernels[1])
                .grid([n, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buffers[0])
                .arg_ptr(self.companions[1].ptr()?)
                .arg_ptr(self.buffers[1])
                .arg_ptr(self.buffers[2])
                .arg_u32(n)
                .arg_u32(e)
                .arg_u32(topk)
                .arg_u32(u32::from(self.normalize))
                .arg_f32(self.scale)
                .launch(self.stream)?;
            Ok(())
        })?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "router",
            &[
                ("logits", F32, self.buffers[0], &[rows, self.experts]),
                ("ids", I32, self.buffers[1], &routes),
                ("weights", F32, self.buffers[2], &routes),
            ],
            self.stream,
        )?;
        timer.measure("route_counts", || {
            KernelLaunch::new(gpu, self.kernels[7])
                .grid([e.div_ceil(128), 1, 1])
                .block([128, 1, 1])
                .arg_ptr(self.buffers[1])
                .arg_ptr(self.buffers[9])
                .arg_u32(n * topk)
                .arg_u32(e)
                .launch(self.stream)?;
            Ok(())
        })?;
        timer.measure("gate_projection", || {
            self.project(gpu, 0, input, self.buffers[3], rows)
        })?;
        timer.measure("up_projection", || {
            self.project(gpu, 1, input, self.buffers[4], rows)
        })?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "gate_up",
            &[
                ("gate", F32, self.buffers[3], &midshape),
                ("up", F32, self.buffers[4], &midshape),
            ],
            self.stream,
        )?;
        timer.measure("activation_down_hadamard", || self.activate_down(gpu, rows))?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "down_input_hadamard",
            &[("xh", F32, self.buffers[7], &midshape)],
            self.stream,
        )?;
        timer.measure("down_mma", || self.matmul(gpu, 2, rows))?;
        timer.measure("weighted_down_hadamard", || {
            self.output_hadamard(gpu, 2, self.buffers[5], rows)
        })?;
        self.shared
            .forward_local_profiled(gpu, input, self.buffers[6], rows, enabled)?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "down_shared",
            &[
                ("down_weighted", F32, self.buffers[5], &downshape),
                ("shared", Bf16, self.buffers[6], &hshape),
            ],
            self.stream,
        )?;
        let count = rows
            .checked_mul(self.hidden)
            .context("EXL3 MoE combine size overflow")?;
        timer.measure("combine", || {
            KernelLaunch::new(gpu, self.kernels[5])
                .grid([
                    u32::try_from(count.div_ceil(256)).context("EXL3 combine grid overflow")?,
                    1,
                    1,
                ])
                .block([256, 1, 1])
                .arg_ptr(self.buffers[5])
                .arg_ptr(self.buffers[6])
                .arg_ptr(output)
                .arg_u32(n)
                .arg_u32(h)
                .arg_u32(topk)
                .arg_u32(1)
                .launch(self.stream)?;
            Ok(())
        })?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "local_output",
            &[("output", Bf16, output, &hshape)],
            self.stream,
        )?;
        Ok(invocation)
    }
}
