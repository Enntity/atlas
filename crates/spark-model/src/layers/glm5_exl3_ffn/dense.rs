// SPDX-License-Identifier: AGPL-3.0-only

//! Native SwiGLU execution with one workspace shared by gate/up.

mod setup;

use super::*;
use spark_runtime::weights::exl3::native::Exl3LinearWorkspace;

enum DenseProjections {
    Packed {
        weights: Box<[Exl3DeviceShard; 3]>,
        workspaces: Box<[Exl3LinearWorkspace; 2]>,
    },
    Reconstructed(super::shared::SharedBf16Weights),
}
pub struct NativeDenseFfn {
    pub(super) layer: Option<usize>,
    pub(super) capture: Option<diagnostics::Exl3FfnRouter>,
    projections: DenseProjections,
    activations: [DevicePtr; 2],
    merged_activation: DevicePtr,
    silu: KernelHandle,
    activation: NativeDenseActivation,
    role: NativeDenseFfnRole,
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) rank: usize,
    pub(super) world: usize,
    pub(super) capacity: usize,
    pub(super) stream: u64,
    active: bool,
}

impl NativeDenseFfn {
    pub fn is_reconstructed(&self) -> bool {
        matches!(self.projections, DenseProjections::Reconstructed(_))
    }
    fn project(
        &self,
        gpu: &dyn GpuBackend,
        index: usize,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        match &self.projections {
            DenseProjections::Packed {
                weights,
                workspaces,
            } => workspaces[usize::from(index == 2)].launch_bf16(
                gpu,
                &weights[index],
                input,
                output,
                rows,
            ),
            DenseProjections::Reconstructed(weights) => {
                ensure!(
                    index == 0 || index == 2,
                    "shared gate/up is one merged projection"
                );
                let (inner, columns, weight, destination) = if index == 2 {
                    (self.intermediate, self.hidden, weights.pointers[1], output)
                } else {
                    (
                        self.hidden,
                        self.intermediate * 2,
                        weights.pointers[0],
                        self.merged_activation,
                    )
                };
                gpu.bf16_gemm_act_weight_t(spark_runtime::gpu::Bf16Gemm {
                    numerics: spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic,
                    input,
                    weight,
                    output: destination,
                    rows: u32::try_from(rows).context("shared rows ABI overflow")?,
                    columns: columns as u32,
                    inner: inner as u32,
                    stream: self.stream,
                })?;
                if index == 0 {
                    super::shared::split_merged_bf16(
                        gpu,
                        self.merged_activation,
                        self.activations[0],
                        self.activations[1],
                        rows,
                        self.intermediate,
                        self.stream,
                    )?;
                }
                Ok(())
            }
        }
    }
    pub(super) fn from_shared(
        gpu: &dyn GpuBackend,
        mut weights: super::shared::SharedBf16Weights,
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
    ) -> Result<Self> {
        let mut activations = [DevicePtr::NULL; 2];
        let mut merged_activation = DevicePtr::NULL;
        let build = (|| -> Result<KernelHandle> {
            weights.validate()?;
            ensure!(
                capacity > 0 && u32::try_from(capacity).is_ok(),
                "shared capacity must fit positive ABI"
            );
            gpu.require_bf16_gemm_numerics(spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic)?;
            activation.validate()?;
            let silu = gpu.kernel("exl3", "exl3_silu_mul_dense")?;
            let bytes = capacity
                .checked_mul(weights.intermediate)
                .and_then(|n| n.checked_mul(2))
                .context("shared FFN activation bytes overflow")?;
            for p in &mut activations {
                *p = gpu.alloc(bytes)?;
            }
            merged_activation = gpu.alloc(
                bytes
                    .checked_mul(2)
                    .context("shared merged activation bytes overflow")?,
            )?;
            Ok(silu)
        })();
        match build {
            Ok(silu) => Ok(Self {
                layer: None,
                capture: None,
                hidden: weights.hidden,
                intermediate: weights.intermediate,
                rank: weights.rank,
                world: weights.world,
                projections: DenseProjections::Reconstructed(weights),
                activations,
                merged_activation,
                silu,
                activation,
                role: NativeDenseFfnRole::SharedExperts,
                capacity,
                stream,
                active: true,
            }),
            Err(error) => {
                let mut cleanup = None;
                retain_error(&mut cleanup, weights.release(gpu));
                for p in activations.into_iter().chain([merged_activation]) {
                    if !p.is_null() {
                        retain_error(&mut cleanup, gpu.free(p));
                    }
                }
                Err(match cleanup {
                    None => error,
                    Some(e) => error.context(format!("shared FFN rollback: {e:#}")),
                })
            }
        }
    }
    pub fn role(&self) -> NativeDenseFfnRole {
        self.role
    }
    pub fn activation(&self) -> NativeDenseActivation {
        self.activation
    }
    pub fn validate(&self, rows: usize, stream: u64, rank: usize, world: usize) -> Result<()> {
        ensure!(
            self.active
                && rows <= self.capacity
                && stream == self.stream
                && rank == self.rank
                && world == self.world,
            "EXL3 FFN lifetime/capacity/stream/topology mismatch"
        );
        match &self.projections {
            DenseProjections::Packed { weights, .. } => {
                for w in weights.iter() {
                    w.buffers()?;
                }
            }
            DenseProjections::Reconstructed(weights) => weights.validate()?,
        }
        Ok(())
    }
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
    pub(super) fn forward_local_captured(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<Option<usize>> {
        self.forward_local_profiled(gpu, input, output, rows, false)
    }
    pub(super) fn forward_local_profiled(
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
        ensure!(
            !input.is_null() && !output.is_null(),
            "EXL3 FFN input/output must be non-null"
        );
        let timer = profile::FfnProfile::new(
            gpu,
            self.stream,
            enabled,
            profile::ProfileMetadata {
                layer: self.layer,
                rows,
                rank: self.rank,
                world: self.world,
                routes: 0,
                role: if self.is_reconstructed() {
                    "shared_reconstructed"
                } else {
                    "dense_packed"
                },
                precision: None,
                thin_capacity: None,
            },
        );
        let invocation = self
            .capture
            .as_ref()
            .map(diagnostics::Exl3FfnRouter::begin)
            .transpose()?
            .flatten();
        let h = [rows, self.hidden];
        let mid = [rows, self.intermediate];
        use diagnostics::DiagnosticDtype::Bf16;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "input",
            &[("input", Bf16, input, &h)],
            self.stream,
        )?;
        timer.measure(
            if self.is_reconstructed() {
                "merged_gate_up_gemm_split"
            } else {
                "gate_projection"
            },
            || self.project(gpu, 0, input, self.activations[0], rows),
        )?;
        if !self.is_reconstructed() {
            timer.measure("up_projection", || {
                self.project(gpu, 1, input, self.activations[1], rows)
            })?;
        }
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "gate_up",
            &[
                ("gate", Bf16, self.activations[0], &mid),
                ("up", Bf16, self.activations[1], &mid),
            ],
            self.stream,
        )?;
        let count = rows
            .checked_mul(self.intermediate)
            .context("EXL3 FFN activation count overflow")?;
        timer.measure("activation", || {
            self.activation.launch(
                gpu,
                self.silu,
                self.activations[0],
                self.activations[1],
                count,
                self.stream,
            )
        })?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "activation",
            &[("activation", Bf16, self.activations[0], &mid)],
            self.stream,
        )?;
        timer.measure("down_projection", || {
            self.project(gpu, 2, self.activations[0], output, rows)
        })?;
        diagnostics::capture(
            &self.capture,
            invocation,
            gpu,
            "local_output",
            &[("output", Bf16, output, &h)],
            self.stream,
        )?;
        Ok(invocation)
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for NativeDenseFfn {
    fn label(&self) -> &'static str {
        "GLM native EXL3 FFN"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        self.active = false;
        let mut error = None;
        match &mut self.projections {
            DenseProjections::Packed {
                weights,
                workspaces,
            } => {
                for w in weights.iter_mut() {
                    retain_error(&mut error, w.release(gpu));
                }
                for w in workspaces.iter_mut() {
                    retain_error(&mut error, w.release(gpu));
                }
            }
            DenseProjections::Reconstructed(weights) => {
                retain_error(&mut error, weights.release(gpu))
            }
        }
        for ptr in self
            .activations
            .iter_mut()
            .chain(std::iter::once(&mut self.merged_activation))
        {
            if !ptr.is_null() {
                match gpu.free(*ptr) {
                    Ok(()) => *ptr = DevicePtr::NULL,
                    Err(e) => retain_error(&mut error, Err(e)),
                }
            }
        }
        finish(error)
    }
}
