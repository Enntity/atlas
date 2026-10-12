// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in typed upstream capture; all I/O stays in the diagnostic router.
use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::weights::exl3::ffn_io::{DiagnosticDtype, DiagnosticTensor, Exl3FfnRouter};

type TensorSpec<'a> = (&'static str, DiagnosticDtype, DevicePtr, &'a [usize]);
pub(super) struct UpstreamTrace(Exl3FfnRouter);
impl UpstreamTrace {
    pub(super) fn from_env(config: &ModelConfig, layer: usize) -> Result<Option<Self>> {
        let router = Exl3FfnRouter::from_env_named(
            layer,
            config.num_hidden_layers,
            config.tp_rank,
            config.tp_world_size,
            "ATLAS_EXL3_UPSTREAM_DUMP_DIR",
            "ATLAS_EXL3_UPSTREAM_DUMP_LAYERS",
        )?
        .map(Self);
        ensure!(
            router.is_none() || layer == 0,
            "EXL3 upstream capture supports explicit layer0 only"
        );
        Ok(router)
    }
    pub(super) fn begin(&self) -> Result<Option<usize>> {
        self.0.begin()
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_gated_norm_binding(
        &self,
        invocation: Option<usize>,
        gpu: &dyn GpuBackend,
        weight: DevicePtr,
        kernel: KernelHandle,
        tokens: u32,
        heads: u32,
        dim: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        let Some(invocation) = invocation else {
            return Ok(());
        };
        ensure!(
            eps.is_finite() && eps > 0.0,
            "gated norm binding epsilon invalid"
        );
        ensure!(
            kernel.0 != 0 && tokens > 0 && heads > 0 && dim > 0,
            "gated norm binding dispatch invalid"
        );
        ensure!(
            tokens.checked_mul(heads).is_some(),
            "gated norm binding grid overflow"
        );
        let (grid, block) = ops::kda_sigmoid_gated_norm_launch_dims(tokens, heads);
        self.0.capture_with_metadata(
            gpu,
            invocation,
            "gated_norm_binding",
            &[DiagnosticTensor {
                name: "weight",
                dtype: DiagnosticDtype::Bf16,
                shape: vec![dim as usize],
                ptr: weight,
            }],
            stream,
            &serde_json::json!({
                "schema": "atlas-kda-gated-norm-binding-v1",
                "eps_f32_bits": eps.to_bits(), "tokens": tokens, "heads": heads, "dim": dim,
                "kernel_handle": kernel.0, "kernel_module": "kda",
                "kernel_symbol": "kda_sigmoid_gated_rms_norm", "grid": grid, "block": block,
                "activation": "sigmoid", "affine": true, "bias": false,
                "residual": false, "prenorm": false,
                "input_dtype": "bf16", "gate_dtype": "bf16", "weight_dtype": "bf16",
                "accumulator_dtype": "f32", "output_dtype": "bf16"
            }),
        )
    }
    pub(super) fn capture(
        &self,
        invocation: Option<usize>,
        gpu: &dyn GpuBackend,
        stage: &str,
        tensors: &[TensorSpec<'_>],
        stream: u64,
    ) -> Result<()> {
        let Some(invocation) = invocation else {
            return Ok(());
        };
        let tensors: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, ptr, shape)| DiagnosticTensor {
                name,
                dtype: *dtype,
                ptr: *ptr,
                shape: shape.to_vec(),
            })
            .collect();
        self.0.capture(gpu, invocation, stage, &tensors, stream)
    }
}
impl Glm5KdaLayer {
    pub(super) fn capture_upstream(
        &self,
        invocation: Option<usize>,
        stage: &str,
        tensors: &[TensorSpec<'_>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(trace) = &self.upstream_trace {
            trace.capture(invocation, ctx.gpu, stage, tensors, stream)?;
        }
        Ok(())
    }
    pub(super) fn capture_upstream_hc(
        &self,
        invocation: Option<usize>,
        stage: &str,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.capture_upstream(
            invocation,
            stage,
            &[
                (
                    "normed",
                    DiagnosticDtype::Bf16,
                    normed,
                    &[rows, self.hidden_size],
                ),
                (
                    "highway",
                    DiagnosticDtype::Bf16,
                    ctx.buffers.hc_streams(),
                    &[rows, self.hc.hc_mult, self.hidden_size],
                ),
                (
                    "post",
                    DiagnosticDtype::F32,
                    ctx.buffers.hc_post(),
                    &[rows, self.hc.hc_mult],
                ),
                (
                    "comb",
                    DiagnosticDtype::F32,
                    ctx.buffers.hc_comb(),
                    &[rows, self.hc.hc_mult, self.hc.hc_mult],
                ),
            ],
            ctx,
            stream,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    #[test]
    fn gated_norm_binding_captures_resident_weight_and_effective_epsilon() {
        let gpu = MockGpuBackend::new();
        let root = std::env::temp_dir().join(format!("atlas-norm-binding-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let trace = UpstreamTrace(Exl3FfnRouter::new(root.clone(), 0, 1, 2).unwrap());
        let ptr = gpu.alloc(4).unwrap();
        let weight = [0, 0x3f, 0, 0x40];
        gpu.copy_h2d(&weight, ptr).unwrap();
        let eps = 0.00003125f32;
        // Disabled capture returns before validation, GPU access or filesystem writes.
        trace
            .capture_gated_norm_binding(
                None,
                &gpu,
                DevicePtr::NULL,
                KernelHandle(0),
                0,
                0,
                0,
                f32::NAN,
                8,
            )
            .unwrap();
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        assert_eq!(gpu.d2h_blocking_count() + gpu.d2h_async_count(), 0);
        std::fs::write(root.join("capture-enabled"), b"").unwrap();
        let invocation = trace.begin().unwrap();
        trace
            .capture_gated_norm_binding(invocation, &gpu, ptr, KernelHandle(123), 3, 4, 2, eps, 8)
            .unwrap();
        assert_eq!(
            std::fs::read(root.join("ffn-L0-rank1-000000-gated_norm_binding-weight.bin")).unwrap(),
            weight
        );
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("ffn-L0-rank1-000000-gated_norm_binding.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["binding"]["eps_f32_bits"], eps.to_bits());
        assert_eq!(metadata["binding"]["kernel_handle"], 123);
        assert_eq!(metadata["binding"]["grid"], serde_json::json!([12, 1, 1]));
        assert_eq!(metadata["binding"]["block"], serde_json::json!([128, 1, 1]));
        // Exercise the real launch wrapper and require its dispatched geometry to
        // match the captured binding, rather than trusting copied constants.
        ops::kda_sigmoid_gated_norm(&gpu, KernelHandle(123), ptr, ptr, ptr, ptr, 3, 4, 2, eps, 8)
            .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(metadata["binding"]["kernel_handle"], launches[0].func);
        assert_eq!(
            metadata["binding"]["grid"],
            serde_json::json!(launches[0].grid)
        );
        assert_eq!(
            metadata["binding"]["block"],
            serde_json::json!(launches[0].block)
        );
        assert_eq!(metadata["binding"]["activation"], "sigmoid");
        assert_eq!(metadata["tensors"][0]["shape"], serde_json::json!([2]));
        assert_eq!(metadata["tensors"][0]["dtype"], "bf16-le");
        for invalid in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            assert!(
                trace
                    .capture_gated_norm_binding(
                        Some(1),
                        &gpu,
                        ptr,
                        KernelHandle(123),
                        3,
                        4,
                        2,
                        invalid,
                        8
                    )
                    .is_err()
            );
        }
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 3);
        gpu.free(ptr).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn disabled_trace_performs_no_io_and_preserves_marker_identity() {
        let gpu = MockGpuBackend::new();
        let root = std::env::temp_dir().join(format!("atlas-upstream-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let trace = UpstreamTrace(Exl3FfnRouter::new(root.clone(), 0, 1, 2).unwrap());
        let ptr = gpu.alloc(4).unwrap();
        gpu.copy_h2d(&[0, 0x3f, 0, 0x40], ptr).unwrap();
        assert_eq!(trace.begin().unwrap(), None);
        trace
            .capture(
                None,
                &gpu,
                "embedding",
                &[("hidden", DiagnosticDtype::Bf16, ptr, &[1, 2])],
                8,
            )
            .unwrap();
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        std::fs::write(root.join("capture-enabled"), b"").unwrap();
        let invocation = trace.begin().unwrap();
        trace
            .capture(
                invocation,
                &gpu,
                "embedding",
                &[("hidden", DiagnosticDtype::Bf16, ptr, &[1, 2])],
                8,
            )
            .unwrap();
        assert_eq!(
            std::fs::read(root.join("ffn-L0-rank1-000000-embedding-hidden.bin")).unwrap(),
            [0, 0x3f, 0, 0x40]
        );
        assert_eq!(trace.begin().unwrap(), Some(1));
        gpu.free(ptr).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
