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
