// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in typed upstream capture; all I/O stays in the diagnostic router.
use super::*;
use crate::layers::glm5_exl3_mhc::MhcResidualPrecision;
use crate::layers::qwen3_attention::HcSiteWeights;
use atlas_core::config::ModelConfig;
use spark_runtime::weights::exl3::ffn_io::{DiagnosticDtype, DiagnosticTensor, Exl3FfnRouter};

type TensorSpec<'a> = (&'static str, DiagnosticDtype, DevicePtr, &'a [usize]);

/// Only the L0→L1 boundary is qualified for upstream capture
/// (docs/plans/l1-escalation-tooling-spec.md D1); G1's wider layer set is a
/// separate reviewed change. The stage calls themselves are layer-generic.
fn ensure_supported_layer(layer: usize) -> Result<()> {
    ensure!(
        layer <= 1,
        "EXL3 upstream capture supports the L0->L1 boundary (layers 0..=1)"
    );
    Ok(())
}

/// mHC boundary facts recorded beside the captured (x, residual, post, comb)
/// tensors so an L1 bisection binds rows and the residual-precision path
/// instead of inferring them (D1). `rows` is the actual local token count
/// (spec D1); the precision value comes from
/// `Glm5KdaLayer::mhc_residual_precision`, the same selection `pre_norm`
/// receives.
pub(super) struct HcBoundaryFacts {
    rows: usize,
    hc_mult: usize,
    sinkhorn_iters: u32,
    hc_eps: f32,
    rms_eps: f32,
    residual_precision: MhcResidualPrecision,
    request_id: Option<String>,
}

fn hc_boundary_binding(facts: &HcBoundaryFacts) -> Result<serde_json::Value> {
    ensure!(facts.rows > 0, "mHC boundary binding rows invalid");
    ensure!(facts.hc_mult > 0, "mHC boundary binding hc_mult invalid");
    ensure!(
        facts.sinkhorn_iters > 0,
        "mHC boundary binding sinkhorn iterations invalid"
    );
    ensure!(
        facts.hc_eps.is_finite()
            && facts.hc_eps > 0.0
            && facts.rms_eps.is_finite()
            && facts.rms_eps > 0.0,
        "mHC boundary binding epsilon invalid"
    );
    let mut value = serde_json::json!({
        "schema": "atlas-mhc-boundary-binding-v1",
        "rows": facts.rows,
        "hc_mult": facts.hc_mult,
        "sinkhorn_iters": facts.sinkhorn_iters,
        "hc_eps_bits": facts.hc_eps.to_bits(),
        "rms_eps_bits": facts.rms_eps.to_bits(),
        "residual_precision": format!("{:?}", facts.residual_precision),
        "sequence_parallel": false,
    });
    if let Some(request_id) = facts.request_id.as_deref() {
        value["request_id"] = serde_json::Value::String(request_id.to_string());
    }
    Ok(value)
}
pub(super) struct UpstreamTrace(Exl3FfnRouter);
impl UpstreamTrace {
    pub(super) fn from_env(config: &ModelConfig, layer: usize) -> Result<Option<Self>> {
        tracing::trace!(target: "upstream", "from_env: layer={} tp_rank={} tp_world_size={}", layer, config.tp_rank, config.tp_world_size);
        let router = Exl3FfnRouter::from_env_named(
            layer,
            config.num_hidden_layers,
            config.tp_rank,
            config.tp_world_size,
            "ATLAS_EXL3_UPSTREAM_DUMP_DIR",
            "ATLAS_EXL3_UPSTREAM_DUMP_LAYERS",
        )?
        .map(Self);
        if router.is_some() {
            tracing::info!(target: "upstream", "from_env: upstream capture enabled for layer={}", layer);
            ensure_supported_layer(layer)?;
        } else {
            tracing::debug!(target: "upstream", "from_env: upstream capture disabled for layer={}", layer);
        }
        Ok(router)
    }
    pub(super) fn begin(&self) -> Result<Option<usize>> {
        let result = self.0.begin();
        tracing::trace!(target: "upstream", "begin: invocation={:?}", result);
        result
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
            tracing::trace!(target: "upstream", "capture_gated_norm_binding: no invocation, skipping");
            return Ok(());
        };
        tracing::debug!(target: "upstream", "capture_gated_norm_binding: invocation={} tokens={} heads={} dim={} eps={} kernel={}", invocation, tokens, heads, dim, eps, kernel.0);
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
        tracing::trace!(target: "upstream", "capture_gated_norm_binding: grid={:?} block={:?}", grid, block);
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
    fn specs(tensors: &[TensorSpec<'_>]) -> Vec<DiagnosticTensor> {
        tensors
            .iter()
            .map(|(name, dtype, ptr, shape)| DiagnosticTensor {
                name,
                dtype: *dtype,
                ptr: *ptr,
                shape: shape.to_vec(),
            })
            .collect()
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
            tracing::trace!(target: "upstream", "capture: no invocation, skipping stage={}", stage);
            return Ok(());
        };
        tracing::trace!(target: "upstream", "capture: invocation={} stage={} tensors={}", invocation, stage, tensors.len());
        self.0
            .capture(gpu, invocation, stage, &Self::specs(tensors), stream)
    }
    /// Capture the mHC boundary tensors with their binding facts (D1).
    /// Disabled capture returns before any validation, GPU access or
    /// filesystem write (the module invariant).
    pub(super) fn capture_boundary(
        &self,
        invocation: Option<usize>,
        gpu: &dyn GpuBackend,
        stage: &str,
        tensors: &[TensorSpec<'_>],
        stream: u64,
        facts: &HcBoundaryFacts,
    ) -> Result<()> {
        let Some(invocation) = invocation else {
            tracing::trace!(target: "upstream", "capture_boundary: no invocation, skipping stage={}", stage);
            return Ok(());
        };
        tracing::debug!(target: "upstream", "capture_boundary: invocation={} stage={} rows={} hc_mult={} sinkhorn_iters={} hc_eps={} rms_eps={} residual_precision={:?} request_id={:?}", invocation, stage, facts.rows, facts.hc_mult, facts.sinkhorn_iters, facts.hc_eps, facts.rms_eps, facts.residual_precision, facts.request_id);
        let binding = hc_boundary_binding(facts)?;
        tracing::trace!(target: "upstream", "capture_boundary: tensors={}", tensors.len());
        self.0.capture_with_metadata(
            gpu,
            invocation,
            stage,
            &Self::specs(tensors),
            stream,
            &binding,
        )
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
        tracing::trace!(target: "upstream", "capture_upstream: layer={} invocation={:?} stage={} tensors={} upstream_trace={}", self.layer_idx, invocation, stage, tensors.len(), self.upstream_trace.is_some());
        if let Some(trace) = &self.upstream_trace {
            trace.capture(invocation, ctx.gpu, stage, tensors, stream)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_upstream_hc(
        &self,
        invocation: Option<usize>,
        stage: &str,
        site: &HcSiteWeights,
        normed: DevicePtr,
        rows: usize,
        local_rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(trace) = &self.upstream_trace else {
            tracing::trace!(target: "upstream", "capture_upstream_hc: no upstream trace for layer={}", self.layer_idx);
            return Ok(());
        };
        tracing::debug!(target: "upstream", "capture_upstream_hc: layer={} invocation={:?} stage={} rows={} local_rows={} hidden_size={} hc_mult={} request_id={:?}", self.layer_idx, invocation, stage, rows, local_rows, self.hidden_size, self.hc.hc_mult, ctx.request_id);
        let facts = HcBoundaryFacts {
            rows: local_rows as usize,
            hc_mult: self.hc.hc_mult,
            sinkhorn_iters: self.hc.sinkhorn_iters as u32,
            hc_eps: self.hc.hc_eps,
            rms_eps: ctx.config.rms_norm_eps as f32,
            residual_precision: self.mhc_residual_precision(site, local_rows),
            request_id: ctx.request_id.as_deref().map(String::from),
        };
        trace.capture_boundary(
            invocation,
            ctx.gpu,
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
            stream,
            &facts,
        )
    }
}
#[cfg(test)]
#[path = "upstream_tests.rs"]
mod tests;
