// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in typed upstream capture; all I/O stays in the diagnostic router.
use super::*;
use crate::weight_loader::glm5::o_proj_binding;
use anyhow::Context;
use atlas_core::config::ModelConfig;
use spark_runtime::weights::exl3::ffn_io::{DiagnosticDtype, DiagnosticTensor, Exl3FfnRouter};
use std::path::PathBuf;

type TensorSpec<'a> = (&'static str, DiagnosticDtype, DevicePtr, &'a [usize]);

/// Load-time C1 binding facts (weight file, resident shard, projection
/// config) staged by `o_proj_binding` and stamped per invocation here. The
/// small load record is re-read per captured invocation: one ~1 KiB read
/// avoids a process-wide cache without adding synchronizing work.
struct OProjBinding {
    dir: PathBuf,
}

pub(super) struct UpstreamTrace {
    router: Exl3FfnRouter,
    rank: usize,
    o_proj: Option<OProjBinding>,
}
impl UpstreamTrace {
    pub(super) fn from_env(config: &ModelConfig, layer: usize) -> Result<Option<Self>> {
        let router = Exl3FfnRouter::from_env_named(
            layer,
            config.num_hidden_layers,
            config.tp_rank,
            config.tp_world_size,
            "ATLAS_EXL3_UPSTREAM_DUMP_DIR",
            "ATLAS_EXL3_UPSTREAM_DUMP_LAYERS",
        )?;
        ensure!(
            router.is_none() || layer == 0,
            "EXL3 upstream capture supports explicit layer0 only"
        );
        let Some(router) = router else {
            return Ok(None);
        };
        // The C1 o_proj binding record is another explicit opt-in on top of
        // the upstream dump; unset = no record and no load-record reads.
        let o_proj = o_proj_binding::gate_dir(layer)?.map(|dir| OProjBinding { dir });
        Ok(Some(Self {
            router,
            rank: config.tp_rank,
            o_proj,
        }))
    }
    pub(super) fn begin(&self) -> Result<Option<usize>> {
        self.router.begin()
    }
    pub(super) fn o_proj_enabled(&self) -> bool {
        self.o_proj.is_some()
    }
    /// Record the invocation-stamped `atlas-local-o-proj-binding-v1` row for
    /// the candidate engine. JSON only: the weight/resident bytes were
    /// written once at load time, so the hot path copies no tensors.
    pub(super) fn capture_o_proj_binding(
        &self,
        invocation: Option<usize>,
        tokens: u32,
        receipt: &serde_json::Value,
    ) -> Result<()> {
        let (Some(invocation), Some(binding)) = (invocation, self.o_proj.as_ref()) else {
            return Ok(());
        };
        let path = binding.dir.join(o_proj_binding::load_file(self.rank));
        let bytes = std::fs::read(&path)
            .with_context(|| format!("local o_proj load record {}", path.display()))?;
        let load: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("local o_proj load record {}", path.display()))?;
        o_proj_binding::validate_load_record(&load, self.rank)?;
        let record = serde_json::json!({
            "schema": o_proj_binding::BINDING_SCHEMA,
            "engine": "candidate",
            "rank": self.rank,
            "layer": 0,
            "invocation": invocation,
            "tokens": tokens,
            "projection": load["projection"],
            "files": {"weight": load["files"]["weight"]},
            "resident": load["resident"],
        });
        let mut receipt = receipt.clone();
        if let Some(map) = receipt.as_object_mut() {
            map.insert(
                "weight_basis".to_string(),
                load.pointer("/resident/recipe/name")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        self.router
            .write_binding_with_receipt(invocation, "o_proj_binding", &record, &receipt)
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
        self.router.capture_with_metadata(
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
        self.router
            .capture(gpu, invocation, stage, &tensors, stream)
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
    /// Diagnostic-only C1 local o_proj binding record (off unless both the
    /// upstream dump and `ATLAS_LOCAL_O_PROJ_BINDING_DIR` are configured).
    /// JSON-only: no tensor copies, no synchronizing work, no effect on the
    /// projection that follows.
    pub(super) fn capture_o_proj_binding(
        &self,
        invocation: Option<usize>,
        tokens: u32,
        decode: bool,
        capture_verify_intermediates: bool,
    ) -> Result<()> {
        let Some(trace) = self.upstream_trace.as_ref() else {
            return Ok(());
        };
        if !trace.o_proj_enabled() {
            return Ok(());
        }
        let (exl3, has_transposed) = match &self.weights.o_proj {
            Glm5Projection::Exl3(_) => (true, false),
            Glm5Projection::Nvfp4 {
                prefill_nvfp4_t, ..
            } => (false, prefill_nvfp4_t.is_some()),
        };
        let receipt = super::o_proj_receipt::o_proj_gemm_receipt(
            capture_verify_intermediates,
            decode,
            tokens,
            exl3,
            has_transposed,
            !decode
                && tokens >= 2048
                && std::env::var("ATLAS_GLM_KDA_PREFILL_LT_FP8").as_deref() == Ok("1"),
            self.w4a16_gemm_k,
            self.w4a16_gemv_k,
            self.w4a16_gemm_t_m128_k,
            self.hidden_size as u32,
            (self.heads * self.dim) as u32,
        );
        trace.capture_o_proj_binding(invocation, tokens, &receipt)
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
#[path = "upstream_tests.rs"]
mod tests;
