// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in tensor-core attention for GLM paged prefill only.
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kv_cache::KvCacheDtype,
};

pub struct GlmSparsePrefillTc<'a> {
    pub config: &'a ModelConfig,
    pub dtype: KvCacheDtype,
    /// The caller populated both cache sides from the same zero-RoPE latent
    /// using mla_cache_assemble_batched and the conventional paged cache writer.
    pub identical_kv_latent: bool,
    pub query: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub indices: DevicePtr,
    pub output: DevicePtr,
    pub block_table: DevicePtr,
    pub rows: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub index_width: u32,
    pub block_size: u32,
    pub scale: f32,
}

fn parse(model: &str, name: &str, value: Option<&str>) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be 0 or 1, got {other:?}"),
    }
}

fn enabled(model: &str, name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse(model, name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(model, name, None),
        Err(error) => Err(error.into()),
    }
}

fn kernel_spec(kv_reuse: bool) -> (&'static str, &'static str, u32) {
    if kv_reuse {
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
            69376,
        )
    } else {
        (
            "glm_sparse_prefill_tc",
            "glm_sparse_mla_prefill_bf16_head32_tc",
            101120,
        )
    }
}

/// Selected IDs and their causal bounds are supplied by the existing native
/// semantic-index producer. This adapter never changes or rebuilds selection.
pub fn try_glm_sparse_prefill_tc(
    gpu: &dyn GpuBackend,
    args: &GlmSparsePrefillTc<'_>,
    stream: u64,
) -> Result<bool> {
    let model = &args.config.model_type;
    dispatch(
        gpu,
        args,
        stream,
        enabled(model, "ATLAS_GLM_SPARSE_PREFILL_TC")?,
        enabled(model, "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE")?,
    )
}

fn dispatch(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    stream: u64,
    enabled: bool,
    kv_reuse: bool,
) -> Result<bool> {
    ensure!(
        !kv_reuse || enabled,
        "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1 requires ATLAS_GLM_SPARSE_PREFILL_TC=1"
    );
    if !enabled || a.rows == 1 {
        return Ok(false);
    }
    ensure!(
        !kv_reuse || a.identical_kv_latent,
        "GLM sparse KV reuse requires the identical zero-RoPE latent cache writer"
    );
    ensure!(
        a.config.model_type == "glm5_next"
            && a.config.hidden_size == 4096
            && a.config.kv_lora_rank == 512
            && a.config.qk_rope_head_dim == 0
            && a.config.index_topk == 2048
            && a.config.index_kpool == 4
            && a.dtype == KvCacheDtype::Bf16
            && (2..=65535).contains(&a.rows)
            && a.heads == 32
            && a.head_dim == 512
            && a.index_width == 2051
            && a.block_size == 16
            && a.scale == 0.0625,
        "GLM sparse TC prefill requires BF16 NoPE512, 32 heads, 2051 selected slots, block16 and scale1/16"
    );
    ensure!(
        [
            a.query,
            a.k_cache,
            a.v_cache,
            a.indices,
            a.output,
            a.block_table
        ]
        .iter()
        .all(|p| p.0 != 0),
        "GLM sparse TC prefill has missing storage"
    );
    let (module, symbol, shared_mem) = kernel_spec(kv_reuse);
    let kernel = gpu.kernel(module, symbol)?;
    ensure!(kernel.0 != 0, "GLM sparse TC prefill kernel is unavailable");
    KernelLaunch::new(gpu, kernel)
        .grid([1, a.rows, 1])
        .block([256, 1, 1])
        .shared_mem(shared_mem)
        .arg_ptr(a.query)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.indices)
        .arg_ptr(a.output)
        .arg_ptr(a.block_table)
        .arg_u32(a.rows)
        .arg_u32(a.heads)
        .arg_u32(a.head_dim)
        .arg_u32(a.index_width)
        .arg_u32(a.block_size)
        .arg_f32(a.scale)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "glm_sparse_prefill_tc_tests.rs"]
mod tests;
