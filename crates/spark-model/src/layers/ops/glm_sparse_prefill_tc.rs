// SPDX-License-Identifier: AGPL-3.0-only
//! Independently opt-in tensor-core GLM paged prefill and repaired K3 row attention.
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kv_cache::KvCacheDtype,
};

#[path = "glm_sparse_decode_split.rs"]
mod decode_split;
pub use decode_split::*;

pub struct GlmSparsePrefillTc<'a> {
    pub config: &'a ModelConfig,
    pub dtype: KvCacheDtype,
    /// The caller populated both cache sides from the same zero-RoPE latent
    /// using mla_cache_assemble[_batched] and the conventional paged cache writer.
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
    validate_geometry(a)?;
    ensure!(
        (2..=65535).contains(&a.rows),
        "GLM sparse TC prefill requires 2..=65535 rows"
    );
    validate_storage(a)?;
    launch(gpu, a, stream, kv_reuse, a.rows)?;
    Ok(true)
}

fn validate_config(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.model_type == "glm5_next"
            && config.hidden_size == 4096
            && config.kv_lora_rank == 512
            && config.qk_rope_head_dim == 0
            && config.index_topk == 2048
            && config.index_kpool == 4,
        "GLM sparse TC requires NoPE512 hidden4096, topk2048 and pool4"
    );
    Ok(())
}

fn validate_geometry(a: &GlmSparsePrefillTc<'_>) -> Result<()> {
    validate_config(a.config)?;
    ensure!(
        a.dtype == KvCacheDtype::Bf16
            && a.heads == 32
            && a.head_dim == 512
            && a.index_width == 2051
            && a.block_size == 16
            && a.scale == 0.0625,
        "GLM sparse TC requires BF16, 32 heads, 2051 selected slots, block16 and scale1/16"
    );
    Ok(())
}

fn validate_storage(a: &GlmSparsePrefillTc<'_>) -> Result<()> {
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
        "GLM sparse TC has missing storage"
    );
    Ok(())
}

fn launch(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    stream: u64,
    kv_reuse: bool,
    grid_rows: u32,
) -> Result<()> {
    let (module, symbol, shared_mem) = kernel_spec(kv_reuse);
    let kernel = gpu.kernel(module, symbol)?;
    ensure!(kernel.0 != 0, "GLM sparse TC prefill kernel is unavailable");
    KernelLaunch::new(gpu, kernel)
        .grid([1, grid_rows, 1])
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
        .launch(stream)
}

/// Single source of truth for the independent repaired-decode opt-in.
pub fn glm_sparse_decode_tc_enabled(model: &str) -> Result<bool> {
    enabled(model, "ATLAS_GLM_SPARSE_DECODE_TC")
}

/// The repaired K3 caller supplies one temporal row after existing semantic
/// selection. Ordinary decode and prefill do not call this entry.
pub fn try_glm_sparse_decode_tc(
    gpu: &dyn GpuBackend,
    args: &GlmSparsePrefillTc<'_>,
    stream: u64,
) -> Result<bool> {
    dispatch_decode(
        gpu,
        args,
        stream,
        glm_sparse_decode_tc_enabled(&args.config.model_type)?,
    )
}

fn dispatch_decode(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    stream: u64,
    enabled: bool,
) -> Result<bool> {
    if !enabled {
        return Ok(false);
    }
    validate_geometry(a)?;
    ensure!(
        a.rows == 1 && a.identical_kv_latent,
        "GLM sparse TC decode requires one row and identical zero-RoPE K/V"
    );
    validate_storage(a)?;
    ensure!(
        [a.query, a.k_cache, a.v_cache]
            .iter()
            .all(|p| p.0.is_multiple_of(16))
            && [a.indices, a.output, a.block_table]
                .iter()
                .all(|p| p.0.is_multiple_of(4)),
        "GLM sparse TC decode requires aligned vector loads and stores"
    );
    launch(gpu, a, stream, true, 1)?;
    Ok(true)
}

/// Complete lazy code loading and shared-memory initialization before serving.
/// Factory policy validates repaired MTP2/BF16 admission before calling this.
pub fn initialize_glm_sparse_decode_tc(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<()> {
    initialize_decode(
        gpu,
        config,
        glm_sparse_decode_tc_enabled(&config.model_type)?,
    )
}

fn initialize_decode(gpu: &dyn GpuBackend, config: &ModelConfig, enabled: bool) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    validate_config(config)?;
    let empty = GlmSparsePrefillTc {
        config,
        dtype: KvCacheDtype::Bf16,
        identical_kv_latent: true,
        query: DevicePtr::NULL,
        k_cache: DevicePtr::NULL,
        v_cache: DevicePtr::NULL,
        indices: DevicePtr::NULL,
        output: DevicePtr::NULL,
        block_table: DevicePtr::NULL,
        rows: 0,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    };
    let stream = gpu.default_stream();
    // Init-only zero rows: the kernel uniformly returns at token_row >= rows,
    // before pointer arithmetic, global reads/writes, or barriers. One CTA still
    // forces the real code/shared-memory launch path. Data dispatch rejects 0.
    launch(gpu, &empty, stream, true, 1)?;
    gpu.synchronize(stream)
}

#[cfg(test)]
#[path = "glm_sparse_prefill_tc_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm_sparse_decode_tc_tests.rs"]
mod decode_tests;
