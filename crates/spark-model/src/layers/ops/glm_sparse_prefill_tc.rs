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
    /// using `mla_cache_assemble[_batched]` and the conventional paged cache writer.
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

/// Opt-in pipelined `fp8_g128` prefill kernel, bit-identical to the kv_pad one.
const PIPE: &str = "ATLAS_GLM_SPARSE_PREFILL_PIPE";
/// The pipe kernel's selected-slot capacity (`GLM_PIPE_SLOTS`); it also
/// requires 16-token cache blocks and returns without writing otherwise.
const PIPE_SLOTS: u32 = 2080;

fn kernel_spec(
    kv_reuse: bool,
    dtype: KvCacheDtype,
    pipe: bool,
) -> (&'static str, &'static str, u32) {
    if dtype == KvCacheDtype::Fp8G128 && pipe {
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_fp8g128_head32_tc_pipe",
            80128,
        )
    } else if dtype == KvCacheDtype::Fp8G128 {
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad",
            69376,
        )
    } else if kv_reuse {
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
        enabled(model, PIPE)?,
    )
}

/// Whether a sparse owner of `rows` reads an `fp8_g128` cache through the
/// BF16 view. The native library reads only the view; every other 2048+-row
/// piece runs BF16 kv_pad on it, which the pipelined kernel reproduces bit for
/// bit from the FP8 cache, so under the pipe only native pieces need it.
pub fn glm_sparse_owner_needs_view(
    model: &str,
    rows: u32,
    native_admits: impl FnOnce() -> Result<bool>,
) -> Result<bool> {
    needs_view(rows, enabled(model, PIPE)?, native_admits)
}

fn needs_view(rows: u32, pipe: bool, native_admits: impl FnOnce() -> Result<bool>) -> Result<bool> {
    Ok(rows >= 2048 && (!pipe || native_admits()?))
}

fn dispatch(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    stream: u64,
    enabled: bool,
    kv_reuse: bool,
    pipe: bool,
) -> Result<bool> {
    ensure!(
        !kv_reuse || enabled,
        "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1 requires ATLAS_GLM_SPARSE_PREFILL_TC=1"
    );
    ensure!(
        !pipe || enabled,
        "{PIPE}=1 requires ATLAS_GLM_SPARSE_PREFILL_TC=1"
    );
    // An `fp8_g128` cache has no other sparse reader, so it takes single rows
    // too and needs the K=V (kv_reuse) kernel.
    let fp8 = a.dtype == KvCacheDtype::Fp8G128;
    if !enabled || (a.rows == 1 && !fp8) {
        return Ok(false);
    }
    ensure!(
        (!kv_reuse && !fp8) || a.identical_kv_latent,
        "GLM sparse KV reuse requires the identical zero-RoPE latent cache writer"
    );
    ensure!(
        kv_reuse || !fp8,
        "fp8_g128 GLM sparse prefill requires ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1"
    );
    validate_geometry(a)?;
    ensure!(
        !pipe || (a.index_width <= PIPE_SLOTS && a.block_size == 16),
        "{PIPE}=1 requires at most {PIPE_SLOTS} selected slots and block16"
    );
    ensure!(
        (if fp8 { 1 } else { 2 }..=65535).contains(&a.rows),
        "GLM sparse TC prefill requires up to 65535 rows"
    );
    validate_storage(a)?;
    launch(gpu, a, stream, kv_reuse, pipe, a.rows)?;
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
        matches!(a.dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
            && a.heads == 32
            && a.head_dim == 512
            && a.index_width == 2051
            && a.block_size == 16
            && a.scale == 0.0625,
        "GLM sparse TC requires BF16 or fp8_g128, 32 heads, 2051 selected slots, block16 and scale1/16"
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
    pipe: bool,
    grid_rows: u32,
) -> Result<()> {
    let (module, symbol, shared_mem) = kernel_spec(kv_reuse, a.dtype, pipe);
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

/// Splits of the selected IDs for `rows` query rows: the count (at most 16,
/// the merge's cap) minimizing waves x 32-key tiles per CTA, with one CTA
/// per SM (69 KB of shared memory) on GB10's 48 SMs. 1 = the unsplit kernel.
pub(crate) fn sparse_split_count(rows: u32, heads: u32, index_width: u32) -> u32 {
    let ctas = rows * heads.div_ceil(32);
    let tiles = index_width.div_ceil(32);
    let cost = |s: u32| (ctas * s).div_ceil(48) * tiles.div_ceil(s);
    (1..=16u32).min_by_key(|&s| (cost(s), s)).unwrap_or(1)
}

/// Bytes of split scratch: FP32 partial outputs, their LSEs, merged LSEs.
pub(crate) fn sparse_split_scratch_bytes(
    splits: u32,
    rows: u32,
    heads: u32,
    head_dim: u32,
) -> usize {
    let rh = rows as usize * heads as usize;
    splits as usize * rh * (head_dim as usize + 1) * 4 + rh * 4
}

/// `try_glm_sparse_prefill_tc` for few-row callers (verify owners): split
/// over the selected IDs (`*_split` kernel + `glm_sparse_decode_split_merge`)
/// when `sparse_split_count` > 1 and `scratch` fits, so the rows fill the GPU
/// instead of one CTA each. Partial sums in a different order than the
/// unsplit kernel. `ATLAS_GLM_SPARSE_VERIFY_SPLIT=0` disables.
pub fn try_glm_sparse_prefill_tc_split(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
) -> Result<bool> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on =
        *ON.get_or_init(|| std::env::var("ATLAS_GLM_SPARSE_VERIFY_SPLIT").as_deref() != Ok("0"));
    let model = &a.config.model_type;
    let (tc, kv_reuse, pipe) = (
        enabled(model, "ATLAS_GLM_SPARSE_PREFILL_TC")?,
        enabled(model, "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE")?,
        enabled(model, PIPE)?,
    );
    let splits = sparse_split_count(a.rows, a.heads, a.index_width);
    let need = sparse_split_scratch_bytes(splits, a.rows, a.heads, a.head_dim);
    if !on
        || !tc
        || !kv_reuse
        || !a.identical_kv_latent
        || splits <= 1
        || scratch.0 == 0
        || !scratch.0.is_multiple_of(16)
        || scratch_bytes < need
    {
        return dispatch(gpu, a, stream, tc, kv_reuse, pipe);
    }
    let rh = a.rows as usize * a.heads as usize;
    let part_lse = scratch.offset(splits as usize * rh * a.head_dim as usize * 4);
    let out_lse = part_lse.offset(splits as usize * rh * 4);
    launch_sparse_partials(gpu, a, splits, None, scratch, part_lse, stream)?;
    decode_split::launch_merge(
        gpu,
        decode_split::merge_kernel(gpu)?,
        scratch,
        part_lse,
        a.output,
        out_lse,
        a.rows,
        a.rows * a.heads,
        splits,
        stream,
    )?;
    Ok(true)
}

/// The `*_split` kernel alone: `splits` normalized FP32 partial outputs
/// `[splits, rows, heads, 512]` and their natural LSEs `[splits, rows, heads]`
/// (`-inf` for a partition with no valid ID) for a caller-side LSE merge.
/// With `row_counts` (`u32[rows]`) row `r` selects only its first
/// `row_counts[r]` IDs and the splits partition that prefix.
pub(crate) fn launch_sparse_partials(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    splits: u32,
    row_counts: Option<DevicePtr>,
    part_o: DevicePtr,
    part_lse: DevicePtr,
    stream: u64,
) -> Result<()> {
    validate_geometry(a)?;
    validate_storage(a)?;
    ensure!(
        (1..=16).contains(&splits) && a.rows > 0 && a.identical_kv_latent,
        "GLM sparse partials need 1..=16 splits, rows and identical K/V"
    );
    let (module, _, shared_mem) = kernel_spec(true, a.dtype, false);
    let symbol = match (a.dtype == KvCacheDtype::Fp8G128, row_counts.is_some()) {
        (true, false) => "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split",
        (false, false) => "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split",
        (true, true) => "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted",
        (false, true) => "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split_counted",
    };
    // The counted entry points are the shard's (`glm_kv_shard.cu`).
    let module = row_counts.map_or(module, |_| super::glm_kv_shard::MODULE);
    let kernel = gpu.op_cache().kernel(gpu, module, symbol)?;
    ensure!(kernel.0 != 0, "GLM sparse split kernel is unavailable");
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([a.heads.div_ceil(32), a.rows, splits])
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
        .arg_ptr(part_o)
        .arg_ptr(part_lse);
    match row_counts {
        Some(counts) => launch.arg_ptr(counts),
        None => launch,
    }
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
    launch(gpu, a, stream, true, false, 1)?;
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

/// Validate the pipe's flag combination and warm its kernel before serving, so
/// its first launch (lazy load, 80 KB shared-memory opt-in) never falls inside
/// a CUDA-graph capture (verify rows reach it when the split does not apply).
pub fn initialize_glm_sparse_prefill_pipe(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
) -> Result<()> {
    let model = &config.model_type;
    if !enabled(model, PIPE)? {
        return Ok(());
    }
    ensure!(
        enabled(model, "ATLAS_GLM_SPARSE_PREFILL_TC")?,
        "{PIPE}=1 requires ATLAS_GLM_SPARSE_PREFILL_TC=1"
    );
    warm(gpu, config, KvCacheDtype::Fp8G128, true)?;
    tracing::info!("{PIPE}=1: pipelined fp8_g128 sparse prefill warmed before KV sizing");
    Ok(())
}

fn initialize_decode(gpu: &dyn GpuBackend, config: &ModelConfig, enabled: bool) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    warm(gpu, config, KvCacheDtype::Bf16, false)
}

fn warm(gpu: &dyn GpuBackend, config: &ModelConfig, dtype: KvCacheDtype, pipe: bool) -> Result<()> {
    validate_config(config)?;
    let empty = GlmSparsePrefillTc {
        config,
        dtype,
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
    launch(gpu, &empty, stream, true, pipe, 1)?;
    gpu.synchronize(stream)
}

#[cfg(test)]
#[path = "glm_sparse_prefill_tc_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm_sparse_decode_tc_tests.rs"]
mod decode_tests;
