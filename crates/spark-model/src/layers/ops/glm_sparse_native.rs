// SPDX-License-Identifier: AGPL-3.0-only
//! Optional native FP8 attention for ordinary long GLM continuation prefills.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferSizes;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;

use super::GlmSparsePrefillTc;
use crate::layer::ForwardContext;
#[path = "glm_sparse_native_loader.rs"]
mod loader;
#[path = "glm_sparse_native_plan.rs"]
mod plan;
use plan::{Plan, Span};

fn validate_geometry(c: &ModelConfig) -> Result<()> {
    ensure!(
        c.model_type == "glm5_next"
            && c.hidden_size == 4096
            && c.hc_mult == 4
            && c.tp_world_size == 2
            && c.ep_world_size == 2
            && c.num_attention_heads == 32
            && c.num_key_value_heads == 32
            && c.head_dim == 256
            && c.kv_lora_rank == 512
            && c.qk_rope_head_dim == 0
            && c.index_topk == 2048
            && c.index_kpool == 4
            && c.index_n_heads == 32
            && c.index_head_dim == 128,
        "native sparse requires GLM TP2/EP2 H4096/HC4,32heads,NoPE512,index2048/pool4"
    );
    Ok(())
}

fn validate_startup(
    c: &ModelConfig,
    rows: usize,
    seq: usize,
    block: usize,
    active: usize,
    dtype: KvCacheDtype,
    layer_dtypes: &[KvCacheDtype],
) -> Result<()> {
    validate_geometry(c)?;
    ensure!(
        // At least a 4K chunk; wider chunks reach the native attention in
        // pieces of at most 4096 rows (`PREFILL_ATTENTION_ROWS`).
        rows >= 4100
            && c.max_batch_tokens == rows
            && seq == plan::MAX_CONTEXT
            && block == 16
            // Only the arena capacity below depends on the sequence count;
            // the library runs each sequence's attention on its own.
            && (1..=crate::layer::glm_long_owner::MAX_OWNERS).contains(&active)
            // An fp8_g128 cache reaches the library through its BF16 view.
            && matches!(dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
            && layer_dtypes.iter().all(|&d| d == dtype),
        "native sparse requires a >=4100-row arena,32K context,C4,block16,BF16/fp8_g128 cache"
    );
    let sizes = BufferSizes::from_config(c, rows, seq, block, active);
    let (required, _) =
        plan::required_bytes(rows, seq, 1, seq / block).map_err(anyhow::Error::msg)?;
    let capacities = [
        sizes.ssm_deinterleaved,
        sizes.expert_down_out,
        sizes.ssm_conv_out_f32,
        sizes.expert_up_out,
        sizes.qkv_output,
        sizes.expert_gate_out,
        sizes.ssm_qkvz,
        sizes.attn_output,
    ];
    for (available, index) in capacities.into_iter().zip([0, 2, 4, 5, 6, 7, 8, 9]) {
        ensure!(
            available >= required[index],
            "native sparse startup arena capacity exceeded at operand{index}"
        );
    }
    Ok(())
}

/// Whether `ATLAS_GLM_SPARSE_NATIVE=1` selects the optional library.
pub fn glm_sparse_native_enabled() -> Result<bool> {
    loader::enabled()
}

fn qualified_context(seq: usize) -> usize {
    seq.min(plan::MAX_CONTEXT)
}

/// Load/configure the optional module on the serving context before KV sizing.
/// The module init allocates no device buffers and never launches attention.
#[allow(clippy::too_many_arguments)]
pub fn initialize_glm_sparse_native(
    c: &ModelConfig,
    rows: usize,
    seq: usize,
    block: usize,
    active: usize,
    dtype: KvCacheDtype,
    layer_dtypes: &[KvCacheDtype],
) -> Result<()> {
    if !loader::enabled()? {
        return Ok(());
    }
    let qualified_seq = qualified_context(seq);
    if qualified_seq != seq {
        tracing::warn!(
            max_seq_len = seq,
            native_context_limit = plan::MAX_CONTEXT,
            "GLM native sparse prefill is qualified through the native context limit; longer rows use the Atlas sparse/plain fallback"
        );
    }
    validate_startup(c, rows, qualified_seq, block, active, dtype, layer_dtypes)?;
    ensure!(
        cfg!(feature = "cuda"),
        "native sparse requires the CUDA backend"
    );
    loader::initialize()
}

fn execute(plan: &Plan, run: loader::Run) -> Result<()> {
    // SAFETY: Plan checks geometry, capacities, alignment, and disjoint live
    // byte ranges. ABI1 consumes the host struct synchronously and queues only
    // onto its supplied serving stream. The retained mapping owns this function.
    let status = unsafe { run(&plan.abi) };
    ensure!(
        status == 0,
        "native sparse forward failed ({status}); no fallback after native submission"
    );
    Ok(())
}

/// Called only after the continued-prefill BF16 KV writer. K3 verification uses
/// its separate multi-sequence MLA path; small rows/capture keep existing ops.
#[allow(clippy::too_many_arguments)]
pub fn try_glm_sparse_native(
    ctx: &ForwardContext<'_>,
    a: &GlmSparsePrefillTc<'_>,
    seq_start: usize,
    physical_blocks: usize,
    block_table_count: usize,
    cache_block_bytes: usize,
    capture_verify_intermediates: bool,
    stream: u64,
) -> Result<bool> {
    if !loader::enabled()? {
        return Ok(false);
    }
    if plan::admit(
        a.rows as usize,
        seq_start,
        ctx.graph_capture,
        ctx.gpu.stream_is_capturing(stream),
        capture_verify_intermediates,
    )
    .is_none()
    {
        return Ok(false);
    }
    validate_geometry(ctx.config)?;
    ensure!(
        a.dtype == KvCacheDtype::Bf16
            && a.identical_kv_latent
            && a.heads == 32
            && a.head_dim == 512
            && a.index_width == 2051
            && a.block_size == 16
            && a.scale == 0.0625
            && cache_block_bytes == 16 * 512 * 2,
        "native sparse invocation does not match the qualified cache/selection geometry"
    );
    let b = ctx.buffers;
    let s = b.sizes();
    ensure!(
        a.query == b.ssm_deinterleaved()
            && a.indices == b.expert_down_out()
            && a.output == b.attn_output(),
        "native sparse requires audited prefill arena owners"
    );
    let span = |ptr: DevicePtr, bytes: usize| Span { ptr: ptr.0, bytes };
    let cache_bytes = physical_blocks
        .checked_mul(cache_block_bytes)
        .ok_or_else(|| anyhow::anyhow!("native sparse physical cache size overflow"))?;
    let table_bytes = block_table_count
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("native sparse block table size overflow"))?;
    let plan = Plan::new(
        a.rows as usize,
        seq_start,
        physical_blocks,
        block_table_count,
        [
            span(a.query, s.ssm_deinterleaved),
            span(a.k_cache, cache_bytes),
            span(a.indices, s.expert_down_out),
            span(a.block_table, table_bytes),
            span(b.ssm_conv_out_f32(), s.ssm_conv_out_f32),
            span(b.expert_up_out(), s.expert_up_out),
            span(b.qkv_output(), s.qkv_output),
            span(b.expert_gate_out(), s.expert_gate_out),
            span(b.ssm_qkvz(), s.ssm_qkvz),
            span(a.output, s.attn_output),
        ],
        stream,
    )
    .map_err(anyhow::Error::msg)?;
    execute(&plan, loader::run()?)?;
    Ok(true)
}

#[cfg(test)]
#[path = "glm_sparse_native_tests.rs"]
mod tests;
