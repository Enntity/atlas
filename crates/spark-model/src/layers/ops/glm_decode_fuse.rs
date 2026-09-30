// SPDX-License-Identifier: AGPL-3.0-only

//! GLM fused decode tier (`ATLAS_GLM_DECODE_FUSE=1`, default off): twins of
//! small per-layer kernels that write the same bytes with fewer dependent
//! loads or one launch fewer (`scripts/dev/glm_decode_fuse_bench.cu` checks
//! every output bit against the chains they replace).
//!
//! `ATLAS_GLM_DECODE_FUSE_MASK` (default 7) selects groups for bisection:
//!
//! * `1` HC post: `hc_post_bf16` / `hc_post_bf16_add_bf16` over 1..=32 rows
//!   as `glm_hc_decode_post_bf16`. Every GLM BF16-highway post goes through
//!   [`super::hc_post`] / [`super::hc_post_bf16_add`]: attention and FFN
//!   sites of KDA, MLA and dense layers, in verify, one-row decode and
//!   prefill chunks of up to 32 rows.
//! * `2` HC partial: the decode seam's `glm_hc_decode_{post_,}partial_bf16`
//!   as the `_rows_bf16` twins, wherever `ATLAS_GLM_HC_DECODE_SEAM=1` runs
//!   (1..=32 rows).
//! * `4` MoE post: the EP unpermute-reduce and the shared-expert blend of the
//!   TP-split shared expert (`ATLAS_GLM_SHARED_TP_SPLIT=1`) as
//!   `moe_unpermute_blend_ep_vec8`: the grouped routed FFN of verify blocks
//!   and short prefill chunks. One-row decode runs another MoE path.
//!
//! A twin runs only where its kernel is shipped and its shape and alignment
//! hold; otherwise the original launch is issued unchanged.

use anyhow::{Result, anyhow, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

pub const HC_POST: u32 = 1;
pub const HC_PARTIAL: u32 = 2;
pub const MOE_POST: u32 = 4;
const ALL: u32 = HC_POST | HC_PARTIAL | MOE_POST;

fn parse(fuse: Option<&str>, mask: Option<&str>) -> Result<u32> {
    let on = match fuse {
        None | Some("0") => false,
        Some("1") => true,
        Some(value) => bail!("ATLAS_GLM_DECODE_FUSE must be 0 or 1, got {value:?}"),
    };
    let groups = match mask {
        None => ALL,
        Some(value) => {
            let groups: u32 = value.parse().map_err(|_| {
                anyhow!("ATLAS_GLM_DECODE_FUSE_MASK must be a group mask in 0..=7, got {value:?}")
            })?;
            ensure!(
                groups & !ALL == 0,
                "ATLAS_GLM_DECODE_FUSE_MASK must be a group mask in 0..=7, got {value:?}"
            );
            groups
        }
    };
    Ok(if on { groups } else { 0 })
}

/// The fused groups that are on. The environment is read once; a malformed
/// value fails every launch that asks.
fn groups() -> Result<u32> {
    static GROUPS: OnceLock<std::result::Result<u32, String>> = OnceLock::new();
    let groups = GROUPS.get_or_init(|| {
        let var = |name| std::env::var(name).ok();
        let groups = parse(
            var("ATLAS_GLM_DECODE_FUSE").as_deref(),
            var("ATLAS_GLM_DECODE_FUSE_MASK").as_deref(),
        );
        if let Ok(groups @ 1..) = groups {
            tracing::info!("ATLAS_GLM_DECODE_FUSE: fused decode groups {groups:#x}");
        }
        groups.map_err(|error| error.to_string())
    });
    groups.clone().map_err(|error| anyhow!(error))
}

fn aligned16(ptrs: &[DevicePtr]) -> bool {
    ptrs.iter().all(|p| p.0.is_multiple_of(16))
}

/// `hc_post` (`peer` null) or `hc_post_bf16_add` as `glm_hc_decode_post_bf16`
/// when the HC post group is on and `kernel` is the GLM BF16-highway kernel
/// for an in-place post of 1..=32 rows. `ptrs`: block output, peer block
/// output, residual, post, comb, out. Returns whether it launched.
pub(super) fn hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ptrs: [DevicePtr; 6],
    dims: [u32; 3],
    stream: u64,
) -> Result<bool> {
    hc_post_for(groups()?, gpu, kernel, ptrs, dims, stream)
}

fn hc_post_for(
    groups: u32,
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ptrs: [DevicePtr; 6],
    [tokens, hidden_size, hc_mult]: [u32; 3],
    stream: u64,
) -> Result<bool> {
    let [block_out, peer, residual, post, comb, out] = ptrs;
    if groups & HC_POST == 0
        || !(1..=32).contains(&tokens)
        || hidden_size != 4096
        || hc_mult != 4
        || residual != out
        || !aligned16(&[block_out, peer, residual])
    {
        return Ok(false);
    }
    let original = if peer.is_null() {
        "hc_post_bf16"
    } else {
        "hc_post_bf16_add_bf16"
    };
    let cache = gpu.op_cache();
    let (Ok(original), Ok(twin)) = (
        cache.kernel(gpu, "hyper_connection", original),
        cache.kernel(gpu, "glm_hc_prefill_vec", "glm_hc_decode_post_bf16"),
    ) else {
        return Ok(false);
    };
    if kernel.0 != original.0 || twin.0 == 0 {
        return Ok(false);
    }
    KernelLaunch::new(gpu, twin)
        .grid([2 * tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(peer)
        .arg_ptr(out)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(tokens)
        .launch(stream)?;
    Ok(true)
}

/// The decode seam's partial kernel (`post`: with the finishing site's post):
/// the `_rows_bf16` twin when the HC partial group is on, the highway is BF16
/// and `hc_fn` is 16-byte aligned, else the established kernel.
pub fn hc_decode_partial(
    gpu: &dyn GpuBackend,
    model_type: &str,
    post: bool,
    hc_fn: DevicePtr,
) -> Result<KernelHandle> {
    let bf16 = super::hc_bf16_for(model_type);
    if let Some(twin) = partial_twin(groups()?, gpu, bf16, post, hc_fn) {
        return Ok(twin);
    }
    let base = if post {
        "glm_hc_decode_post_partial"
    } else {
        "glm_hc_decode_partial"
    };
    gpu.kernel(
        "glm_hc_prefill_vec",
        &super::hc_kernel_name(model_type, base),
    )
}

fn partial_twin(
    groups: u32,
    gpu: &dyn GpuBackend,
    bf16_highway: bool,
    post: bool,
    hc_fn: DevicePtr,
) -> Option<KernelHandle> {
    if groups & HC_PARTIAL == 0 || !bf16_highway || !aligned16(&[hc_fn]) {
        return None;
    }
    let twin = if post {
        "glm_hc_decode_post_partial_rows_bf16"
    } else {
        "glm_hc_decode_partial_rows_bf16"
    };
    gpu.op_cache()
        .kernel(gpu, "glm_hc_prefill_vec", twin)
        .ok()
        .filter(|kernel| kernel.0 != 0)
}

/// The EP unpermute-reduce followed by `moe_batched_blend(output, shared,
/// normed, gate)` as one `moe_unpermute_blend_ep_vec8` launch, when the MoE
/// post group is on and the shape and alignment hold. `ptrs`: expert output,
/// output, token_to_perm, topk ids, topk weights, shared output, normed
/// input, shared-expert gate weight (nullable). `dims`: hidden size, tokens,
/// top-k, local expert range. Returns whether it launched.
pub fn moe_unpermute_blend(
    gpu: &dyn GpuBackend,
    ptrs: [DevicePtr; 8],
    dims: [u32; 5],
    stream: u64,
) -> Result<bool> {
    moe_unpermute_blend_for(groups()?, gpu, ptrs, dims, stream)
}

fn moe_unpermute_blend_for(
    groups: u32,
    gpu: &dyn GpuBackend,
    ptrs: [DevicePtr; 8],
    [hidden_size, tokens, top_k, local_start, local_end]: [u32; 5],
    stream: u64,
) -> Result<bool> {
    let [
        expert_output,
        output,
        token_to_perm,
        ids,
        weights,
        shared,
        normed,
        gate,
    ] = ptrs;
    if groups & MOE_POST == 0
        || hidden_size != 4096
        || !(1..=8).contains(&top_k)
        || tokens == 0
        || !aligned16(&[expert_output, output, shared, normed, gate])
    {
        return Ok(false);
    }
    let Ok(kernel) = gpu
        .op_cache()
        .kernel(gpu, "moe", "moe_unpermute_blend_ep_vec8")
    else {
        return Ok(false);
    };
    KernelLaunch::new(gpu, kernel)
        .grid([tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_output)
        .arg_ptr(output)
        .arg_ptr(token_to_perm)
        .arg_ptr(ids)
        .arg_ptr(weights)
        .arg_ptr(shared)
        .arg_ptr(normed)
        .arg_ptr(gate)
        .arg_u32(tokens)
        .arg_u32(top_k)
        .arg_u32(local_start)
        .arg_u32(local_end)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "glm_decode_fuse_tests.rs"]
mod tests;
