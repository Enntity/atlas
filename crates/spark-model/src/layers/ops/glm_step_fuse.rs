// SPDX-License-Identifier: AGPL-3.0-only

//! GLM step-fuse tier (`ATLAS_GLM_STEP_FUSE=1`, default off): the verify
//! step's mHC seams in fewer launches and shorter dependent chains, with
//! every output byte unchanged (`scripts/dev/glm_decode_fuse_bench.cu`
//! checks them against the chains they replace, groups 32 and 64).
//!
//! Under `ATLAS_PDL=1` a verify step's ~1,400 small kernels are resident
//! before their predecessors end, so what each costs is not its host launch
//! but the chain after its `griddepcontrol.wait`: the release, its dependent
//! loads, its barriers. Each mHC seam (two a layer) is a partial kernel over
//! 64 CTAs, a finalizer over one CTA a row, then the caller's RMS norm.
//!
//! `ATLAS_GLM_STEP_FUSE_MASK` (default 3) selects groups for bisection:
//!
//! * `1` HC norm: the decode seam's finalizer (`glm_hc_decode_finalize_bf16`)
//!   and the RMS norm its caller runs next over the same rows
//!   (`rms_norm_vanilla` or its `_regs` twin at hidden 4096) as one
//!   `glm_hc_decode_finalize_norm_bf16` launch of 1024 threads a row: the
//!   norm is the `_regs` row body over the collapsed row, kept in shared
//!   memory, and the 25 sums are staged by all threads before the same
//!   ordered adds. One launch fewer at every seam (90 a 45-layer step). The
//!   caller hands its norm in as a [`SeamNorm`] and issues it with
//!   [`SeamNorm::run`], which launches nothing when the seam wrote it.
//! * `2` HC touch: the seam's `_rows_bf16` partial twins
//!   (`ATLAS_GLM_DECODE_FUSE` group 2) as `_rows_touch_bf16`, which pull the
//!   next site's 1.5 MiB `hc_fn` into L2 before their PDL wait, as the
//!   `ATLAS_GLM_DECODE_GEMV_BATCH` twins do with GEMV weights
//!   ([`super::gemv_touch`]). Loads only, discarded: the body is the twin's.
//!
//! A twin runs only where the target ships it and its shapes hold; otherwise
//! the original launches are issued unchanged.
//!
//! Prior art (docs/glm-prior-art.md): the touch before the PDL wait is
//! TensorFold's L2 weight touch (<https://github.com/jayleaton/glm53-tensorfold-spark>
//! patch 0440; Apache-2.0), as in [`super::gemv_touch`]; MiaAI-Lab's
//! `TF_GLM_L2PF` (patch 0046) likewise reads a seam's next weights into L2
//! while the previous site's partials are gathered. No code copied.

use std::cell::Cell;
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::glm_decode_fuse::env_groups;
use crate::weight_map::DenseWeight;

pub const HC_NORM: u32 = 1;
pub const HC_TOUCH: u32 = 2;
const ALL: u32 = HC_NORM | HC_TOUCH;
const NAME: &str = "ATLAS_GLM_STEP_FUSE";
const MODULE: &str = "glm_hc_prefill_vec";

/// The step-fuse groups that are on, read once (see
/// [`super::glm_decode_fuse::env_groups`]).
fn groups() -> Result<u32> {
    static GROUPS: OnceLock<std::result::Result<u32, String>> = OnceLock::new();
    env_groups(&GROUPS, NAME, ALL, "fused step groups")
}

/// The seam's partial kernel: `partial`, or its touch twin when the HC touch
/// group is on, `partial` is one of the `_rows_bf16` twins and the target
/// ships the touch twin (same arguments, grid and bytes).
pub fn hc_partial(gpu: &dyn GpuBackend, partial: KernelHandle) -> Result<KernelHandle> {
    Ok(hc_partial_for(groups()?, gpu, partial))
}

fn hc_partial_for(groups: u32, gpu: &dyn GpuBackend, partial: KernelHandle) -> KernelHandle {
    if groups & HC_TOUCH == 0 || partial.0 == 0 {
        return partial;
    }
    let cache = gpu.op_cache();
    let resolve = |name| cache.kernel(gpu, MODULE, name).ok().filter(|k| k.0 != 0);
    TOUCH
        .into_iter()
        .find(|&(rows, _)| resolve(rows).map(|k| k.0) == Some(partial.0))
        .and_then(|(_, touch)| resolve(touch))
        .unwrap_or(partial)
}

/// `(rows twin, its touch twin)` of the seam's partial kernels.
const TOUCH: [(&str, &str); 2] = [
    (
        "glm_hc_decode_partial_rows_bf16",
        "glm_hc_decode_partial_rows_touch_bf16",
    ),
    (
        "glm_hc_decode_post_partial_rows_bf16",
        "glm_hc_decode_post_partial_rows_touch_bf16",
    ),
];

/// The RMS norm a seam's caller runs next over the seam's output rows
/// (`input`, which the seam's finalizer writes), into `out`.
pub struct SeamNorm<'a> {
    kernel: KernelHandle,
    weight: &'a DenseWeight,
    out: DevicePtr,
    eps: f32,
    /// `(input, rows)` once the seam's finalizer wrote `out`.
    fused: Cell<Option<(DevicePtr, u32)>>,
}

impl<'a> SeamNorm<'a> {
    pub fn new(kernel: KernelHandle, weight: &'a DenseWeight, out: DevicePtr, eps: f32) -> Self {
        Self {
            kernel,
            weight,
            out,
            eps,
            fused: Cell::new(None),
        }
    }

    /// `ops::rms_norm(input -> out)` over `rows` rows, unless the seam
    /// already wrote `out` from the same rows.
    pub fn run(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: u32,
        hidden_size: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(done) = self.fused.get() {
            ensure!(
                done == (input, rows),
                "{NAME}: the seam normed {} rows of {:#x}, the caller asks for {rows} of {:#x}",
                done.1,
                done.0.0,
                input.0
            );
            return Ok(());
        }
        super::rms_norm(
            gpu,
            self.kernel,
            input,
            self.weight,
            self.out,
            rows,
            hidden_size,
            self.eps,
            stream,
        )
    }
}

/// The seam's finalizer fused with `norm` (the HC norm group): launches
/// `glm_hc_decode_finalize_norm_bf16` and returns true when the group is on,
/// the target ships it, the highway is BF16 (`bf16_highway`), `hidden_size`
/// is 4096, `norm` is `rms_norm_vanilla` or its `_regs` twin writing another
/// buffer than the seam's output, and there are 1..=32 rows. Else nothing
/// is launched. `ptrs`: highway, partials, hc_scale, hc_base, seam output
/// (`hidden`), post, comb. `eps`: the mHC RMS eps and the Sinkhorn eps.
#[allow(clippy::too_many_arguments)]
pub fn hc_finalize_norm(
    gpu: &dyn GpuBackend,
    norm: Option<&SeamNorm>,
    bf16_highway: bool,
    ptrs: [DevicePtr; 7],
    [rows, hidden_size, sinkhorn_iters]: [u32; 3],
    [norm_eps, hc_eps]: [f32; 2],
    stream: u64,
) -> Result<bool> {
    let Some(norm) = norm else {
        return Ok(false);
    };
    hc_finalize_norm_for(
        groups()?,
        gpu,
        norm,
        bf16_highway,
        ptrs,
        [rows, hidden_size, sinkhorn_iters],
        [norm_eps, hc_eps],
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn hc_finalize_norm_for(
    groups: u32,
    gpu: &dyn GpuBackend,
    norm: &SeamNorm,
    bf16_highway: bool,
    [streams, partial, hc_scale, hc_base, hidden, post, comb]: [DevicePtr; 7],
    [rows, hidden_size, sinkhorn_iters]: [u32; 3],
    [norm_eps, hc_eps]: [f32; 2],
    stream: u64,
) -> Result<bool> {
    if groups & HC_NORM == 0
        || !bf16_highway
        || hidden_size != 4096
        || !(1..=32).contains(&rows)
        || norm.out == hidden
        || norm.fused.get().is_some()
    {
        return Ok(false);
    }
    let cache = gpu.op_cache();
    let shipped = |module, name| cache.kernel(gpu, module, name).ok().filter(|k| k.0 != 0);
    let vanilla = [
        ("rms_norm_vanilla", "rms_norm_vanilla"),
        ("glm_rms_norm_regs", "rms_norm_vanilla_regs"),
    ];
    let Some(kernel) = shipped(MODULE, "glm_hc_decode_finalize_norm_bf16") else {
        return Ok(false);
    };
    if !vanilla
        .into_iter()
        .any(|(module, name)| shipped(module, name).map(|k| k.0) == Some(norm.kernel.0))
    {
        return Ok(false);
    }
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(partial)
        .arg_ptr(hc_scale)
        .arg_ptr(hc_base)
        .arg_ptr(hidden)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(rows)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .arg_ptr(norm.weight.weight)
        .arg_ptr(norm.out)
        .arg_f32(norm.eps)
        .launch(stream)?;
    norm.fused.set(Some((hidden, rows)));
    Ok(true)
}

#[cfg(test)]
#[path = "glm_step_fuse_tests.rs"]
mod tests;
