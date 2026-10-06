// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next exact fused decode tier (`ATLAS_QWEN4EXP_DECODE_FUSE=1`,
//! default off): one launch where a single-token decode layer ran a chain of
//! small dependent kernels, writing the same bytes
//! (`scripts/dev/qwen4exp_decode_fuse_bench.cu` checks every output bit
//! against the chains replaced). The GLM tier (`glm_decode_fuse`) applied to
//! this model's layer stack.
//!
//! `ATLAS_QWEN4EXP_DECODE_FUSE_MASK` (default 7) selects groups for
//! bisection:
//!
//! * `1` GDN: `dense_gemv_ba_gates`, `causal_conv1d_update_l2norm_f32`,
//!   `gated_delta_rule_decode_f32` and `gated_rms_norm_f32_input_sigmoid` as
//!   `qwen4exp_gdn_decode_fused` (the 36 GDN mixers' decode step).
//! * `2` mHC seam: the mixer's `hc_post` and the MoE site's `hc_pre` stage as
//!   `hc_post_stage_vec` (every decode layer, T <= 4).
//! * `4` MoE: under EP, the routed weighted sum reads a null shared row
//!   instead of a memset zero buffer, and the post-all-reduce shared-expert
//!   blend and the layer's mHC post run as `moe_blend_hc_post`.
//!
//! A fused kernel runs only where it is shipped (the qwen3.8-flash-next
//! target) and its shape and alignment hold; otherwise the original launches
//! are issued unchanged. Exact, and the collectives are unchanged, so the
//! ranks need not agree on it (no startup-parity entry).

use anyhow::{Result, anyhow, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

pub const GDN: u32 = 1;
pub const HC_SEAM: u32 = 2;
pub const MOE: u32 = 4;
const ALL: u32 = GDN | HC_SEAM | MOE;

fn parse(fuse: Option<&str>, mask: Option<&str>) -> Result<u32> {
    let on = match fuse {
        None | Some("0") => false,
        Some("1") => true,
        Some(value) => bail!("ATLAS_QWEN4EXP_DECODE_FUSE must be 0 or 1, got {value:?}"),
    };
    let groups = match mask {
        None => ALL,
        Some(value) => {
            let groups: u32 = value.parse().map_err(|_| {
                anyhow!(
                    "ATLAS_QWEN4EXP_DECODE_FUSE_MASK must be a group mask in 0..=7, got {value:?}"
                )
            })?;
            ensure!(
                groups & !ALL == 0,
                "ATLAS_QWEN4EXP_DECODE_FUSE_MASK must be a group mask in 0..=7, got {value:?}"
            );
            groups
        }
    };
    Ok(if on { groups } else { 0 })
}

/// Whether fused `group` is on. The environment is read once; a malformed
/// value fails every launch that asks.
pub fn on(group: u32) -> Result<bool> {
    static GROUPS: OnceLock<std::result::Result<u32, String>> = OnceLock::new();
    let groups = GROUPS.get_or_init(|| {
        let var = |name| std::env::var(name).ok();
        let groups = parse(
            var("ATLAS_QWEN4EXP_DECODE_FUSE").as_deref(),
            var("ATLAS_QWEN4EXP_DECODE_FUSE_MASK").as_deref(),
        );
        if let Ok(groups @ 1..) = groups {
            tracing::info!("ATLAS_QWEN4EXP_DECODE_FUSE: fused decode groups {groups:#x}");
        }
        groups.map_err(|error| error.to_string())
    });
    Ok(groups.clone().map_err(|error| anyhow!(error))? & group != 0)
}

/// `module::name`, resolved once per process (0 when not shipped). Looked up
/// only once its group is on, so other targets leave no failed audit row.
fn kernel(slot: &OnceLock<KernelHandle>, gpu: &dyn GpuBackend, name: &str) -> KernelHandle {
    *slot.get_or_init(|| crate::layers::try_kernel(gpu, module_of(name), name))
}

fn module_of(name: &str) -> &'static str {
    if name == "hc_post_stage_vec" {
        "hyper_connection"
    } else {
        "qwen4exp_decode_fuse"
    }
}

fn aligned(ptrs: &[DevicePtr], to: u64) -> bool {
    ptrs.iter().all(|p| p.0.is_multiple_of(to))
}

/// The GDN mixer's decode buffers between its two projections.
pub struct GdnDecode {
    pub h_state: DevicePtr,
    pub conv_state: DevicePtr,
    /// `[Q | K | V | Z]` of this token (Z is the norm gate).
    pub qkvz: DevicePtr,
    pub conv_w: DevicePtr,
    pub ba_in: DevicePtr,
    pub ba_w: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub gate_out: DevicePtr,
    pub beta_out: DevicePtr,
    pub norm_w: DevicePtr,
    pub out: DevicePtr,
}

/// Geometry the fused kernel is written for: 128-wide heads, three value
/// heads per key head (one thread-block cluster per key head), a 4-tap conv.
const GDN_D: u32 = 128;
const GDN_REPEAT: u32 = 3;
const GDN_DCONV: u32 = 4;

/// `qwen4exp_gdn_decode_fused` in place of the BA gates, conv, recurrence and
/// gated-norm launches of one decode token, when the GDN group is on and the
/// shape fits. Returns whether it launched; on `false` nothing was launched.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode(
    gpu: &dyn GpuBackend,
    b: &GdnDecode,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    ba_k: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    if !on(GDN)? {
        return Ok(false);
    }
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = kernel(&K, gpu, "qwen4exp_gdn_decode_fused");
    let conv_dim = 2 * nk * kd + nv * vd;
    let fits = k.0 != 0
        && gdn_geometry_fits(nk, nv, kd, vd, d_conv, ba_k)
        && aligned(&[b.conv_state, b.ba_in, b.ba_w], 16)
        && aligned(&[b.qkvz.offset(conv_dim as usize * 2), b.norm_w, b.out], 8);
    if !fits {
        return Ok(false);
    }
    KernelLaunch::new(gpu, k)
        .grid([nv, 1, 1])
        .block([GDN_D, 1, 1])
        .arg_ptr(b.h_state)
        .arg_ptr(b.conv_state)
        .arg_ptr(b.qkvz)
        .arg_ptr(b.conv_w)
        .arg_ptr(b.ba_in)
        .arg_ptr(b.ba_w)
        .arg_ptr(b.a_log)
        .arg_ptr(b.dt_bias)
        .arg_ptr(b.gate_out)
        .arg_ptr(b.beta_out)
        .arg_ptr(b.qkvz.offset(conv_dim as usize * 2))
        .arg_ptr(b.norm_w)
        .arg_ptr(b.out)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(ba_k)
        .arg_u32(kd)
        .arg_f32(l2_eps)
        .arg_f32(eps)
        .launch(stream)?;
    Ok(true)
}

/// The geometry `qdf_gdn_block` is written for (both launch forms).
fn gdn_geometry_fits(nk: u32, nv: u32, kd: u32, vd: u32, d_conv: u32, ba_k: u32) -> bool {
    kd == GDN_D
        && vd == GDN_D
        && nk != 0
        && nv == GDN_REPEAT * nk
        && d_conv == GDN_DCONV
        && ba_k.is_multiple_of(8)
}

/// Sequences one `qwen4exp_gdn_decode_fused_rows` launch takes
/// (`QDF_ROWS_MAX`: the by-value state table's length).
pub const GDN_ROWS_MAX: usize = 8;

/// A batched decode step's GDN buffers between the projections: row `r` is
/// sequence `r`'s token, against `states[r]` (recurrence, conv window).
pub struct GdnDecodeRows<'a> {
    pub states: &'a [(DevicePtr, DevicePtr)],
    /// `[rows, qkvz_stride]` BF16: `[Q | K | V | Z]` per row.
    pub qkvz: DevicePtr,
    pub qkvz_stride: u32,
    pub conv_w: DevicePtr,
    /// `[rows, ba_k]` BF16: the mixer input rows.
    pub ba_in: DevicePtr,
    pub ba_w: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    /// `[rows, 2 * nv]` FP32: gate then beta per row.
    pub gates: DevicePtr,
    pub norm_w: DevicePtr,
    /// `[rows, nv * 128]` BF16.
    pub out: DevicePtr,
}

/// `ATLAS_QWEN4EXP_BATCH_SMALL`: the GDN step of every row of a batched
/// decode (one token per sequence) as `qwen4exp_gdn_decode_fused_rows`
/// launches of up to [`GDN_ROWS_MAX`] sequences, in place of four launches
/// per sequence. Each row's bytes are the four kernels' (the fused step's
/// contract), whether or not the decode-fuse tier is on. Returns whether it
/// launched; on `false` nothing was launched. The caller checks the lever and
/// that its per-sequence arm is the four-kernel FP32 one.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_rows(
    gpu: &dyn GpuBackend,
    b: &GdnDecodeRows<'_>,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    ba_k: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = *K.get_or_init(|| {
        crate::layers::try_kernel(
            gpu,
            "qwen4exp_decode_fuse",
            "qwen4exp_gdn_decode_fused_rows",
        )
    });
    let conv_dim = 2 * nk * kd + nv * vd;
    let fits = k.0 != 0
        && !b.states.is_empty()
        && gdn_geometry_fits(nk, nv, kd, vd, d_conv, ba_k)
        && b.states.iter().all(|&(_, conv)| conv.0.is_multiple_of(16))
        && aligned(&[b.ba_in, b.ba_w], 16)
        && b.qkvz_stride.is_multiple_of(4)
        && aligned(&[b.qkvz.offset(conv_dim as usize * 2), b.norm_w, b.out], 8);
    if !fits {
        return Ok(false);
    }
    let (bf16, fp32) = (2usize, 4usize);
    for (chunk, states) in b.states.chunks(GDN_ROWS_MAX).enumerate() {
        let first = chunk * GDN_ROWS_MAX;
        // QdfRowStates: h[8], then conv[8]; unused slots stay null.
        let mut table = [0u64; 2 * GDN_ROWS_MAX];
        for (i, &(h, conv)) in states.iter().enumerate() {
            table[i] = h.0;
            table[GDN_ROWS_MAX + i] = conv.0;
        }
        KernelLaunch::new(gpu, k)
            .grid([nv, states.len() as u32, 1])
            .block([GDN_D, 1, 1])
            .arg_words(&table)
            .arg_ptr(b.qkvz.offset(first * b.qkvz_stride as usize * bf16))
            .arg_ptr(b.conv_w)
            .arg_ptr(b.ba_in.offset(first * ba_k as usize * bf16))
            .arg_ptr(b.ba_w)
            .arg_ptr(b.a_log)
            .arg_ptr(b.dt_bias)
            .arg_ptr(b.gates.offset(first * 2 * nv as usize * fp32))
            .arg_ptr(b.norm_w)
            .arg_ptr(b.out.offset(first * (nv * vd) as usize * bf16))
            .arg_u32(nk)
            .arg_u32(nv)
            .arg_u32(ba_k)
            .arg_u32(kd)
            .arg_u32(b.qkvz_stride)
            .arg_f32(l2_eps)
            .arg_f32(eps)
            .launch(stream)?;
    }
    Ok(true)
}

/// Must match `HC_V_STAGE_SPLIT` and `QHC_MAX_MULT` in hyper_connection.cu.
const HC_SPLIT: u32 = 8;
const HC_MAX_MULT: u32 = 8;

/// The mixer's mHC post folded into the next site's stage: the block output
/// and injection vector `hc_post` would have consumed.
#[derive(Clone, Copy)]
pub struct HcPostFold {
    pub block_out: DevicePtr,
    pub inj: DevicePtr,
}

static HC_POST_STAGE: OnceLock<KernelHandle> = OnceLock::new();

/// Whether the mHC seam group is on and [`hc_post_stage`] can take this
/// post and stage (kernel shipped, shape and alignment).
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_post_stage_fits(
    gpu: &dyn GpuBackend,
    fold: HcPostFold,
    streams: DevicePtr,
    norm_w: DevicePtr,
    normed: DevicePtr,
    hidden_size: u32,
    hc_mult: u32,
) -> Result<bool> {
    if !on(HC_SEAM)? {
        return Ok(false);
    }
    let hc_dim = hc_mult * hidden_size;
    Ok(kernel(&HC_POST_STAGE, gpu, "hc_post_stage_vec").0 != 0
        && hc_mult <= HC_MAX_MULT
        && hidden_size.is_multiple_of(4)
        && hc_dim.is_multiple_of(4 * HC_SPLIT)
        && aligned(&[streams, norm_w, normed], 16)
        && aligned(&[fold.block_out], 8))
}

/// `hc_post_stage_vec` in place of `hc_post` (in place on `streams`) and the
/// stage of the next `hc_pre`, writing `normed` exactly as either stage
/// kernel does. Only after [`hc_post_stage_fits`].
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_post_stage(
    gpu: &dyn GpuBackend,
    fold: HcPostFold,
    streams: DevicePtr,
    norm_w: DevicePtr,
    normed: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let k = kernel(&HC_POST_STAGE, gpu, "hc_post_stage_vec");
    KernelLaunch::new(gpu, k)
        .grid([num_tokens, HC_SPLIT, 1])
        .block([1024, 1, 1])
        .arg_ptr(fold.block_out)
        .arg_ptr(streams)
        .arg_ptr(fold.inj)
        .arg_ptr(norm_w)
        .arg_ptr(normed)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(eps)
        .launch(stream)
}

/// The layer's mHC post the MoE forward may run itself (MoE group): the
/// highway, the injection vector and the stream count.
#[derive(Clone, Copy)]
pub struct MoeHcPost {
    pub streams: DevicePtr,
    pub inj: DevicePtr,
    pub hc_mult: u32,
}

const BLEND_BLOCK: u32 = 256;

static MOE_BLEND_HC_POST: OnceLock<KernelHandle> = OnceLock::new();

/// Whether the EP routed sum may pass a null shared row to
/// `moe_weighted_sum_blend` instead of a memset zero buffer: the MoE group is
/// on and this is the target that ships the fused tier (whose
/// `moe_weighted_sum_blend` reads null as +0.0, the bytes a zeroed BF16 row
/// gives).
pub fn null_shared_row(gpu: &dyn GpuBackend) -> Result<bool> {
    Ok(on(MOE)? && kernel(&MOE_BLEND_HC_POST, gpu, "moe_blend_hc_post").0 != 0)
}

/// `moe_blend_hc_post` in place of the EP `moe_batched_blend` of
/// `num_tokens` rows and the layer's following `hc_post` (in place on the
/// highway). Returns whether it launched.
#[allow(clippy::too_many_arguments)]
pub fn moe_blend_hc_post(
    gpu: &dyn GpuBackend,
    post: MoeHcPost,
    output: DevicePtr,
    shared_out: DevicePtr,
    normed: DevicePtr,
    gate_weight: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<bool> {
    if !on(MOE)? {
        return Ok(false);
    }
    let k = kernel(&MOE_BLEND_HC_POST, gpu, "moe_blend_hc_post");
    let fits = k.0 != 0 && hidden_size.is_multiple_of(4) && aligned(&[post.streams], 16);
    if !fits {
        return Ok(false);
    }
    KernelLaunch::new(gpu, k)
        .grid([num_tokens, hidden_size.div_ceil(4 * BLEND_BLOCK), 1])
        .block([BLEND_BLOCK, 1, 1])
        .arg_ptr(output)
        .arg_ptr(shared_out)
        .arg_ptr(normed)
        .arg_ptr(gate_weight)
        .arg_ptr(post.streams)
        .arg_ptr(post.inj)
        .arg_u32(hidden_size)
        .arg_u32(post.hc_mult)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_groups() {
        assert_eq!(parse(None, None).unwrap(), 0);
        assert_eq!(parse(Some("0"), Some("7")).unwrap(), 0);
        assert_eq!(parse(Some("1"), None).unwrap(), ALL);
        assert_eq!(parse(Some("1"), Some("5")).unwrap(), GDN | MOE);
        assert!(parse(Some("yes"), None).is_err());
        assert!(parse(Some("1"), Some("8")).is_err());
        assert!(parse(Some("1"), Some("x")).is_err());
    }
}
