// SPDX-License-Identifier: AGPL-3.0-only

//! Launch sequences for the EXL3 reference-policy (FLA bit-parity) KDA
//! kernels in `kernels/gb10/glm-5.3-flash/exl3/kda_fla.cu`. Grids, blocks and argument
//! order mirror the byte-exact GPU tests in `tests/exl3_validation/kda_fla/`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Head dimension the kernels are written for (K = V = 128).
pub const KDA_FLA_DIM: usize = 128;
/// Reference chunk length (FLA_CHUNK_SIZE).
pub const KDA_FLA_CHUNK: usize = 64;
const DELTA_BV: u32 = 32;
const OUT_BV: u32 = 64;
const L2_EPS: f32 = 1.0e-6;
/// RCP_LN2 as the f32 the reference passes for the chunk cumsum scale
/// (1.4426950408889634 rounds to `LOG2_E`, 0x3FB8AA3B).
const CUMSUM_SCALE: f32 = std::f32::consts::LOG2_E;

#[derive(Clone, Copy, Debug)]
pub struct KdaFlaKernels {
    conv_prefill: KernelHandle,
    conv_decode: KernelHandle,
    l2norm: KernelHandle,
    beta: KernelHandle,
    gate: KernelHandle,
    kkt_inter: KernelHandle,
    kkt_intra: KernelHandle,
    solve_tril: KernelHandle,
    recompute_w_u: KernelHandle,
    delta_h: KernelHandle,
    output: KernelHandle,
    recurrent_decode: KernelHandle,
    /// Multi-row verify variants (0 when the module predates them).
    conv_rows_snap: KernelHandle,
    recurrent_rows_snap: KernelHandle,
}

impl KdaFlaKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| gpu.kernel("kda_fla", name);
        Ok(Self {
            conv_prefill: k("kda_fla_conv_prefill")?,
            conv_decode: k("kda_fla_conv_decode")?,
            l2norm: k("kda_fla_l2norm")?,
            beta: k("kda_fla_beta")?,
            gate: k("kda_fla_gate_cumsum")?,
            kkt_inter: k("kda_fla_kkt_inter")?,
            kkt_intra: k("kda_fla_kkt_intra")?,
            solve_tril: k("kda_fla_solve_tril")?,
            recompute_w_u: k("kda_fla_recompute_w_u")?,
            delta_h: k("kda_fla_delta_h")?,
            output: k("kda_fla_output")?,
            recurrent_decode: k("kda_fla_recurrent_decode")?,
            conv_rows_snap: k("kda_fla_conv_rows_snap").unwrap_or(KernelHandle(0)),
            recurrent_rows_snap: k("kda_fla_recurrent_decode_rows").unwrap_or(KernelHandle(0)),
        })
    }
}

/// Byte offsets of the prefill intermediates inside one scratch region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdaFlaScratchLayout {
    q: usize,
    k: usize,
    beta: usize,
    g: usize,
    a: usize,
    aqk: usize,
    ai: usize,
    w: usize,
    u: usize,
    kg: usize,
    h: usize,
    v_new: usize,
    /// Bytes of `A` (and of `Aqk`), cleared before the KKT kernels.
    a_bytes: usize,
    pub total_bytes: usize,
}

impl KdaFlaScratchLayout {
    pub fn new(tokens: usize, heads: usize) -> Result<Self> {
        ensure!(
            tokens > 1 && heads > 0,
            "KDA FLA prefill needs >1 row and heads"
        );
        let align = |n: usize| n.div_ceil(256) * 256;
        let plane = tokens * heads * KDA_FLA_DIM;
        let chunks = tokens.div_ceil(KDA_FLA_CHUNK);
        let a_bytes = tokens * heads * KDA_FLA_CHUNK * 4;
        let sizes = [
            plane * 2,                                      // q (l2, bf16)
            plane * 2,                                      // k (l2, bf16)
            tokens * heads * 4,                             // beta f32
            plane * 4,                                      // g cumsum f32
            a_bytes,                                        // A f32
            a_bytes,                                        // Aqk f32
            tokens * heads * KDA_FLA_CHUNK * 2,             // Ai bf16
            plane * 2,                                      // w
            plane * 2,                                      // u
            plane * 2,                                      // kg
            chunks * heads * KDA_FLA_DIM * KDA_FLA_DIM * 2, // h per chunk bf16
            plane * 2,                                      // v_new
        ];
        let mut offsets = [0usize; 12];
        let mut at = 0usize;
        for (o, s) in offsets.iter_mut().zip(sizes) {
            *o = at;
            at += align(s);
        }
        let [q, k, beta, g, a, aqk, ai, w, u, kg, h, v_new] = offsets;
        Ok(Self {
            q,
            k,
            beta,
            g,
            a,
            aqk,
            ai,
            w,
            u,
            kg,
            h,
            v_new,
            a_bytes,
            total_bytes: at,
        })
    }
}

/// One sequence's KDA core operands in Atlas layouts.
#[derive(Clone, Copy, Debug)]
pub struct KdaFlaIo {
    /// Packed q|k|v projection rows [T, 3*heads*128] BF16.
    pub qkv: DevicePtr,
    /// Raw gate [T, heads, 128] BF16.
    pub raw_gate: DevicePtr,
    /// Raw beta logits [T, heads] BF16.
    pub raw_beta: DevicePtr,
    /// Conv weight [3*heads*128, 4] BF16.
    pub conv_weight: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    /// Conv window [3*heads*128, 4] F32 (last four raw rows, newest last).
    pub conv_state: DevicePtr,
    /// Recurrent state [heads, K, V] F32, updated in place.
    pub state: DevicePtr,
    /// Conv output [T, 3*heads*128] BF16 (scratch, also the v source).
    pub conv_out: DevicePtr,
    /// Core output [T, heads, 128] BF16.
    pub output: DevicePtr,
    pub heads: u32,
    pub lower_bound: f32,
}

fn scale() -> f32 {
    (KDA_FLA_DIM as f64).powf(-0.5) as f32
}

/// Chunked prefill of `tokens > 1` rows (vLLM's prefill classification).
pub fn kda_fla_prefill(
    gpu: &dyn GpuBackend,
    k: &KdaFlaKernels,
    io: &KdaFlaIo,
    tokens: u32,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
) -> Result<()> {
    let h = io.heads;
    let layout = KdaFlaScratchLayout::new(tokens as usize, h as usize)?;
    ensure!(
        layout.total_bytes <= scratch_bytes,
        "KDA FLA scratch too small: need {} have {scratch_bytes}",
        layout.total_bytes
    );
    let at = |o: usize| scratch.offset(o);
    let (q, kn, beta, g) = (at(layout.q), at(layout.k), at(layout.beta), at(layout.g));
    let (a, aqk, ai) = (at(layout.a), at(layout.aqk), at(layout.ai));
    let (w, u, kg, hs, v_new) = (
        at(layout.w),
        at(layout.u),
        at(layout.kg),
        at(layout.h),
        at(layout.v_new),
    );
    let dim = 3 * h * KDA_FLA_DIM as u32;
    let nt = div_ceil(tokens, KDA_FLA_CHUNK as u32);
    KernelLaunch::new(gpu, k.conv_prefill)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.conv_state)
        .arg_ptr(io.qkv)
        .arg_ptr(io.conv_weight)
        .arg_ptr(io.conv_out)
        .arg_u32(dim)
        .arg_u32(tokens)
        // Atlas zeroes the window for a fresh sequence; reading zeros is
        // bitwise identical to the reference's has_initial_state=false path.
        .arg_u32(1)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.l2norm)
        .grid([div_ceil(2 * tokens * h, 8), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(io.conv_out)
        .arg_ptr(q)
        .arg_ptr(kn)
        .arg_u32(tokens)
        .arg_u32(h)
        .arg_f32(L2_EPS)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.beta)
        .grid([div_ceil(tokens * h, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.raw_beta)
        .arg_ptr(beta)
        .arg_u32(tokens * h)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.gate)
        .grid([div_ceil(h * KDA_FLA_DIM as u32, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(io.raw_gate)
        .arg_ptr(io.dt_bias)
        .arg_ptr(io.a_log)
        .arg_ptr(g)
        .arg_u32(tokens)
        .arg_u32(h)
        .arg_f32(io.lower_bound)
        .arg_f32(CUMSUM_SCALE)
        .launch(stream)?;
    // The reference allocates A/Aqk with torch.zeros; the KKT kernels write
    // only their own blocks.
    gpu.memset_async(a, 0, layout.a_bytes, stream)?;
    gpu.memset_async(aqk, 0, layout.a_bytes, stream)?;
    for (kernel, blocks, threads) in [(k.kkt_inter, 16, 32), (k.kkt_intra, 4, 256)] {
        KernelLaunch::new(gpu, kernel)
            .grid([nt, blocks, h])
            .block([threads, 1, 1])
            .arg_ptr(q)
            .arg_ptr(kn)
            .arg_ptr(g)
            .arg_ptr(beta)
            .arg_ptr(a)
            .arg_ptr(aqk)
            .arg_f32(scale())
            .arg_u32(tokens)
            .arg_u32(h)
            .launch(stream)?;
    }
    KernelLaunch::new(gpu, k.solve_tril)
        .grid([nt, h, 1])
        .block([64, 1, 1])
        .arg_ptr(a)
        .arg_ptr(ai)
        .arg_u32(tokens)
        .arg_u32(h)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.recompute_w_u)
        .grid([nt, h, 1])
        .block([128, 1, 1])
        .arg_ptr(kn)
        .arg_ptr(io.conv_out.offset(2 * h as usize * KDA_FLA_DIM * 2))
        .arg_u32(dim)
        .arg_ptr(beta)
        .arg_ptr(ai)
        .arg_ptr(g)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(kg)
        .arg_u32(tokens)
        .arg_u32(h)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.delta_h)
        .grid([KDA_FLA_DIM as u32 / DELTA_BV, h, 1])
        .block([128, 1, 1])
        .arg_ptr(kg)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(g)
        .arg_ptr(io.state)
        .arg_ptr(hs)
        .arg_ptr(v_new)
        .arg_u32(tokens)
        .arg_u32(h)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.output)
        .grid([KDA_FLA_DIM as u32 / OUT_BV, nt, h])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(v_new)
        .arg_ptr(g)
        .arg_ptr(aqk)
        .arg_ptr(hs)
        .arg_ptr(io.output)
        .arg_f32(scale())
        .arg_u32(tokens)
        .arg_u32(h)
        .launch(stream)
}

/// One row (vLLM's `query_len == 1` decode classification).
pub fn kda_fla_decode(
    gpu: &dyn GpuBackend,
    k: &KdaFlaKernels,
    io: &KdaFlaIo,
    stream: u64,
) -> Result<()> {
    let dim = 3 * io.heads * KDA_FLA_DIM as u32;
    KernelLaunch::new(gpu, k.conv_decode)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.conv_state)
        .arg_ptr(io.qkv)
        .arg_ptr(io.conv_weight)
        .arg_ptr(io.conv_out)
        .arg_u32(dim)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.recurrent_decode)
        .grid([KDA_FLA_DIM as u32 / 32, io.heads, 1])
        .block([128, 1, 1])
        .arg_ptr(io.conv_out)
        .arg_ptr(io.raw_gate)
        .arg_ptr(io.raw_beta)
        .arg_ptr(io.a_log)
        .arg_ptr(io.dt_bias)
        .arg_ptr(io.state)
        .arg_ptr(io.output)
        .arg_u32(io.heads)
        .launch(stream)
}

/// Speculative verify of `rows` decode rows in two launches: the conv and
/// recurrence of each row are those of [`kda_fla_decode`], with the conv
/// window after every row written to `conv_snaps + t * conv_snap_stride`
/// and the recurrent state after rows `0..rows-1` to
/// `h_snaps + t * h_snap_stride` (strides in bytes, f32-aligned). Returns
/// `false` (nothing launched) when the kernels are unavailable.
#[allow(clippy::too_many_arguments)]
pub fn kda_fla_verify_rows(
    gpu: &dyn GpuBackend,
    k: &KdaFlaKernels,
    io: &KdaFlaIo,
    rows: u32,
    conv_snaps: DevicePtr,
    conv_snap_stride: usize,
    h_snaps: DevicePtr,
    h_snap_stride: usize,
    stream: u64,
) -> Result<bool> {
    if k.conv_rows_snap.0 == 0 || k.recurrent_rows_snap.0 == 0 {
        return Ok(false);
    }
    anyhow::ensure!(
        conv_snap_stride.is_multiple_of(4) && h_snap_stride.is_multiple_of(4),
        "KDA FLA verify snapshot strides must be f32-aligned"
    );
    let dim = 3 * io.heads * KDA_FLA_DIM as u32;
    KernelLaunch::new(gpu, k.conv_rows_snap)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(io.conv_state)
        .arg_ptr(io.qkv)
        .arg_ptr(io.conv_weight)
        .arg_ptr(io.conv_out)
        .arg_u32(dim)
        .arg_u32(rows)
        .arg_ptr(conv_snaps)
        .arg_u64((conv_snap_stride / 4) as u64)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.recurrent_rows_snap)
        .grid([KDA_FLA_DIM as u32 / 32, io.heads, 1])
        .block([128, 1, 1])
        .arg_ptr(io.conv_out)
        .arg_ptr(io.raw_gate)
        .arg_ptr(io.raw_beta)
        .arg_ptr(io.a_log)
        .arg_ptr(io.dt_bias)
        .arg_ptr(io.state)
        .arg_ptr(io.output)
        .arg_u32(io.heads)
        .arg_u32(rows)
        .arg_ptr(h_snaps)
        .arg_u64((h_snap_stride / 4) as u64)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_aligned_disjoint_and_sized() {
        let l = KdaFlaScratchLayout::new(129, 32).unwrap();
        let offs = [
            l.q, l.k, l.beta, l.g, l.a, l.aqk, l.ai, l.w, l.u, l.kg, l.h, l.v_new,
        ];
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
        assert!(offs.iter().all(|o| o % 256 == 0));
        // 3 chunks of per-head 128x128 BF16 h plus the per-row planes.
        assert_eq!(l.v_new - l.h, 3 * 32 * 128 * 128 * 2);
        assert!(l.total_bytes >= l.v_new + 129 * 32 * 128 * 2);
        assert_eq!(l.a_bytes, 129 * 32 * 64 * 4);
    }

    #[test]
    fn layout_rejects_decode_and_empty_geometry() {
        assert!(KdaFlaScratchLayout::new(1, 32).is_err());
        assert!(KdaFlaScratchLayout::new(5, 0).is_err());
    }

    #[test]
    fn scale_matches_reference_f32() {
        // 128 ** -0.5 rounded to f32, the value the byte-exact GPU tests pass.
        assert_eq!(scale().to_bits(), 0x3DB5_04F3);
        assert_eq!(CUMSUM_SCALE.to_bits(), 0x3FB8_AA3B);
    }
}
