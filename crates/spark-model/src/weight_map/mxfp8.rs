// SPDX-License-Identifier: AGPL-3.0-only

//! MXFP8 weight twins (OCP microscaling FP8): E4M3 values plus one E8M0
//! scale per 32 input columns.
//!
//! Two sources produce them. GLM quantizes BF16 projections into MXFP8 twins
//! at load (`ops::mxfp8_quantize`). Checkpoints can also SHIP them: ModelOpt's
//! `quant_algo: "MXFP8"` (nvidia Qwen3.8-Flash-Next-FP4-Local's whole
//! backbone) stores `X.weight` F8_E4M3 `[n, k]` next to `X.weight_scale` U8
//! UE8M0 `[n, k/32]` — byte-for-byte the untiled [`Mxfp8Weight`] layout.
//!
//! For those, [`mxfp8_source`] is the per-tensor probe ("is this projection
//! MXFP8, and where are its bytes"), and [`dequant_mxfp8_to_bf16`] is the
//! exact BF16 expansion every existing BF16 arm can load unchanged.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::DenseWeight;

/// MXFP8 twin of a projection: E4M3 `[n, k]` + E8M0 `[n, k/32]`.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Weight {
    pub data: DevicePtr,
    pub scales: DevicePtr,
    /// The bytes are in `mxfp8_tile16`'s chunk-tiled layout
    /// (`ops::mxfp8_tile_if_requested`), served only by the `_tiled` tiers.
    pub tiled: bool,
}

/// Values per E8M0 scale block.
const BLOCK: usize = 32;

/// E8M0 exponents whose scale keeps EVERY E4M3 value exact in BF16.
///
/// E4M3 values are multiples of 2^-9 with at most 4 significant bits, up to
/// 448 = 1.75 * 2^8. Times 2^s they stay exact in BF16 (8 significant bits,
/// f32's exponent range, subnormals down to 2^-133) iff `s - 9 >= -133` and
/// `8 + s <= 127`, i.e. `-124 <= s <= 119`, i.e. biased `3 <= e <= 246`.
/// Real checkpoints sit far inside it (FP4-Local: 111..=119), so anything
/// outside is refused rather than rounded: this path promises exactness.
const EXACT_E8M0: std::ops::RangeInclusive<u8> = 3..=246;

/// A checkpoint-native MXFP8 projection in the store.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Source {
    /// `data` = `X.weight`, `scales` = `X.weight_scale`, untiled.
    pub weight: Mxfp8Weight,
    pub n: usize,
    pub k: usize,
}

/// Is `{prefix}.weight` a checkpoint MXFP8 projection?
///
/// `Ok(None)` for anything else, including the other FP8 layouts (BF16/F32
/// block or per-row scales, `weight_scale_inv`, DeepSeek-V4's `.scale`).
/// MXFP8 is recognised by an E4M3 weight with a ONE-BYTE `weight_scale`
/// (U8 or F8_E8M0): no other FP8 scheme in the tree stores its scale that way.
/// A tensor that matches that signature but has the wrong scale geometry is an
/// error — it would otherwise fall through to a layout it is not.
pub fn mxfp8_source(store: &WeightStore, prefix: &str) -> Result<Option<Mxfp8Source>> {
    let (Ok(w), Ok(s)) = (
        store.get(&format!("{prefix}.weight")),
        store.get(&format!("{prefix}.weight_scale")),
    ) else {
        return Ok(None);
    };
    if w.dtype != WeightDtype::FP8E4M3
        || !matches!(s.dtype, WeightDtype::UInt8 | WeightDtype::FP8E8M0)
    {
        return Ok(None);
    }
    ensure!(
        w.shape.len() == 2 && w.shape[1].is_multiple_of(BLOCK),
        "{prefix}: MXFP8 weight must be [n, k] with k % 32 == 0, got {:?}",
        w.shape
    );
    let (n, k) = (w.shape[0], w.shape[1]);
    ensure!(
        s.shape == [n, k / BLOCK],
        "{prefix}: MXFP8 scale is {:?}, expected [{n}, {}] for a [{n}, {k}] weight",
        s.shape,
        k / BLOCK
    );
    Ok(Some(Mxfp8Source {
        weight: Mxfp8Weight {
            data: w.ptr,
            scales: s.ptr,
            tiled: false,
        },
        n,
        k,
    }))
}

/// UE8M0 scale bytes as f32 multipliers, refusing any that would make the
/// BF16 expansion inexact (see [`EXACT_E8M0`]; 0xFF is the NaN code).
pub(crate) fn mxfp8_scales_f32(scales: &[u8]) -> Result<Vec<f32>> {
    if let Some(at) = scales.iter().position(|e| !EXACT_E8M0.contains(e)) {
        bail!(
            "MXFP8 scale block {at} has E8M0 exponent {} (2^{}), outside the \
             {EXACT_E8M0:?} range where the BF16 expansion is exact",
            scales[at],
            scales[at] as i32 - 127
        );
    }
    Ok(scales
        .iter()
        .map(|&e| f32::from_bits((e as u32) << 23))
        .collect())
}

/// Host reference: MXFP8 `[n, k]` to BF16 bits. The kernel path below must
/// match it bit for bit; it is also what the golden test pins against
/// PyTorch's own `float8_e4m3fn -> float32 -> bfloat16`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn mxfp8_to_bf16_host(
    data: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<u16>> {
    ensure!(
        data.len() == n * k && scales.len() * BLOCK == n * k,
        "MXFP8 host dequant: {} data / {} scale bytes for [{n}, {k}]",
        data.len(),
        scales.len()
    );
    let mult = mxfp8_scales_f32(scales)?;
    Ok(data
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let v = atlas_core::numeric::fp8_e4m3_to_f32(b) * mult[i / BLOCK];
            // Exact by construction (`EXACT_E8M0`), so RNE never rounds here:
            // the result's low 16 f32 bits are zero.
            atlas_core::numeric::f32_to_bf16(v)
        })
        .collect())
}

/// Expand a checkpoint MXFP8 projection to a fresh BF16 `[n, k]` buffer.
///
/// EXACT: every E4M3 value times a power of two in [`EXACT_E8M0`] is
/// representable in BF16, so no rounding happens anywhere — the BF16 copy is
/// the checkpoint's own math, not an approximation of it.
///
/// Runs the shared `dequant_fp8_blockscaled_bf16` kernel (in `common/`, so
/// every kernel target has it) with a `1 x 32` block and the E8M0 bytes widened
/// to f32 on the host: the scale matrix is k/32 bytes a row, so the D2H, the
/// widening and the H2D are a few MB for the largest projection.
///
/// The source stays in the store. Whether to release it once the BF16 copy
/// exists is the model loader's call: it is the copy an MXFP8-native decode
/// path would read.
pub fn dequant_mxfp8_to_bf16(gpu: &dyn GpuBackend, src: &Mxfp8Source) -> Result<DenseWeight> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let (n, k) = (src.n, src.k);
    let mut e8m0 = vec![0u8; n * k / BLOCK];
    gpu.copy_d2h(src.weight.scales, &mut e8m0)
        .context("MXFP8 scale D2H")?;
    let mult = mxfp8_scales_f32(&e8m0)?;
    let mult_bytes: Vec<u8> = mult.iter().flat_map(|f| f.to_le_bytes()).collect();
    let scale_f32 = gpu.alloc(mult_bytes.len())?;
    gpu.copy_h2d(&mult_bytes, scale_f32)?;

    let out = gpu.alloc(n * k * 2)?;
    let stream = gpu.default_stream();
    let launched = gpu
        .kernel(
            "dequant_fp8_blockscaled_bf16",
            "dequant_fp8_blockscaled_bf16",
        )
        .and_then(|kernel| {
            KernelLaunch::new(gpu, kernel)
                .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
                .block([64, 4, 1])
                .arg_ptr(src.weight.data)
                .arg_ptr(scale_f32)
                .arg_ptr(out)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(1) // block_n: one scale row per weight row
                .arg_u32(BLOCK as u32) // block_k
                .arg_u32((k / BLOCK) as u32) // scale row stride
                .arg_u32(1) // scale_is_fp32
                .launch(stream)
        })
        .and_then(|()| gpu.synchronize(stream));
    // The widened scales are dead once the kernel has run (synchronized
    // above), and are freed on the error path too.
    gpu.free(scale_f32)?;
    if let Err(e) = launched {
        let _ = gpu.free(out);
        return Err(e.context(format!("MXFP8 -> BF16 dequant [{n}, {k}]")));
    }
    Ok(DenseWeight { weight: out })
}

#[cfg(test)]
#[path = "mxfp8_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "mxfp8_gpu_tests.rs"]
mod gpu_tests;
