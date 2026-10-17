// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen vision geometry: the area-bound / long-side sizing and the fit to the
//! encoder's patch capacity. Split out of `vision_preprocess` like
//! `glm_sizing`.

use super::*;

/// Compute the target (H, W) so that:
/// - The area bound is respected: `max_pixels` when the caller supplies one,
///   otherwise the long side is clamped to [`FALLBACK_MAX_DIM`].
/// - The long side never exceeds [`ABS_MAX_DIM`], bound or not.
/// - Both sides are multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - Aspect ratio is preserved (rounded to nearest grid_unit).
/// - The image is never upscaled.
///
/// `max_pixels` is an area, matching the `size.longest_edge` /
/// `shortest_edge` convention HF's Qwen2VL/Qwen3VL processors use (both are
/// pixel counts, not edge lengths, despite the names). It comes from the
/// checkpoint's `preprocessor_config.json` or the operator's
/// `--vision-max-pixels`; the operator's value wins.
///
/// ★ `max_pixels` REPLACES the long-side clamp rather than combining with it.
/// Combining was the bug: `dim_scale.min(pixel_scale)` meant a checkpoint
/// permitting 4096² could never exceed 1280 on the long side, so the model's
/// own declared bound could only ever lower the resolution, never raise it.
/// `pub(crate)` alias name used by the video path — see [`target_size_for`].
///
/// For a Qwen video's frames: each temporal group is one encoder sequence, so
/// it is fitted to the encoder's capacity like a still ([`qwen_fit`]).
pub(crate) fn target_size_for(
    orig_h: u32,
    orig_w: u32,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    qwen_fit(vcfg, orig_h, orig_w, max_pixels)
}

pub(super) fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    snapped_target(orig_h, orig_w, grid_unit, max_pixels, f32::round)
}

/// [`target_size_with_max_pixels`] with each side snapped by `snap`.
fn snapped_target(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
    snap: fn(f32) -> f32,
) -> (u32, u32) {
    let long_side = orig_h.max(orig_w) as f32;
    let area = (orig_h as f32) * (orig_w as f32);
    let bound_scale = match max_pixels.filter(|&p| p > 0) {
        // Model- or operator-declared AREA bound governs.
        Some(p) => ((p as f32) / area).sqrt(),
        // Nothing declared: fall back to the historical long-side clamp.
        None => (FALLBACK_MAX_DIM as f32) / long_side,
    };
    // Safety net, always applied.
    let abs_scale = (ABS_MAX_DIM as f32) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0); // never upscale
    let side = |orig: u32| (snap(orig as f32 * scale / grid_unit as f32) as u32).max(1) * grid_unit;
    (side(orig_h), side(orig_w))
}

/// The Qwen target of a still (or a video's frames), fitted to the encoder's
/// patch capacity (`derive_max_patches`, at most `CEILING_MAX_PATCHES`).
///
/// A size that fits is exactly [`target_size_with_max_pixels`]'s (images
/// that were served before are byte-identical). One that does not (a
/// 2880x1800 screenshot under the checkpoint's 4096^2 bound, or the 1280px
/// fallback at patch 14) is resized to the capacity's area instead, and
/// floored as HF's `smart_resize` does when rounding lands above it
/// (2064x2032 -> 2016x2048, not 2080x2048).
pub(super) fn qwen_fit(
    vcfg: &VisionConfig,
    orig_h: u32,
    orig_w: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let ps = vcfg.patch_size.max(1);
    let grid_unit = (ps * vcfg.spatial_merge_size) as u32;
    let p_max = crate::layers::vision_encoder::derive_max_patches(max_pixels, ps).0;
    let patches = |(h, w): (u32, u32)| (h as usize / ps) * (w as usize / ps);
    let base = target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels);
    if patches(base) <= p_max {
        return base;
    }
    let bound = max_pixels
        .filter(|&p| p > 0)
        .unwrap_or(usize::MAX)
        .min(p_max.saturating_mul(ps * ps));
    let mut fit = snapped_target(orig_h, orig_w, grid_unit, Some(bound), f32::round);
    if fit.0 as usize * fit.1 as usize > bound {
        fit = snapped_target(orig_h, orig_w, grid_unit, Some(bound), f32::floor);
    }
    // A strip's short side cannot go below one grid unit, so the area can
    // still exceed the bound: shorten the long side until the grid fits.
    let unit = grid_unit as usize;
    let unit_patches = (unit / ps) * (unit / ps);
    let max_units = (p_max / unit_patches.max(1)).max(1);
    let (uh, uw) = (fit.0 as usize / unit, fit.1 as usize / unit);
    if uh * uw > max_units {
        let short = uh.min(uw);
        let long = (max_units / short).max(1);
        let (uh, uw) = if uh <= uw {
            (short, long)
        } else {
            (long, short)
        };
        fit = ((uh * unit) as u32, (uw * unit) as u32);
    }
    fit
}
