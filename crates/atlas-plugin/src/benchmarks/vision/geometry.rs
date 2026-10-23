// SPDX-License-Identifier: AGPL-3.0-only

//! Legacy unbounded grid arithmetic retained for video fixtures.
//! Still-image fidelity uses `image_geometry` with the resolved checkpoint
//! processor policy; it must not assume these bound-free counts.

/// Snap a side to the patch grid, the way the preprocessor does when no
/// downscale is required: `round(side / grid) * grid`.
///
/// `f32::round` is half-away-from-zero, which is what makes 336 → 352 rather
/// than 320. Pinned by `the_rounding_mode_is_pinned`; if
/// the engine ever switches to half-even this model must move with it or every
/// expectation silently drifts by one grid unit.
pub fn snap(side: u32, grid: u32) -> u32 {
    (((side as f32) / (grid as f32)).round() as u32).max(1) * grid
}

/// Merged vision tokens for a `w × h` image at the given geometry.
///
/// Both sides snap to `patch × merge` first, then each `patch × patch` square
/// is one patch and each `merge × merge` block of patches becomes one token.
/// For Qwen3-VL geometry (patch 16, merge 2) a 448×448 image is 784 patches
/// and **196 tokens** — the figure measured against a live server on
/// 2026-08-14, which anchors this whole model.
pub fn expected_vision_tokens(w: u32, h: u32, patch: u32, merge: u32) -> u32 {
    let grid = patch * merge;
    let sw = snap(w, grid);
    let sh = snap(h, grid);
    (sw / patch) * (sh / patch) / (merge * merge)
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
