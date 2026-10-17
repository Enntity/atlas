// SPDX-License-Identifier: AGPL-3.0-only

//! Size an image from its header before decoding it.
//!
//! A few-KB PNG may declare 16384x16384; decoded and patched it is hundreds of
//! MiB of f32. Admission must know an item's vision tokens BEFORE that, so it
//! can refuse a request over its limits without allocating anything, and
//! charge the in-flight budget for exactly what the decode will hold.
//! [`probe_image`] reads only the header (format and dimensions) and fixes the
//! geometry with the same function the decode uses (`still_target`);
//! [`ProbedImage::preprocess`] then decodes and checks the two agree.

use super::*;

/// An image whose geometry is known and whose pixels are not yet decoded.
pub struct ProbedImage {
    pub(super) bytes: Vec<u8>,
    pub(super) fmt: image::ImageFormat,
    pub(super) max_pixels: Option<usize>,
    /// Patch grid the decode will produce (sides may swap under EXIF
    /// orientation; their product cannot).
    pub grid_h: usize,
    pub grid_w: usize,
}

impl std::fmt::Debug for ProbedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ProbedImage {{ {:?}, {} bytes, grid {}x{} }}",
            self.fmt,
            self.bytes.len(),
            self.grid_h,
            self.grid_w
        )
    }
}

impl ProbedImage {
    /// Vision tokens (merged rows) the image will occupy.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        image_pad_count(self.grid_h, self.grid_w, spatial_merge_size)
    }

    /// Patches the encoder will see.
    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Decode and patch: `(pixels, grid_h, grid_w)` as
    /// [`preprocess_image_with_max_pixels`] returns them.
    pub fn preprocess(self, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
        preprocess_probed(self, vcfg)
    }
}

/// Read `data_uri`'s header and fix the geometry its decode will have.
/// Allocates only the compressed bytes.
pub fn probe_image(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<ProbedImage> {
    // Before anything divides by them.
    validate_geometry(vcfg)?;
    let (_mime, bytes) = decode_data_uri_bytes(data_uri)?;
    let fmt = sniff_format(&bytes)?;
    let (w, h) = limited_reader(&bytes, fmt)
        .into_dimensions()
        .context("image decode failed")?;
    ensure!(
        w > 0 && h > 0 && w <= DECODE_MAX_SIDE && h <= DECODE_MAX_SIDE,
        "image dimensions {w}x{h} are outside 1..={DECODE_MAX_SIDE}"
    );
    let (th, tw) = still_target(vcfg, h, w, max_pixels)?;
    Ok(ProbedImage {
        bytes,
        fmt,
        max_pixels,
        grid_h: th as usize / vcfg.patch_size,
        grid_w: tw as usize / vcfg.patch_size,
    })
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
