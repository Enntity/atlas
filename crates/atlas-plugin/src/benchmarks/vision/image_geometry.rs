// SPDX-License-Identifier: AGPL-3.0-only

//! Independent still-image reference driven by the served processor policy.
//! This module must not call the engine's resize implementation.

use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct ImageProcessor {
    pub algorithm: String,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub encoder_capacity_pixels: u64,
}

impl ImageProcessor {
    pub fn tokens(&self, w: u32, h: u32) -> Result<usize> {
        ensure!(
            self.algorithm == "qwen-smart-resize-v1",
            "unsupported image processor {}",
            self.algorithm
        );
        let grid = self
            .patch_size
            .checked_mul(self.spatial_merge_size)
            .filter(|&g| g > 0)
            .ok_or_else(|| anyhow::anyhow!("invalid processor grid"))?;
        ensure!(
            w > 0 && h > 0 && u64::from(w.max(h)) <= 200 * u64::from(w.min(h)),
            "invalid image dimensions"
        );
        let unit = u64::from(grid) * u64::from(grid);
        let max = self.max_pixels.min(self.encoder_capacity_pixels);
        ensure!(
            max >= unit,
            "processor capacity is smaller than one merged patch"
        );
        let min = self.min_pixels.max(unit).min(max);
        // Reference arithmetic follows the declared HF policy, independently
        // of the production preprocessor and its observed row counts.
        let g = f64::from(grid);
        let width = f64::from(w);
        let height = f64::from(h);
        let mut cols = (width / g).round_ties_even() as u64;
        let mut rows = (height / g).round_ties_even() as u64;
        let rounded_area = cols * rows * unit;
        if rounded_area > max {
            let scale = (width * height / max as f64).sqrt();
            cols = ((width / scale / g).floor() as u64).max(1);
            rows = ((height / scale / g).floor() as u64).max(1);
        } else if rounded_area < min {
            let scale = (min as f64 / (width * height)).sqrt();
            cols = (width * scale / g).ceil() as u64;
            rows = (height * scale / g).ceil() as u64;
        }
        // A narrow strip may exceed encoder capacity after its short edge is
        // rounded up to one grid unit. Shorten the long edge to fit.
        let capacity_units = self.encoder_capacity_pixels / unit;
        if cols * rows > capacity_units {
            if cols <= rows {
                cols = cols.max(1).min(capacity_units);
                rows = (capacity_units / cols).max(1);
            } else {
                rows = rows.max(1).min(capacity_units);
                cols = (capacity_units / rows).max(1);
            }
        }
        Ok(usize::try_from(cols * rows)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> ImageProcessor {
        ImageProcessor {
            algorithm: "qwen-smart-resize-v1".into(),
            patch_size: 16,
            spatial_merge_size: 2,
            min_pixels: 65536,
            max_pixels: 16777216,
            encoder_capacity_pixels: 4194304,
        }
    }

    #[test]
    fn checkpoint_bounds_and_python_rounding_define_the_ladder() {
        let p = checkpoint();
        for (w, h, count) in [
            (224, 224, 64),
            (336, 336, 100),
            (8, 8, 64),
            (64, 2048, 128),
            (1600, 900, 1400),
            (4096, 4096, 4096),
        ] {
            assert_eq!(p.tokens(w, h).unwrap(), count, "{w}x{h}");
            assert_eq!(p.tokens(h, w).unwrap(), count, "transpose {w}x{h}");
        }
        assert_eq!(
            p.tokens(336, 336).unwrap() - p.tokens(224, 224).unwrap(),
            36
        );
        assert_ne!(
            p.tokens(1600, 900).unwrap(),
            920,
            "must catch the old 1280-side clamp"
        );
    }

    #[test]
    fn missing_or_unsupported_policy_cannot_silently_pass() {
        assert!(serde_json::from_str::<ImageProcessor>("{}").is_err());
        let mut p = checkpoint();
        p.algorithm = "unknown".into();
        assert!(p.tokens(224, 224).is_err());
        p.algorithm = "qwen-smart-resize-v1".into();
        p.patch_size = 0;
        assert!(p.tokens(224, 224).is_err());
    }
}
