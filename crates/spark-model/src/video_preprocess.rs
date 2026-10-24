// SPDX-License-Identifier: AGPL-3.0-only

//! Video → patch tensor, the temporal sibling of [`crate::vision_preprocess`].
//!
//! # What a video is, to this encoder
//!
//! GLM's visual tower has no temporal attention between groups. Frames fuse
//! inside a patch: the flattened patch dimension is
//! `C × temporal_patch_size × patch²`, so a patch spans `tp` consecutive
//! frames. A still image fills that axis by replicating itself `tp` times.
//!
//! A video of `n` sampled frames becomes `grid_t = ceil(n / tp)` temporal
//! groups; each model repeats its last sampled frame into an odd group.
//! Each group is a full `grid_h × grid_w` patch plane. The
//! native encoder runs each group as its own attention sequence, and Qwen's
//! prompt expansion gives every group its own timestamped pad range.
//!
//! # Container support
//!
//! Two backends, chosen by MAGIC BYTES rather than the declared MIME:
//!
//! - **GIF** decodes in-process, pure Rust, always available, no dependency.
//! - **Everything else** (MP4/MOV, WebM/Matroska, AVI — H.264, H.265, VP9,
//!   AV1) goes to ffmpeg as a subprocess, which is opt-in.
//!
//! Sniffing the bytes rather than trusting the label means a client that
//! sends an mp4 as `video/gif`, or as `application/octet-stream`, still gets
//! the right decoder. See `video_decode_ffmpeg` for why a subprocess rather
//! than a linked decoder, and issue #515.

use anyhow::{Context, Result, ensure};
use atlas_core::config::VisionConfig;
use image::RgbImage;

use crate::vision_preprocess::{
    decode_data_uri_bytes, normalize_channel, patch_coordinates, qwen_fit,
};

/// Frames per second to sample at, when the caller has no better idea.
/// Matches the `fps: 2` every Qwen3-VL `video_processor` block declares.
pub const DEFAULT_FPS: f32 = 2.0;

/// Sampling floor and ceiling, also from the checkpoints' own video processor
/// (`min_frames: 4`, `max_frames: 768`). The floor matters more than it looks:
/// with `temporal_patch_size = 2`, fewer than 2 frames cannot fill a single
/// temporal group, and a 1-frame "video" would silently become a still.
pub const DEFAULT_MIN_FRAMES: usize = 4;
pub const DEFAULT_MAX_FRAMES: usize = 768;

/// A decoded, ready-to-encode video.
pub struct PreprocessedVideo {
    /// One entry per temporal group, each shaped exactly like a preprocessed
    /// still: `[grid_h * grid_w, C * tp * patch * patch]`.
    pub groups: Vec<Vec<f32>>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// One timestamp per temporal group, in source-video seconds. The prompt
    /// expanders for Qwen and GLM retain this metadata in encoder order.
    pub timestamps: Vec<f32>,
}

/// Summarised rather than derived: the payload is megabytes of f32 and a
/// derived `Debug` would dump all of it into any test failure or log line
/// that happens to format one.
impl std::fmt::Debug for PreprocessedVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreprocessedVideo {{ grid_t: {}, grid_h: {}, grid_w: {}, groups: {} x {} f32, timestamps: {} }}",
            self.grid_t,
            self.grid_h,
            self.grid_w,
            self.groups.len(),
            self.groups.first().map_or(0, Vec::len),
            self.timestamps.len()
        )
    }
}

impl PreprocessedVideo {
    /// Merged tokens this video contributes: one per `merge × merge` block of
    /// patches, per temporal group.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.grid_t * (self.grid_h / sms) * (self.grid_w / sms)
    }
}

/// Pick which frame indices to keep so the clip plays at `fps`.
///
/// `native_fps` is what the container says it runs at. Sampling is by
/// NEAREST-INDEX over a uniform grid rather than by dropping every Nth frame:
/// the latter quantises badly when the ratio is not an integer (a 30fps clip
/// sampled at 2fps by "keep every 15th" is fine, at 2.5fps it is not).
///
/// The result is clamped into `[min_frames, max_frames]` and then to a
/// multiple of `temporal_patch_size`, because a partial group cannot be
/// encoded. Returns indices into the decoded frame list.
pub fn sample_indices(
    n_frames: usize,
    native_fps: f32,
    target_fps: f32,
    min_frames: usize,
    max_frames: usize,
    temporal_patch_size: usize,
) -> Vec<usize> {
    if n_frames == 0 {
        return Vec::new();
    }
    let tp = temporal_patch_size.max(1);
    let native_fps = if native_fps.is_finite() && native_fps > 0.0 {
        native_fps
    } else {
        DEFAULT_FPS
    };
    let target_fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        DEFAULT_FPS
    };

    let duration = n_frames as f32 / native_fps;
    let wanted = (duration * target_fps).round().max(1.0) as usize;

    // Clamp to the checkpoint's band, but never ask for more frames than
    // exist — upsampling a short clip by repeating frames would inflate the
    // token count with no new information.
    let wanted = wanted
        .clamp(min_frames.max(1), max_frames.max(1))
        .min(n_frames);

    // Round DOWN to a whole number of temporal groups; a partial group has no
    // representation. Never below one group, or there is nothing to encode.
    let wanted = (wanted / tp).max(1) * tp;
    let wanted = wanted.min((n_frames / tp).max(1) * tp).min(n_frames);

    if wanted >= n_frames {
        return (0..n_frames).collect();
    }
    // Uniform positions across the clip, nearest index, deduplicated in order.
    let mut out = Vec::with_capacity(wanted);
    for i in 0..wanted {
        let pos = if wanted == 1 {
            0.0
        } else {
            (i as f32) * ((n_frames - 1) as f32) / ((wanted - 1) as f32)
        };
        out.push((pos.round() as usize).min(n_frames - 1));
    }
    out
}

/// Legacy grouping helper: drop a trailing partial temporal group.
/// The model-specific preprocessors pad their final sampled group instead.
///
/// The decoder ([`crate::video_sample`]) already chose which source frames to
/// return, at the operator's rate and spanning the clip. A partial group
/// cannot be encoded as video — the patch tensor's temporal axis is exactly
/// `temporal_patch_size` frames — so a remainder is dropped here rather than
/// padded. Frames are never duplicated to fill a group, which would charge
/// the model for motion it never saw.
pub fn group_indices(n_frames: usize, temporal_patch_size: usize) -> Vec<usize> {
    let tp = temporal_patch_size.max(1);
    let whole = n_frames / tp * tp;
    // A clip with fewer frames than one group keeps them all: refusing it is
    // the caller's job (the `temporal group` error in `preprocess_video`),
    // and reporting nothing here would hide why it was refused.
    if whole == 0 {
        return (0..n_frames).collect();
    }
    (0..whole).collect()
}

/// Decode every frame of a container, choosing a backend by what the bytes
/// actually are.
///
/// Returns the frames, with one SOURCE time per frame. Both backends now go
/// through the same coverage rule: the plan spans the whole clip at no more
/// than `target_fps`, and the times it returns say where each returned frame
/// actually sits. GIF decodes in-process (pure Rust, no dependency); every
/// other container goes to ffmpeg.
///
/// Dispatch is on MAGIC BYTES, not the declared MIME. A client that labels an
/// mp4 `video/gif`, or sends `application/octet-stream`, still gets the right
/// decoder; and a GIF mislabelled as mp4 does not needlessly spawn a process.
pub fn decode_frames(
    data_uri: &str,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<(Vec<RgbImage>, Vec<f32>)> {
    let (mime, bytes) = decode_data_uri_bytes(data_uri)?;
    ensure!(!bytes.is_empty(), "the video payload is empty");

    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return decode_gif(&bytes, target_fps, ffmpeg);
    }

    // Everything else goes to the subprocess backend. If it is disabled the
    // error names the flag AND the container, so the operator is not left
    // guessing which of the two problems they have.
    let kind = sniff_container(&bytes, &mime);
    crate::video_decode_ffmpeg::decode_with_plan(&bytes, target_fps, ffmpeg)
        .with_context(|| format!("decoding {kind}"))
}

/// Best-effort container name for error messages. Cosmetic only — nothing
/// branches on it — so an unrecognized blob is described as such rather than
/// guessed at.
fn sniff_container(bytes: &[u8], mime: &str) -> String {
    let by_magic = if bytes.len() > 12 && &bytes[4..8] == b"ftyp" {
        Some("an MP4/MOV container")
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("a Matroska/WebM container")
    } else if bytes.starts_with(b"RIFF") {
        Some("an AVI container")
    } else {
        None
    };
    match (by_magic, mime.is_empty()) {
        (Some(k), _) => k.to_string(),
        (None, false) => format!("a {mime} payload"),
        (None, true) => "an unrecognized container".to_string(),
    }
}

mod gif;
mod gif_access;
mod glm;
mod patchify;
use gif::decode_gif;
use patchify::patchify_group;

/// Full pipeline: a base64 `data:` URI holding an animated container becomes
/// temporal groups of patches.
pub fn preprocess_video(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<PreprocessedVideo> {
    ensure!(
        vcfg.patch_size > 0 && vcfg.spatial_merge_size > 0 && vcfg.temporal_patch_size > 0,
        "vision_config geometry is invalid (patch/merge/temporal size is 0)"
    );
    if vcfg.is_glm5_next {
        return glm::preprocess_glm_video(data_uri, vcfg, max_pixels, target_fps, ffmpeg);
    }
    let (frames, frame_times) = decode_frames(data_uri, target_fps, ffmpeg)?;
    let tp = vcfg.temporal_patch_size;
    ensure!(!frames.is_empty(), "frame sampling selected no frames");

    // Qwen keeps the coverage decoder's source frames and repeats the last
    // frame to complete an odd temporal group. GLM uses its reference sampler.
    let first = &frames[0];
    let (th, tw) = qwen_fit(
        vcfg,
        first.height(),
        first.width(),
        max_pixels,
        Some(frames.len()),
    )?;
    let ps = vcfg.patch_size;
    let grid_h = th as usize / ps;
    let grid_w = tw as usize / ps;
    let grid_t = frames.len().div_ceil(tp);
    if let Some(budget) = ffmpeg.max_total_decoded_pixels {
        let volume = grid_t
            .saturating_mul(tp)
            .saturating_mul(th as usize)
            .saturating_mul(tw as usize);
        ensure!(
            volume <= budget,
            "video grid needs {volume} pixels, above the {budget}-pixel volume bound; \
             reduce frames or use a less extreme aspect ratio"
        );
    }
    let groups = (0..grid_t)
        .map(|g| {
            let resized: Vec<RgbImage> = (0..tp)
                .map(|k| {
                    let frame = &frames[(g * tp + k).min(frames.len() - 1)];
                    image::imageops::resize(frame, tw, th, image::imageops::FilterType::CatmullRom)
                })
                .collect();
            patchify_group(vcfg, &resized, grid_h, grid_w)
        })
        .collect();
    Ok(PreprocessedVideo {
        groups,
        grid_t,
        grid_h,
        grid_w,
        // Qwen uses the average source time of each group's end frames.
        timestamps: (0..grid_t)
            .map(|g| {
                let first = frame_times[g * tp];
                let last = frame_times[(g * tp + tp - 1).min(frames.len() - 1)];
                (first + last) / 2.0
            })
            .collect(),
    })
}

#[cfg(test)]
#[path = "video_preprocess_tests.rs"]
mod tests;
