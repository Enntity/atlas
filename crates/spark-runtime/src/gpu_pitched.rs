// SPDX-License-Identifier: AGPL-3.0-only

//! Host-staging geometry shared by every [`crate::gpu::GpuBackend`]: the
//! shape of a pitched host↔device copy and the alignment of page-locked
//! staging. Kept out of `gpu.rs` (500-LoC cap); re-exported from there.

/// Alignment of the heap stand-in for page-locked memory (mock / backends
/// without pinning). Real pinned memory is page-aligned, and callers rely on
/// it: O_DIRECT I/O straight into a staging buffer needs a 4 KiB boundary.
pub const HOST_PINNED_ALIGN: usize = 4096;

/// Geometry of a pitched (2-D) copy between host staging and a device pool:
/// `height` rows of `width` bytes, rows `host_pitch` apart on the host and
/// `dev_pitch` apart on the device (each pitch ≥ `width`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pitched {
    pub host_pitch: usize,
    pub dev_pitch: usize,
    pub width: usize,
    pub height: usize,
}

impl Pitched {
    /// Host bytes the copy spans: the last row ends `width` into its pitch.
    pub fn host_span(self) -> usize {
        match self.height {
            0 => 0,
            h => (h - 1) * self.host_pitch + self.width,
        }
    }
}
