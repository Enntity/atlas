// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit packed-linear accumulation and reference-padding contracts.
use anyhow::{Context, Result, ensure};

/// One source-bound cooperative launch geometry. Numerical qualification is
/// specific to the supplied geometry and checkpoint, not inferred from shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exl3CooperativeProfile {
    pub tile_k: u32,
    pub tile_n: u32,
    pub slices: u32,
    pub half_partial: bool,
}

/// Profiles for the installed operator's padded-row buckets 2, 4, 8, 16.
/// Larger rows share bucket 16, as the source-bound autotune hash specifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exl3LinearNumerics {
    pub logical_n: usize,
    pub logical_offset: usize,
    pub bits: u8,
    pub reference_min_rows: usize,
    pub reference_pad: bool,
    /// None declares that bucket unavailable; it never selects a fallback.
    pub row_profiles: [Option<Exl3CooperativeProfile>; 4],
    /// Speculative-verify alternative for the padded 5..8-row bucket
    /// (`ATLAS_EXL3_VERIFY_PROFILES=1`), used only inside a [`verify_scope`]:
    /// prefill chunks and serial decode keep `row_profiles`. None: no
    /// alternative.
    pub verify_rows_5_8: Option<Exl3CooperativeProfile>,
}

thread_local! {
    static VERIFY_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the current thread's forward as a speculative verify until dropped,
/// so `verify_rows_5_8` profiles apply (launches recorded into a graph keep
/// the choice made at capture time).
pub struct VerifyScope {
    previous: bool,
}

pub fn verify_scope() -> VerifyScope {
    VerifyScope {
        previous: VERIFY_ACTIVE.with(|v| v.replace(true)),
    }
}

impl Drop for VerifyScope {
    fn drop(&mut self) {
        let previous = self.previous;
        VERIFY_ACTIVE.with(|v| v.set(previous));
    }
}

fn verify_active() -> bool {
    VERIFY_ACTIVE.with(|v| v.get())
}

/// Whether the current thread is inside a [`verify_scope`] (a speculative
/// verify forward). Callers use it to keep verify-only numerics choices out
/// of prefill and serial decode.
pub fn in_verify_scope() -> bool {
    verify_active()
}

/// Profile slots: 0..4 are `row_profiles`, 4 is `verify_rows_5_8`.
pub const EXL3_PROFILE_SLOTS: usize = 5;

impl Exl3LinearNumerics {
    pub(super) fn validate(self, k: usize, physical_n: usize, capacity: usize) -> Result<()> {
        ensure!(
            (1..=65535).contains(&capacity),
            "EXL3 capacity must fit CUDA grid.y"
        );
        ensure!(
            (1..=8).contains(&self.bits),
            "EXL3 numerical profile bits must be explicit 1..=8"
        );
        ensure!(
            (1..=128).contains(&self.reference_min_rows),
            "EXL3 reference minimum rows must be explicit 1..=128"
        );
        ensure!(
            self.logical_n > 0 && self.logical_n <= u32::MAX as usize,
            "EXL3 logical width must be positive and fit the kernel ABI"
        );
        ensure!(
            self.logical_offset.is_multiple_of(128),
            "EXL3 logical component offset must align to H128"
        );
        ensure!(
            self.logical_offset
                .checked_add(physical_n)
                .is_some_and(|end| end <= self.logical_n),
            "EXL3 physical component exceeds logical reference width"
        );
        for profile in self
            .row_profiles
            .into_iter()
            .chain([self.verify_rows_5_8])
            .flatten()
        {
            ensure!(
                matches!(
                    (profile.tile_k, profile.tile_n),
                    (16, 128) | (32, 128) | (32, 256) | (16, 512)
                ),
                "unsupported EXL3 cooperative tile geometry"
            );
            ensure!(
                (1..=48).contains(&profile.slices) && !profile.half_partial,
                "EXL3 generic F32 operator requires explicit 1..=48 slices and unrounded partials"
            );
            ensure!(
                k.is_multiple_of(profile.tile_k as usize)
                    && self.logical_n.is_multiple_of(profile.tile_n as usize),
                "EXL3 reference dimensions do not match cooperative tile geometry"
            );
            ensure!(
                (k / profile.tile_k as usize)
                    .checked_mul(self.logical_n / profile.tile_n as usize)
                    .is_some_and(|tiles| tiles <= u32::MAX as usize),
                "EXL3 flattened logical tile count exceeds the kernel ABI"
            );
        }
        for rows in [1, 3, 5, 9, capacity] {
            if rows <= capacity {
                self.profile_for_rows(rows)?;
            }
        }
        Ok(())
    }

    pub fn profile_for_rows(self, rows: usize) -> Result<Exl3CooperativeProfile> {
        self.profile_at(self.profile_slot(rows)?)
            .context("EXL3 selected row bucket has no explicit numerical profile")
    }

    /// The profile slot for `rows`: the padded-row bucket, or slot 4 (the
    /// verify alternative) for the 5..8-row bucket inside a verify scope.
    pub fn profile_slot(self, rows: usize) -> Result<usize> {
        let bucket = self.bucket_for_rows(rows)?;
        Ok(
            if bucket == 2 && self.verify_rows_5_8.is_some() && verify_active() {
                4
            } else {
                bucket
            },
        )
    }

    pub fn profile_at(self, slot: usize) -> Option<Exl3CooperativeProfile> {
        match slot {
            0..4 => self.row_profiles[slot],
            4 => self.verify_rows_5_8,
            _ => None,
        }
    }

    /// Index into `row_profiles` for `rows` (after the reference padding).
    pub fn bucket_for_rows(self, rows: usize) -> Result<usize> {
        ensure!(
            rows > 0 && (1..=128).contains(&self.reference_min_rows),
            "EXL3 profile lookup requires positive rows and explicit minimum"
        );
        let padded = if self.reference_pad {
            rows.max(self.reference_min_rows)
        } else {
            rows
        };
        Ok(match padded {
            1..=2 => 0,
            3..=4 => 1,
            5..=8 => 2,
            _ => 3,
        })
    }

    pub(super) fn kernel_args(self, rows: usize) -> Result<[u32; 6]> {
        let profile = self.profile_for_rows(rows)?;
        Ok([
            self.logical_n as u32,
            self.logical_offset as u32,
            profile.tile_k,
            profile.tile_n,
            profile.slices,
            u32::from(profile.half_partial),
        ])
    }
}
