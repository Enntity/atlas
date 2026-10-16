// SPDX-License-Identifier: AGPL-3.0-only

//! F4 diagnostic (`ATLAS_GLM_EXL3_DFLASH_EXPERT_STATS=1`): unique routed
//! experts per multi-row MoE launch (the verify's expert overlap). Reads the
//! routing ids back with a stream sync, so it perturbs timing: never enable
//! it in a timed run.

use std::sync::atomic::{AtomicU64, Ordering};

static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static UNIQUE: AtomicU64 = AtomicU64::new(0);
static ROUTES: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_EXL3_DFLASH_EXPERT_STATS").as_deref() == Ok("1"))
}

/// Count the distinct valid expert ids among one launch's routes.
pub fn record(ids: &[i32], experts: usize) {
    let mut seen = vec![false; experts];
    let mut unique = 0u64;
    for &id in ids {
        if id >= 0 && (id as usize) < experts && !seen[id as usize] {
            seen[id as usize] = true;
            unique += 1;
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    UNIQUE.fetch_add(unique, Ordering::Relaxed);
    ROUTES.fetch_add(ids.len() as u64, Ordering::Relaxed);
}

/// `(launches, unique experts, routes)` since the last take.
pub fn take() -> (u64, u64, u64) {
    (
        LAUNCHES.swap(0, Ordering::Relaxed),
        UNIQUE.swap(0, Ordering::Relaxed),
        ROUTES.swap(0, Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn counts_distinct_valid_ids() {
        super::take();
        super::record(&[1, 2, 2, -1, 5, 1, 300], 288);
        assert_eq!(super::take(), (1, 3, 7));
    }
}
