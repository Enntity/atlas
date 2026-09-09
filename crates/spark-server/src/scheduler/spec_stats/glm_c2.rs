// SPDX-License-Identifier: AGPL-3.0-only
//! Cumulative committed selected transactions. No tokens, timers or GPU work.
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct GlmC2Stats {
    pair_commits: AtomicU64,
    // Physical owner order: index = accepted_slot0 * 5 + accepted_slot1.
    pair_accept: [AtomicU64; 25],
    serial_commits: AtomicU64,
    serial_accept: [AtomicU64; 5],
    bootstrap_commits: AtomicU64,
    // Cohort-size order: index0=C3, index1=C4. Acceptance bins count owners,
    // not transactions; neither field is a Pair/Serial or emitted-token count.
    owner_commits: [AtomicU64; 2],
    owner_accept: [[AtomicU64; 5]; 2],
}

#[derive(Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub pair_commits: u64,
    pub pair_accept: [u64; 25],
    pub serial_commits: u64,
    pub serial_accept: [u64; 5],
    pub bootstrap_commits: u64,
    pub owner_commits: [u64; 2],
    pub owner_accept: [[u64; 5]; 2],
}

impl GlmC2Stats {
    pub fn owners_committed(&self, accepted: &[usize]) {
        // Called only after the typed3/4 cohort and every count passed commit.
        let group = accepted.len() - 3;
        for &count in accepted {
            self.owner_accept[group][count].fetch_add(1, Ordering::Relaxed);
        }
        self.owner_commits[group].fetch_add(1, Ordering::Relaxed);
    }
    pub fn pair_committed(&self, accepted: [usize; 2]) {
        self.pair_accept[accepted[0] * 5 + accepted[1]].fetch_add(1, Ordering::Relaxed);
        self.pair_commits.fetch_add(1, Ordering::Relaxed);
    }
    pub fn serial_committed(&self, accepted: usize) {
        self.serial_accept[accepted].fetch_add(1, Ordering::Relaxed);
        self.serial_commits.fetch_add(1, Ordering::Relaxed);
    }
    pub fn bootstrap_committed(&self) {
        self.bootstrap_commits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            pair_commits: self.pair_commits.load(Ordering::Relaxed),
            pair_accept: std::array::from_fn(|i| self.pair_accept[i].load(Ordering::Relaxed)),
            serial_commits: self.serial_commits.load(Ordering::Relaxed),
            serial_accept: std::array::from_fn(|i| self.serial_accept[i].load(Ordering::Relaxed)),
            bootstrap_commits: self.bootstrap_commits.load(Ordering::Relaxed),
            owner_commits: std::array::from_fn(|i| self.owner_commits[i].load(Ordering::Relaxed)),
            owner_accept: std::array::from_fn(|i| {
                std::array::from_fn(|count| self.owner_accept[i][count].load(Ordering::Relaxed))
            }),
        }
    }

    /// Called only at an idle-wave boundary or immediately before the selected
    /// nonreturning shutdown. Cumulative values allow exact wave subtraction.
    /// Commit counts are NOT emitted-token counts or healthy-release authority.
    pub fn summary(&self, boundary: &'static str) {
        let s = self.snapshot();
        tracing::info!(
            boundary,
            pair_commits = s.pair_commits,
            pair_accept_slot0_major = ?s.pair_accept,
            serial_commits = s.serial_commits,
            serial_accept = ?s.serial_accept,
            bootstrap_commits = s.bootstrap_commits,
            owner_commits_c3_c4 = ?s.owner_commits,
            owner_accept_c3_c4 = ?s.owner_accept,
            "GLM C2 committed summary (cumulative; acceptance counts exclude seed)"
        );
    }
}
