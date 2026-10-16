// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-only acceptance scoring (F4 phase 1).
//!
//! Each decode step proposes `k` drafts after the target's serial token; the
//! drafts are never verified. They are scored here against the tokens the
//! serial decode subsequently produces: a draft made with its bonus token at
//! absolute position `base` predicts positions `base + 1 ..= base + k`.
//! Greedy block verification would accept the longest matching prefix, so
//! the per-step accepted length is that prefix length and the classic
//! acceptance length is `tau = 1 + accepted` (the bonus is always emitted).
//!
//! Pure host-side bookkeeping: identical inputs give identical statistics on
//! every rank, with no device or communicator access.

use std::collections::VecDeque;

/// Token that never matches (pads a short proposal to `k` positions).
const NO_TOKEN: u32 = u32::MAX;

#[derive(Debug, Clone)]
struct Pending {
    /// Absolute position of the bonus token the drafts follow.
    base: usize,
    drafts: Vec<u32>,
    /// Positions resolved so far (`base + 1 ..= base + resolved`).
    resolved: usize,
    /// Leading drafts matched so far.
    accepted: usize,
    /// No mismatch seen yet.
    open: bool,
    hits: Vec<bool>,
}

/// Aggregate acceptance statistics over fully resolved draft steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptStats {
    k: usize,
    /// Draft steps whose `k` positions all resolved.
    pub steps: u64,
    /// `hist[a]` = steps with exactly `a` accepted drafts (`a` in `0..=k`).
    pub hist: Vec<u64>,
    /// `hits[j]` = steps whose draft `j + 1` equals the actual token at that
    /// position, regardless of earlier drafts (unconditional).
    pub hits: Vec<u64>,
    /// Draft steps dropped because the sequence ended before they resolved.
    pub truncated: u64,
    /// Draft steps proposed (resolved, truncated, or still pending).
    pub proposed: u64,
}

impl AcceptStats {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            steps: 0,
            hist: vec![0; k + 1],
            hits: vec![0; k],
            truncated: 0,
            proposed: 0,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Fold another sequence's statistics in (same `k` only).
    pub fn merge(&mut self, other: &AcceptStats) {
        if other.k != self.k {
            return;
        }
        self.steps += other.steps;
        self.truncated += other.truncated;
        self.proposed += other.proposed;
        for (a, b) in self.hist.iter_mut().zip(&other.hist) {
            *a += b;
        }
        for (a, b) in self.hits.iter_mut().zip(&other.hits) {
            *a += b;
        }
    }

    /// Mean accepted drafts per step (0..=k).
    pub fn mean_accepted(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        let sum: u64 = self
            .hist
            .iter()
            .enumerate()
            .map(|(a, &n)| a as u64 * n)
            .sum();
        sum as f64 / self.steps as f64
    }

    /// Mean acceptance length including the bonus token.
    pub fn tau(&self) -> f64 {
        1.0 + self.mean_accepted()
    }

    /// `prefix[j]` = P(accepted >= j + 1): draft `j + 1` would be accepted
    /// under greedy prefix verification.
    pub fn prefix_curve(&self) -> Vec<f64> {
        let steps = self.steps.max(1) as f64;
        (1..=self.k)
            .map(|j| self.hist[j..].iter().sum::<u64>() as f64 / steps)
            .collect()
    }

    /// `cond[j]` = P(accepted >= j + 1 | accepted >= j).
    pub fn conditional_curve(&self) -> Vec<f64> {
        let prefix = self.prefix_curve();
        let mut prev = if self.steps == 0 { 0.0 } else { 1.0 };
        prefix
            .iter()
            .map(|&p| {
                let c = if prev > 0.0 { p / prev } else { 0.0 };
                prev = p;
                c
            })
            .collect()
    }

    /// `raw[j]` = P(draft `j + 1` equals the actual token), unconditional.
    pub fn raw_curve(&self) -> Vec<f64> {
        let steps = self.steps.max(1) as f64;
        self.hits.iter().map(|&n| n as f64 / steps).collect()
    }

    /// Body of a `DFLASH_ACCEPT` log line (everything after the prefix).
    pub fn summary(&self) -> String {
        fn fmt(v: &[f64]) -> String {
            let parts: Vec<String> = v.iter().map(|x| format!("{x:.3}")).collect();
            format!("[{}]", parts.join(","))
        }
        format!(
            "k={} steps={} tau={:.3} mean_acc={:.3} prefix={} cond={} raw={} hist={:?} truncated={} proposed={}",
            self.k,
            self.steps,
            self.tau(),
            self.mean_accepted(),
            fmt(&self.prefix_curve()),
            fmt(&self.conditional_curve()),
            fmt(&self.raw_curve()),
            self.hist,
            self.truncated,
            self.proposed,
        )
    }
}

/// Per-sequence scorer: drafts wait here until the serial decode produces
/// the tokens at the positions they predicted.
#[derive(Debug, Clone)]
pub struct DraftScorer {
    k: usize,
    pending: VecDeque<Pending>,
    stats: AcceptStats,
}

impl DraftScorer {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "draft scorer needs k >= 1");
        Self {
            k,
            pending: VecDeque::new(),
            stats: AcceptStats::new(k),
        }
    }

    pub fn stats(&self) -> &AcceptStats {
        &self.stats
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// The serial decode produced `token` at absolute position `pos`.
    /// Resolves every pending draft that predicted `pos`.
    pub fn observe(&mut self, pos: usize, token: u32) {
        for p in self.pending.iter_mut() {
            if pos <= p.base || pos > p.base + self.k {
                continue;
            }
            let j = pos - p.base; // 1-based draft depth
            if j != p.resolved + 1 {
                // A gap (position skipped): treat the draft as unresolvable.
                p.open = false;
                p.resolved = self.k + 1;
                continue;
            }
            p.resolved = j;
            let hit = p.drafts[j - 1] == token;
            p.hits[j - 1] = hit;
            if p.open && hit {
                p.accepted += 1;
            } else {
                p.open = false;
            }
        }
        while let Some(front) = self.pending.front() {
            if front.resolved < self.k {
                break;
            }
            let done = self.pending.pop_front().expect("front exists");
            if done.resolved > self.k {
                self.stats.truncated += 1;
                continue;
            }
            self.stats.steps += 1;
            self.stats.hist[done.accepted] += 1;
            for (count, hit) in self.stats.hits.iter_mut().zip(&done.hits) {
                *count += u64::from(*hit);
            }
        }
    }

    /// Record the drafts proposed after the bonus token at position `base`.
    /// A short proposal is padded with never-matching positions.
    pub fn record(&mut self, base: usize, drafts: &[u32]) {
        let mut padded: Vec<u32> = drafts.iter().copied().take(self.k).collect();
        padded.resize(self.k, NO_TOKEN);
        self.stats.proposed += 1;
        self.pending.push_back(Pending {
            base,
            drafts: padded,
            resolved: 0,
            accepted: 0,
            open: true,
            hits: vec![false; self.k],
        });
    }

    /// Sequence ended: unresolved drafts count as truncated.
    pub fn finish(&mut self) {
        self.stats.truncated += self.pending.len() as u64;
        self.pending.clear();
    }
}

/// Phase 2 verify-row parity (`ATLAS_GLM_EXL3_DFLASH_VERIFY_PROBE`): a probe
/// runs the `k + 1`-row verify forward over `[bonus, d_1..d_k]` at position
/// `base`, then restores the target state and the serial decode continues.
/// Verify row `j` predicts the token at `base + j + 1`; it is checkable only
/// while drafts `d_1..d_j` equal the serial tokens (greedy verify accepts
/// exactly that prefix), and then its argmax must equal the serial token.
#[derive(Debug, Clone, Default)]
pub struct VerifyParity {
    pending: VecDeque<ParityEntry>,
    pub probes: u64,
    pub rows_checked: u64,
    pub rows_matched: u64,
    /// Probes whose verify-accepted prefix equals the serial-accepted prefix.
    pub accept_agree: u64,
    pub accept_total: u64,
    /// First few mismatches: (position, verify row, serial token, verify argmax).
    pub mismatches: Vec<(usize, usize, u32, u32)>,
}

#[derive(Debug, Clone)]
struct ParityEntry {
    base: usize,
    drafts: Vec<u32>,
    argmax: Vec<u32>,
    /// Rows resolved so far.
    resolved: usize,
    /// Drafts matched the serial tokens so far.
    open: bool,
    /// Serial-accepted prefix length so far.
    serial_accepted: usize,
}

impl VerifyParity {
    const MAX_MISMATCHES: usize = 8;

    /// Verify of `[bonus at base, drafts..]` produced row argmaxes `argmax`
    /// (`argmax.len() == drafts.len() + 1`).
    pub fn record(&mut self, base: usize, drafts: &[u32], argmax: &[u32]) {
        if argmax.len() != drafts.len() + 1 {
            return;
        }
        self.probes += 1;
        self.pending.push_back(ParityEntry {
            base,
            drafts: drafts.to_vec(),
            argmax: argmax.to_vec(),
            resolved: 0,
            open: true,
            serial_accepted: 0,
        });
    }

    /// The serial decode produced `token` at `pos`.
    pub fn observe(&mut self, pos: usize, token: u32) {
        for e in self.pending.iter_mut() {
            let rows = e.argmax.len();
            if pos <= e.base || pos > e.base + rows || !e.open {
                continue;
            }
            let j = pos - e.base - 1; // verify row
            if j != e.resolved {
                e.open = false;
                continue;
            }
            e.resolved += 1;
            self.rows_checked += 1;
            if e.argmax[j] == token {
                self.rows_matched += 1;
            } else if self.mismatches.len() < Self::MAX_MISMATCHES {
                self.mismatches.push((pos, j, token, e.argmax[j]));
            }
            if j < e.drafts.len() && e.drafts[j] == token {
                e.serial_accepted += 1;
            } else {
                e.open = false;
            }
        }
        while let Some(front) = self.pending.front() {
            if front.open && front.resolved < front.argmax.len() {
                break;
            }
            let done = self.pending.pop_front().expect("front exists");
            let verify_accepted = done
                .drafts
                .iter()
                .zip(&done.argmax)
                .take_while(|(d, a)| d == a)
                .count();
            self.accept_total += 1;
            self.accept_agree += u64::from(verify_accepted == done.serial_accepted);
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "probes={} rows_checked={} rows_matched={} accept_agree={}/{} mismatches={:?}",
            self.probes,
            self.rows_checked,
            self.rows_matched,
            self.accept_agree,
            self.accept_total,
            self.mismatches
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a scorer over an actual token stream starting at `start`, with
    /// `propose(pos)` giving the drafts made after the bonus at `pos`.
    fn run(
        k: usize,
        start: usize,
        actual: &[u32],
        propose: impl Fn(usize) -> Vec<u32>,
    ) -> AcceptStats {
        let mut s = DraftScorer::new(k);
        for (i, &tok) in actual.iter().enumerate() {
            let pos = start + i;
            s.observe(pos, tok);
            s.record(pos, &propose(pos));
        }
        s.finish();
        s.stats().clone()
    }

    #[test]
    fn perfect_drafter_accepts_everything() {
        let actual: Vec<u32> = (0..20).collect();
        let st = run(3, 10, &actual, |pos| {
            (1..=3).map(|j| (pos - 10 + j) as u32).collect()
        });
        // Steps whose 3 predicted positions all arrived: 20 - 3.
        assert_eq!(st.steps, 17);
        assert_eq!(st.truncated, 3);
        assert_eq!(st.proposed, 20);
        assert_eq!(st.hist, vec![0, 0, 0, 17]);
        assert!((st.tau() - 4.0).abs() < 1e-12);
        assert_eq!(st.prefix_curve(), vec![1.0, 1.0, 1.0]);
        assert_eq!(st.raw_curve(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn prefix_stops_at_first_miss_but_raw_counts_later_hits() {
        let mut s = DraftScorer::new(3);
        s.record(5, &[100, 999, 102]);
        s.observe(6, 100);
        s.observe(7, 101);
        assert_eq!(s.stats().steps, 0);
        s.observe(8, 102);
        let st = s.stats();
        assert_eq!(st.steps, 1);
        assert_eq!(st.hist, vec![0, 1, 0, 0]);
        assert_eq!(st.hits, vec![1, 0, 1]);
        assert!((st.tau() - 2.0).abs() < 1e-12);
        assert_eq!(st.prefix_curve(), vec![1.0, 0.0, 0.0]);
        assert_eq!(st.conditional_curve(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn first_miss_gives_zero_accepted() {
        let mut s = DraftScorer::new(2);
        s.record(0, &[7, 8]);
        s.observe(1, 1);
        s.observe(2, 8);
        assert_eq!(s.stats().hist, vec![1, 0, 0]);
        assert_eq!(s.stats().hits, vec![0, 1]);
    }

    #[test]
    fn short_proposal_is_padded_with_misses() {
        let mut s = DraftScorer::new(3);
        s.record(0, &[1]);
        s.observe(1, 1);
        s.observe(2, 2);
        s.observe(3, 3);
        assert_eq!(s.stats().hist, vec![0, 1, 0, 0]);
    }

    #[test]
    fn position_gap_truncates_instead_of_misscoring() {
        let mut s = DraftScorer::new(3);
        s.record(0, &[1, 2, 3]);
        s.observe(1, 1);
        s.observe(3, 3); // position 2 never observed
        assert_eq!(s.stats().steps, 0);
        assert_eq!(s.stats().truncated, 1);
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn overlapping_drafts_resolve_in_order() {
        let actual = [10u32, 11, 12, 13, 14, 15];
        // Every draft predicts actual[i+1], then a miss.
        let st = run(2, 0, &actual, |pos| {
            vec![actual.get(pos + 1).copied().unwrap_or(0), 0]
        });
        assert_eq!(st.steps, 4);
        assert_eq!(st.hist, vec![0, 4, 0]);
        assert!((st.mean_accepted() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn merge_and_summary_are_deterministic() {
        let mut a = AcceptStats::new(2);
        a.steps = 2;
        a.hist = vec![1, 0, 1];
        a.hits = vec![1, 1];
        let mut b = a.clone();
        b.merge(&a);
        assert_eq!(b.steps, 4);
        assert_eq!(b.hist, vec![2, 0, 2]);
        assert_eq!(b.summary(), b.clone().summary());
        assert!(b.summary().starts_with("k=2 steps=4 tau=2.000"));
        // Mismatched k is ignored.
        b.merge(&AcceptStats::new(3));
        assert_eq!(b.steps, 4);
    }

    #[test]
    fn empty_stats_are_finite() {
        let st = AcceptStats::new(4);
        assert_eq!(st.tau(), 1.0);
        assert!(st.prefix_curve().iter().all(|v| *v == 0.0));
        assert!(st.conditional_curve().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn verify_parity_checks_only_the_serial_accepted_prefix() {
        let mut v = VerifyParity::default();
        // bonus at 10, drafts predict 11..13; verify argmax rows predict 11..14.
        v.record(10, &[1, 2, 3], &[1, 2, 9, 4]);
        v.observe(11, 1); // row 0 vs serial 1: match; draft 1 accepted
        v.observe(12, 2); // row 1 match; draft 2 accepted
        v.observe(13, 7); // row 2 predicted 9, serial 7: mismatch; draft 3 rejected
        v.observe(14, 4); // closed: not checked
        assert_eq!((v.rows_checked, v.rows_matched), (3, 2));
        assert_eq!(v.mismatches, vec![(13, 2, 7, 9)]);
        // verify accepted [1,2] (3 != 9), serial accepted 2: agree.
        assert_eq!((v.accept_agree, v.accept_total), (1, 1));
    }

    #[test]
    fn verify_parity_full_accept_checks_every_row() {
        let mut v = VerifyParity::default();
        v.record(0, &[5, 6], &[5, 6, 7]);
        for (p, t) in [(1, 5), (2, 6), (3, 7)] {
            v.observe(p, t);
        }
        assert_eq!((v.rows_checked, v.rows_matched), (3, 3));
        assert_eq!((v.accept_agree, v.accept_total), (1, 1));
        assert!(v.mismatches.is_empty());
    }
}
