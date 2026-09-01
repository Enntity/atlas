// SPDX-License-Identifier: AGPL-3.0-only

//! Collective-safe bounded phase interleaving.

use std::time::{Duration, Instant};

use super::{ActiveSeqTiming, PendingRequestInfo, SchedulingPolicy};

/// Explicit decode/prefill cadence for a phase-interleaved scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseInterleaveConfig {
    pub decode_steps: usize,
    pub prefill_steps: usize,
    pub prefill_slice_tokens: usize,
}

impl PhaseInterleaveConfig {
    pub fn new(
        decode_steps: usize,
        prefill_steps: usize,
        prefill_slice_tokens: usize,
    ) -> Result<Self, &'static str> {
        if decode_steps == 0 {
            return Err("decode_steps must be greater than zero");
        }
        if prefill_steps == 0 {
            return Err("prefill_steps must be greater than zero");
        }
        if prefill_slice_tokens == 0 {
            return Err("prefill_slice_tokens must be greater than zero");
        }
        decode_steps
            .checked_add(prefill_steps)
            .ok_or("phase-interleave cycle length overflow")?;
        Ok(Self {
            decode_steps,
            prefill_steps,
            prefill_slice_tokens,
        })
    }
}

/// Pure per-iteration phase machine. A cycle starts with decode so a newly
/// arrived prompt cannot immediately stall an already-streaming response.
pub struct PhaseInterleaveController {
    config: PhaseInterleaveConfig,
    cycle_step: usize,
}

impl PhaseInterleaveController {
    pub fn new(config: PhaseInterleaveConfig) -> Self {
        Self {
            config,
            cycle_step: 0,
        }
    }

    /// Whether this iteration may execute one prefill slab.
    pub fn allow_prefill(&mut self, has_active_decode: bool, has_prefill_work: bool) -> bool {
        if !has_active_decode {
            self.cycle_step = 0;
            return true;
        }
        if !has_prefill_work {
            self.cycle_step = 0;
            return false;
        }
        let allow = self.cycle_step >= self.config.decode_steps;
        let cycle_len = self.config.decode_steps + self.config.prefill_steps;
        self.cycle_step = (self.cycle_step + 1) % cycle_len;
        allow
    }
}

/// GLM-style collective-safe phase interleaving. Request order remains FIFO;
/// fairness comes from bounded phase cadence rather than prompt reordering.
pub struct PhaseInterleavePolicy {
    config: PhaseInterleaveConfig,
    tbt_deadline: Duration,
}

impl PhaseInterleavePolicy {
    pub fn new(config: PhaseInterleaveConfig, tbt_deadline_ms: u64) -> Self {
        Self {
            config,
            tbt_deadline: Duration::from_millis(tbt_deadline_ms),
        }
    }
}

impl SchedulingPolicy for PhaseInterleavePolicy {
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool {
        let now = Instant::now();
        active_timings
            .iter()
            .all(|timing| now.duration_since(timing.last_token_time) < self.tbt_deadline)
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        (0..requests.len().min(capacity)).collect()
    }

    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        if active_timings.is_empty() {
            full_chunk
        } else {
            full_chunk.min(self.config.prefill_slice_tokens)
        }
    }

    fn phase_interleave_config(&self) -> Option<PhaseInterleaveConfig> {
        Some(self.config)
    }

    fn name(&self) -> &str {
        "phase-interleave"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cadence_is_bounded_and_decode_first() {
        let cfg = PhaseInterleaveConfig::new(4, 1, 32).unwrap();
        let mut controller = PhaseInterleaveController::new(cfg);
        let decisions: Vec<bool> = (0..10)
            .map(|_| controller.allow_prefill(true, true))
            .collect();
        assert_eq!(
            decisions,
            vec![
                false, false, false, false, true, false, false, false, false, true
            ]
        );
    }

    #[test]
    fn cadence_resets_when_work_disappears() {
        let cfg = PhaseInterleaveConfig::new(2, 1, 32).unwrap();
        let mut controller = PhaseInterleaveController::new(cfg);
        assert!(!controller.allow_prefill(true, true));
        assert!(!controller.allow_prefill(true, false));
        assert!(!controller.allow_prefill(true, true));
        assert!(!controller.allow_prefill(true, true));
        assert!(controller.allow_prefill(true, true));
    }

    #[test]
    fn idle_decode_never_blocks_prefill() {
        let cfg = PhaseInterleaveConfig::new(8, 2, 32).unwrap();
        let mut controller = PhaseInterleaveController::new(cfg);
        assert!(controller.allow_prefill(false, true));
        assert!(controller.allow_prefill(false, true));
    }

    #[test]
    fn rejects_degenerate_cycles() {
        assert!(PhaseInterleaveConfig::new(0, 1, 32).is_err());
        assert!(PhaseInterleaveConfig::new(1, 0, 32).is_err());
        assert!(PhaseInterleaveConfig::new(1, 1, 0).is_err());
    }

    #[test]
    fn overlap_uses_bounded_slice_without_shrinking_solo_prefill() {
        let policy = PhaseInterleavePolicy::new(PhaseInterleaveConfig::new(4, 1, 32).unwrap(), 500);
        assert_eq!(policy.prefill_slice_budget(&[], 128), 128);
        let active = [ActiveSeqTiming {
            last_token_time: Instant::now(),
        }];
        assert_eq!(policy.prefill_slice_budget(&active, 128), 32);
    }
}
