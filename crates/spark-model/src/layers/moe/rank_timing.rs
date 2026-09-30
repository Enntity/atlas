// SPDX-License-Identifier: AGPL-3.0-only

//! Per-rank MoE timing for EP2 decode/verify steps
//! (`ATLAS_GLM_MOE_RANK_TIMING=1`, default off = no events, no host work).
//!
//! Under expert TP both ranks run the same routed and shared work in every
//! MoE layer, so a rank waits in the EP collective only for the wire and for
//! however much longer the peer's copy of that work took. This tap measures
//! both on the GPU clock, with no sync: three timed events per MoE layer
//! (entry, before the EP all-reduce, after it) give `busy` = entry to
//! collective and `coll` = collective to summed output. Each rank logs one
//! line per step; joining the two ranks' lines by `step` shows, per layer,
//! which rank was slower and how long the other one waited.
//!
//! A step's events are read when the next step first reaches a layer that
//! still holds a sample: the host has read that step's tokens by then, so
//! they have completed.
//! Layers inside a CUDA-graph capture are not timed (a replay runs no host
//! code), so measure with `ATLAS_GLM_VERIFY_GRAPH` off. The events live as
//! long as the process, like the layer's own stream events.

use std::fmt;

use anyhow::Result;

use super::MoeLayer;
use crate::layer::ForwardContext;

/// Largest row count timed: decode and DFlash verify, not prefill chunks.
const MAX_ROWS: u32 = 64;

/// One model's probes, in the order its MoE layers first ran.
#[derive(Debug, Default)]
pub struct MoeRankTiming(parking_lot::Mutex<Probes>);

#[derive(Debug, Default)]
struct Probes {
    probes: Vec<Probe>,
    /// Steps logged so far. Both ranks run the same forwards, so they count alike.
    steps: u64,
}

#[derive(Debug)]
struct Probe {
    layer: usize,
    /// Timed events: MoE entry, before the EP collective, after it.
    events: [u64; 3],
    /// Rows of the recorded sample; 0 = nothing to collect.
    rows: u32,
}

/// One step's samples in MoE-layer order, in microseconds (-1 = not ready).
#[derive(Debug, PartialEq)]
struct Step {
    step: u64,
    rows: u32,
    busy_us: Vec<i32>,
    coll_us: Vec<i32>,
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "step={} rows={} busy_us={:?} coll_us={:?}",
            self.step, self.rows, self.busy_us, self.coll_us
        )
    }
}

impl Probes {
    /// Index of `layer`'s probe, created on first use from `events`.
    fn slot(&mut self, layer: usize, events: impl FnOnce() -> Result<[u64; 3]>) -> Result<usize> {
        if let Some(i) = self.probes.iter().position(|p| p.layer == layer) {
            return Ok(i);
        }
        self.probes.push(Probe {
            layer,
            events: events()?,
            rows: 0,
        });
        Ok(self.probes.len() - 1)
    }

    /// The finished step, when probe `i` still holds its sample (a new step
    /// has come back round to it): every recorded sample, collected and
    /// cleared. `elapsed(a, b)` is the GPU time from event `a` to event `b`
    /// once both have completed.
    fn take_step(&mut self, i: usize, elapsed: impl Fn(u64, u64) -> Option<f32>) -> Option<Step> {
        let us = |a, b| elapsed(a, b).map_or(-1, |t| t.round() as i32);
        let recorded = || self.probes.iter().filter(|p| p.rows != 0);
        let step = Step {
            step: self.steps,
            rows: Some(self.probes[i].rows).filter(|&rows| rows != 0)?,
            busy_us: recorded().map(|p| us(p.events[0], p.events[1])).collect(),
            coll_us: recorded().map(|p| us(p.events[1], p.events[2])).collect(),
        };
        self.probes.iter_mut().for_each(|p| p.rows = 0);
        self.steps += 1;
        Some(step)
    }
}

/// Start timing one MoE forward of `rows` rows: log the step that just
/// finished if this layer closes it, then mark MoE entry. Returns the
/// probe for [`before_collective`] / [`after_collective`], `None` when this
/// forward is not timed.
pub(super) fn begin(
    layer: &MoeLayer,
    rows: u32,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<Option<usize>> {
    if !ctx.levers.moe_rank_timing
        || rows > MAX_ROWS
        || ctx.comm.is_none()
        || ctx.config.ep_world_size < 2
        || ctx.gpu.stream_is_capturing(stream)
    {
        return Ok(None);
    }
    let gpu = ctx.gpu;
    let mut t = ctx.stats.moe_rank_timing.0.lock();
    let timed = || gpu.create_timed_event();
    let i = t.slot(std::ptr::from_ref(layer) as usize, || {
        Ok([timed()?, timed()?, timed()?])
    })?;
    if let Some(step) = t.take_step(i, |a, b| gpu.event_elapsed_us(a, b).ok().flatten()) {
        tracing::info!("moe-rank-timing: rank={} {step}", ctx.config.ep_rank);
    }
    let entry = t.probes[i].events[0];
    if entry == 0 {
        return Ok(None); // backend without timed events
    }
    gpu.record_event(entry, stream)?;
    Ok(Some(i))
}

/// Mark the point where this rank's MoE output is ready for the collective.
pub(super) fn before_collective(
    probe: Option<usize>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let Some(i) = probe else { return Ok(()) };
    let event = ctx.stats.moe_rank_timing.0.lock().probes[i].events[1];
    ctx.gpu.record_event(event, stream)
}

/// Mark the summed output and leave the sample for the next step to collect.
pub(super) fn after_collective(
    probe: Option<usize>,
    rows: u32,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let Some(i) = probe else { return Ok(()) };
    let mut t = ctx.stats.moe_rank_timing.0.lock();
    ctx.gpu.record_event(t.probes[i].events[2], stream)?;
    t.probes[i].rows = rows;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probes for `layers`, each with events `[10l, 10l+1, 10l+2]`.
    fn probes(layers: &[usize]) -> Probes {
        let mut t = Probes::default();
        for &l in layers {
            let e = 10 * l as u64;
            t.slot(l, || Ok([e, e + 1, e + 2])).unwrap();
        }
        t
    }

    #[test]
    fn slots_keep_first_seen_order() {
        let mut t = probes(&[7, 3, 9]);
        let unused = || -> Result<[u64; 3]> { panic!("a known layer must not make new events") };
        assert_eq!(t.slot(3, unused).unwrap(), 1);
        assert_eq!(t.slot(7, unused).unwrap(), 0);
        assert_eq!(t.slot(4, || Ok([1, 2, 3])).unwrap(), 3);
        assert!(t.slot(5, || anyhow::bail!("no events")).is_err());
        assert_eq!(t.probes.len(), 4, "a failed probe is not kept");
    }

    #[test]
    fn a_step_collects_recorded_layers_in_order_then_clears() {
        let mut t = probes(&[1, 2, 3]);
        assert_eq!(
            t.take_step(0, |_, _| Some(1.0)),
            None,
            "nothing recorded yet"
        );
        t.probes[0].rows = 7;
        t.probes[2].rows = 7;
        // Layer 1 busy 812.4 us, collective 30.6 us; layer 3's collective
        // has not completed.
        let elapsed = |a: u64, b: u64| match (a, b) {
            (10, 11) => Some(812.4),
            (11, 12) => Some(30.6),
            (30, 31) => Some(640.0),
            _ => None,
        };
        assert_eq!(
            t.take_step(1, elapsed),
            None,
            "layer 2 holds no sample: same step"
        );
        let step = t.take_step(0, elapsed).unwrap();
        assert_eq!(
            step,
            Step {
                step: 0,
                rows: 7,
                busy_us: vec![812, 640],
                coll_us: vec![31, -1]
            }
        );
        assert_eq!(
            step.to_string(),
            "step=0 rows=7 busy_us=[812, 640] coll_us=[31, -1]"
        );
        assert_eq!(t.take_step(0, elapsed), None, "samples are collected once");
        // Layer 1 stops being timed (captured into a graph): the first layer
        // that still holds a sample closes the step.
        t.probes[1].rows = 4;
        t.probes[2].rows = 4;
        assert_eq!(t.take_step(0, elapsed), None);
        let step = t.take_step(1, |_, _| None).unwrap();
        assert_eq!((step.step, step.rows, step.busy_us), (1, 4, vec![-1, -1]));
    }

    #[test]
    fn both_ranks_number_steps_and_layers_alike() {
        // Same forwards on both ranks, different timings.
        let (mut r0, mut r1) = (probes(&[5, 6]), probes(&[5, 6]));
        for step in 0..3 {
            for t in [&mut r0, &mut r1] {
                t.probes.iter_mut().for_each(|p| p.rows = 8);
            }
            let s0 = r0.take_step(0, |_, _| Some(700.0)).unwrap();
            let s1 = r1.take_step(0, |_, _| Some(950.0)).unwrap();
            assert_eq!((s0.step, s0.rows, s0.busy_us.len()), (step, 8, 2));
            assert_eq!((s1.step, s1.rows, s1.busy_us.len()), (s0.step, s0.rows, 2));
        }
    }
}
