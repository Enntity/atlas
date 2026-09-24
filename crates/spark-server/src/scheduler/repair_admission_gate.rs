// SPDX-License-Identifier: AGPL-3.0-only

//! Admission fence for the repaired GLM TP2/EP2 scheduler.
//!
//! The repair lane owns shared prompt-capture and collective command state.
//! A request that needs grammar, tools, vision, or explicit serial decode is
//! therefore admitted as a separate plain-decode class. The fence keeps the
//! two classes from sharing a prefill or decode wave, including while a
//! sequence is parked for swap or decode-time preemption.

use super::types::{ActiveSeq, PreemptedSeq, PrefillInProgress, SwappedSeq};
use crate::api::InferenceRequest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Mtp,
    Plain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OccupiedLane {
    Idle,
    Mtp,
    Plain,
    Mixed,
}

fn request_lane(req: &InferenceRequest) -> Lane {
    if req.has_grammar_spec()
        || req.tools_present()
        || req.require_tool_call()
        || req.disable_mtp()
        || req.suppress_tool_call()
        || req.has_image_pixels()
        || request_exceeds_repair_context(req.prompt_len(), req.max_tokens())
    {
        Lane::Plain
    } else {
        // Adapter and beam shapes are still rejected by repair_admission. They
        // remain in the MTP class here so a rejected item cannot reorder a
        // later plain request across the admission fence.
        Lane::Mtp
    }
}

fn request_exceeds_repair_context(prompt_tokens: usize, max_tokens: usize) -> bool {
    prompt_tokens.saturating_add(max_tokens)
        > spark_model::speculative::glm_repair_policy::MAX_LONG_CONTEXT
}

fn merge_lane(current: OccupiedLane, next: Lane) -> OccupiedLane {
    match (current, next) {
        (OccupiedLane::Idle, Lane::Mtp) | (OccupiedLane::Mtp, Lane::Mtp) => OccupiedLane::Mtp,
        (OccupiedLane::Idle, Lane::Plain) | (OccupiedLane::Plain, Lane::Plain) => {
            OccupiedLane::Plain
        }
        (OccupiedLane::Mtp, Lane::Plain) | (OccupiedLane::Plain, Lane::Mtp) => OccupiedLane::Mixed,
        (OccupiedLane::Mixed, _) => OccupiedLane::Mixed,
    }
}

fn lane_for_state(
    disable_mtp: bool,
    suppress_tool_call: bool,
    grammar: bool,
    tools_present: bool,
    require_tool_call: bool,
    tool_request: bool,
) -> Lane {
    if disable_mtp
        || suppress_tool_call
        || grammar
        || tools_present
        || require_tool_call
        || tool_request
    {
        Lane::Plain
    } else {
        Lane::Mtp
    }
}

fn occupied_lane(
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    preempted: &[PreemptedSeq],
) -> OccupiedLane {
    let mut lane = OccupiedLane::Idle;
    for a in active {
        lane = merge_lane(
            lane,
            lane_for_state(
                a.disable_mtp,
                a.suppress_tool_call,
                a.grammar_state.is_some(),
                a.tools_present,
                a.require_tool_call,
                a.tool_request,
            ),
        );
    }
    for p in prefilling {
        lane = merge_lane(
            lane,
            lane_for_state(
                p.disable_mtp,
                p.suppress_tool_call,
                p.grammar_state.is_some(),
                p.tools_present,
                p.require_tool_call,
                false,
            ),
        );
    }
    for s in swapped {
        lane = merge_lane(
            lane,
            lane_for_state(
                s.disable_mtp,
                s.suppress_tool_call,
                false,
                s.tools_present,
                s.require_tool_call,
                s.tool_request,
            ),
        );
    }
    for p in preempted {
        lane = merge_lane(
            lane,
            lane_for_state(
                p.a.disable_mtp,
                p.a.suppress_tool_call,
                p.a.grammar_state.is_some(),
                p.a.tools_present,
                p.a.require_tool_call,
                p.a.tool_request,
            ),
        );
    }
    lane
}

fn limit_lanes(lanes: &[Lane], occupied: OccupiedLane, capacity: usize) -> (usize, usize) {
    if capacity == 0 || lanes.is_empty() {
        return (0, 0);
    }
    let wanted = match occupied {
        OccupiedLane::Idle => lanes[0],
        OccupiedLane::Mtp => Lane::Mtp,
        // Plain decode currently has request-local state that is not proven
        // safe to share even with another plain owner.  Keep the wave
        // serialized until every existing plain owner is idle.
        OccupiedLane::Plain => return (0, 0),
        // A mixed wave is already an invariant violation. Do not admit a new
        // owner until the existing wave drains back to one class.
        OccupiedLane::Mixed => return (0, 0),
    };
    let class_cap = if matches!(occupied, OccupiedLane::Idle) && wanted == Lane::Plain {
        // The existing model prefill still may populate the shared MTP capture
        // before a fallback request takes the native decode path. Admit one
        // fresh plain request at a time until that side effect is removed or
        // made per-owner; this is a safety bound, not a throughput claim.
        capacity.min(1)
    } else {
        capacity
    };
    let prefix = lanes
        .iter()
        .take_while(|lane| **lane == wanted)
        .count()
        .min(class_cap);
    (prefix, class_cap)
}

/// Return `(eligible_queue_prefix, effective_capacity)` for this scheduler
/// tick. The caller only exposes that prefix to its normal scheduling policy;
/// the first opposite-class request remains at the front of the queue.
pub(super) fn limit_requests(
    requests: &[InferenceRequest],
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    preempted: &[PreemptedSeq],
    capacity: usize,
    repair_enabled: bool,
) -> (usize, usize) {
    if !repair_enabled {
        return (requests.len(), capacity);
    }
    let lanes: Vec<Lane> = requests.iter().map(request_lane).collect();
    limit_lanes(
        &lanes,
        occupied_lane(active, prefilling, swapped, preempted),
        capacity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_lane_follows_fifo_class_boundary() {
        assert_eq!(
            limit_lanes(
                &[Lane::Mtp, Lane::Mtp, Lane::Plain, Lane::Mtp],
                OccupiedLane::Idle,
                4,
            ),
            (2, 4)
        );
        assert_eq!(
            limit_lanes(
                &[Lane::Plain, Lane::Plain, Lane::Mtp],
                OccupiedLane::Idle,
                4,
            ),
            (1, 1)
        );
    }

    #[test]
    fn occupied_class_cannot_admit_the_other_collective() {
        assert_eq!(
            limit_lanes(&[Lane::Plain, Lane::Plain], OccupiedLane::Mtp, 4),
            (0, 4)
        );
        assert_eq!(
            limit_lanes(&[Lane::Mtp, Lane::Mtp], OccupiedLane::Plain, 4),
            (0, 0)
        );
        assert_eq!(limit_lanes(&[Lane::Plain], OccupiedLane::Plain, 4), (0, 0));
    }

    #[test]
    fn mixed_occupancy_fails_closed() {
        assert_eq!(
            limit_lanes(&[Lane::Mtp, Lane::Plain], OccupiedLane::Mixed, 4),
            (0, 0)
        );
    }

    #[test]
    fn dynamic_tool_state_is_plain_decode_lane() {
        assert_eq!(
            lane_for_state(false, true, false, false, false, false),
            Lane::Plain
        );
        assert_eq!(
            lane_for_state(false, false, true, false, false, false),
            Lane::Plain
        );
        assert_eq!(
            lane_for_state(false, false, false, false, false, true),
            Lane::Plain
        );
        assert_eq!(
            lane_for_state(false, false, false, false, false, false),
            Lane::Mtp
        );
    }

    #[test]
    fn budget_past_repair_context_is_plain_decode_lane() {
        let cap = spark_model::speculative::glm_repair_policy::MAX_LONG_CONTEXT;
        assert!(!request_exceeds_repair_context(cap - 400, 400));
        assert!(request_exceeds_repair_context(cap - 399, 400));
        assert!(request_exceeds_repair_context(cap, 1));
        assert_eq!(request_lane_for_test(cap - 399, 400), Lane::Plain);
    }

    fn request_lane_for_test(prompt_tokens: usize, max_tokens: usize) -> Lane {
        if request_exceeds_repair_context(prompt_tokens, max_tokens) {
            Lane::Plain
        } else {
            Lane::Mtp
        }
    }
}
