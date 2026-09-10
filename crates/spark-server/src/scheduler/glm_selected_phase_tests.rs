// SPDX-License-Identifier: AGPL-3.0-only
//! Real selected F5/E6/E7 verdicts and worker replay; byte-backed, not numerics.
use super::*;

#[test]
fn glm_selected_phase_boundary_commits_only_pre_boundary_inputs() {
    if process::isolated(
        "scheduler::glm_owner_step::tests::phase_boundary::glm_selected_phase_boundary_commits_only_pre_boundary_inputs",
    ) {
        return;
    }
    for physical in [&[0usize, 2, 3][..], &[0, 1][..], &[0][..]] {
        // Unchanged selection control first, then native opener and natural end.
        for boundary_kind in 0..3 {
            let mut r = Run::new(physical, true);
            r.cold();
            let index = r.active.iter().position(|a| a.seq.slot_idx == 0).unwrap();
            let a = &mut r.active[index];
            let tokens = glm_c2_serial::issued(a, r.model.vocab_size()).unwrap();
            let boundary = tokens[1]; // Actual issued draft; never forge a receipt.
            // Explicit scheduler-history conditioning, not a model-output
            // oracle. Canonical SequenceState and the real issued receipt stay
            // untouched. Verify honors presence penalties, not caller bias.
            a.output_tokens = vec![boundary];
            a.presence_penalty = -100.0;
            let base = a.seq.seq_len;
            let previous_output = a.output_tokens.len();
            let remaining = a.remaining;
            a.eos_tokens = vec![100];
            a.enable_thinking = true;
            a.inside_thinking = true;
            a.think_ended = false;
            a.think_just_ended = false;
            a.thinking_tokens = 20; // Above the legacy A4 floor, below budget.
            a.thinking_budget = Some(128);
            a.force_end_thinking = false;
            a.think_start_token = Some(99);
            a.think_end_token = Some(if boundary_kind == 2 { boundary } else { 98 });
            a.tools_present = boundary_kind == 1;
            a.tool_call_start_token = Some(if boundary_kind == 1 { boundary } else { 97 });
            a.tool_call_end_token = Some(96);
            a.require_tool_call = false;
            a.suppress_tool_call = false;
            r.sched.limits.glm_tool_boundary = Some(if boundary_kind == 1 { boundary } else { 97 });
            let mut ctx = context(&r.sched);
            ctx.glm_tool_boundary = r.sched.limits.glm_tool_boundary;
            ctx.think_start_token = Some(99);
            ctx.think_end_token = a.think_end_token;
            ctx.tool_call_start_token = a.tool_call_start_token;
            ctx.tool_call_end_token = Some(96);
            glm_c2_serial::step_selected_serial(&r.model, &mut r.active, &r.sched, &ctx).unwrap();
            let packets = r.tx.packets();
            let accepted = if physical.len() == 1 {
                assert_eq!(packets[1], [0xffff_fff5]);
                packets[4][0] as usize
            } else {
                assert_eq!(
                    packets[1],
                    [if physical.len() == 2 {
                        0xffff_ffe6
                    } else {
                        0xffff_ffe7
                    }]
                );
                packets[3][2] as usize
            };
            let a = &r.active[index];
            if boundary_kind == 0 {
                assert!(
                    accepted >= 1,
                    "nonboundary control must genuinely accept its issued draft"
                );
            } else {
                assert_eq!(
                    accepted, 0,
                    "boundary is the bonus, never commit later stale-phase rows"
                );
                assert_eq!(&a.output_tokens[previous_output..], &[boundary]);
                assert_eq!(a.remaining, remaining - 1);
                assert!(!a.inside_thinking && a.think_ended);
                assert_eq!(a.thinking_tokens, 20);
                assert_eq!(a.seq.seq_len, base + 1);
                assert_eq!(&a.seq.tokens[base..], &tokens[..1]);
                assert_eq!(a.last_token, boundary);
                assert_eq!(a.pending_drafts.len(), 4);
            }
            assert_eq!(a.seq.seq_len, base + accepted + 1);
            assert_eq!(&a.seq.tokens[base..], &tokens[..accepted + 1]);
            let e1 = if physical.len() == 1 { 5 } else { 4 };
            assert_eq!(packets[e1], [0]);
            assert_eq!(packets[e1 + 1], [0xffff_ffe1]);
            assert_eq!(packets[e1 + 2].last(), Some(&a.last_token));
            r.replay(if physical.len() == 1 {
                2
            } else {
                1 + physical.len()
            });
            r.close();
        }
    }
}
