// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::layer::MoeLoraRoute;

fn state(seq: &mut SequenceState) -> &mut Glm5MtpProposerState {
    seq.proposer_state
        .as_mut()
        .unwrap()
        .as_any_mut()
        .downcast_mut()
        .unwrap()
}

#[test]
fn actual_arm_requires_absent_live_and_historical_owners_for_fold_or_skip() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 1);
            for route in [
                crate::lora::resolve_moe_lora_route(-1, -1, false),
                crate::lora::resolve_moe_lora_route(-1, 0, true),
            ] {
                ctx.moe_lora_route = route;
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                for owner in 0..4 {
                    // A fresh actual request stamp prevents exhaustion from
                    // suppressing any of these independent negative probes.
                    seq.mtp_capture_gen += 1;
                    let generation = seq.mtp_capture_gen;
                    prepared_with_source(head, ctx, gpu, state(&mut seq), generation, 3);
                    let mut owners = no_adapter_owners();
                    match owner {
                        0 => owners.pool = true,
                        1 => owners.overlays = true,
                        2 => owners.rotatable = true,
                        _ => owners.install_attempted = true,
                    }
                    assert!(
                        super::super::arm_prepared(
                            &mut seq,
                            3,
                            3,
                            4,
                            saved,
                            0,
                            false,
                            ctx,
                            7,
                            || owners
                        )
                        .is_err()
                    );
                    assert!(gpu.events.lock().is_empty());
                }
            }
        });
    }
}

#[test]
fn actual_arm_still_rejects_refuse_config_capacity_and_active_sequences() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        ctx.moe_lora_route = crate::lora::resolve_moe_lora_route(1, 0, true);
        assert_eq!(ctx.moe_lora_route, MoeLoraRoute::Refuse);
        assert!(arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).is_err());
        ctx.moe_lora_route = crate::lora::resolve_moe_lora_route(0, 0, true);
        seq.adapter_slot = 0;
        assert!(arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).is_err());
        seq.adapter_slot = -1;
        seq.adapter_id = 123;
        assert!(arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).is_err());
        seq.adapter_id = 0;
        let mut config = ctx.config.clone();
        config.adapter_max_rank = 4;
        let changed = ForwardContext {
            config: &config,
            midchunk_capture: None,
            ..*ctx
        };
        assert!(arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, &changed, 7).is_err());
        assert!(gpu.events.lock().is_empty());
    });
}

#[test]
fn disabled_or_exhausted_trace_does_not_read_model_ownership() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        state(&mut seq).hidden_trace.enabled = false;
        super::super::arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7, || {
            panic!("disabled ownership read")
        })
        .unwrap();
        state(&mut seq).hidden_trace.enabled = true;
        for _ in 0..8 {
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        }
        super::super::arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7, || {
            panic!("exhausted ownership read")
        })
        .unwrap();
    });
}
