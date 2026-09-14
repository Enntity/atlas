// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::glm5_mtp::repair_state::RepairPhase;
use crate::speculative::glm_repair::{GlmPairRepair, RepairInput, RepairSpan};

#[test]
fn real_writer_bootstrap_then_every_verdict_stages_only_accepted_rows() {
    for (drafts, accepted) in [1, 2, 4]
        .into_iter()
        .flat_map(|d| (0..=d).map(move |a| (d, a)))
    {
        fixture_rows(8, |head, ctx, gpu, seen| {
            let capture = gpu.alloc(64 * 1024).unwrap();
            let bonus = gpu.alloc(1024).unwrap();
            gpu.copy_h2d(&[0x3c; 1024], bonus).unwrap();
            let mut state = head.alloc_state_inner(gpu).unwrap();
            let tokens = vec![0, 1, 2];
            let input = RepairInput {
                token: 3,
                tokens: &tokens,
                prompt_len: 2,
                position: 3,
                drafts,
                generation: 1,
                capture_generation: 1,
                captured_rows: 2,
                context_tokens: 64,
                capture: RepairSpan {
                    ptr: capture,
                    bytes: 64 * 1024,
                },
                normalized: RepairSpan {
                    ptr: ctx.buffers.norm_output(),
                    bytes: ctx.buffers.sizes().norm_output,
                },
                bonus: RepairSpan {
                    ptr: bonus,
                    bytes: 1024,
                },
                hidden_row: 0,
            };
            let copies = gpu.d2d_count();
            let short_verify = RepairInput {
                normalized: RepairSpan {
                    bytes: drafts * 1024,
                    ..input.normalized
                },
                ..input
            };
            assert!(head.validate_prepare(&short_verify, &state, ctx).is_err());
            head.validate_prepare(&input, &state, ctx).unwrap();
            assert_eq!(gpu.d2d_count(), copies);
            assert!(state.block_table.is_empty());
            head.prepare(&input, &mut state, ctx, 0).unwrap();
            assert_eq!(state.seq_len, 2);
            assert_eq!(*seen.lock(), [0, 1]);
            assert!(matches!(state.repair, RepairPhase::Proposed(_)));
            // Stand in for the completed autoregressive proposal writes.
            state.seq_len += drafts;
            state.last_num_drafted = drafts;
            let verified = &[3, 4, 5, 6, 7][..drafts + 1];
            let mut committed_tokens = tokens.clone();
            committed_tokens.extend_from_slice(&verified[..accepted + 1]);
            state
                .record_verified(
                    1,
                    1,
                    3,
                    verified,
                    accepted,
                    committed_tokens.len(),
                    drafts + 1,
                )
                .unwrap();
            state.repair.acknowledge(accepted).unwrap();
            let target: Vec<u8> = (0..5).flat_map(|i| vec![0x70 + i; 1024]).collect();
            gpu.copy_h2d(&target, ctx.buffers.norm_output()).unwrap();
            let input = RepairInput {
                token: 0,
                tokens: &committed_tokens,
                position: committed_tokens.len(),
                captured_rows: 0,
                hidden_row: accepted,
                ..input
            };
            let start = seen.lock().len();
            let wrong_depth = RepairInput {
                drafts: if drafts == 1 { 4 } else { 1 },
                ..input
            };
            assert!(head.validate_prepare(&wrong_depth, &state, ctx).is_err());
            assert_eq!(seen.lock().len(), start);
            head.prepare(&input, &mut state, ctx, 0).unwrap();
            assert_eq!(state.seq_len, 3 + accepted);
            assert_eq!(
                &seen.lock()[start..],
                &(3..3 + accepted as i64).collect::<Vec<_>>()
            );
            assert_eq!(
                &gpu.read_alloc(capture).unwrap()[..accepted * 1024],
                &target[..accepted * 1024]
            );
            assert_eq!(gpu.read_alloc(bonus).unwrap(), vec![0x3c; 1024]);
            // Terminal retirement after a prepared proposal releases blocks
            // without inventing a verified zero-acceptance event.
            head.free_state(gpu, None, &mut state).unwrap();
            assert!(state.block_table.is_empty());
            assert_eq!(state.seq_len, 0);
            assert!(matches!(state.repair, RepairPhase::Capture));
        });
    }
}

#[test]
fn real_prepare_rejects_alias_and_stale_capture_before_copy_or_allocation() {
    fixture_rows(8, |head, ctx, gpu, _| {
        let capture = gpu.alloc(64 * 1024).unwrap();
        let bonus = gpu.alloc(1024).unwrap();
        let mut state = head.alloc_state_inner(gpu).unwrap();
        let mut input = RepairInput {
            token: 3,
            tokens: &[0, 1, 2],
            prompt_len: 2,
            position: 3,
            drafts: 4,
            generation: 1,
            capture_generation: 2,
            captured_rows: 2,
            context_tokens: 64,
            capture: RepairSpan {
                ptr: capture,
                bytes: 64 * 1024,
            },
            normalized: RepairSpan {
                ptr: ctx.buffers.norm_output(),
                bytes: ctx.buffers.sizes().norm_output,
            },
            bonus: RepairSpan {
                ptr: bonus,
                bytes: 1024,
            },
            hidden_row: 0,
        };
        assert!(head.prepare(&input, &mut state, ctx, 0).is_err());
        input.capture_generation = 1;
        input.capture.ptr = ctx.buffers.norm_output();
        assert!(head.prepare(&input, &mut state, ctx, 0).is_err());
        assert!(state.block_table.is_empty());
        assert_eq!(state.seq_len, 0);
        assert_eq!(gpu.d2d_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
        input.capture.ptr = capture;
        head.prepare(&input, &mut state, ctx, 0).unwrap();
        state.seq_len = 6;
        state.last_num_drafted = 4;
        state
            .record_verified(1, 1, 3, &[3, 4, 5, 6, 7], 1, 5, 5)
            .unwrap();
        state.repair.acknowledge(1).unwrap();
        // A disappearing source allocation models a D2D failure after all
        // host checks; never publish the planned canonical cursor on error.
        let input = RepairInput {
            tokens: &[0, 1, 2, 3, 4],
            position: 5,
            hidden_row: 1,
            captured_rows: 0,
            normalized: RepairSpan {
                ptr: DevicePtr(0x7fff_0000_0000),
                bytes: 5 * 1024,
            },
            ..input
        };
        assert!(head.prepare(&input, &mut state, ctx, 0).is_err());
        assert_eq!(state.seq_len, 6);
        assert!(matches!(state.repair, RepairPhase::Failed));
        assert!(head.validate_prepare(&input, &state, ctx).is_err());
        head.free_state(gpu, None, &mut state).unwrap();
        assert!(state.block_table.is_empty());
    });
}
