// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn actual_bootstrap_capture_binding_and_token_slice_fail_before_reads() {
    for fault in 0..11 {
        fixture(17, |gpu, ctx, cache, blocks, source, tokens, phase| {
            phase
                .primer_before(
                    &tokens[1..17],
                    DeviceSpan {
                        ptr: source,
                        bytes: 16 * 8192,
                    },
                    ctx,
                    7,
                )
                .unwrap();
            phase.primer_after(cache, &blocks[..1], ctx, 7).unwrap();
            let span = crate::speculative::glm_repair::RepairSpan {
                ptr: source,
                bytes: 17 * 8192,
            };
            let mut input = crate::speculative::glm_repair::RepairInput {
                token: tokens[17],
                tokens,
                prompt_len: 17,
                position: 18,
                drafts: 4,
                generation: 1,
                capture_generation: 1,
                captured_rows: 17,
                context_tokens: 2044,
                capture: span,
                normalized: span,
                bonus: span,
                hidden_row: 0,
            };
            match fault {
                0 => input.generation = 2,
                1 => input.capture_generation = 2,
                2 => input.captured_rows = 16,
                3 => input.prompt_len = 18,
                4 => input.position = 17,
                5 => input.hidden_row = 1,
                6 => input.capture.ptr = source.offset(2),
                7 => input.capture.bytes += 2,
                8 => input.tokens = &tokens[..17],
                9 => gpu.capturing.store(true, Ordering::Relaxed),
                _ => {}
            }
            gpu.events.lock().clear();
            let supplied = if fault == 10 { &[7][..] } else { &tokens[17..] };
            assert!(
                phase
                    .bootstrap_before(
                        &input,
                        supplied,
                        DeviceSpan {
                            ptr: source.offset(16 * 8192),
                            bytes: 8192
                        },
                        16,
                        ctx,
                        7
                    )
                    .is_err(),
                "fault{fault}"
            );
            assert!(gpu.events.lock().is_empty());
            assert!(phase.evidence(0, 1, 17).is_err());
            assert!(bootstrap(phase, 17, source, tokens, ctx).is_err());
            assert!(gpu.events.lock().is_empty());
        });
    }
}

#[test]
fn inactive_prompt_helpers_are_all_inert_without_owner_queries_or_allocations() {
    fixture(17, |gpu, ctx, cache, blocks, source, tokens, _| {
        let mut phase = Prompt::default();
        let queries = gpu.capture_queries.load(Ordering::Relaxed);
        gpu.capturing.store(true, Ordering::Relaxed);
        phase
            .primer_before(
                &[],
                DeviceSpan {
                    ptr: DevicePtr::NULL,
                    bytes: 0,
                },
                ctx,
                7,
            )
            .unwrap();
        phase.primer_after(cache, &[], ctx, 7).unwrap();
        bootstrap(&mut phase, 17, source, tokens, ctx).unwrap();
        phase.bootstrap_after(cache, blocks, ctx, 7).unwrap();
        assert!(gpu.events.lock().is_empty());
        assert!(phase.scratch.is_none());
        assert_eq!(gpu.capture_queries.load(Ordering::Relaxed), queries);
    });
}
