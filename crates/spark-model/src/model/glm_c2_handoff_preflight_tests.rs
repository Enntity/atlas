// SPDX-License-Identifier: AGPL-3.0-only
//! Selected producer refusals must precede target/capture/KV work.
use super::{fixture::*, isolated};
use crate::traits::Model;
use std::sync::atomic::Ordering;

#[test]
fn actual_selected_prefill_preflights_cold_shape_profile_and_capture() {
    if isolated(
        "preflight_tests::actual_selected_prefill_preflights_cold_shape_profile_and_capture",
    ) {
        return;
    }
    for rank in 0..2 {
        for case in 0..11 {
            let mut f = Fixture::new(rank);
            f.model
                .prefill(&[4, 3, 2, 1], &mut f.seqs[1], CALLER)
                .unwrap();
            let slab = f.gpu.slab();
            let before = f.gpu.read_span(slab, SLAB_BYTES);
            match case {
                0 => f.seqs[0].cached_prefix_tokens = 1,
                1 => f.seqs[0].marconi_skip_to = 1,
                2 => f.seqs[0].prefix_lookup_applied = true,
                3 => f.model.comm = None,
                4 => {
                    f.gpu.capturing.store(true, Ordering::Relaxed);
                }
                5 => f.seqs[0].adapter_id = 7,
                _ => {}
            }
            f.gpu.clear();
            let result = match case {
                6 => f
                    .model
                    .prefill_chunk(&[1, 2, 3, 4], &mut f.seqs[0], 1, 3, true, CALLER),
                7 => f
                    .model
                    .prefill_chunk(&[1, 2, 3, 4], &mut f.seqs[0], 0, 2, false, CALLER),
                8 => f
                    .model
                    .prefill_twophase(&[1, 2, 3, 4], &mut f.seqs[0], 2, CALLER),
                9 => f.model.prefill(&[1; 9], &mut f.seqs[0], CALLER),
                10 => {
                    f.seqs[0].prompt_len = 0;
                    f.model.prefill(&[1], &mut f.seqs[0], CALLER)
                }
                _ => f.model.prefill(&[1, 2, 3, 4], &mut f.seqs[0], CALLER),
            };
            assert!(
                result.is_err(),
                "selected preflight case{case} rank{rank} unexpectedly accepted"
            );
            assert!(
                f.gpu.trace().is_empty(),
                "selected preflight case{case} did backend work: {:?}",
                f.gpu.trace()
            );
            assert!(
                f.gpu.read_span(slab, SLAB_BYTES) == before,
                "peer publication changed"
            );
            assert_eq!(f.seqs[0].mtp_capture_gen, 0);
        }
    }
}

#[test]
fn selected_mixed_and_batched_producers_refuse_before_any_backend_work() {
    if isolated(
        "preflight_tests::selected_mixed_and_batched_producers_refuse_before_any_backend_work",
    ) {
        return;
    }
    for rank in 0..2 {
        for case in 0..3 {
            let mut f = Fixture::new(rank);
            let (a, b) = f.seqs.split_at_mut(1);
            f.gpu.clear();
            let result = match case {
                0 => f
                    .model
                    .decode_batch(&[1, 2], &mut [&mut a[0], &mut b[0]], CALLER)
                    .map(|_| ()),
                1 => f
                    .model
                    .mixed_forward(
                        &[1],
                        &mut [&mut a[0]],
                        &[4, 3, 2, 1],
                        &mut b[0],
                        0,
                        4,
                        true,
                        CALLER,
                    )
                    .map(|_| ()),
                _ => f
                    .model
                    .prefill_batch_chunk_rows(
                        &mut [crate::traits::PrefillSlice {
                            prompt_tokens: &[1, 2, 3, 4],
                            seq: &mut a[0],
                            chunk_start: 0,
                            chunk_len: 4,
                            is_last_chunk: true,
                        }],
                        CALLER,
                        0,
                    )
                    .map(|_| ()),
            };
            assert!(result.is_err(), "unsupported selected producer case{case}");
            assert!(
                f.gpu.trace().is_empty(),
                "unsupported selected producer performed work: {:?}",
                f.gpu.trace()
            );
        }
    }
}

#[test]
fn actual_decode_refuses_active_capture_before_target_work() {
    if isolated("preflight_tests::actual_decode_refuses_active_capture_before_target_work") {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        f.model
            .prefill(&[1, 2, 3, 4], &mut f.seqs[0], CALLER)
            .unwrap();
        let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        f.gpu.capturing.store(true, Ordering::Relaxed);
        f.gpu.clear();
        assert!(f.model.decode(5, &mut f.seqs[0], CALLER).is_err());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.seqs[0].seq_len, 4);
        assert!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES) == before);
    }
}
