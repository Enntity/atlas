// SPDX-License-Identifier: AGPL-3.0-only
//! The real retained-tail/repair writers keep two request owners disjoint.
use super::*;
use crate::speculative::glm_repair::{GlmPairRepair, RepairInput, RepairSpan};

#[test]
fn retained_requests_survive_foreign_capture_and_accepted_row_staging() {
    fixture_rows(8, |head, ctx, gpu, _seen| {
        let shared = gpu.alloc(64 * 1024).unwrap();
        let bonus = gpu.alloc(1024).unwrap();
        let mut states = [
            head.alloc_state_inner(gpu).unwrap(),
            head.alloc_state_inner(gpu).unwrap(),
        ];
        for (index, state) in states.iter_mut().enumerate() {
            state
                .block_table
                .push(head.kv_cache.lock().alloc_block().unwrap());
            state.seq_len = 1;
            gpu.copy_h2d(&vec![0x31 + index as u8; 1024], shared)
                .unwrap();
            state
                .retain_repair_prompt_tail(gpu, shared, index as u64 + 1, 2, 1024, 0)
                .unwrap();
        }
        gpu.copy_h2d(&vec![0xef; 64 * 1024], shared).unwrap();
        for (index, state) in states.iter_mut().enumerate() {
            let generation = index as u64 + 1;
            let tokens = vec![0, 1, 2];
            let input = RepairInput {
                token: 3,
                tokens: &tokens,
                prompt_len: 2,
                position: 3,
                drafts: 2,
                generation,
                capture_generation: generation,
                captured_rows: 0,
                context_tokens: 64,
                capture: RepairSpan {
                    ptr: shared,
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
            let ptr = state.repair_owned.as_ref().unwrap().ptr;
            assert_eq!(
                &gpu.read_alloc(ptr).unwrap()[..1024],
                &vec![0x31 + index as u8; 1024]
            );
            head.prepare(&input, state, ctx, 0).unwrap();
            state.seq_len += 2;
            state.last_num_drafted = 2;
            state
                .record_verified(generation, generation, 3, &[3, 4, 5], 2, 6, 3)
                .unwrap();
            state.repair.acknowledge(2).unwrap();
            let target = vec![0x71 + index as u8; 3 * 1024];
            gpu.copy_h2d(&target, ctx.buffers.norm_output()).unwrap();
            let committed = vec![0, 1, 2, 3, 4, 5];
            head.prepare(
                &RepairInput {
                    token: 6,
                    tokens: &committed,
                    position: 6,
                    hidden_row: 2,
                    ..input
                },
                state,
                ctx,
                0,
            )
            .unwrap();
            assert_eq!(&gpu.read_alloc(ptr).unwrap()[1024..], &target[..2048]);
            assert_eq!(gpu.read_alloc(shared).unwrap(), vec![0xef; 64 * 1024]);
        }
        for state in &mut states {
            let ptr = state.repair_owned.as_ref().unwrap().ptr;
            head.free_state(gpu, None, state).unwrap();
            assert!(gpu.read_alloc(ptr).is_none());
        }
    });
}
