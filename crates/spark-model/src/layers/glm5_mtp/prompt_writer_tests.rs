// SPDX-License-Identifier: AGPL-3.0-only
//! Real primer and repair hooks, with byte-writing body sentinel (not GPU math).
use super::*;
use crate::layers::glm5_mtp::hidden_trace;
use crate::speculative::glm_repair::{GlmPairRepair, RepairInput, RepairSpan};
use sha2::{Digest, Sha256};

fn kv_reference(domain: &[u8], index: usize, rows: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update((index as u64).to_le_bytes());
    h.update(512u32.to_le_bytes());
    h.update(2u32.to_le_bytes());
    for _ in 0..rows * 2 {
        h.update([0xab; 1024]);
    }
    h.finalize().into()
}

#[test]
fn actual_primer_repair_observe_after_writes_and_reset_makes_long_request_inert() {
    for p in [2, 17, 148, 256] {
        fixture_geometry(true, 5, 4096, 2044, |head, ctx, gpu, seen| {
            let capture = gpu.alloc(2044 * 8192).unwrap();
            let bonus = gpu.alloc(8192).unwrap();
            gpu.copy_h2d(&vec![0x51; p * 8192], capture).unwrap();
            let mut state = head.alloc_state_inner(gpu).unwrap();
            hidden_trace::fixture_set_enabled(&mut state, true);
            state
                .hidden_trace
                .prompt
                .arm(
                    crate::model::glm_mtp_prompt_trace::fixture_capture(capture, 2044, p, 1, 0)
                        .unwrap(),
                )
                .unwrap();
            let tokens: Vec<_> = (0..=p).map(|i| (i % 8) as u32).collect();
            let allocations = gpu.alloc_count();
            assert_eq!(
                head.prefill_kv_batched(&tokens[..p], capture, &mut state, ctx, 37)
                    .unwrap(),
                p - 1
            );
            assert_eq!(state.seq_len, p - 1);
            assert_eq!(gpu.alloc_count(), allocations);
            let input = RepairInput {
                token: tokens[p],
                tokens: &tokens,
                prompt_len: p,
                position: p + 1,
                drafts: 4,
                generation: 1,
                capture_generation: 1,
                captured_rows: p,
                context_tokens: 2044,
                capture: RepairSpan {
                    ptr: capture,
                    bytes: 2044 * 8192,
                },
                normalized: RepairSpan {
                    ptr: ctx.buffers.norm_output(),
                    bytes: ctx.buffers.sizes().norm_output,
                },
                bonus: RepairSpan {
                    ptr: bonus,
                    bytes: 8192,
                },
                hidden_row: 0,
            };
            head.prepare(&input, &mut state, ctx, 37).unwrap();
            assert_eq!(state.seq_len, p);
            assert_eq!(seen.lock().len(), p);
            assert_eq!(gpu.alloc_count(), allocations);
            let hashes = hidden_trace::fixture_prompt_hashes(&state, 1, p).unwrap();
            assert_eq!(
                hashes[4],
                kv_reference(b"atlas/glm53/mtp-kv/prefix/v1\0", p - 1, p - 1)
            );
            assert_eq!(
                hashes[5],
                kv_reference(b"atlas/glm53/mtp-kv/appended/v1\0", p - 1, 1)
            );
            assert_eq!(
                hashes[6],
                kv_reference(b"atlas/glm53/mtp-kv/prefix/v1\0", p, p)
            );
            assert!(head.prepare(&input, &mut state, ctx, 37).is_err());
            head.free_state(gpu, None, &mut state).unwrap();
            assert!(!state.hidden_trace.prompt.active());
            assert!(hidden_trace::fixture_prompt_hashes(&state, 1, p).is_err());
            let long = vec![1; 257];
            assert_eq!(
                head.prefill_kv_batched(&long, capture, &mut state, ctx, 37)
                    .unwrap(),
                256
            );
            assert!(!state.hidden_trace.prompt.active());
            head.free_state(gpu, None, &mut state).unwrap();
        });
    }
}

#[test]
fn actual_writer_failures_poison_without_publishing_expected_rows() {
    for bootstrap in [false, true] {
        fixture_geometry(true, 5, 4096, 2044, |head, ctx, gpu, seen| {
            let capture = gpu.alloc(2044 * 8192).unwrap();
            let bonus = gpu.alloc(8192).unwrap();
            let mut state = head.alloc_state_inner(gpu).unwrap();
            let owner =
                crate::model::glm_mtp_prompt_trace::fixture_capture(capture, 2044, 17, 1, 0)
                    .unwrap();
            state.hidden_trace.prompt.arm(owner).unwrap();
            let tokens = vec![1; 18];
            if bootstrap {
                head.prefill_kv_batched(&tokens[..17], capture, &mut state, ctx, 37)
                    .unwrap();
            }
            seen.lock().push(-1);
            let result = if bootstrap {
                let input = RepairInput {
                    token: 1,
                    tokens: &tokens,
                    prompt_len: 17,
                    position: 18,
                    drafts: 4,
                    generation: 1,
                    capture_generation: 1,
                    captured_rows: 17,
                    context_tokens: 2044,
                    capture: RepairSpan {
                        ptr: capture,
                        bytes: 2044 * 8192,
                    },
                    normalized: RepairSpan {
                        ptr: ctx.buffers.norm_output(),
                        bytes: ctx.buffers.sizes().norm_output,
                    },
                    bonus: RepairSpan {
                        ptr: bonus,
                        bytes: 8192,
                    },
                    hidden_row: 0,
                };
                head.prepare(&input, &mut state, ctx, 37)
            } else {
                head.prefill_kv_batched(&tokens[..17], capture, &mut state, ctx, 37)
                    .map(|_| ())
            };
            assert!(
                format!("{:#}", result.unwrap_err()).contains("injected actual KV body failure")
            );
            assert_eq!(state.seq_len, if bootstrap { 16 } else { 0 });
            assert!(hidden_trace::fixture_prompt_hashes(&state, 1, 17).is_err());
            assert!(state.hidden_trace.prompt.arm(owner).is_err());
            head.free_state(gpu, None, &mut state).unwrap();
            assert!(!state.hidden_trace.prompt.active());
        });
    }
}
