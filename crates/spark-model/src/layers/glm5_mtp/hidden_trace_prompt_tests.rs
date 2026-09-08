// SPDX-License-Identifier: AGPL-3.0-only
//! Real source bytes and real checked KV readers, not numerical simulation.
use super::*;
const SIDE_ROW: usize = 1024;
const SIDE_BLOCK: usize = 16384;
#[path = "hidden_trace_kv_test_gpu.rs"]
mod support;
use std::sync::atomic::Ordering;
use support::{Gpu, cache, context};

#[path = "hidden_trace_prompt_owner_tests.rs"]
mod owner_tests;

fn source_reference(domain: &[u8], index: usize, bytes: &[u8]) -> [u8; 32] {
    let mut raw = domain.to_vec();
    raw.extend_from_slice(&(index as u64).to_le_bytes());
    raw.extend_from_slice(&4096u32.to_le_bytes());
    raw.extend_from_slice(&2u32.to_le_bytes());
    raw.extend_from_slice(bytes);
    Sha256::digest(raw).into()
}
fn token_reference(domain: &[u8], index: usize, tokens: &[u32]) -> [u8; 32] {
    let mut raw = domain.to_vec();
    raw.extend_from_slice(&(index as u64).to_le_bytes());
    for t in tokens {
        raw.extend_from_slice(&t.to_le_bytes());
    }
    Sha256::digest(raw).into()
}

#[test]
fn real_prompt_phase_reads_sources_and_composes_same_prefix_as_body_probe() {
    for p in [2, 17, 33, 129, 148, 256] {
        let gpu = Gpu::new();
        context(&gpu, |ctx| {
            let (cache, blocks) = cache(&gpu, p, true);
            let bytes: Vec<_> = (0..p * 8192).map(|i| (i ^ (i >> 8)) as u8).collect();
            let source = gpu.inner.alloc(bytes.len()).unwrap();
            gpu.inner.copy_h2d(&bytes, source).unwrap();
            let capture =
                crate::model::glm_mtp_prompt_trace::fixture_capture(source, p, p, 1, 0).unwrap();
            let mut phase = Prompt::default();
            phase.arm(capture).unwrap();
            assert!(phase.active());
            let tokens: Vec<_> = (0..=p).map(|i| (i % 8) as u32).collect();
            phase
                .primer_before(
                    &tokens[1..p],
                    DeviceSpan {
                        ptr: source,
                        bytes: (p - 1) * 8192,
                    },
                    ctx,
                    7,
                )
                .unwrap();
            phase
                .primer_after(&cache, &blocks[..(p - 1).div_ceil(16)], ctx, 7)
                .unwrap();
            let input = crate::speculative::glm_repair::RepairInput {
                token: 7,
                tokens: &tokens,
                prompt_len: p,
                position: p + 1,
                drafts: 4,
                generation: 1,
                capture_generation: 1,
                captured_rows: p,
                context_tokens: 2044,
                capture: crate::speculative::glm_repair::RepairSpan {
                    ptr: source,
                    bytes: p * 8192,
                },
                normalized: crate::speculative::glm_repair::RepairSpan {
                    ptr: ctx.buffers.norm_output(),
                    bytes: ctx.buffers.sizes().norm_output,
                },
                bonus: crate::speculative::glm_repair::RepairSpan {
                    ptr: ctx.buffers.hidden_states(),
                    bytes: 8192,
                },
                hidden_row: 0,
            };
            phase
                .bootstrap_before(
                    &input,
                    &tokens[p..],
                    DeviceSpan {
                        ptr: source.offset((p - 1) * 8192),
                        bytes: 8192,
                    },
                    p - 1,
                    ctx,
                    7,
                )
                .unwrap();
            phase.bootstrap_after(&cache, &blocks, ctx, 7).unwrap();
            let e = phase.evidence(0, 1, p).unwrap();
            assert_eq!(
                e.primer_source,
                source_reference(
                    b"atlas/glm53/mtp-source/primer/v1\0",
                    p,
                    &bytes[..(p - 1) * 8192]
                )
            );
            assert_eq!(
                e.bootstrap_source,
                source_reference(
                    b"atlas/glm53/mtp-source/bootstrap/v1\0",
                    p - 1,
                    &bytes[(p - 1) * 8192..]
                )
            );
            assert_eq!(
                e.primer_tokens,
                token_reference(
                    b"atlas/glm53/mtp-source/primer-tokens/v1\0",
                    p - 1,
                    &tokens[1..p]
                )
            );
            assert_eq!(
                e.bootstrap_token,
                token_reference(
                    b"atlas/glm53/mtp-source/bootstrap-token/v1\0",
                    p,
                    &tokens[p..]
                )
            );
            assert!(
                [
                    e.primer_source,
                    e.bootstrap_source,
                    e.primer_tokens,
                    e.bootstrap_token,
                    e.primer_kv,
                    e.bootstrap_kv,
                    e.written_prefix
                ]
                .iter()
                .all(|v| *v != [0; 32])
            );
            let bytes_read: usize = gpu
                .events
                .lock()
                .iter()
                .map(|e| match e {
                    support::Event::Read(_, n, 7) => *n,
                    _ => panic!("unexpected side effect {e:?}"),
                })
                .sum();
            assert_eq!(bytes_read, p * 10240);
            assert!(phase.scratch.is_none() && phase.continuation.is_none());
            let body = super::super::kv::Probe::before(&cache, &blocks, p, ctx, 7).unwrap();
            assert_eq!(e.written_prefix, body.prefix);
        });
    }
}

fn fixture(
    p: usize,
    run: impl FnOnce(&Gpu, &ForwardContext, &PagedKvCache, &[u32], DevicePtr, &[u32], &mut Prompt),
) {
    let gpu = Gpu::new();
    context(&gpu, |ctx| {
        let (cache, blocks) = cache(&gpu, p, true);
        let source = gpu.inner.alloc(p * 8192).unwrap();
        gpu.inner.copy_h2d(&vec![0x51; p * 8192], source).unwrap();
        let tokens: Vec<_> = (0..=p).map(|i| (i % 8) as u32).collect();
        let mut phase = Prompt::default();
        phase
            .arm(crate::model::glm_mtp_prompt_trace::fixture_capture(source, p, p, 1, 0).unwrap())
            .unwrap();
        run(&gpu, ctx, &cache, &blocks, source, &tokens, &mut phase);
    });
}
fn bootstrap(
    phase: &mut Prompt,
    p: usize,
    source: DevicePtr,
    tokens: &[u32],
    ctx: &ForwardContext,
) -> Result<()> {
    let span = crate::speculative::glm_repair::RepairSpan {
        ptr: source,
        bytes: p * 8192,
    };
    let input = crate::speculative::glm_repair::RepairInput {
        token: tokens[p],
        tokens,
        prompt_len: p,
        position: p + 1,
        drafts: 4,
        generation: 1,
        capture_generation: 1,
        captured_rows: p,
        context_tokens: 2044,
        capture: span,
        normalized: span,
        bonus: span,
        hidden_row: 0,
    };
    phase.bootstrap_before(
        &input,
        &tokens[p..],
        DeviceSpan {
            ptr: source.offset((p - 1) * 8192),
            bytes: 8192,
        },
        p - 1,
        ctx,
        7,
    )
}
fn observe(
    phase: &mut Prompt,
    p: usize,
    source: DevicePtr,
    tokens: &[u32],
    cache: &PagedKvCache,
    blocks: &[u32],
    ctx: &ForwardContext,
) -> Result<()> {
    phase.primer_before(
        &tokens[1..p],
        DeviceSpan {
            ptr: source,
            bytes: (p - 1) * 8192,
        },
        ctx,
        7,
    )?;
    phase.primer_after(cache, blocks, ctx, 7)?;
    bootstrap(phase, p, source, tokens, ctx)?;
    phase.bootstrap_after(cache, blocks, ctx, 7)
}

#[test]
fn every_source_and_immediate_writer_copy_failure_is_spent_without_replacement() {
    fixture(256, |gpu, ctx, cache, blocks, source, tokens, _| {
        // 64 primer-source chunks +32 primer K/V reads +1 source +2 row reads.
        for fault in 1..=99 {
            let mut phase = Prompt::default();
            let capture =
                crate::model::glm_mtp_prompt_trace::fixture_capture(source, 256, 256, 1, 0)
                    .unwrap();
            phase.arm(capture).unwrap();
            gpu.events.lock().clear();
            gpu.fail.store(fault, Ordering::Relaxed);
            assert!(
                observe(&mut phase, 256, source, tokens, cache, blocks, ctx).is_err(),
                "fault {fault}"
            );
            assert_eq!(gpu.events.lock().len(), fault);
            assert!(
                gpu.events
                    .lock()
                    .iter()
                    .all(|e| matches!(e, support::Event::Read(_, _, 7)))
            );
            assert!(phase.evidence(0, 1, 256).is_err());
            assert!(phase.arm(capture).is_err());
            assert!(
                phase
                    .primer_before(
                        &tokens[1..256],
                        DeviceSpan {
                            ptr: source,
                            bytes: 255 * 8192
                        },
                        ctx,
                        7
                    )
                    .is_err()
            );
            assert_eq!(gpu.events.lock().len(), fault);
            phase.fail();
            assert!(phase.scratch.is_none() && phase.continuation.is_none());
        }
    });
}

#[test]
fn composed_observation_detects_mutation_between_writer_and_first_body() {
    for mutate_before_bootstrap in [true, false] {
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
            if mutate_before_bootstrap {
                gpu.inner
                    .copy_h2d(
                        &[0xee],
                        cache.k_pool_ptr(0).offset(blocks[0] as usize * 16384),
                    )
                    .unwrap();
            }
            bootstrap(phase, 17, source, tokens, ctx).unwrap();
            phase.bootstrap_after(cache, blocks, ctx, 7).unwrap();
            if !mutate_before_bootstrap {
                gpu.inner
                    .copy_h2d(
                        &[0xee],
                        cache.v_pool_ptr(0).offset(blocks[1] as usize * 16384),
                    )
                    .unwrap();
            }
            assert_ne!(
                phase.evidence(0, 1, 17).unwrap().written_prefix,
                kv::Probe::before(cache, blocks, 17, ctx, 7).unwrap().prefix
            );
        });
    }
}

#[test]
fn actual_prompt_owner_phase_capture_and_stream_fail_before_reads() {
    for fault in 0..8 {
        fixture(17, |gpu, ctx, cache, blocks, source, tokens, phase| {
            let span = DeviceSpan {
                ptr: source,
                bytes: 16 * 8192,
            };
            let result = match fault {
                0 => phase.primer_before(&tokens[1..16], span, ctx, 7),
                1 => phase.primer_before(
                    &tokens[1..17],
                    DeviceSpan {
                        ptr: source.offset(2),
                        ..span
                    },
                    ctx,
                    7,
                ),
                2 => phase.primer_after(cache, blocks, ctx, 7),
                3 => bootstrap(phase, 17, source, tokens, ctx),
                4 => {
                    gpu.capturing.store(true, Ordering::Relaxed);
                    phase.primer_before(&tokens[1..17], span, ctx, 7)
                }
                5 => {
                    let bad = ForwardContext {
                        graph_capture: true,
                        midchunk_capture: None,
                        ..*ctx
                    };
                    phase.primer_before(&tokens[1..17], span, &bad, 7)
                }
                6 => {
                    phase.primer_before(&tokens[1..17], span, ctx, 7).unwrap();
                    gpu.events.lock().clear();
                    phase.primer_after(cache, blocks, ctx, 8)
                }
                _ => {
                    phase.fail();
                    phase.primer_before(&tokens[1..17], span, ctx, 7)
                }
            };
            assert!(result.is_err(), "fault {fault}");
            assert!(gpu.events.lock().is_empty());
            assert!(phase.evidence(0, 1, 17).is_err());
        });
    }
}

#[test]
fn actual_prefix_owner_and_logical_map_are_bound_across_writer_and_body() {
    for fault in 0..4 {
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
            bootstrap(phase, 17, source, tokens, ctx).unwrap();
            let (foreign, _) = support::cache(gpu, 17, false);
            let mut changed = blocks.to_vec();
            changed.swap(0, 1);
            if fault < 2 {
                gpu.events.lock().clear();
                assert!(
                    phase
                        .bootstrap_after(
                            if fault == 0 { &foreign } else { cache },
                            if fault == 1 { &changed } else { blocks },
                            ctx,
                            7
                        )
                        .is_err()
                );
            } else {
                phase.bootstrap_after(cache, blocks, ctx, 7).unwrap();
                gpu.events.lock().clear();
                assert!(
                    phase
                        .validate_body(
                            if fault == 2 { &foreign } else { cache },
                            if fault == 3 { &changed } else { blocks },
                            ctx,
                            7
                        )
                        .is_err()
                );
            }
            assert!(gpu.events.lock().is_empty());
        });
    }
}
