// SPDX-License-Identifier: AGPL-3.0-only
//! ATLAS_GLM_LAYER_FORK=index through the actual GLM paged prefill: a dense
//! owner's semantic-index maintenance leaves the compute stream for the side
//! stream between the KV cache write and the W_uk absorb, unchanged, and
//! shares no buffer with the compute launches it overlaps; a selecting owner
//! keeps the in-line order. Recorded launches, not CUDA numerics.
use super::write_floor_tests::{PagedRun, paged_run};
use super::*;
use crate::layers::glm_layer_fork::ForkLane;

const LATENT_WRITE: u64 = 821;
const POOL_FINALIZE: u64 = 802;
const STREAM: u64 = 0;
const LANE: ForkLane = ForkLane {
    side: 5,
    fork: 31,
    join: 32,
};

#[derive(Clone, Debug, PartialEq)]
enum Step {
    Launch(u64, u64, Vec<Vec<u8>>),
    Fence(String),
}

/// One paged run's launches (stream, kernel, arguments) and fences in call
/// order, and the compute stream's absorb output (`ssm_deinterleaved`).
fn steps(
    dtype: KvCacheDtype,
    seq_len_start: usize,
    lane: Option<ForkLane>,
) -> (Vec<Step>, Vec<u8>) {
    let edit = |layer: &mut Qwen3AttentionLayer| layer.index_fork = lane;
    paged_run(dtype, seq_len_start, 3, 0, edit, |run: PagedRun<'_>| {
        let typed = run.gpu.1.lock().unwrap()[run.typed_before..].to_vec();
        let mut launches = typed.into_iter();
        let steps = run.gpu.2.lock().unwrap()[run.log_before..]
            .iter()
            .map(|entry| match entry.strip_prefix('L') {
                Some(stream) => {
                    let (kernel, args) = launches.next().unwrap();
                    Step::Launch(stream.parse().unwrap(), kernel, args)
                }
                // Untyped launches and fences, as logged.
                None => Step::Fence(entry.clone()),
            })
            .collect();
        (
            steps,
            run.arena.ssm_deinterleaved().0.to_ne_bytes().to_vec(),
        )
    })
    .unwrap()
}

fn on(steps: &[Step], stream: u64) -> Vec<Step> {
    steps
        .iter()
        .filter(|s| matches!(s, Step::Launch(at, ..) if *at == stream))
        .cloned()
        .collect()
}

/// The 8-byte arguments (pointers) of `launches`.
fn pointers(launches: &[Step]) -> std::collections::BTreeSet<Vec<u8>> {
    launches
        .iter()
        .flat_map(|s| match s {
            Step::Launch(_, _, args) => args.iter().filter(|a| a.len() == 8).cloned().collect(),
            Step::Fence(_) => Vec::new(),
        })
        .collect()
}

#[test]
fn a_dense_owner_forks_its_index_maintenance_beside_q_b() {
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8G128] {
        let (off, absorbed) = steps(dtype, 100, None);
        let (fork, _) = steps(dtype, 100, Some(LANE));
        assert!(off.iter().all(|s| match s {
            Step::Launch(stream, ..) => *stream == STREAM,
            Step::Fence(f) => f.starts_with(&format!("U{STREAM} ")),
        }));
        // Off, the index block runs right after the latent write, through the
        // pool finalize.
        let pos = |steps: &[Step], kernel: u64| {
            steps
                .iter()
                .position(|s| matches!(s, Step::Launch(_, k, _) if *k == kernel))
                .unwrap()
        };
        let (write, finalize) = (pos(&off, LATENT_WRITE), pos(&off, POOL_FINALIZE));
        let block: Vec<Step> = off[write + 1..=finalize]
            .iter()
            .map(|s| match s {
                Step::Launch(_, k, a) => Step::Launch(LANE.side, *k, a.clone()),
                Step::Fence(f) => panic!("{dtype:?}: untyped {f} in the index block"),
            })
            .collect();
        assert!(
            block.len() >= 4,
            "{dtype:?}: key / gate projections, norm, writes"
        );
        // The first compute launch after it that writes the absorbed queries.
        let absorb = finalize
            + 1
            + off[finalize + 1..]
                .iter()
                .position(|s| matches!(s, Step::Launch(_, _, a) if a.contains(&absorbed)))
                .unwrap();
        let window = &off[finalize + 1..absorb];
        assert!(!window.is_empty(), "{dtype:?}: q_b runs beside the index");
        // Forked: fence, the block on the side stream, the window, fence,
        // the absorb and the rest — every launch otherwise unchanged.
        let fence = |s: &str| Step::Fence(s.to_owned());
        let mut want = off[..=write].to_vec();
        want.extend([fence("R0 31"), fence("W5 31")]);
        want.extend(block.iter().cloned());
        want.extend(window.iter().cloned());
        want.extend([fence("R5 32"), fence("W0 32")]);
        want.extend(off[absorb..].iter().cloned());
        assert_eq!(fork, want, "{dtype:?}");
        // The two chains share no buffer.
        let shared: Vec<_> = pointers(&block)
            .intersection(&pointers(window))
            .cloned()
            .collect();
        assert!(shared.is_empty(), "{dtype:?}: {shared:?}");
        assert_eq!(on(&fork, LANE.side), block);
    }
}

#[test]
fn a_selecting_owner_keeps_the_in_line_order() {
    // Rows end past index_topk (2048): the selection reads the index. BF16:
    // the fp8_g128 sparse reader is an env-gated kernel.
    let (off, _) = steps(KvCacheDtype::Bf16, 2100, None);
    assert_eq!(steps(KvCacheDtype::Bf16, 2100, Some(LANE)).0, off);
}
