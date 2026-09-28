// SPDX-License-Identifier: AGPL-3.0-only
//! Actual selected Model producer failures; fixture sentinels are not numerics.
use super::{fixture::*, isolated};
use crate::layers::glm5_mtp::Glm5MtpProposerState;
use crate::speculative::DraftProposer;
use crate::traits::{Model, SequenceState};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
enum Entry {
    Full,
    Chunk,
}
#[derive(Clone, Copy, Debug)]
enum Boundary {
    Target,
    Capture,
    Tail,
    Completion,
}
const PROMPT: [u32; 4] = [1, 2, 3, 4];

fn produce(f: &mut Fixture, entry: Entry, owner: usize, prompt: &[u32]) -> Result<DevicePtr> {
    match entry {
        Entry::Full => f.model.prefill(prompt, &mut f.seqs[owner], CALLER),
        Entry::Chunk => {
            f.model
                .prefill_chunk(prompt, &mut f.seqs[owner], 0, prompt.len(), true, CALLER)
        }
    }
}

fn private(seq: &SequenceState) -> &Glm5MtpProposerState {
    seq.proposer_state
        .as_ref()
        .unwrap()
        .as_any()
        .downcast_ref::<Glm5MtpProposerState>()
        .unwrap()
}

fn peer_first(rank: usize, entry: Entry) -> Fixture {
    let mut f = Fixture::new(rank);
    f.gpu.write_span(f.gpu.slab(), &vec![0xa5; SLAB_BYTES]);
    produce(&mut f, entry, 1, &[4, 3, 2, 1]).unwrap();
    assert_eq!(
        private(&f.seqs[1]).seq_len,
        3,
        "actual peer P-1 primer completed"
    );
    f.gpu.clear();
    f
}

fn unique(events: &[Event], label: &str, predicate: impl Fn(&Event) -> bool) -> usize {
    let found: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| predicate(event))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "actual {label} boundary must be unique; events={events:#?}"
    );
    found[0]
}

fn failure_ordinal(rank: usize, entry: Entry, boundary: Boundary) -> usize {
    let mut f = peer_first(rank, entry);
    // The control executes the real cold complete API, including all preflight
    // guards and P-1 writes. A stubbed/unsupported entry cannot supply an ordinal.
    produce(&mut f, entry, 0, &PROMPT).unwrap();
    assert_eq!(private(&f.seqs[0]).seq_len, 3);
    let events = f.gpu.trace();
    let capture = f.model.mtp_prefill_hidden;
    let slab = f.gpu.slab();
    let target = unique(&events, "target", |event| {
        matches!(event, Event::Target(4, 0, DEFAULT))
    });
    // GLM capture is the actual RMSNorm writer into the capture owner, not
    // the generic non-GLM D2D copy. The sentinel backend implements that
    // numerical boundary internally without manufacturing a Copy event.
    let captured = unique(&events, "GLM capture RMSNorm", |event| {
        matches!(event,
        Event::Kernel(name, ptrs, DEFAULT)
            if matches!(name.as_str(), "rms_norm_vanilla" | "rms_norm")
                && ptrs.as_slice() == [f.model.buffers.hidden_states(), f.model.final_norm.weight, capture])
    });
    let tail = unique(&events, "tail D2D", |event| {
        matches!(event,
        Event::Copy(src, dst, ROW_BYTES, DEFAULT)
            if *src == capture.offset(3 * ROW_BYTES) && *dst == slab)
    });
    assert!(target < captured && captured < tail);
    let expected: Vec<_> = PROMPT
        .iter()
        .enumerate()
        .flat_map(|(row, token)| vec![*token as u8 + 0x20 + row as u8; ROW_BYTES])
        .collect();
    assert_eq!(
        f.gpu.read_span(capture, 4 * ROW_BYTES),
        expected,
        "successful capture control must write all four sentinel rows"
    );
    assert_eq!(
        events.get(tail + 1),
        Some(&Event::Sync(DEFAULT)),
        "actual tail publication requires immediate default-stream completion"
    );
    assert_eq!(
        f.gpu.read_span(slab, ROW_BYTES),
        f.gpu.read_span(capture.offset(3 * ROW_BYTES), ROW_BYTES)
    );
    match boundary {
        Boundary::Target => target + 1,
        Boundary::Capture => captured + 1,
        Boundary::Tail => tail + 1,
        Boundary::Completion => tail + 2,
    }
}

fn assert_peer_unchanged(f: &Fixture, bytes: &[u8], blocks: &BTreeSet<u32>) {
    assert_eq!(
        f.gpu
            .read_span(f.gpu.slab().offset(6 * ROW_BYTES), 6 * ROW_BYTES),
        bytes
    );
    assert_eq!(private(&f.seqs[1]).seq_len, 3);
    assert_eq!(
        private(&f.seqs[1])
            .block_table
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        *blocks
    );
    assert_eq!(f.seqs[1].tokens, [4, 3, 2, 1]);
    assert_eq!(f.seqs[1].seq_len, 4);
}

fn boundary_is_terminal(boundary: Boundary) {
    for rank in 0..2 {
        for entry in [Entry::Full, Entry::Chunk] {
            let ordinal = failure_ordinal(rank, entry, boundary);
            let mut f = peer_first(rank, entry);
            let slab = f.gpu.slab();
            let peer_bytes = f.gpu.read_span(slab.offset(6 * ROW_BYTES), 6 * ROW_BYTES);
            let peer_blocks: BTreeSet<_> =
                private(&f.seqs[1]).block_table.iter().copied().collect();
            let failed_blocks: BTreeSet<_> =
                private(&f.seqs[0]).block_table.iter().copied().collect();
            assert_eq!((peer_blocks.len(), failed_blocks.len()), (128, 128));
            assert!(peer_blocks.is_disjoint(&failed_blocks));
            f.gpu.fail.store(ordinal, Ordering::Relaxed);
            let error = produce(&mut f, entry, 0, &PROMPT).unwrap_err();
            assert!(
                format!("{error:#}").contains("injected fixture operation failure"),
                "must reach actual injected {boundary:?}/{entry:?}/rank{rank}, got {error:#}"
            );
            assert!(
                f.gpu.trace().len() >= ordinal,
                "failure boundary actually executed"
            );
            assert_peer_unchanged(&f, &peer_bytes, &peer_blocks);
            // In the completion case bytes may already have been copied into
            // row0: bytes present is NOT a successfully published hidden view.
            f.gpu.clear();
            f.model.gpu.synchronize(DEFAULT).unwrap();
            f.gpu.clear();
            assert!(
                produce(&mut f, entry, 0, &PROMPT).is_err(),
                "later completion cannot permit retry after {boundary:?}/{entry:?}/rank{rank}"
            );
            assert!(
                !f.gpu.trace().iter().any(|event| matches!(
                    event,
                    Event::Target(..) | Event::Body(..) | Event::Kv(..) | Event::Copy(..)
                )),
                "failed request retry must reject before target/capture/primer/publication work"
            );
            assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
            assert_peer_unchanged(&f, &peer_bytes, &peer_blocks);
            // Retiring only the healthy peer yields precisely one reusable
            // reserve. This catches failed-slot reuse and duplicate returns.
            f.model.free_sequence(&mut f.seqs[1]).unwrap();
            let mut replacement = f.head.alloc_state(f.model.gpu.as_ref()).unwrap();
            let blocks: BTreeSet<_> = replacement
                .as_any()
                .downcast_ref::<Glm5MtpProposerState>()
                .unwrap()
                .block_table
                .iter()
                .copied()
                .collect();
            assert_eq!(blocks, peer_blocks);
            assert!(blocks.is_disjoint(&failed_blocks));
            assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
            f.head
                .free_state(f.model.gpu.as_ref(), None, replacement.as_mut())
                .unwrap();
            // Healthy-owner control above precedes the separately terminal
            // Model cleanup error; that error now closes the whole session.
            let free = f.head.paired_test_free_blocks();
            f.gpu.clear();
            assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
            assert!(f.model.alloc_sequence().is_err());
            assert!(f.gpu.trace().is_empty());
            assert_eq!(f.head.paired_test_free_blocks(), free);
            assert!(!f.model.ssm_pool.slot_is_free(0));
            assert!(
                !f.gpu
                    .trace()
                    .iter()
                    .any(|event| matches!(event, Event::Free(ptr) if *ptr == slab))
            );
        }
    }
}

#[test]
fn actual_target_failure_poisoned_before_model_prefill_returns() {
    if isolated("writer_fault_tests::actual_target_failure_poisoned_before_model_prefill_returns") {
        return;
    }
    boundary_is_terminal(Boundary::Target);
}

#[test]
fn actual_capture_norm_failure_poisoned_before_model_prefill_returns() {
    if isolated(
        "writer_fault_tests::actual_capture_norm_failure_poisoned_before_model_prefill_returns",
    ) {
        return;
    }
    boundary_is_terminal(Boundary::Capture);
}

#[test]
fn actual_tail_copy_failure_poisoned_before_model_prefill_returns() {
    if isolated(
        "writer_fault_tests::actual_tail_copy_failure_poisoned_before_model_prefill_returns",
    ) {
        return;
    }
    boundary_is_terminal(Boundary::Tail);
}

#[test]
fn actual_tail_completion_failure_poisoned_before_model_prefill_returns() {
    if isolated(
        "writer_fault_tests::actual_tail_completion_failure_poisoned_before_model_prefill_returns",
    ) {
        return;
    }
    boundary_is_terminal(Boundary::Completion);
}
