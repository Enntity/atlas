// SPDX-License-Identifier: AGPL-3.0-only
//! Actual K5/detachment faults, controlled at real recorder operations.
//! Byte sentinels prove dense ownership and refusal, not CUDA completion/numerics.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::speculative::DraftProposer;
use crate::traits::Model;
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Target,
    Normalization,
    Readback,
    Accepted,
    Bonus,
    Completion,
}
impl Boundary {
    fn verification(self) -> bool {
        matches!(self, Self::Target | Self::Normalization | Self::Readback)
    }
    fn event(self, f: &Fixture, owner: usize, h: &flow::History, accepted: usize) -> Event {
        let normalized = f.model.buffers.norm_output();
        let owned = f.gpu.slab().offset(owner * 6 * ROW_BYTES);
        match self {
            Self::Target => Event::Target(1, h.base, DEFAULT),
            Self::Normalization => Event::Kernel(
                "rms_norm_vanilla".into(),
                vec![
                    f.model.buffers.hidden_states(),
                    f.model.final_norm.weight,
                    normalized,
                ],
                DEFAULT,
            ),
            Self::Readback => Event::Read(f.model.buffers.scratch(), 20, DEFAULT),
            Self::Accepted => {
                assert_eq!(accepted, 4, "a=0 has no accepted-row copy to fault");
                Event::Copy(
                    normalized,
                    owned.offset(ROW_BYTES),
                    accepted * ROW_BYTES,
                    DEFAULT,
                )
            }
            Self::Bonus => Event::Copy(
                normalized.offset(accepted * ROW_BYTES),
                owned.offset(5 * ROW_BYTES),
                ROW_BYTES,
                DEFAULT,
            ),
            Self::Completion => Event::Sync(DEFAULT),
        }
    }
}

fn verify(f: &mut Fixture, owner: usize, h: &flow::History) -> Result<()> {
    let predictions =
        f.model
            .decode_verify_graphed_kgamma(&h.issued, &mut f.seqs[owner], CALLER)?;
    let rows = flow::normalized(h);
    assert_eq!(
        predictions,
        rows.iter()
            .map(|row| u32::from(row[0] % 8))
            .collect::<Vec<_>>()
    );
    assert_eq!(f.seqs[owner].seq_len, h.base + 5);
    assert_eq!(&f.seqs[owner].tokens[h.base..], h.issued);
    assert_eq!(
        f.gpu
            .read_span(f.model.buffers.norm_output(), 5 * ROW_BYTES),
        rows.concat()
    );
    Ok(())
}

fn rollback(f: &mut Fixture, owner: usize, h: &flow::History, accepted: usize) {
    f.seqs[owner].seq_len = h.base + accepted + 1;
    f.seqs[owner].tokens.truncate(h.base + accepted + 1);
}

fn setup(
    rank: usize,
    owner: usize,
    accepted: usize,
    boundary: Boundary,
) -> (Fixture, [flow::History; 2]) {
    let (mut f, histories) = flow::prepare(rank, [1 - owner, owner]);
    if !boundary.verification() {
        verify(&mut f, owner, &histories[owner]).unwrap();
        rollback(&mut f, owner, &histories[owner], accepted);
    }
    f.gpu.clear();
    (f, histories)
}

fn perform(
    f: &mut Fixture,
    owner: usize,
    h: &flow::History,
    accepted: usize,
    boundary: Boundary,
) -> Result<()> {
    if boundary.verification() {
        verify(f, owner, h)
    } else {
        f.model
            .record_glm_mtp_verified(&mut f.seqs[owner], h.base, &h.issued, accepted)
    }
}

fn unique(events: &[Event], event: &Event) -> usize {
    let indices: Vec<_> = events
        .iter()
        .enumerate()
        .filter_map(|(index, found)| (found == event).then_some(index))
        .collect();
    assert_eq!(
        indices.len(),
        1,
        "positive control boundary {event:?}; trace={events:?}"
    );
    indices[0]
}

fn control(rank: usize, owner: usize, accepted: usize, boundary: Boundary) -> usize {
    let (mut f, mut histories) = setup(rank, owner, accepted, boundary);
    let expected = boundary.event(&f, owner, &histories[owner], accepted);
    perform(&mut f, owner, &histories[owner], accepted, boundary).unwrap();
    let trace = f.gpu.trace();
    let ordinal = unique(&trace, &expected) + 1;
    if boundary.verification() {
        let target = unique(
            &trace,
            &Boundary::Target.event(&f, owner, &histories[owner], accepted),
        );
        let norm = unique(
            &trace,
            &Boundary::Normalization.event(&f, owner, &histories[owner], accepted),
        );
        let read = unique(
            &trace,
            &Boundary::Readback.event(&f, owner, &histories[owner], accepted),
        );
        assert!(target < norm && norm < read);
        rollback(&mut f, owner, &histories[owner], accepted);
        f.model
            .record_glm_mtp_verified(
                &mut f.seqs[owner],
                histories[owner].base,
                &histories[owner].issued,
                accepted,
            )
            .unwrap();
    } else {
        let bonus = unique(
            &trace,
            &Boundary::Bonus.event(&f, owner, &histories[owner], accepted),
        );
        assert_eq!(trace.get(bonus + 1), Some(&Event::Sync(DEFAULT)));
        assert_eq!(trace.len(), if accepted == 0 { 2 } else { 3 });
        if accepted > 0 {
            assert_eq!(
                trace[0],
                Boundary::Accepted.event(&f, owner, &histories[owner], accepted)
            );
        }
    }
    flow::detached(&f, owner, &histories[owner], accepted);
    flow::acknowledge(&mut f, owner, accepted, owner == 0);
    flow::continue_owner(&mut f, owner, &mut histories[owner], accepted);
    ordinal
}

struct Peer {
    tokens: Vec<u32>,
    target: usize,
    private: usize,
    blocks: BTreeSet<u32>,
    kv: Vec<Vec<u8>>,
    pointers: Vec<(DevicePtr, DevicePtr)>,
    slab: Vec<u8>,
}
impl Peer {
    fn snapshot(f: &Fixture, owner: usize) -> Self {
        let state = flow::private(&f.seqs[owner]);
        Self {
            tokens: f.seqs[owner].tokens.clone(),
            target: f.seqs[owner].seq_len,
            private: state.seq_len,
            blocks: state.block_table.iter().copied().collect(),
            kv: flow::bytes(f, owner, state.seq_len),
            pointers: f
                .head
                .paired_test_kv_rows(
                    f.seqs[owner].proposer_state.as_ref().unwrap().as_ref(),
                    f.model.gpu.as_ref(),
                    state.seq_len,
                )
                .unwrap(),
            slab: f
                .gpu
                .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
        }
    }
    fn unchanged(&self, f: &Fixture, owner: usize) {
        let state = flow::private(&f.seqs[owner]);
        assert_eq!(f.seqs[owner].tokens, self.tokens);
        assert_eq!(f.seqs[owner].seq_len, self.target);
        assert_eq!(state.seq_len, self.private);
        assert_eq!(
            state.block_table.iter().copied().collect::<BTreeSet<_>>(),
            self.blocks
        );
        // Saved actual canonical addresses remain a byte oracle after Model
        // cleanup terminally revokes both leases; do not request new authority.
        for ((k, v), expected) in self.pointers.iter().zip(&self.kv) {
            assert_eq!(f.gpu.read_span(*k, 1024), *expected);
            assert_eq!(f.gpu.read_span(*v, 1024), *expected);
        }
        assert_eq!(
            f.gpu
                .read_span(f.gpu.slab().offset(owner * 6 * ROW_BYTES), 6 * ROW_BYTES),
            self.slab
        );
    }
}

fn terminal(f: &mut Fixture, histories: &[flow::History; 2], owner: usize, accepted: usize) {
    f.gpu.fail.store(usize::MAX, Ordering::Relaxed);
    f.model.gpu.synchronize(DEFAULT).unwrap(); // Completion now succeeds; failure must remain sticky.
    f.gpu.clear();
    for who in [owner, 1 - owner] {
        assert!(
            f.model
                .decode_verify_graphed_kgamma(&histories[who].issued, &mut f.seqs[who], CALLER)
                .is_err()
        );
        assert!(
            f.model
                .run_mtp_propose_inner(
                    histories[who].issued[0],
                    f.seqs[who].seq_len,
                    4,
                    &mut f.seqs[who],
                    None
                )
                .is_err()
        );
        assert!(f.model.decode(1, &mut f.seqs[who], CALLER).is_err());
    }
    assert!(
        f.model
            .record_glm_mtp_verified(
                &mut f.seqs[owner],
                histories[owner].base,
                &histories[owner].issued,
                accepted
            )
            .is_err()
    );
    assert!(
        f.model
            .trim_proposer_state(&mut f.seqs[owner], accepted, 0)
            .is_err()
    );
    assert!(
        f.model
            .commit_accepted_prefix(&mut f.seqs[owner], accepted + 1, 5)
            .is_err()
    );
    assert!(
        f.gpu.trace().is_empty(),
        "failed active transaction cannot authorize retry or peer producer"
    );
    assert!(f.model.free_sequence(&mut f.seqs[owner]).is_err());
    assert!(
        f.head.alloc_state(f.model.gpu.as_ref()).is_err(),
        "failed reserve must not become reusable"
    );
}

fn fault(rank: usize, owner: usize, accepted: usize, boundary: Boundary) {
    let ordinal = control(rank, owner, accepted, boundary);
    let (mut f, histories) = setup(rank, owner, accepted, boundary);
    let expected = boundary.event(&f, owner, &histories[owner], accepted);
    let peer = Peer::snapshot(&f, 1 - owner);
    f.gpu.fail.store(ordinal, Ordering::Relaxed);
    let error = perform(&mut f, owner, &histories[owner], accepted, boundary).unwrap_err();
    assert!(
        format!("{error:#}").contains("injected fixture operation failure"),
        "preflight/control error is not an injected boundary failure: {error:#}"
    );
    assert_eq!(
        f.gpu.trace().get(ordinal - 1),
        Some(&expected),
        "fresh fixture must fail at the same exact semantic event, with its own pointers"
    );
    peer.unchanged(&f, 1 - owner);
    terminal(&mut f, &histories, owner, accepted);
    peer.unchanged(&f, 1 - owner);
}

#[test]
fn actual_k5_target_normalization_readback_faults_are_terminal() {
    if flow::isolated(
        "verdict_fault_tests::actual_k5_target_normalization_readback_faults_are_terminal",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for accepted in [0, 4] {
                for boundary in [
                    Boundary::Target,
                    Boundary::Normalization,
                    Boundary::Readback,
                ] {
                    fault(rank, owner, accepted, boundary);
                }
            }
        }
    }
}

#[test]
fn actual_verdict_accepted_bonus_completion_faults_are_terminal() {
    if flow::isolated(
        "verdict_fault_tests::actual_verdict_accepted_bonus_completion_faults_are_terminal",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for accepted in [0, 4] {
                for boundary in [Boundary::Accepted, Boundary::Bonus, Boundary::Completion] {
                    if accepted == 0 && matches!(boundary, Boundary::Accepted) {
                        continue;
                    }
                    fault(rank, owner, accepted, boundary);
                }
            }
        }
    }
}
