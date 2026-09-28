// SPDX-License-Identifier: AGPL-3.0-only
//! Flow helpers stay unit-test-only; the command recorder has one shared implementation.
use super::{fixture::*, verdict_continuation_tests as flow};
pub(crate) use crate::model::glm_c2_test_support::wire::Wire;
use crate::traits::{Model, SequenceState};
use anyhow::Result;
use std::sync::atomic::Ordering;

pub(super) fn bootstrapped(rank: usize, order: [usize; 2]) -> Fixture {
    let mut f = Fixture::new(rank);
    f.gpu.deterministic_logits.store(true, Ordering::Relaxed);
    let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
    for owner in order {
        f.seqs[owner].prompt_len = prompts[owner].len();
        f.model
            .prefill(&prompts[owner], &mut f.seqs[owner], CALLER)
            .unwrap();
        f.model
            .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
            .unwrap();
    }
    f.gpu.clear();
    f
}

pub(super) fn worker(f: &mut Fixture) -> Result<bool> {
    let mut slots =
        std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only)).map(Some);
    let result = f.model.ep_worker_step(&mut slots);
    f.seqs = slots.map(Option::unwrap);
    result
}

pub(super) fn same_private(a: &Fixture, b: &Fixture, owner: usize) {
    let rows = flow::private(&a.seqs[owner]).seq_len;
    assert_eq!(flow::private(&b.seqs[owner]).seq_len, rows);
    assert_eq!(flow::bytes(a, owner, rows), flow::bytes(b, owner, rows));
    for row in 0..6 {
        assert_eq!(flow::slab(a, owner, row), flow::slab(b, owner, row));
    }
}
