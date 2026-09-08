// SPDX-License-Identifier: AGPL-3.0-only
//! Actual command-buffer bytes, replayed locally; not collective agreement.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::traits::{Model, SequenceState};
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::DevicePtr;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct Wire {
    rank: usize,
    gpu: Arc<Recorder>,
    pending: Mutex<VecDeque<Vec<u32>>>,
    pub seen: Mutex<Vec<Vec<u32>>>,
    pub fail: AtomicUsize,
}
impl Wire {
    pub fn install(f: &mut Fixture, rank: usize) -> Arc<Self> {
        let wire = Arc::new(Self {
            rank,
            gpu: f.gpu.clone(),
            pending: Mutex::new(VecDeque::new()),
            seen: Mutex::new(Vec::new()),
            fail: AtomicUsize::new(0),
        });
        f.model.comm = Some(wire.clone());
        f.model.ep_protocol_v2 = true;
        wire
    }
    pub fn clear(&self) {
        assert!(self.pending.lock().is_empty());
        self.seen.lock().clear();
        self.fail.store(0, Ordering::Relaxed);
    }
    pub fn queue(&self, packets: &[Vec<u32>]) {
        self.clear();
        self.pending.lock().extend(packets.iter().cloned());
    }
    pub fn packets(&self) -> Vec<Vec<u32>> {
        self.seen.lock().clone()
    }
    pub fn done(&self) {
        assert!(self.pending.lock().is_empty());
    }
    fn transfer(&self, pointer: u64, bytes: usize, root: usize) -> Result<()> {
        ensure!(root == 0 && bytes % 4 == 0, "unexpected command boundary");
        let words = if self.rank == 0 {
            self.gpu
                .read_span(DevicePtr(pointer), bytes)
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        } else {
            let words = self.pending.lock().pop_front().expect("unscripted receive");
            ensure!(words.len() * 4 == bytes, "wrong received payload extent");
            words
        };
        self.seen.lock().push(words.clone());
        ensure!(
            self.seen.lock().len() != self.fail.load(Ordering::Relaxed),
            "injected command transfer failure"
        );
        if self.rank == 1 {
            let data: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            self.gpu.write_span(DevicePtr(pointer), &data);
        }
        Ok(())
    }
}
macro_rules! inert {
    ($($name:ident($($arg:ty),*));*) => {
        impl CommBackend for Wire {
            $(fn $name(&self, $(_: $arg),*) -> Result<()> { Ok(()) })*
            fn rank(&self) -> usize { self.rank }
            fn world_size(&self) -> usize { 2 }
            fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
                self.transfer(ptr, 4, 0)
            }
            fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
                self.transfer(ptr, bytes, root)
            }
            fn all_gather(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
                self.gpu.gather_sentinel(src, dst, bytes)
            }
        }
    };
}
inert! { all_reduce(u64, usize); reduce_scatter(u64, u64, usize); barrier();
send_to(u64, usize, usize, u64); recv_from(u64, usize, usize, u64) }

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
