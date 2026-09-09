// SPDX-License-Identifier: AGPL-3.0-only
//! Actual command-buffer bytes, replayed locally; not collective agreement.
use super::fixture::*;
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::DevicePtr;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(crate) struct Wire {
    rank: usize,
    gpu: Arc<Recorder>,
    pending: Mutex<VecDeque<Vec<u32>>>,
    pub seen: Mutex<Vec<Vec<u32>>>,
    pub fail: AtomicUsize,
    cold_prefix: AtomicBool,
    roots: Mutex<Vec<usize>>,
}
impl Wire {
    pub fn install(f: &mut Fixture, rank: usize) -> Arc<Self> {
        let wire = Arc::new(Self {
            rank,
            gpu: f.gpu.clone(),
            pending: Mutex::new(VecDeque::new()),
            seen: Mutex::new(Vec::new()),
            fail: AtomicUsize::new(0),
            cold_prefix: AtomicBool::new(false),
            roots: Mutex::new(Vec::new()),
        });
        f.model.comm = Some(wire.clone());
        f.model.ep_protocol_v2 = true;
        wire
    }
    pub fn clear(&self) {
        assert!(self.pending.lock().is_empty());
        self.seen.lock().clear();
        self.roots.lock().clear();
        self.fail.store(0, Ordering::Relaxed);
    }
    pub fn queue(&self, packets: &[Vec<u32>]) {
        self.clear();
        self.pending.lock().extend(packets.iter().cloned());
    }
    pub fn packets(&self) -> Vec<Vec<u32>> {
        self.seen.lock().clone()
    }
    pub fn enable_cold_prefix(&self) {
        self.cold_prefix.store(true, Ordering::Relaxed);
    }
    pub fn roots(&self) -> Vec<usize> {
        self.roots.lock().clone()
    }
    pub fn done(&self) {
        assert!(self.pending.lock().is_empty());
    }
    fn transfer(&self, pointer: u64, bytes: usize, root: usize) -> Result<()> {
        let cold_root = root == 1 && bytes == 4 && self.cold_prefix.load(Ordering::Relaxed);
        ensure!(
            (root == 0 || cold_root) && bytes % 4 == 0,
            "unexpected command boundary"
        );
        if cold_root && self.rank == 1 {
            ensure!(
                self.gpu.read_span(DevicePtr(pointer), 4) == [0; 4],
                "worker cold prefix source must be zero"
            );
        }
        let words = if cold_root && self.rank == 0 {
            vec![0] // Explicit cold peer response, not general collective agreement.
        } else if self.rank == 0 {
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
        ensure!(
            !cold_root || words == [0],
            "cold prefix response must be zero"
        );
        self.roots.lock().push(root);
        self.seen.lock().push(words.clone());
        ensure!(
            self.seen.lock().len() != self.fail.load(Ordering::Relaxed),
            "injected command transfer failure"
        );
        if self.rank == 1 || cold_root {
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
