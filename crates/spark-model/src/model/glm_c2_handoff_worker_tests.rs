// SPDX-License-Identifier: AGPL-3.0-only
//! Actual v2 worker F0/token dispatch and paired producer ownership.
//! Sequence leases are genuinely allocated by Fixture; F1 allocation, SSM,
//! NCCL numerics, accepted-row verdicts and paired E1 consumption are NOT tested.
use super::{fixture::*, isolated};
use crate::traits::{Model, SequenceState};
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::DevicePtr;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Message {
    Idle(u32),
    Payload(usize, Vec<u32>),
}

struct Scripted {
    gpu: Arc<Recorder>,
    pending: Mutex<VecDeque<Message>>,
    received: Mutex<Vec<Message>>,
    fail_at: AtomicUsize,
    collectives: AtomicUsize,
}

impl Scripted {
    fn install(f: &mut Fixture) -> Arc<Self> {
        let comm = Arc::new(Self {
            gpu: f.gpu.clone(),
            pending: Mutex::new(VecDeque::new()),
            received: Mutex::new(Vec::new()),
            fail_at: AtomicUsize::new(0),
            collectives: AtomicUsize::new(0),
        });
        f.model.comm = Some(comm.clone());
        f.model.ep_protocol_v2 = true;
        comm
    }

    fn queue(&self, messages: Vec<Message>) {
        assert!(self.pending.lock().is_empty());
        self.received.lock().clear();
        self.pending.lock().extend(messages);
    }

    fn receive(&self, ptr: u64, bytes: usize, root: usize, idle: bool) -> Result<()> {
        let message = self
            .pending
            .lock()
            .pop_front()
            .expect("unscripted broadcast");
        let words = match &message {
            Message::Idle(word) => {
                ensure!(idle && root == 0 && bytes == 4, "wrong idle boundary");
                vec![*word]
            }
            Message::Payload(expected_root, words) => {
                ensure!(
                    !idle && root == *expected_root,
                    "wrong payload boundary/root"
                );
                words.clone()
            }
        };
        ensure!(bytes == words.len() * 4, "wrong actual broadcast extent");
        let mut received = self.received.lock();
        received.push(message);
        ensure!(
            received.len() != self.fail_at.load(Ordering::Relaxed),
            "injected worker receive failure"
        );
        let data: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        if root == 1 {
            // Real ep_min_u32 has already uploaded this rank's local value.
            ensure!(
                self.gpu.read_span(DevicePtr(ptr), bytes) == data,
                "worker rooted prefix-match payload differed"
            );
        } else {
            self.gpu.write_span(DevicePtr(ptr), &data);
        }
        Ok(())
    }

    fn done(&self, expected: &[Message]) {
        assert!(self.pending.lock().is_empty());
        assert_eq!(*self.received.lock(), expected);
    }

    fn inert_collective(&self) -> Result<()> {
        self.collectives.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl CommBackend for Scripted {
    fn rank(&self) -> usize {
        1
    }
    fn world_size(&self) -> usize {
        2
    }
    fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
        self.receive(ptr, 4, 0, true)
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        self.receive(ptr, bytes, root, false)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        self.inert_collective()
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        self.inert_collective()
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        self.inert_collective()
    }
    fn barrier(&self) -> Result<()> {
        self.inert_collective()
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        self.inert_collective()
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        self.inert_collective()
    }
}

fn command(slot: usize, cmd: u32) -> Vec<Message> {
    vec![Message::Idle(slot as u32), Message::Payload(0, vec![cmd])]
}

fn prefill(slot: usize, prompt: &[u32]) -> Vec<Message> {
    let mut messages = command(slot, 0xfffffff0);
    messages.extend([
        Message::Payload(0, vec![prompt.len() as u32]),
        Message::Payload(0, vec![0]),
        Message::Payload(0, vec![prompt.len() as u32]),
        Message::Payload(0, prompt.to_vec()),
        // Actual cold chunked-prefill prefix agreement, both rooted broadcasts.
        Message::Payload(0, vec![0]),
        Message::Payload(1, vec![0]),
    ]);
    messages
}

fn disable_mtp(slot: usize) -> Vec<Message> {
    vec![
        Message::Idle(slot as u32),
        Message::Payload(0, vec![0xfffffff6]),
        Message::Payload(0, vec![1]),
    ]
}

fn take_slots(f: &mut Fixture) -> [Option<SequenceState>; 2] {
    std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only)).map(Some)
}

fn assert_publication(f: &Fixture, source: DevicePtr, row: usize) {
    let events = f.gpu.trace();
    let dst = f.gpu.slab().offset(row * ROW_BYTES);
    let copies: Vec<_> = events
        .iter()
        .enumerate()
        .filter(
            |(_, event)| matches!(event, Event::Copy(_, to, n, _) if *to == dst && *n == ROW_BYTES),
        )
        .collect();
    assert_eq!(copies.len(), 1, "exactly one actual ownership publication");
    let (index, event) = copies[0];
    assert_eq!(*event, Event::Copy(source, dst, ROW_BYTES, DEFAULT));
    assert_eq!(events.get(index + 1), Some(&Event::Sync(DEFAULT)));
}

#[test]
fn worker_applies_native_only_fence_before_prefill() {
    if isolated("worker_tests::worker_applies_native_only_fence_before_prefill") {
        return;
    }
    let mut f = Fixture::new(1);
    let comm = Scripted::install(&mut f);
    let mut slots = take_slots(&mut f);
    let messages = disable_mtp(1);
    comm.queue(messages.clone());

    assert!(f.model.ep_worker_step(&mut slots).unwrap());
    comm.done(&messages);
    assert!(slots[1].as_ref().unwrap().disable_mtp);
    // The fence is metadata only; no model collective or MTP producer ran.
    assert_eq!(comm.collectives.load(Ordering::Relaxed), 0);
}

#[test]
fn worker_f0_and_bootstrap_publish_two_real_owners_before_peer_overwrite() {
    if isolated(
        "worker_tests::worker_f0_and_bootstrap_publish_two_real_owners_before_peer_overwrite",
    ) {
        return;
    }
    let mut f = Fixture::new(1);
    let comm = Scripted::install(&mut f);
    let mut slots = take_slots(&mut f);
    let slab = f.gpu.slab();
    f.gpu.write_span(slab, &vec![0xa5; SLAB_BYTES]);
    let prompts = [[1, 2, 3, 4], [4, 3, 2, 1]];
    let mut tails = Vec::new();
    for owner in 0..2 {
        let messages = prefill(owner, &prompts[owner]);
        comm.queue(messages.clone());
        f.gpu.clear();
        assert!(f.model.ep_worker_step(&mut slots).unwrap());
        comm.done(&messages);
        let seq = slots[owner].as_ref().unwrap();
        assert_eq!(seq.tokens, prompts[owner]);
        assert_eq!(seq.seq_len, 4);
        let private = seq
            .proposer_state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<crate::layers::glm5_mtp::Glm5MtpProposerState>()
            .unwrap();
        assert_eq!(private.seq_len, 3, "actual worker P-1 KV primer completed");
        let source = f.model.mtp_prefill_hidden.offset(3 * ROW_BYTES);
        let tail = f.gpu.read_span(source, ROW_BYTES);
        assert_eq!(tail, vec![prompts[owner][3] as u8 + 0x23; ROW_BYTES]);
        assert_eq!(
            f.gpu
                .read_span(slab.offset(owner * 6 * ROW_BYTES), ROW_BYTES),
            tail
        );
        assert_publication(&f, source, owner * 6);
        tails.push(tail);
    }
    assert_ne!(tails[0], tails[1]);
    let mut bonus: [Vec<u8>; 2] = std::array::from_fn(|_| Vec::new());
    // Reverse order makes slot addressing observable independently of prefill order.
    for owner in [1, 0] {
        let token = 5 + owner as u32;
        let messages = command(owner, token);
        comm.queue(messages.clone());
        f.gpu.clear();
        assert!(f.model.ep_worker_step(&mut slots).unwrap());
        comm.done(&messages);
        let seq = slots[owner].as_ref().unwrap();
        assert_eq!(seq.seq_len, 5);
        assert_eq!(seq.tokens.last(), Some(&token));
        let source = f.model.buffers.norm_output();
        bonus[owner] = f.gpu.read_span(source, ROW_BYTES);
        assert_eq!(bonus[owner], vec![token as u8 + 0x24; ROW_BYTES]);
        assert_publication(&f, source, owner * 6 + 5);
    }
    assert_ne!(bonus[0], bonus[1]);
    for owner in 0..2 {
        assert_eq!(
            f.gpu
                .read_span(slab.offset(owner * 6 * ROW_BYTES), ROW_BYTES),
            tails[owner]
        );
        assert_eq!(
            f.gpu
                .read_span(slab.offset((owner * 6 + 5) * ROW_BYTES), ROW_BYTES),
            bonus[owner]
        );
        assert_eq!(
            f.gpu
                .read_span(slab.offset((owner * 6 + 1) * ROW_BYTES), 4 * ROW_BYTES),
            vec![0xa5; 4 * ROW_BYTES],
            "accepted-row staging stays unpublished"
        );
    }
}

#[test]
fn worker_selected_e1_requires_versioned_payload_before_proposer_work() {
    if isolated("worker_tests::worker_selected_e1_requires_versioned_payload_before_proposer_work")
    {
        return;
    }
    let mut f = Fixture::new(1);
    let comm = Scripted::install(&mut f);
    let mut slots = take_slots(&mut f);
    let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
    let mut messages = command(1, 0xffffffe1);
    messages.push(Message::Payload(0, vec![0, 1, 0, 1, 0, 5, 4, 7]));
    comm.queue(messages.clone());
    f.gpu.clear();
    let error = format!("{:#}", f.model.ep_worker_step(&mut slots).unwrap_err());
    assert!(error.contains("paired E1 version"), "{error}");
    comm.done(&messages);
    assert_eq!(comm.collectives.load(Ordering::Relaxed), 0);
    assert!(f.gpu.trace().iter().all(|event| matches!(
        event,
        Event::Sync(DEFAULT) | Event::Read(_, 4 | 32, DEFAULT)
    )));
    assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
    assert!(slots.iter().all(|seq| seq.as_ref().unwrap().seq_len == 0));
}

#[test]
fn worker_f0_receive_failures_do_not_enter_target_or_publish_hidden() {
    if isolated("worker_tests::worker_f0_receive_failures_do_not_enter_target_or_publish_hidden") {
        return;
    }
    // Outer slot/cmd, three header words, bulk tokens: six real receive boundaries.
    for fail_at in 1..=6 {
        let mut f = Fixture::new(1);
        let comm = Scripted::install(&mut f);
        let mut slots = take_slots(&mut f);
        let slab = f.gpu.slab();
        f.gpu.write_span(slab, &vec![0xa5; SLAB_BYTES]);
        comm.queue(prefill(1, &[4, 3, 2, 1]));
        comm.fail_at.store(fail_at, Ordering::Relaxed);
        f.gpu.clear();
        let error = format!("{:#}", f.model.ep_worker_step(&mut slots).unwrap_err());
        assert!(error.contains("injected worker receive failure"), "{error}");
        assert_eq!(comm.received.lock().len(), fail_at);
        assert_eq!(comm.collectives.load(Ordering::Relaxed), 0);
        assert!(!f.gpu.trace().iter().any(|event| matches!(
            event,
            Event::Target(..) | Event::Body(..) | Event::Kv(..) | Event::Copy(..)
        )));
        assert_eq!(f.gpu.read_span(slab, SLAB_BYTES), vec![0xa5; SLAB_BYTES]);
        assert!(slots.iter().all(|seq| seq.as_ref().unwrap().seq_len == 0));
    }
}
