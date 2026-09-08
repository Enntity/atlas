// SPDX-License-Identifier: AGPL-3.0-only
//! Actual rank1 F5 worker, not rank0 masquerading as a receiver or C2 E1 admission.
//! Collectives copy deterministic sentinel halves; no NCCL or KDA numerical proof.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::traits::{Model, SequenceState};
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::DevicePtr;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Receive {
    idle: bool,
    words: Vec<u32>,
}
struct Scripted {
    gpu: Arc<Recorder>,
    pending: Mutex<VecDeque<Receive>>,
    seen: Mutex<Vec<Receive>>,
    fail_receive: AtomicUsize,
}
impl Scripted {
    fn install(f: &mut Fixture) -> Arc<Self> {
        let comm = Arc::new(Self {
            gpu: f.gpu.clone(),
            pending: Mutex::new(VecDeque::new()),
            seen: Mutex::new(Vec::new()),
            fail_receive: AtomicUsize::new(0),
        });
        f.model.comm = Some(comm.clone());
        f.model.ep_protocol_v2 = true;
        comm
    }
    fn queue(&self, messages: &[Receive]) {
        assert!(self.pending.lock().is_empty());
        self.seen.lock().clear();
        self.pending.lock().extend(messages.iter().cloned());
    }
    fn receive(&self, ptr: u64, n: usize, root: usize, idle: bool) -> Result<()> {
        let next = self
            .pending
            .lock()
            .pop_front()
            .expect("unscripted worker receive");
        ensure!(
            root == 0 && next.idle == idle && next.words.len() * 4 == n,
            "wrong actual command/payload boundary"
        );
        let bytes: Vec<_> = next
            .words
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect();
        self.seen.lock().push(next);
        ensure!(
            self.seen.lock().len() != self.fail_receive.load(Ordering::Relaxed),
            "injected accepted-count receive failure"
        );
        self.gpu.write_span(DevicePtr(ptr), &bytes);
        Ok(())
    }
    fn done(&self, expected: &[Receive]) {
        assert!(self.pending.lock().is_empty());
        assert_eq!(*self.seen.lock(), expected);
    }
}
macro_rules! inert {
    ($($name:ident($($arg:ty),*));*) => {
        impl CommBackend for Scripted {
            $(fn $name(&self, $(_: $arg),*) -> Result<()> { Ok(()) })*
            fn rank(&self) -> usize { 1 }
            fn world_size(&self) -> usize { 2 }
            fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
                self.receive(ptr, 4, 0, true)
            }
            fn broadcast(&self, ptr: u64, n: usize, root: usize) -> Result<()> {
                self.receive(ptr, n, root, false)
            }
            fn all_gather(&self, src: u64, dst: u64, n: usize) -> Result<()> {
                self.gpu.gather_sentinel(src, dst, n)
            }
        }
    };
}
inert! { all_reduce(u64, usize); reduce_scatter(u64, u64, usize); barrier();
send_to(u64, usize, usize, u64); recv_from(u64, usize, usize, u64) }

fn command(owner: usize, opcode: u32) -> Vec<Receive> {
    vec![
        Receive {
            idle: true,
            words: vec![owner as u32],
        },
        Receive {
            idle: false,
            words: vec![opcode],
        },
    ]
}

fn verdict(
    f: &mut Fixture,
    comm: &Scripted,
    owner: usize,
    history: &flow::History,
    accepted: usize,
) {
    let mut messages = command(owner, 0xfffffff5);
    messages.extend([
        Receive {
            idle: false,
            words: vec![5],
        },
        Receive {
            idle: false,
            words: history.issued.clone(),
        },
        Receive {
            idle: false,
            words: vec![accepted as u32],
        },
    ]);
    comm.queue(&messages);
    let mut slots =
        std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only)).map(Some);
    f.gpu.clear();
    let result = f.model.ep_worker_step(&mut slots);
    f.seqs = slots.map(Option::unwrap);
    assert!(result.unwrap());
    comm.done(&messages);
    assert_eq!(f.seqs[owner].seq_len, history.base + accepted + 1);
    assert_eq!(
        &f.seqs[owner].tokens[history.base..],
        &history.issued[..accepted + 1]
    );
    let rows = flow::normalized(history);
    for (row, expected) in rows.iter().enumerate() {
        assert_eq!(
            f.gpu.read_span(
                f.model.buffers.norm_output().offset(row * ROW_BYTES),
                ROW_BYTES
            ),
            *expected,
            "actual worker K5 final normalization, not painted scratch"
        );
    }
    let target_calls: Vec<_> = f
        .gpu
        .trace()
        .into_iter()
        .filter(|e| matches!(e, Event::Target(_, _, _)))
        .collect();
    assert_eq!(
        target_calls,
        (0..5)
            .map(|row| Event::Target(1, history.base + row, DEFAULT))
            .collect::<Vec<_>>()
    );
    flow::detached(f, owner, history, accepted);
}

#[test]
fn all25_rank1_worker_f5_repair_off_both_owner_orders() {
    if flow::isolated("verdict_worker_tests::all25_rank1_worker_f5_repair_off_both_owner_orders") {
        return;
    }
    assert_eq!(std::env::var("ATLAS_GLM_MTP_REPAIR").unwrap(), "0");
    for order in [[0, 1], [1, 0]] {
        for a in 0..5 {
            for b in 0..5 {
                let (mut f, mut histories) = flow::prepare(1, order);
                let comm = Scripted::install(&mut f);
                let accepted = [a, b];
                for owner in order {
                    verdict(&mut f, &comm, owner, &histories[owner], accepted[owner]);
                }
                for owner in order.into_iter().rev() {
                    flow::detached(&f, owner, &histories[owner], accepted[owner]);
                    // Worker already ran real record -> trim -> commit. No duplicate ack.
                    flow::continue_owner(&mut f, owner, &mut histories[owner], accepted[owner]);
                }
            }
        }
    }
}

#[test]
fn repeated_worker_transactions_detach_before_peer_and_keep_e1_closed() {
    if flow::isolated(
        "verdict_worker_tests::repeated_worker_transactions_detach_before_peer_and_keep_e1_closed",
    ) {
        return;
    }
    for first in [[0, 1], [1, 0]] {
        let (mut f, mut histories) = flow::prepare(1, first);
        let comm = Scripted::install(&mut f);
        for (round, accepted) in [[0, 4], [4, 0], [1, 3], [3, 1], [4, 4], [0, 0]]
            .into_iter()
            .enumerate()
        {
            let order = if round % 2 == 0 {
                first
            } else {
                [first[1], first[0]]
            };
            for owner in order {
                verdict(&mut f, &comm, owner, &histories[owner], accepted[owner]);
            }
            for owner in order.into_iter().rev() {
                flow::detached(&f, owner, &histories[owner], accepted[owner]);
                flow::continue_owner(&mut f, owner, &mut histories[owner], accepted[owner]);
            }
        }
        let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        let cursors = f.seqs.each_ref().map(|seq| flow::private(seq).seq_len);
        let messages = command(0, 0xffffffe1);
        comm.queue(&messages);
        let mut slots =
            std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only)).map(Some);
        f.gpu.clear();
        let error = f.model.ep_worker_step(&mut slots).unwrap_err();
        f.seqs = slots.map(Option::unwrap);
        assert!(format!("{error:#}").contains("requires max_batch_size=1"));
        comm.done(&messages); // No draft count/token/position payload admitted.
        assert!(!f.gpu.trace().iter().any(|e| matches!(
            e,
            Event::Body(_, _) | Event::Target(_, _, _) | Event::Kernel(_, _, _)
        )));
        assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
        assert_eq!(
            f.seqs.each_ref().map(|seq| flow::private(seq).seq_len),
            cursors
        );
    }
}

#[test]
fn accepted_count_receive_read_and_invalid_value_fail_after_actual_k5() {
    if flow::isolated(
        "verdict_worker_tests::accepted_count_receive_read_and_invalid_value_fail_after_actual_k5",
    ) {
        return;
    }
    for owner in 0..2 {
        for accepted in [0, 4] {
            for mode in 0..3 {
                let (mut control, mut control_history) = flow::prepare(1, [owner, 1 - owner]);
                let control_comm = Scripted::install(&mut control);
                verdict(
                    &mut control,
                    &control_comm,
                    owner,
                    &control_history[owner],
                    accepted,
                );
                let trace = control.gpu.trace();
                let ordinal = trace
                    .iter()
                    .rposition(|event| *event == Event::Read(control.model.ep_cmd_buf, 4, DEFAULT))
                    .unwrap()
                    + 1;
                let verify_read = trace
                    .iter()
                    .position(|event| {
                        *event == Event::Read(control.model.buffers.scratch(), 20, DEFAULT)
                    })
                    .unwrap();
                assert!(verify_read + 1 < ordinal);
                flow::continue_owner(&mut control, owner, &mut control_history[owner], accepted);

                let (mut f, histories) = flow::prepare(1, [owner, 1 - owner]);
                let comm = Scripted::install(&mut f);
                let peer = 1 - owner;
                let peer_len = flow::private(&f.seqs[peer]).seq_len;
                let peer_kv = flow::bytes(&f, peer, peer_len);
                let peer_blocks = flow::private(&f.seqs[peer]).block_table.clone();
                let peer_tokens = f.seqs[peer].tokens.clone();
                let peer_slab = f
                    .gpu
                    .read_span(f.gpu.slab().offset(peer * 6 * ROW_BYTES), 6 * ROW_BYTES);
                let mut messages = command(owner, 0xfffffff5);
                messages.extend([
                    Receive {
                        idle: false,
                        words: vec![5],
                    },
                    Receive {
                        idle: false,
                        words: histories[owner].issued.clone(),
                    },
                    Receive {
                        idle: false,
                        words: vec![if mode == 2 { 5 } else { accepted as u32 }],
                    },
                ]);
                comm.queue(&messages);
                f.gpu.clear();
                if mode == 0 {
                    comm.fail_receive.store(5, Ordering::Relaxed);
                }
                if mode == 1 {
                    f.gpu.fail.store(ordinal, Ordering::Relaxed);
                }
                let mut slots =
                    std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only))
                        .map(Some);
                let error = f.model.ep_worker_step(&mut slots).unwrap_err();
                f.seqs = slots.map(Option::unwrap);
                comm.done(&messages);
                let error = format!("{error:#}");
                match mode {
                    0 => assert!(error.contains("injected accepted-count receive failure")),
                    1 => {
                        assert!(error.contains("injected fixture operation failure"));
                        assert_eq!(
                            f.gpu.trace().get(ordinal - 1),
                            Some(&Event::Read(f.model.ep_cmd_buf, 4, DEFAULT))
                        );
                    }
                    _ => assert!(error.contains("accepted 5 drafts"), "{error}"),
                }
                let events = f.gpu.trace();
                assert!(
                    events.contains(&Event::Read(f.model.buffers.scratch(), 20, DEFAULT)),
                    "real K5 completed before verdict receive"
                );
                assert_eq!(f.seqs[owner].seq_len, histories[owner].base + 5);
                assert_eq!(
                    f.gpu
                        .read_span(f.model.buffers.norm_output(), 5 * ROW_BYTES),
                    flow::normalized(&histories[owner]).concat()
                );
                f.gpu.fail.store(usize::MAX, Ordering::Relaxed);
                f.model.gpu.synchronize(DEFAULT).unwrap();
                f.gpu.clear();
                for who in [owner, peer] {
                    assert!(
                        f.model
                            .decode_verify_graphed_kgamma(
                                &histories[who].issued,
                                &mut f.seqs[who],
                                CALLER
                            )
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
                    f.gpu.trace().is_empty(),
                    "later completion cannot reopen issued F5 transaction"
                );
                assert_eq!(flow::bytes(&f, peer, peer_len), peer_kv);
                assert_eq!(flow::private(&f.seqs[peer]).block_table, peer_blocks);
                assert_eq!(f.seqs[peer].tokens, peer_tokens);
                assert_eq!(
                    f.gpu
                        .read_span(f.gpu.slab().offset(peer * 6 * ROW_BYTES), 6 * ROW_BYTES),
                    peer_slab
                );
            }
        }
    }
}
