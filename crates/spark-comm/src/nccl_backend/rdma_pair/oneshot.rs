// SPDX-License-Identifier: AGPL-3.0-only

//! Graph-capturable one-shot exchange over the RDMA pair
//! (`ATLAS_RDMA_ONESHOT=1`, default off).
//!
//! The legacy pair keeps per-call state on the host (sequence numbers baked
//! into stream memops, a job queue), so it refuses CUDA-graph capture. Here a
//! call enqueues three stream operations whose arguments depend only on the
//! payload size, so eager calls and graph replays run the same protocol:
//!
//! 1. wait until the host `stage` word is 0 (the previous send has left `S`);
//! 2. copy the payload into the staging buffer `S` on the copy engine (SM
//!    stores to pinned memory were read stale by the NIC, see `super`);
//! 3. `rdma_oneshot_bf16`: block 0 publishes `stage = bytes` (the copy is
//!    complete by stream order), then all wait for the peer's flags and land
//!    the peer's payload (BF16 add, bit-identical to `bf16_add_inplace`, or a
//!    copy). `ATLAS_RDMA_ONESHOT_STAGE_FENCE=1` instead publishes `stage` with
//!    a fenced stream memop between 2 and 3, the legacy `ready` ordering
//!    (about 5-7 us slower per call on GB10).
//!
//! `ATLAS_RDMA_ONESHOT_SM_STAGE=1` makes it one stream operation: the kernel
//! waits for `stage` == 0 and stages the payload itself (write-through
//! stores, a system-scope fence, then `stage`), and is launched with
//! programmatic dependent launch (`ATLAS_RDMA_ONESHOT_PDL`, default on), so
//! the producer -> all-reduce -> consumer chain has no copy-engine hops or
//! launch gaps. The landing is the same code, so the result is bit-identical
//! to the copy-engine path whenever the NIC reads what the SMs stored; mode
//! `2` (plain stores) exists only as the control of the stress test for that
//! (`nic_loopback_sm_stage_stress`). Checked equal on both ranks.
//!
//! The proxy serves a non-zero `stage` as op `served + 1`: it WRITEs `S` into
//! the peer's receive slot `seq % 2` (striped over the rails from
//! `stripe_min`), then a flag `seq << 24 | bytes` per rail, and clears `stage`
//! once those complete. The kernel keeps `seq` in device memory and the proxy
//! counts stages, so both advance once per op whether it was launched eagerly
//! or replayed. Two receive slots suffice: we only stage `seq + 2` after our
//! kernel `seq + 1`, which waited on the peer's flag `seq + 1`, which the peer
//! only staged after its kernel `seq` finished reading slot `seq % 2`.
//!
//! A late rank never waits for its own send: the peer's data has landed and
//! its kernel exits; only its next stage waits for `S` to drain. Both ranks
//! must issue the same one-shot ops in the same order, and one at a time on
//! the GPU: the sequence and `S` are shared by every stream, so ops on
//! different streams (decode on the default stream, prefill on the
//! scheduler's prefill stream) must be serialized by the caller with events
//! or syncs, as they are for the legacy channel.
//!
//! A chain of up to [`MAX_RANKS`] ranks ([`Shape`], `rdma_chain_bf16`) runs
//! the same protocol with a receive slot pair and a flag line per piece
//! ([`pieces`]) for every other rank: each rank stages once and its proxy sends `S` to both
//! neighbours, which forward it on ([`relay`]), so every rank lands the same
//! sum of all payloads. The two-slot argument holds per origin: a rank only
//! stages `seq + 2` after its kernel `seq + 1`, which waited on every rank's
//! flag `seq + 1`, each staged after that rank's kernel `seq` and forwarded
//! by every rank in between after it arrived there.
//!
//! Nothing hangs on a fault; the poison word names it and the kernel traps.
//! The kernel poisons on a size mismatch at a sequence number, a flag more
//! than one op ahead (the peer cannot stage `seq + 2` before our flag
//! `seq + 1`), or a peer that never arrives within
//! `ATLAS_RDMA_ONESHOT_TIMEOUT_MS`. The host poisons when it can no longer
//! serve: the proxy failed (it then clears `stage`, so a stream blocked in
//! step 1 reaches a kernel), or a launch failed after a fenced `stage` was
//! published. Every kernel checks the word at launch and while it waits.
//!
//! Prior art (ideas only, no code; docs/glm-prior-art.md): the protocol
//! follows b12x RoCEnante (<https://github.com/local-inference-lab/b12x>,
//! `b12x/comm/roce/`, `docs/rocenante.md`; RoCEnante by Jason Cook,
//! local-inference-lab/b12x#295; Apache-2.0): pinned parity slots, a host
//! proxy that WRITEs each stripe then a seq flag on the same QP, a
//! device-resident sequence for graph replay, and poison on timeout, for the
//! hazard in <https://github.com/local-inference-lab/b12x/issues/313> (which
//! reached us through jayleaton/glm53-tensorfold-spark `docs/SFXNZ-AUDIT.md`).
//! The kernel's block structure (block 0 polls the host flags and republishes
//! seq in device memory; the last block out stores seq) follows mmastrac's
//! arx one-shot all-reduce (<https://github.com/mmastrac/glm-5.3-flash-4x-gx10>,
//! `experimental/arx/arx_vllm.cu`, PR #4; unlicensed, so idea only).

pub(super) use super::bootstrap::{MAX_RAILS, MAX_RANKS};
use super::{SPLIT_MIN, cu, cuMemcpyAsync};
use crate::nccl_backend::pair_ops::oneshot_kernel_name;
use crate::nccl_backend::{cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2};
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[path = "oneshot/wire.rs"]
mod wire;
pub(super) use wire::{
    A2A_BIT, Config, FP8_BIT, FUSED_BIT, MAX_PIECES, OpKind, PIPE_BIT, RS_BIT, RS_PIECES, Shape,
    decode_size, fp8_wire, is_fused, origin_slot, pieces, region_bytes, rs_per_segment, rs_pieces,
    rs_wire_pieces, stripes,
};
pub(in crate::nccl_backend) use wire::{A2a, A2aPlane};
use wire::{
    chain_flag_line, ctrl_off, describe_poison, fail_at, flag_line, flag_word, poison_at,
    poison_host, recv_off,
};

/// `CUlaunchAttribute` / `CUlaunchConfig` (driver API), as in atlas-core's
/// `registry/launch.rs`.
#[repr(C)]
struct CuLaunchAttribute {
    id: u32,
    pad: [u8; 4],
    value: [u8; 64],
}

#[repr(C)]
struct CuLaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem_bytes: u32,
    stream: u64,
    attrs: *mut CuLaunchAttribute,
    num_attrs: u32,
}

const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;

unsafe extern "C" {
    fn cuLaunchKernelEx(
        config: *const CuLaunchConfig,
        f: u64,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuStreamWaitValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
    fn cuStreamWriteValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
}

const CU_STREAM_WAIT_VALUE_EQ: u32 = 0x1;
const CU_STREAM_WAIT_VALUE_GEQ: u32 = 0x0;
/// Payload sizes ride in the low 24 bits of a flag word, below [`RS_BIT`].
const MAX_LIMIT: usize = RS_BIT - 64;
/// The capture contract promises at least this much (8 decode rows x 5 x
/// 4096 BF16 on the k5 route is 320 KiB).
const MIN_MAX: usize = 512 << 10;
/// A chain payload of this many bytes per piece is cut finer ([`pieces`]).
const PIECE: usize = 128 << 10;
const THREADS: usize = 256;
/// Kernel state in device memory (see `rdma_oneshot.cu`): seq, finished
/// blocks, go, staged blocks, then the reduce-scatter's per-piece writer
/// counts.
const STATE_BYTES: usize = 1024;
/// The fused reduce-scatter's producer counters (u32 a piece) and published
/// pieces (u64 a piece) in the state (`ONESHOT_STATE_PRODUCED`, `_OWN`).
const STATE_PRODUCED: usize = 256;
const STATE_OWN: usize = 512;
const _: () = assert!(
    STATE_PRODUCED + 4 * MAX_PIECES <= STATE_OWN && STATE_OWN + 8 * MAX_PIECES <= STATE_BYTES
);
const MAX_BLOCKS: usize = 16;

/// The reduce-scatter's and the all-to-all's grid (`ATLAS_RDMA_RS_BLOCKS`,
/// default 48, 1..=256; local, as the split of an op's pieces over blocks
/// changes no value). Their kernels stage and land every piece through the
/// pinned buffers: at 16 blocks the SP seam's 64 MB reduce-scatter took
/// 4009 us (FP8 3953), at 48 3857 us (FP8 3495; four nodes, 2026-10-18).
fn rs_blocks() -> usize {
    static BLOCKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BLOCKS.get_or_init(|| {
        std::env::var("ATLAS_RDMA_RS_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|b| (1..=256).contains(b))
            .unwrap_or(48)
    })
}

/// What an op lands on each rank.
#[derive(Clone, Copy)]
enum Land {
    /// The sum of every rank's payload (on a pair: the peer's, added or
    /// copied).
    Sum,
    /// Rank `root`'s payload (a chain broadcast).
    Root(usize),
    /// Each other rank's payload `stride` bytes per rank from rank 0's (a
    /// chain gather).
    Gather(i64),
    /// The four-rank chain's reduce-scatter of segments `stride` bytes
    /// apart (`oneshot_rs4`).
    ReduceScatter(i64),
    /// The four-rank chain's all-to-all (`oneshot_a2a4`).
    AllToAll(A2a),
    /// The fused reduce-scatter (`oneshot_rs4_fused`) of segments `stride`
    /// bytes apart, the op of `epoch`.
    Fused(i64, u32),
}

/// The fused reduce-scatter's kernel argument (`OneshotRsFused`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FusedArg {
    done: u64,
    epoch: u32,
    pad: u32,
}

/// Enqueue side, owned by the backend.
pub(in crate::nccl_backend) struct OneShot {
    cfg: Config,
    shape: Shape,
    /// Host and device addresses of the channel's part of the pinned region.
    host: usize,
    dev: u64,
    /// Device-resident kernel state: sequence (u64), finished-block counter
    /// (u32 at 8), the sequence block 0 saw arrive (u64 at 16) and the
    /// staged-block counter (u32 at 24).
    state: u64,
    kernel: AtomicU64,
    /// Channel 0 of a chain with `ATLAS_RDMA_RS_FUSED_SEG`: the fused
    /// region and its completion words.
    fused: Option<FusedRegion>,
    /// Fused ops begun (each op's `epoch`), and whether the channel has
    /// run no op since the last that every rank hears from its neighbours
    /// (`fused` module doc).
    epoch: AtomicU32,
    fused_last: std::sync::atomic::AtomicBool,
}

/// This rank's fused region (`fused` module): host and device addresses,
/// the segment capacity, and the u32 completion word of each piece of its
/// own segment (device memory).
#[derive(Clone, Copy, Debug)]
pub(super) struct FusedRegion {
    pub(super) dev: u64,
    pub(super) cap: usize,
    pub(super) done: u64,
}

impl Drop for OneShot {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.state) };
        if let Some(f) = self.fused {
            unsafe { cuMemFree_v2(f.done) };
        }
    }
}

mod channel;
pub(super) mod fused;
mod fused_ops;
mod ops;
pub(super) mod relay;
pub(super) use channel::Channel;

#[cfg(test)]
#[path = "oneshot_gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "oneshot_tests.rs"]
mod tests;
