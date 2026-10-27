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

/// Pieces a chain op may cut its payload into, one flag line each per
/// origin (`ONESHOT_ORIGIN_LINES` in the kernel); a pair's rails take the
/// first [`MAX_RAILS`] lines.
pub(super) const MAX_PIECES: usize = 32;

/// Control page after `S` and the receive slots, one 64-byte line per word:
/// `stage` (u32, GPU -> proxy), `poison` (kernel -> host), per origin one
/// flag per piece (neighbour NIC -> kernel), one flag source word per piece
/// for our own sends and, on a chain, per origin one per piece for the sends
/// that forward it (proxy -> NIC).
const STAGE: usize = 0;
const POISON: usize = 64;
const FLAGS: usize = 128;
const ORIGIN_LINES: usize = 64 * MAX_PIECES;
const FLAG_SRC: usize = FLAGS + ORIGIN_LINES * (MAX_RANKS - 1);
const FWD_SRC: usize = FLAG_SRC + 64 * MAX_PIECES;
/// The pipelined ops' ready words, one u64 per piece (kernel -> proxy: the
/// op whose staged or computed piece is in `S`); `ONESHOT_READY` in
/// `rdma_oneshot.cu`.
const READY: usize = FWD_SRC + ORIGIN_LINES * (MAX_RANKS - 1);
const CTRL_PAGE: usize = 16384;
const _: () = assert!(READY == 14464 && READY + 8 * MAX_PIECES <= CTRL_PAGE);
/// A chain reduce-scatter op's flag and `stage` size carry this bit (its
/// payloads stay below it), so no proxy or kernel takes its pieces for a
/// flood op's ([`relay`]).
pub(super) const RS_BIT: usize = 1 << 23;
/// A `stage` word with this bit (above every size code): the kernel stages
/// the op piece by piece and publishes each in its `READY` word, and the
/// proxy sends each piece as it is published (the pipelined all-gather).
pub(super) const PIPE_BIT: usize = 1 << 31;
/// A reduce-scatter op's payload divides into pieces of every segment of
/// four ranks in 64-byte multiples at this granularity (two a segment).
pub(super) const RS_PIECES: usize = 8;
/// A reduce-scatter piece is cut no finer than this.
const RS_PIECE: usize = 128 << 10;

/// Pieces a segment of a `bytes` four-rank reduce-scatter op is cut into:
/// 8, 4 or 2, the most that leaves every piece whole 64-byte units of at
/// least [`RS_PIECE`] (2 below that), the same on every rank.
pub(super) fn rs_per_segment(bytes: usize) -> usize {
    let seg = bytes / MAX_RANKS;
    [8, 4]
        .into_iter()
        .find(|&k| seg.is_multiple_of(k * 64) && seg / k >= RS_PIECE)
        .unwrap_or(2)
}

/// A reduce-scatter op's pieces: [`rs_per_segment`] equal 64-byte
/// multiples of each segment, in segment order.
pub(super) fn rs_pieces(bytes: usize) -> Vec<(usize, usize)> {
    stripes(bytes, MAX_RANKS * rs_per_segment(bytes), 0)
}

/// An all-to-all op's size field: [`RS_BIT`] and this bit, which a
/// segmented payload (whole 512-byte units) never sets.
pub(super) const A2A_BIT: usize = 1;

/// A reduce-scatter op's size field carries this bit too when its pieces
/// travel as E4M3 with one f32 scale per 128 values (`oneshot_rs4_fp8`,
/// [`OneShot::reduce_scatter`] with `fp8`); the size stays the BF16 payload's.
pub(super) const FP8_BIT: usize = 2;

/// A fused reduce-scatter's size field carries this bit beside [`RS_BIT`],
/// and the payload in 512-byte units from bit 3 ([`fused::code`]): one op of
/// a whole seam is larger than the plain size field.
pub(super) const FUSED_BIT: usize = 4;

/// Whether size field `code` names a fused reduce-scatter.
pub(super) fn is_fused(code: usize) -> bool {
    code & RS_BIT != 0 && code & (A2A_BIT | FUSED_BIT) == FUSED_BIT
}

/// Wire bytes of `bytes` of BF16 (a multiple of 1024) as E4M3 values and
/// their f32 scales, one per 128 values.
pub(super) fn fp8_wire(bytes: usize) -> usize {
    bytes / 2 + bytes / 64
}

/// The `(offset, len)` pieces on the wire of the reduce-scatter or
/// all-to-all op whose size field is `code`: [`rs_pieces`], or under
/// [`FP8_BIT`] their [`fp8_wire`] images.
pub(super) fn rs_wire_pieces(code: usize) -> Vec<(usize, usize)> {
    let parts = rs_pieces(decode_size(code).0);
    if code & RS_BIT != 0 && code & (A2A_BIT | FP8_BIT) == FP8_BIT {
        parts
            .into_iter()
            .map(|(o, l)| (fp8_wire(o), fp8_wire(l)))
            .collect()
    } else {
        parts
    }
}

/// What a flag's or `stage`'s size field names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpKind {
    /// Every rank's whole payload to every rank (all-reduce, gather,
    /// broadcast).
    Flood,
    /// The four-rank reduce-scatter (`relay::rs_role`).
    ReduceScatter,
    /// The four-rank all-to-all: segment `d` to rank `d`.
    AllToAll,
}

/// A flag's or `stage`'s size field: `(bytes, kind)`.
pub(super) fn decode_size(code: usize) -> (usize, OpKind) {
    match (code & RS_BIT != 0, code & A2A_BIT != 0) {
        (false, _) => (code, OpKind::Flood),
        (true, false) if code & FUSED_BIT != 0 => {
            (((code & (RS_BIT - 1)) >> 3) * 512, OpKind::ReduceScatter)
        }
        (true, false) => (code & !(RS_BIT | FP8_BIT), OpKind::ReduceScatter),
        (true, true) => (code & !(RS_BIT | A2A_BIT), OpKind::AllToAll),
    }
}

/// One plane of an all-to-all op, as the kernel takes it (`OneshotA2aPlane`
/// in `rdma_oneshot.cu`): this rank's part for rank `d` at `src + d *
/// src_stride`, `bytes` of `row_bytes` rows; rank `o`'s lands at `dst + o *
/// dst_stride`, rows `pitch` apart.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::nccl_backend) struct A2aPlane {
    pub src: u64,
    pub src_stride: i64,
    pub dst: u64,
    pub dst_stride: i64,
    pub row_bytes: u32,
    pub pitch: u32,
    pub bytes: u32,
    pub pad: u32,
}

/// An all-to-all op's two planes (an unused plane has no bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::nccl_backend) struct A2a {
    pub planes: [A2aPlane; 2],
}

impl A2a {
    /// Bytes of one segment (a rank's part for one rank, both planes).
    pub(in crate::nccl_backend) fn segment(&self) -> usize {
        self.planes.iter().map(|p| p.bytes as usize).sum()
    }

    /// Whether the kernel can take it: 16-byte rows, pitches, strides and
    /// addresses, whole rows, and a segment of whole 128-byte units.
    pub(in crate::nccl_backend) fn valid(&self) -> bool {
        let seg = self.segment();
        seg > 0
            && seg.is_multiple_of(128)
            && self.planes.iter().all(|p| {
                p.bytes == 0
                    || (p.row_bytes > 0
                        && p.row_bytes.is_multiple_of(16)
                        && p.pitch >= p.row_bytes
                        && p.pitch.is_multiple_of(16)
                        && p.bytes.is_multiple_of(p.row_bytes)
                        && [p.src, p.dst].iter().all(|a| a.is_multiple_of(16))
                        && [p.src_stride, p.dst_stride].iter().all(|s| s % 16 == 0))
            })
    }
}
/// Poison reasons (high bits) over the op's sequence number; the kernel
/// writes the top three, the host the others.
const POISON_LAUNCH: u64 = 1 << 59;
const POISON_PROXY: u64 = 1 << 60;
const POISON_DESYNC: u64 = 1 << 61;
const POISON_TIMEOUT: u64 = 1 << 62;
const POISON_MISMATCH: u64 = 1 << 63;

/// One-shot settings; read from the shared profile, and checked equal on both
/// ranks at bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config {
    /// Largest one-shot payload in bytes (`ATLAS_RDMA_ONESHOT_MAX`, 1 MiB,
    /// at least 512 KiB).
    pub(super) max: usize,
    /// Payloads from this size stripe over every rail
    /// (`ATLAS_RDMA_ONESHOT_STRIPE_MIN`, default the legacy split size).
    pub(super) stripe_min: usize,
    /// Kernel wait limit (`ATLAS_RDMA_ONESHOT_TIMEOUT_MS`, 30 s; 0 = none).
    pub(super) timeout_ns: u64,
    /// Publish `stage` with a fenced stream memop rather than from the
    /// kernel (`ATLAS_RDMA_ONESHOT_STAGE_FENCE=1`); local only, and ignored
    /// under `sm_stage`.
    pub(super) stage_fence: bool,
    /// The kernel stages the payload (`ATLAS_RDMA_ONESHOT_SM_STAGE`): 0 =
    /// copy engine (default), 1 = write-through SM stores, 2 = plain SM
    /// stores (stress-test control only).
    pub(super) sm_stage: u32,
    /// Launch the kernel with programmatic dependent launch under `sm_stage`
    /// (`ATLAS_RDMA_ONESHOT_PDL`, default 1); local only.
    pub(super) pdl: bool,
    /// The fused reduce-scatter's segment capacity in bytes
    /// (`ATLAS_RDMA_RS_FUSED_SEG`, whole 64 KiB; 0 = off): a chain region of
    /// `fused::entries` of these. Every rank must agree (the chain head).
    pub(super) fused_seg: usize,
}

impl Config {
    /// `None` unless `ATLAS_RDMA_ONESHOT=1`.
    pub(super) fn from_env() -> Option<Self> {
        Self::parse(|key| std::env::var(key).ok())
    }

    fn parse(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        if var("ATLAS_RDMA_ONESHOT").as_deref() != Some("1") {
            return None;
        }
        let num = |key: &str, default: usize| {
            var(key)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default)
        };
        Some(Self {
            max: num("ATLAS_RDMA_ONESHOT_MAX", 1 << 20)
                .clamp(MIN_MAX, MAX_LIMIT)
                .next_multiple_of(64),
            stripe_min: num("ATLAS_RDMA_ONESHOT_STRIPE_MIN", SPLIT_MIN),
            timeout_ns: (num("ATLAS_RDMA_ONESHOT_TIMEOUT_MS", 30_000) as u64)
                .saturating_mul(1_000_000),
            stage_fence: var("ATLAS_RDMA_ONESHOT_STAGE_FENCE").as_deref() == Some("1"),
            sm_stage: match var("ATLAS_RDMA_ONESHOT_SM_STAGE").as_deref() {
                Some("1") => 1,
                Some("2") => 2,
                _ => 0,
            },
            pdl: var("ATLAS_RDMA_ONESHOT_PDL").as_deref() != Some("0"),
            fused_seg: num("ATLAS_RDMA_RS_FUSED_SEG", 0) / (64 << 10) * (64 << 10),
        })
    }

    /// Whether a payload takes the one-shot path. Rank-invariant by
    /// construction: it looks at nothing but the size (never at addresses).
    pub(super) fn eligible(&self, bytes: usize) -> bool {
        bytes > 0 && bytes <= self.max && bytes.is_multiple_of(2)
    }

    /// Bootstrap form, `max`, `stripe_min` and `sm_stage`: both ranks must
    /// send the same.
    pub(super) fn wire(cfg: Option<Self>) -> [u64; 3] {
        cfg.map_or([0; 3], |c| {
            [c.max as u64, c.stripe_min as u64, u64::from(c.sm_stage)]
        })
    }
}

/// Where a rank sits: one of two (a pair), or on a chain where every
/// payload reaches every rank through the ranks in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) ranks: usize,
    pub(super) rank: usize,
    /// Rails of the link to the rank below and to the one above; 0 where
    /// there is none. A pair's one link is both.
    pub(super) rails: [usize; 2],
}

impl Shape {
    pub(super) fn pair(rank: usize, rails: usize) -> Self {
        Self {
            ranks: 2,
            rank,
            rails: [rails; 2],
        }
    }

    /// Receive slot pairs: one per other rank.
    pub(super) fn origins(&self) -> usize {
        self.ranks - 1
    }

    pub(super) fn check(&self) -> Result<()> {
        ensure!(
            (2..=MAX_RANKS).contains(&self.ranks) && self.rank < self.ranks,
            "RDMA one-shot: rank {} of {} (supported: 2..={MAX_RANKS} ranks)",
            self.rank,
            self.ranks
        );
        let used = [self.rank > 0, self.rank + 1 < self.ranks];
        for (side, (&rails, used)) in ["below", "above"].iter().zip(self.rails.iter().zip(used)) {
            ensure!(
                !used || (1..=MAX_RAILS).contains(&rails),
                "RDMA one-shot supports 1..={MAX_RAILS} rails, got {rails} to the rank {side}"
            );
        }
        Ok(())
    }

    /// Flags the kernel waits for per origin: the stripes the pair's link
    /// carries a `bytes` payload in, or a chain's [`pieces`].
    fn stripe_count(&self, bytes: usize, stripe_min: usize) -> u32 {
        let n = if self.ranks > 2 {
            pieces(bytes, stripe_min).len()
        } else {
            stripes(bytes, self.rails[1], stripe_min).len()
        };
        n as u32
    }
}

/// Origin slot of rank `origin`'s payload at rank `at`: ranks in order,
/// skipping `at`. A payload keeps its slot index along its way: it moves
/// away from its origin, so every rank it passes is on the same side of it.
pub(super) fn origin_slot(origin: usize, at: usize) -> usize {
    origin - usize::from(origin > at)
}

/// Bytes of pinned region the one-shot channel appends: `S`, two receive
/// slots per origin and the control page.
pub(super) fn region_bytes(max: usize, origins: usize) -> usize {
    ctrl_off(max, origins) + CTRL_PAGE
}
fn recv_off(max: usize, origin: usize, slot: usize) -> usize {
    (1 + 2 * origin + slot) * max
}
fn ctrl_off(max: usize, origins: usize) -> usize {
    (1 + 2 * origins) * max
}
/// The flag line of `origin`'s rail `rail` in the control page.
fn flag_line(origin: usize, rail: usize) -> usize {
    FLAGS + ORIGIN_LINES * origin + 64 * rail
}

/// Multi-rank completion words match the payload's two parity slots. Ring
/// flood and specialized collectives can deliver one origin over different
/// QPs, so seq+1 may arrive before seq: it must not publish seq's completion
/// or overwrite its flag. The pair retains its original single-word ABI.
fn chain_flag_line(origin: usize, piece: usize, seq: u64) -> usize {
    flag_line(origin, piece) + 8 * (seq as usize & 1)
}

/// `(offset, len)` per rail: one rail below `stripe_min`, else split over all
/// `rails` in 64-byte-aligned parts. Identical on both ranks.
pub(super) fn stripes(bytes: usize, rails: usize, stripe_min: usize) -> Vec<(usize, usize)> {
    let used = if bytes >= stripe_min { rails } else { 1 };
    let part = bytes.div_ceil(used).next_multiple_of(64);
    (0..used)
        .map(|r| r * part)
        .take_while(|&off| off < bytes)
        .map(|off| (off, part.min(bytes - off)))
        .collect()
}

/// A chain's `(offset, len)` pieces: one below `stripe_min`, else two (one
/// per rail of a two-rail link), and payloads of several [`PIECE`]s one per
/// `PIECE`, at most one per flag line; 64-byte aligned. Piece `p` goes on rail
/// `p % rails` of every link, so the layout does not depend on the rails and
/// a rank forwards each piece as soon as it lands: the hops of a large
/// payload overlap.
pub(super) fn pieces(bytes: usize, stripe_min: usize) -> Vec<(usize, usize)> {
    let n = if bytes >= stripe_min {
        (bytes / PIECE).clamp(2, MAX_PIECES)
    } else {
        1
    };
    stripes(bytes, n, 0)
}

/// The flag the proxy writes for op `seq` of `bytes` (see the kernel).
fn flag_word(seq: u64, bytes: usize) -> u64 {
    (seq << 24) | bytes as u64
}

fn describe_poison(word: u64) -> String {
    let why = match word & !(POISON_LAUNCH - 1) {
        POISON_MISMATCH => "the peer sent a different size (ranks diverged)",
        POISON_DESYNC => "the peer's flag is more than one op ahead (sequences diverged)",
        POISON_PROXY => "the RDMA proxy failed (see its error)",
        POISON_LAUNCH => "a kernel launch failed after its stage was published",
        POISON_TIMEOUT => "the peer never arrived (ATLAS_RDMA_ONESHOT_TIMEOUT_MS)",
        _ => "unknown fault",
    };
    format!("op {}: {why}", word & (POISON_LAUNCH - 1))
}

/// Why the channel with control page `ctrl` (host address) stopped, if it did.
fn poison_at(ctrl: usize) -> Option<String> {
    // SAFETY: the control page lives in the pinned region for the pair's
    // lifetime; the kernel writes this word only before trapping.
    let word = unsafe { ((ctrl + POISON) as *const u64).read_volatile() };
    (word != 0).then(|| describe_poison(word))
}

/// Stop the channel from the host: every later kernel traps at launch. An
/// earlier reason is kept.
fn poison_host(ctrl: usize, word: u64) {
    // SAFETY: as in `poison_at`.
    let poison = unsafe { &*((ctrl + POISON) as *const AtomicU64) };
    let _ = poison.compare_exchange(0, word, Ordering::AcqRel, Ordering::Relaxed);
}

/// A proxy that is going away, with `served` ops sent, leaves nothing to
/// clear `stage` again: poison the channel with control page `ctrl`, then
/// release a stream blocked in the stage wait so it reaches a kernel, which
/// traps on the poison.
fn fail_at(ctrl: usize, served: u64) {
    poison_host(ctrl, POISON_PROXY | (served + 1));
    // SAFETY: the control page lives in the pinned region for the channel's
    // lifetime; `stage` is written by the GPU (stream memop) and the proxy.
    unsafe { &*((ctrl + STAGE) as *const AtomicU32) }.store(0, Ordering::Release);
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

impl OneShot {
    pub(super) fn new(cfg: Config, shape: Shape, host: usize, dev: u64) -> Result<Self> {
        shape.check()?;
        let (mut state, zero) = (0u64, [0u8; STATE_BYTES]);
        cu(
            unsafe { cuMemAlloc_v2(&mut state, STATE_BYTES) },
            "cuMemAlloc(one-shot state)",
        )?;
        cu(
            unsafe { cuMemcpyHtoD_v2(state, zero.as_ptr().cast(), STATE_BYTES) },
            "cuMemcpyHtoD(one-shot state)",
        )?;
        Ok(Self {
            cfg,
            shape,
            host,
            dev,
            state,
            kernel: AtomicU64::new(0),
            fused: None,
            epoch: AtomicU32::new(0),
            fused_last: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Give channel 0 of a four-rank chain the fused region at device
    /// address `dev` (`fused::entries` slots of `cap` bytes).
    pub(super) fn with_fused(mut self, dev: u64, cap: usize) -> Result<Self> {
        let mut done = 0u64;
        let bytes = 4 * MAX_PIECES;
        cu(
            unsafe { cuMemAlloc_v2(&mut done, bytes) },
            "cuMemAlloc(fused done)",
        )?;
        let zero = [0u8; 4 * MAX_PIECES];
        cu(
            unsafe { cuMemcpyHtoD_v2(done, zero.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD(fused done)",
        )?;
        self.fused = Some(FusedRegion { dev, cap, done });
        Ok(self)
    }

    /// Provide the kernel (`oneshot_kernel_name`); the channel is
    /// unavailable until then.
    pub(in crate::nccl_backend) fn set_kernel(&self, handle: u64) {
        self.kernel.store(handle, Ordering::Relaxed);
    }

    /// Ranks of the channel.
    pub(in crate::nccl_backend) fn ranks(&self) -> usize {
        self.shape.ranks
    }

    /// Largest payload [`Self::enqueue`] accepts (0 while unavailable).
    pub(in crate::nccl_backend) fn max_bytes(&self) -> usize {
        if self.kernel.load(Ordering::Relaxed) == 0 {
            0
        } else {
            self.cfg.max
        }
    }

    /// Why the channel stopped (the kernel trapped or will), if it did.
    pub(in crate::nccl_backend) fn poisoned(&self) -> Option<String> {
        poison_at(self.ctrl())
    }

    /// Host address of the control page.
    fn ctrl(&self) -> usize {
        self.host + ctrl_off(self.cfg.max, self.shape.origins())
    }

    /// Send `src[..bytes]` to the peer and land its payload in `dst` (added
    /// in place when `add`, else copied), all on `stream` and capturable.
    /// A chain only sums in place (`add`, `src == dst`): every rank's
    /// payload, the same on all. `false` (nothing enqueued) when the size or
    /// the call is ineligible or the kernel is missing; every rank sees the
    /// same answer.
    pub(in crate::nccl_backend) fn enqueue(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks > 2 && (!add || src != dst) {
            return Ok(false);
        }
        self.launch(src, dst, bytes, add, Land::Sum, self.cfg.timeout_ns, stream)
    }

    /// A chain's all-gather of every rank's `base + rank * stride`
    /// `[..bytes]` into the same place on every rank, on `stream` and
    /// capturable; `false` as for [`Self::enqueue`], and on a pair.
    pub(in crate::nccl_backend) fn gather(
        &self,
        base: u64,
        stride: i64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks == 2 {
            return Ok(false);
        }
        let own = base.wrapping_add_signed(stride * self.shape.rank as i64);
        self.launch(
            own,
            base,
            bytes,
            false,
            Land::Gather(stride),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// The four-rank chain's reduce-scatter: segment `d` of every rank's
    /// partial at `seg0 + d * stride` (`bytes / 4` each), summed in the
    /// all-reduce's pairwise tree, lands in rank `d`'s own segment, on
    /// `stream` and capturable. `false` as for [`Self::enqueue`], and unless
    /// four ranks, 16-byte alignment and a payload of whole 64-byte pieces
    /// below [`RS_BIT`]. `fp8`: the pieces travel as E4M3 and f32 scales,
    /// summed in f32 (lossy, `oneshot_rs4_fp8`), and must be whole KiB.
    pub(in crate::nccl_backend) fn reduce_scatter(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        fp8: bool,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks != MAX_RANKS
            || !bytes.is_multiple_of(RS_PIECES * 64)
            || bytes >= RS_BIT
            || !seg0.is_multiple_of(16)
            || stride % 16 != 0
            || fp8 && !bytes.is_multiple_of(MAX_RANKS * rs_per_segment(bytes) * 1024)
        {
            return Ok(false);
        }
        // The kernel takes `add` as the reduce-scatter's FP8 switch.
        self.launch(
            seg0,
            seg0,
            bytes,
            fp8,
            Land::ReduceScatter(stride),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// The four-rank chain's all-to-all of `a2a` (`CommBackend::
    /// all_to_all_async`, one op: its four segments within the one-shot
    /// maximum) on `stream`, capturable; `false` as for [`Self::enqueue`],
    /// and unless four ranks and a shape [`A2a::valid`] admits.
    pub(in crate::nccl_backend) fn all_to_all(&self, a2a: A2a, stream: u64) -> Result<bool> {
        let bytes = MAX_RANKS * a2a.segment();
        if self.shape.ranks != MAX_RANKS || !a2a.valid() || bytes >= RS_BIT {
            return Ok(false);
        }
        self.launch(
            0,
            0,
            bytes,
            false,
            Land::AllToAll(a2a),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// Begin a fused reduce-scatter of `seg`-byte segments at `seg0 + d *
    /// stride` (`CommBackend::rs_fused_begin`): its producer plan, its
    /// pieces in the order the producer finishes them (at the ends the raw
    /// segments as `relay::rs_role` sends them, farthest first; in the
    /// middle the segments it sums on, then its raw one; its own last). `None` off channel 0 of a four-rank
    /// chain with a fused region, for a segment beyond its capacity or of
    /// misaligned pieces, and right after another fused op.
    pub(in crate::nccl_backend) fn rs_fused_begin(
        &self,
        seg0: u64,
        stride: i64,
        seg: usize,
    ) -> Result<Option<crate::RsFused>> {
        let Some(f) = self.fused else {
            return Ok(None);
        };
        let bytes = MAX_RANKS * seg;
        let k = rs_per_segment(bytes);
        if self.kernel.load(Ordering::Relaxed) == 0
            || seg == 0
            || seg > f.cap
            || !bytes.is_multiple_of(RS_PIECES * 64)
            || !(seg / k).is_multiple_of(16)
            || !seg0.is_multiple_of(16)
            || stride % 16 != 0
            || (stride.unsigned_abs() as usize) < seg
            || self.fused_last.load(Ordering::Relaxed)
        {
            return Ok(None);
        }
        let piece = seg / k;
        let rank = self.shape.rank;
        let role = relay::rs_role(rank);
        let ctrl = self.dev + ctrl_off(self.cfg.max, self.shape.origins()) as u64;
        let mut plan = crate::RsFusedPlan {
            ready: ctrl + READY as u64,
            state: self.state,
            stage: ctrl + STAGE as u64,
            poison: ctrl + POISON as u64,
            timeout_ns: self.cfg.timeout_ns,
            code: fused::code(bytes) as u32,
            npieces: (MAX_RANKS * k) as u32,
            piece_bytes: piece as u32,
            mode: self.cfg.sm_stage.max(1),
            pad: 0,
            piece: [crate::RsFusedPiece::default(); crate::RS_FUSED_MAX_PIECES],
        };
        // A middle rank produces the segments it sums on first: their raw
        // inputs arrive first (the ends send those first), and their pairs
        // feed the far side's sums, so its links fill from the start; its
        // raw segment (one hop, for an end's own sum) follows.
        let computed = role.computed.map_or(&[][..], |(_, segs)| segs);
        let order = if computed.is_empty() {
            [(role.raw.1, true), (computed, false), (&[rank][..], false)]
        } else {
            [(computed, false), (role.raw.1, true), (&[rank][..], false)]
        };
        let mut at = plan.piece.iter_mut();
        for (segs, raw) in order {
            for &g in segs {
                for j in 0..k {
                    let slot = at.next().context("fused reduce-scatter: too many pieces")?;
                    *slot = crate::RsFusedPiece {
                        out: seg0.wrapping_add_signed(g as i64 * stride) + (j * piece) as u64,
                        s: if raw {
                            f.dev + (fused::raw_index(rank, g) * f.cap + j * piece) as u64
                        } else {
                            0
                        },
                        p: (g * k + j) as u32,
                        pad: 0,
                    };
                }
            }
        }
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(Some(crate::RsFused {
            plan,
            done: f.done,
            epoch,
            piece_bytes: piece,
            pieces: k,
            seg0,
            stride,
            bytes: seg,
        }))
    }

    /// The chain kernel of the fused op `fused` on `stream`, capturable;
    /// `false` as for [`Self::enqueue`].
    pub(in crate::nccl_backend) fn reduce_scatter_fused(
        &self,
        fused: &crate::RsFused,
        stream: u64,
    ) -> Result<bool> {
        self.launch(
            fused.seg0,
            fused.seg0,
            MAX_RANKS * fused.bytes,
            false,
            Land::Fused(fused.stride, fused.epoch),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// `stream` waits until piece `piece` of this rank's segment of `fused`
    /// is final.
    pub(in crate::nccl_backend) fn rs_fused_wait(
        &self,
        fused: &crate::RsFused,
        piece: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            piece < fused.pieces,
            "fused reduce-scatter: piece {piece} of {}",
            fused.pieces
        );
        cu(
            unsafe {
                cuStreamWaitValue32_v2(
                    stream,
                    fused.done + 4 * piece as u64,
                    fused.epoch,
                    CU_STREAM_WAIT_VALUE_GEQ,
                )
            },
            "cuStreamWaitValue32(fused done)",
        )
    }

    /// A chain's broadcast of rank `root`'s `ptr[..bytes]` into every rank's
    /// `ptr`, on `stream` and capturable; `false` as for [`Self::enqueue`],
    /// and on a pair (its broadcasts stay on NCCL). It waits for the root
    /// without a limit, as ncclBroadcast does: a worker idles here until the
    /// head's next command, and the host times a slow broadcast afterwards.
    pub(in crate::nccl_backend) fn broadcast(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks == 2 || root >= self.shape.ranks {
            return Ok(false);
        }
        self.launch(ptr, ptr, bytes, false, Land::Root(root), 0, stream)
    }

    /// One op: stage `src`, then land what `land` says in `dst`, waiting up
    /// to `timeout_ns` (0: no limit).
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        land: Land,
        timeout_ns: u64,
        stream: u64,
    ) -> Result<bool> {
        let ranks = self.shape.ranks;
        // The kernel's `root`: < ranks a broadcast, == ranks the sum (or
        // the pair's exchange), ranks + 1 a gather `stride` bytes apart.
        let (root, stride) = match land {
            Land::Sum => (ranks, 0),
            Land::Root(root) => (root, 0),
            Land::Gather(stride) => (ranks + 1, stride),
            Land::ReduceScatter(stride) => (ranks + 2, stride),
            Land::AllToAll(_) => (ranks + 3, 0),
            Land::Fused(stride, _) => (ranks + 4, stride),
        };
        let mut a2a = match land {
            Land::AllToAll(a2a) => a2a,
            _ => A2a::default(),
        };
        let (mut fused, region) = match (land, self.fused) {
            (Land::Fused(_, epoch), Some(f)) => (
                FusedArg {
                    done: f.done,
                    epoch,
                    pad: 0,
                },
                Some(f),
            ),
            (Land::Fused(..), None) => return Ok(false),
            _ => (FusedArg::default(), None),
        };
        // The reduce-scatter's and the all-to-all's kernels always stage
        // (pipelined, piece by piece), and run every block.
        let rs = matches!(
            land,
            Land::ReduceScatter(_) | Land::AllToAll(_) | Land::Fused(..)
        );
        let kernel = self.kernel.load(Ordering::Relaxed);
        let fits = match region {
            Some(f) => bytes <= MAX_RANKS * f.cap,
            None => self.cfg.eligible(bytes),
        };
        if kernel == 0 || !fits {
            return Ok(false);
        }
        // Whether a fused op may follow (`fused` module doc): not right
        // after one, until an op every rank hears from its neighbours.
        match land {
            Land::Fused(..) => self.fused_last.store(true, Ordering::Relaxed),
            Land::Root(_) => {}
            _ => self.fused_last.store(false, Ordering::Relaxed),
        }
        let ctrl = self.dev + ctrl_off(self.cfg.max, self.shape.origins()) as u64;
        let stage = ctrl + STAGE as u64;
        let sm = if rs {
            self.cfg.sm_stage.max(1)
        } else {
            self.cfg.sm_stage
        };
        let fence = self.cfg.stage_fence && sm == 0;
        if sm == 0 {
            cu(
                unsafe { cuStreamWaitValue32_v2(stream, stage, 0, CU_STREAM_WAIT_VALUE_EQ) },
                "cuStreamWaitValue32(one-shot stage)",
            )?;
            cu(
                unsafe { cuMemcpyAsync(self.dev, src, bytes, stream) },
                "cuMemcpyAsync(one-shot stage)",
            )?;
        }
        if fence {
            // Default flags fence the copy before the word lands.
            cu(
                unsafe { cuStreamWriteValue32_v2(stream, stage, bytes as u32, 0) },
                "cuStreamWriteValue32(one-shot stage)",
            )?;
        }
        let blocks = if rs {
            rs_blocks()
        } else {
            (bytes / 16).div_ceil(THREADS).clamp(1, MAX_BLOCKS)
        } as u32;
        // The kernel's pieces: a reduce-scatter's (`rs_pieces`), else the
        // flood's or the pair's stripes.
        let rails = if rs {
            (MAX_RANKS * rs_per_segment(bytes)) as u32
        } else {
            self.shape.stripe_count(bytes, self.cfg.stripe_min)
        };
        // A fused op lands in its region, slots `cap` bytes apart.
        let (recv, slot_stride) = match region {
            Some(f) => (f.dev, f.cap as u64),
            None => (
                self.dev + recv_off(self.cfg.max, 0, 0) as u64,
                self.cfg.max as u64,
            ),
        };
        let mut args = (
            dst,
            recv,
            slot_stride,
            bytes as u32,
            add as u32,
            ctrl + FLAGS as u64,
            rails,
            self.state,
            if fence { 0 } else { stage },
            ctrl + POISON as u64,
            timeout_ns,
            src,
            self.dev,
            sm,
            self.shape.ranks as u32,
            self.shape.rank as u32,
            root as u32,
            stride,
        );
        // The driver reads as many as the kernel takes: the pair's first 14.
        let mut params: [*mut c_void; 20] = [
            (&raw mut args.0).cast(),
            (&raw mut args.1).cast(),
            (&raw mut args.2).cast(),
            (&raw mut args.3).cast(),
            (&raw mut args.4).cast(),
            (&raw mut args.5).cast(),
            (&raw mut args.6).cast(),
            (&raw mut args.7).cast(),
            (&raw mut args.8).cast(),
            (&raw mut args.9).cast(),
            (&raw mut args.10).cast(),
            (&raw mut args.11).cast(),
            (&raw mut args.12).cast(),
            (&raw mut args.13).cast(),
            (&raw mut args.14).cast(),
            (&raw mut args.15).cast(),
            (&raw mut args.16).cast(),
            (&raw mut args.17).cast(),
            (&raw mut a2a).cast(),
            (&raw mut fused).cast(),
        ];
        // A forced kernel stage (the reduce-scatter under a copy-engine
        // configuration) launches as the copy-engine path does: no PDL.
        let status = if sm != 0 && self.cfg.sm_stage != 0 && self.cfg.pdl {
            // The kernel enters with `atlas_pdl_enter()` before it reads
            // anything its predecessor wrote.
            let mut value = [0u8; 64];
            value[..4].copy_from_slice(&1i32.to_ne_bytes());
            let mut attr = CuLaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value,
            };
            let config = CuLaunchConfig {
                grid: [blocks, 1, 1],
                block: [THREADS as u32, 1, 1],
                shared_mem_bytes: 0,
                stream,
                attrs: &mut attr,
                num_attrs: 1,
            };
            unsafe { cuLaunchKernelEx(&config, kernel, params.as_mut_ptr(), std::ptr::null_mut()) }
        } else {
            unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    THREADS as u32,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        };
        if status != 0 && fence {
            // `stage` is already on the stream: the proxy will serve an op no
            // kernel consumes, and its count would stay ahead of the device
            // sequence (stale receive slots). Stop the channel instead.
            poison_host(self.ctrl(), POISON_LAUNCH);
        }
        cu(status, oneshot_kernel_name(self.shape.ranks))?;
        Ok(true)
    }
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
pub(super) mod relay;
pub(super) use channel::Channel;

#[cfg(test)]
#[path = "oneshot_gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "oneshot_tests.rs"]
mod tests;
