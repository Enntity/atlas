// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::{MAX_RAILS, MAX_RANKS, SPLIT_MIN};

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
    pub(super) fn stripe_count(&self, bytes: usize, stripe_min: usize) -> u32 {
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
pub(super) fn recv_off(max: usize, origin: usize, slot: usize) -> usize {
    (1 + 2 * origin + slot) * max
}
pub(super) fn ctrl_off(max: usize, origins: usize) -> usize {
    (1 + 2 * origins) * max
}
/// The flag line of `origin`'s rail `rail` in the control page.
pub(super) fn flag_line(origin: usize, rail: usize) -> usize {
    FLAGS + ORIGIN_LINES * origin + 64 * rail
}

/// Multi-rank completion words match the payload's two parity slots. Ring
/// flood and specialized collectives can deliver one origin over different
/// QPs, so seq+1 may arrive before seq: it must not publish seq's completion
/// or overwrite its flag. The pair retains its original single-word ABI.
pub(super) fn chain_flag_line(origin: usize, piece: usize, seq: u64) -> usize {
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
pub(super) fn flag_word(seq: u64, bytes: usize) -> u64 {
    (seq << 24) | bytes as u64
}

pub(super) fn describe_poison(word: u64) -> String {
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
pub(super) fn poison_at(ctrl: usize) -> Option<String> {
    // SAFETY: the control page lives in the pinned region for the pair's
    // lifetime; the kernel writes this word only before trapping.
    let word = unsafe { ((ctrl + POISON) as *const u64).read_volatile() };
    (word != 0).then(|| describe_poison(word))
}

/// Stop the channel from the host: every later kernel traps at launch. An
/// earlier reason is kept.
pub(super) fn poison_host(ctrl: usize, word: u64) {
    // SAFETY: as in `poison_at`.
    let poison = unsafe { &*((ctrl + POISON) as *const AtomicU64) };
    let _ = poison.compare_exchange(0, word, Ordering::AcqRel, Ordering::Relaxed);
}

/// A proxy that is going away, with `served` ops sent, leaves nothing to
/// clear `stage` again: poison the channel with control page `ctrl`, then
/// release a stream blocked in the stage wait so it reaches a kernel, which
/// traps on the poison.
pub(super) fn fail_at(ctrl: usize, served: u64) {
    poison_host(ctrl, POISON_PROXY | (served + 1));
    // SAFETY: the control page lives in the pinned region for the channel's
    // lifetime; `stage` is written by the GPU (stream memop) and the proxy.
    unsafe { &*((ctrl + STAGE) as *const AtomicU32) }.store(0, Ordering::Release);
}
