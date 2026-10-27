// SPDX-License-Identifier: AGPL-3.0-only

//! A chain of four ranks through the real NIC on one host (GPU + RDMA
//! device; `#[ignore]`d): four [`RdmaChain`]s in one process, one stream each
//! on one GPU, bootstrapped through rank 0 over localhost, every link's QPs
//! meeting inside the HCA. It runs the real chain proxies (own sends,
//! forwards, completions reaped as they come) and the chain kernel. Every
//! all-reduce must equal the pairwise tree of `bf16_add_inplace` on the four
//! payloads, (x0 + x1) + (x2 + x3), bit for bit on every rank; every
//! broadcast the root's payload; every gather each rank's part in its place
//! (in sequence-parallel prefill's order: rank 0's part on top); every
//! reduce-scatter the all-reduce's words in each rank's own part (SP's
//! order), interleaved with the flood ops. Eager ops, graph replays, ops above the
//! one-shot maximum, and back-to-back ops with one rank held up.
//!
//!   ATLAS_ONESHOT_CUBIN_DIR=<cubins> ATLAS_RDMA_ONESHOT=1 ATLAS_RDMA_PAIR_CHAIN=1 \
//!   ATLAS_RDMA_LEFT_RAILS=rocep1s0f1,roceP2p1s0f1 ATLAS_RDMA_RIGHT_RAILS=rocep1s0f1,roceP2p1s0f1 \
//!   ATLAS_RDMA_RS_BLOCKS=16 [ATLAS_RDMA_ONESHOT_SM_STAGE=1 ATLAS_RDMA_ONESHOT_PDL=0] \
//!     cargo test -p spark-comm chain_loopback -- --ignored --nocapture
//!
//! With SM staging keep PDL off here: four ranks share one GPU, and under
//! PDL each kernel lets its successor be scheduled before it waits, so a
//! replayed graph's later kernels can fill the GPU before every rank's
//! current one is resident (one rank per GPU, as in serving, cannot). For
//! the same reason the reduce-scatter's grid stays at 16 blocks here
//! (`ATLAS_RDMA_RS_BLOCKS`): four ranks of 48 would not all be resident.

use super::*;
use crate::nccl_backend::rdma_pair::bootstrap::chain::Rendezvous;
use crate::nccl_backend::rdma_pair::chain::RdmaChain;

const RANKS: usize = 4;

/// Rank `rank`'s payload for op `k`: finite BF16 values (no NaN or inf, so
/// the comparison is by bits), every fifth op small integers.
fn payload(rank: usize, k: u64, n: usize) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let x = mix((k << 34) ^ ((rank as u64) << 32) ^ i);
            if k.is_multiple_of(5) {
                f32::to_bits((x % 61) as f32 - 30.0).wrapping_shr(16) as u16
            } else {
                // Sign, an exponent below 0xff, any mantissa.
                (x as u16 & 0x807f) | (((x >> 16) % 0xfe) as u16) << 7
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Sum,
    Broadcast(usize),
    /// `bytes / RANKS` per rank, rank `r`'s at part `RANKS - 1 - r`
    /// (`false`) or part `r` (`true`).
    Gather(bool),
    /// `bytes / RANKS` per rank, rank `r`'s at part `RANKS - 1 - r`.
    ReduceScatter,
}

/// One collective issued by every rank, in place on per-rank buffers.
struct Step {
    bytes: usize,
    kind: Kind,
    buf: [u64; RANKS],
    want: [u64; RANKS],
}

impl Step {
    fn new(bytes: usize, kind: Kind, misalign: u64) -> Self {
        Self {
            bytes,
            kind,
            buf: [0; RANKS].map(|_: u64| alloc(bytes + 256) + misalign),
            want: [0; RANKS].map(|_: u64| alloc(bytes)),
        }
    }

    fn prepare(&self, g: &Gpu, k: u64) {
        for rank in 0..RANKS {
            upload(g, self.buf[rank], &payload(rank, k, self.bytes / 2));
        }
        ck!(cuStreamSynchronize(g.stream));
    }

    fn issue(&self, chains: &[RdmaChain], streams: &[u64; RANKS]) {
        for (rank, chain) in chains.iter().enumerate() {
            self.issue_on(chain, rank, streams[rank]);
        }
    }

    fn issue_on(&self, chain: &RdmaChain, rank: usize, stream: u64) {
        let ptr = self.buf[rank];
        let ok = match self.kind {
            Kind::Sum => chain.all_reduce(ptr, self.bytes, stream),
            Kind::Broadcast(root) => chain.oneshot().broadcast(ptr, self.bytes, root, stream),
            Kind::Gather(up) => {
                let part = self.bytes / RANKS;
                if up {
                    chain.gather(ptr, part as i64, part, stream)
                } else {
                    let top = ptr + ((RANKS - 1) * part) as u64;
                    chain.gather(top, -(part as i64), part, stream)
                }
            }
            Kind::ReduceScatter => {
                let part = self.bytes / RANKS;
                let top = ptr + ((RANKS - 1) * part) as u64;
                chain.reduce_scatter(top, -(part as i64), part, false, stream)
            }
        };
        assert!(ok.unwrap());
    }

    /// Wrong values over all ranks against the reference for op `k`.
    fn check(&self, g: &Gpu, k: u64) -> usize {
        let n = self.bytes / 2;
        let want = match self.kind {
            Kind::Broadcast(root) => payload(root, k, n),
            Kind::Gather(up) => {
                let part = n / RANKS;
                let origin = |at: usize| if up { at } else { RANKS - 1 - at };
                (0..RANKS)
                    .flat_map(|at| payload(origin(at), k, n)[at * part..][..part].to_vec())
                    .collect()
            }
            Kind::Sum | Kind::ReduceScatter => {
                for (rank, &w) in self.want.iter().enumerate() {
                    upload(g, w, &payload(rank, k, n));
                }
                let w = self.want;
                launch_add(g, g.stream, w[0], w[1], n);
                launch_add(g, g.stream, w[2], w[3], n);
                launch_add(g, g.stream, w[0], w[2], n);
                download(g, w[0], n)
            }
        };
        (0..RANKS)
            .map(|rank| {
                let got = download(g, self.buf[rank], n);
                // A reduce-scatter leaves only this rank's own part.
                let (from, len) = if self.kind == Kind::ReduceScatter {
                    ((RANKS - 1 - rank) * n / RANKS, n / RANKS)
                } else {
                    (0, n)
                };
                got[from..][..len]
                    .iter()
                    .zip(&want[from..])
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum()
    }
}

/// Synchronize every rank's stream, naming each rank's one-shot fault if
/// one failed.
fn sync(streams: &[u64; RANKS], chains: &[RdmaChain]) {
    for &s in streams {
        let status = unsafe { cuStreamSynchronize(s) };
        if status != 0 {
            let why: Vec<_> = chains.iter().map(|c| c.oneshot().poisoned()).collect();
            panic!("stream failed with CUDA status {status}; one-shot faults by rank: {why:?}");
        }
    }
}

/// Four chain ranks on this GPU through the NIC, or `None` (skipped).
fn ranks() -> Option<(Gpu, Vec<RdmaChain>)> {
    let g = gpu()?;
    if std::env::var("ATLAS_RDMA_ONESHOT").as_deref() != Ok("1") {
        eprintln!("ATLAS_RDMA_ONESHOT unset: skipped");
        return None;
    }
    let chain_kernel = load("rdma_oneshot.cubin", "rdma_chain_bf16");
    let (ctx, port) = (g.ctx, 20_000 + (std::process::id() % 20_000) as u16);
    let chains: Vec<RdmaChain> = std::thread::scope(|s| {
        let ranks: Vec<_> = (0..RANKS)
            .map(|rank| {
                s.spawn(move || {
                    ck!(cuCtxSetCurrent(ctx));
                    let at = Rendezvous {
                        at: (std::net::Ipv4Addr::LOCALHOST, port).into(),
                        workers: vec![std::net::Ipv4Addr::LOCALHOST.into()],
                        lifelines: Vec::new(),
                    };
                    RdmaChain::connect(rank, RANKS, &at).unwrap()
                })
            })
            .collect();
        ranks.into_iter().map(|r| r.join().unwrap()).collect()
    });
    for chain in &chains {
        for oneshot in chain.channels() {
            oneshot.set_kernel(chain_kernel);
        }
    }
    Some((g, chains))
}

#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_four_ranks() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let max = chains[0].oneshot().max_bytes();
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    let steps = [
        Step::new(65_536, Kind::Sum, 0),
        Step::new(8192, Kind::Sum, 2),
        Step::new(131_072, Kind::Sum, 0),
        Step::new(18, Kind::Sum, 2),
        Step::new(4, Kind::Broadcast(0), 0),
        Step::new(327_680, Kind::Sum, 0),
        Step::new(max, Kind::Sum, 0),
        Step::new(4096, Kind::Broadcast(3), 0),
        Step::new(2 * max + 4096, Kind::Sum, 0),
        Step::new(24_576, Kind::Sum, 0),
        Step::new(12, Kind::Broadcast(2), 2),
        Step::new(32_768, Kind::Broadcast(1), 0),
        Step::new(4 * 8192, Kind::Gather(false), 0),
        Step::new(4 * 18, Kind::Gather(false), 2),
        Step::new(4 * (2 * max + 4096), Kind::Gather(false), 0),
        Step::new(4 * 8192, Kind::Gather(true), 2),
        Step::new(4 * (max + 64), Kind::Gather(true), 0),
        Step::new(4 * 8192, Kind::ReduceScatter, 0),
        Step::new(4 * 128, Kind::ReduceScatter, 0),
        Step::new(4 * (2 * max + 4096), Kind::ReduceScatter, 0),
        Step::new(65_536, Kind::Sum, 0),
        Step::new(4 * 65_536, Kind::ReduceScatter, 0),
    ];
    let rounds = std::env::var("ATLAS_ONESHOT_LOOPBACK_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let (mut k, mut wrong) = (0u64, 0usize);
    let eager = |step: &Step, k: &mut u64| {
        *k += 1;
        step.prepare(&g, *k);
        step.issue(&chains, &streams);
        sync(&streams, &chains);
        step.check(&g, *k)
    };
    for _ in 0..rounds {
        for step in &steps {
            wrong += eager(step, &mut k);
        }
    }
    let execs = [0, 1, 2, 3].map(|rank| {
        capture(
            &Gpu {
                stream: streams[rank],
                ..g
            },
            || {
                for step in &steps {
                    step.issue_on(&chains[rank], rank, streams[rank]);
                }
            },
        )
    });
    for replay in 0..rounds {
        let base = k;
        for (i, step) in steps.iter().enumerate() {
            step.prepare(&g, base + 1 + i as u64);
        }
        for rank in 0..RANKS {
            ck!(cuGraphLaunch(execs[rank], streams[rank]));
        }
        sync(&streams, &chains);
        k += steps.len() as u64;
        for (i, step) in steps.iter().enumerate() {
            wrong += step.check(&g, base + 1 + i as u64);
        }
        wrong += eager(&steps[replay % steps.len()], &mut k);
    }
    // Back to back, one sync per round, one rank's stream held up before
    // every step (the rank rotates, the length varies): the others run
    // ahead, their payloads wait at the proxies, slots are reused with no
    // host sync between.
    let drag = alloc(8 << 20);
    for round in 0..rounds {
        let base = k;
        for (i, step) in steps.iter().enumerate() {
            step.prepare(&g, base + 1 + i as u64);
        }
        for (i, step) in steps.iter().enumerate() {
            let n = (4 << 20) >> ((i + round) % 8);
            launch_add(&g, streams[(i + round) % RANKS], drag, drag, n);
            step.issue(&chains, &streams);
        }
        sync(&streams, &chains);
        k += steps.len() as u64;
        for (i, step) in steps.iter().enumerate() {
            wrong += step.check(&g, base + 1 + i as u64);
        }
    }
    // `ATLAS_RDMA_ONESHOT_CHANNELS=2`: every rank runs the steps on channel
    // 0 on its stream and, at once, on channel 1 on a second stream (the
    // order differs: channel 1 runs them backwards), with no sync between.
    if chains[0].channels().len() > 1 {
        let second = [0; RANKS].map(|_| new_stream());
        let other: Vec<Step> = steps
            .iter()
            .rev()
            .map(|s| Step::new(s.bytes, s.kind, 0))
            .collect();
        for _ in 0..rounds {
            let base = k;
            for (i, step) in steps.iter().chain(&other).enumerate() {
                step.prepare(&g, base + 1 + i as u64);
            }
            for rank in 0..RANKS {
                for (a, b) in steps.iter().zip(&other) {
                    a.issue_on(&chains[rank], rank, streams[rank]);
                    let _one = crate::chain_channel::enter(1);
                    b.issue_on(&chains[rank], rank, second[rank]);
                }
            }
            sync(&streams, &chains);
            sync(&second, &chains);
            for (i, step) in steps.iter().chain(&other).enumerate() {
                wrong += step.check(&g, base + 1 + i as u64);
            }
            k += 2 * steps.len() as u64;
        }
        // Back to back on both channels at once, no sync: the large ops
        // (split all-reduce, pipelined gather and reduce-scatter) of each
        // channel interleave with the other's. A gather in place repeats
        // exactly; the others only must not fault.
        let big: Vec<Step> = [Kind::Sum, Kind::Gather(false), Kind::ReduceScatter]
            .map(|kind| Step::new(4 << 20, kind, 0))
            .into_iter()
            .collect();
        let gathers = [
            Step::new(4 << 20, Kind::Gather(false), 0),
            Step::new(4 << 20, Kind::Gather(false), 0),
        ];
        k += 1;
        gathers.iter().for_each(|g2| g2.prepare(&g, k));
        for _ in 0..200 {
            for rank in 0..RANKS {
                for step in &big {
                    step.issue_on(&chains[rank], rank, streams[rank]);
                }
                gathers[0].issue_on(&chains[rank], rank, streams[rank]);
                let _one = crate::chain_channel::enter(1);
                for step in big.iter().rev() {
                    step.issue_on(&chains[rank], rank, second[rank]);
                }
                gathers[1].issue_on(&chains[rank], rank, second[rank]);
            }
        }
        sync(&streams, &chains);
        sync(&second, &chains);
        wrong += gathers.iter().map(|g2| g2.check(&g, k)).sum::<usize>();
        println!(
            "two channels: {rounds} rounds of both step lists at once, 200 back-to-back rounds"
        );
    }
    let mut timing = String::new();
    for (bytes, kind) in [
        (8192, Kind::Sum),
        (32_768, Kind::Sum),
        (65_536, Kind::Sum),
        (4 << 20, Kind::Sum),
        (4 << 20, Kind::Gather(false)),
        (4 << 20, Kind::ReduceScatter),
    ] {
        let step = Step::new(bytes, kind, 0);
        step.issue(&chains, &streams);
        sync(&streams, &chains);
        let t0 = Instant::now();
        for _ in 0..1000 {
            step.issue(&chains, &streams);
        }
        sync(&streams, &chains);
        let what = match kind {
            Kind::Gather(_) => "gather",
            Kind::ReduceScatter => "reduce-scatter",
            _ => "sum",
        };
        timing += &format!(" {what} {bytes}B {:.1}us", t0.elapsed().as_secs_f64() * 1e3);
    }
    let poison: Vec<_> = chains.iter().map(|c| c.oneshot().poisoned()).collect();
    println!(
        "chain NIC loopback: {k} steps, {wrong} wrong values, poison {poison:?}; back-to-back per op:{timing}"
    );
    assert_eq!(wrong, 0);
    assert!(poison.iter().all(Option::is_none));
}

/// The KDA o_proj exchange of an `n`-row chunk (`ATLAS_ONESHOT_A2A_ROWS`,
/// default 8192): each rank holds `[n, 2048]` E4M3 rows of its heads and
/// their tiled UE8M0 scales (128 x 4 per 512-byte tile); part `d` of the rows
/// belongs to rank `3 - d`, which lands every origin's part as the K slice
/// `[o * 2048, ..)` of its `[n / 4, 8192]` operand and the scale tiles at
/// that slice's tile columns. Checked byte for byte, then timed back to back.
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_all_to_all() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let n: usize = std::env::var("ATLAS_ONESHOT_A2A_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let (kl, rows) = (2048usize, n / RANKS);
    let (q_part, sf_part) = (rows * kl, rows * kl / 32);
    let tile_row = 16 * 512; // one 128-row band of a 2048-column slice
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    // Distinct bytes per rank and position (any u8 is an E4M3 / UE8M0 code).
    let fill = |rank: usize, salt: u64, bytes: usize| -> Vec<u16> {
        (0..bytes / 2)
            .map(|i| mix(salt ^ (rank as u64) << 40 ^ i as u64) as u16)
            .collect()
    };
    let srcs: Vec<(u64, u64, Vec<u16>, Vec<u16>)> = (0..RANKS)
        .map(|r| {
            let (q, sf) = (alloc(RANKS * q_part), alloc(RANKS * sf_part));
            let (hq, hs) = (fill(r, 1, RANKS * q_part), fill(r, 2, RANKS * sf_part));
            upload(&g, q, &hq);
            upload(&g, sf, &hs);
            (q, sf, hq, hs)
        })
        .collect();
    let dsts: Vec<(u64, u64)> = (0..RANKS)
        .map(|_| (alloc(rows * RANKS * kl), alloc(rows * RANKS * kl / 32)))
        .collect();
    ck!(cuStreamSynchronize(g.stream));
    let planes = |r: usize| {
        let (q, sf, _, _) = &srcs[r];
        let (a, a_sf) = dsts[r];
        // Part d (for rank 3 - d) at d parts in: rank d's part counts down.
        [
            crate::AllToAllPlane {
                src: q + ((RANKS - 1) * q_part) as u64,
                src_stride: -(q_part as i64),
                dst: a,
                dst_stride: kl as i64,
                rows,
                row_bytes: kl,
                pitch: RANKS * kl,
            },
            crate::AllToAllPlane {
                src: sf + ((RANKS - 1) * sf_part) as u64,
                src_stride: -(sf_part as i64),
                dst: a_sf,
                dst_stride: tile_row as i64,
                rows: rows / 128,
                row_bytes: tile_row,
                pitch: RANKS * tile_row,
            },
        ]
    };
    let issue = || {
        for (r, chain) in chains.iter().enumerate() {
            assert!(chain.all_to_all(&planes(r), streams[r]).unwrap());
        }
    };
    // The shape needs ops of 1/16 of the chunk at least (two planes of whole
    // rows): at the default 1 MiB one-shot maximum an 8K chunk does not fit.
    let fits: Vec<bool> = (0..RANKS)
        .map(|r| chains[r].all_to_all(&planes(r), streams[r]).unwrap())
        .collect();
    if !fits.iter().all(|&f| f) {
        assert!(fits.iter().all(|&f| !f), "the ranks disagree on the shape");
        eprintln!(
            "{n}-row all-to-all exceeds the one-shot maximum: skipped (ATLAS_RDMA_ONESHOT_MAX)"
        );
        return;
    }
    sync(&streams, &chains);
    issue();
    sync(&streams, &chains);
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut wrong = 0usize;
    for (r, &(a, a_sf)) in dsts.iter().enumerate() {
        let got_q = bytes(&download(&g, a, rows * RANKS * kl / 2));
        let got_sf = bytes(&download(&g, a_sf, rows * RANKS * kl / 64));
        let part = RANKS - 1 - r;
        for (o, src) in srcs.iter().enumerate() {
            let (q, sf) = (bytes(&src.2), bytes(&src.3));
            for row in 0..rows {
                let want = &q[part * q_part + row * kl..][..kl];
                let at = row * RANKS * kl + o * kl;
                wrong += got_q[at..at + kl]
                    .iter()
                    .zip(want)
                    .filter(|(x, y)| x != y)
                    .count();
            }
            for t in 0..rows / 128 {
                let want = &sf[part * sf_part + t * tile_row..][..tile_row];
                let at = t * RANKS * tile_row + o * tile_row;
                wrong += got_sf[at..at + tile_row]
                    .iter()
                    .zip(want)
                    .filter(|(x, y)| x != y)
                    .count();
            }
        }
    }
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        issue();
    }
    sync(&streams, &chains);
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let poison: Vec<_> = chains.iter().map(|c| c.oneshot().poisoned()).collect();
    println!(
        "chain NIC loopback all-to-all: {n} rows, {} KiB a pair a rank, {wrong} wrong bytes, poison {poison:?}; {us:.0} us back to back",
        (q_part + sf_part) >> 10
    );
    assert_eq!(wrong, 0);
    assert!(poison.iter().all(Option::is_none));
}

/// The lossy FP8 reduce-scatter (`reduce_scatter_fp8_async`): every rank's
/// own segment is within the quantization's bound of the exact sum of the
/// four BF16 partials (each of at most three E4M3 roundings on a path off by
/// at most 1/16 of its value, plus the BF16 rounding), close on average, and
/// the same bits on a second run of the same inputs; one op and several.
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_reduce_scatter_fp8() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let max = chains[0].oneshot().max_bytes();
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    let bf16 = |v: u16| f32::from_bits(u32::from(v) << 16) as f64;
    // Magnitudes 2^-4 .. 2^4 within a row, either sign.
    let fill = |rank: usize, salt: u64, n: usize| -> Vec<u16> {
        (0..n as u64)
            .map(|i| {
                let x = mix(salt << 40 ^ (rank as u64) << 36 ^ i);
                let v = ((x % 2001) as f32 - 1000.0) / 1000.0 * 2f32.powi((x >> 20) as i32 % 8 - 4);
                (v.to_bits() >> 16) as u16
            })
            .collect()
    };
    let sizes = [
        4 * 8192,
        4 * 65_536,
        4 * 1_048_576,
        4 * (max / 4 / 8192 * 8192 + 8192),
    ];
    let mut worst = 0f64;
    for (salt, &bytes) in sizes.iter().enumerate() {
        let (n, part) = (bytes / 2, bytes / RANKS);
        let inputs: Vec<Vec<u16>> = (0..RANKS).map(|r| fill(r, salt as u64, n)).collect();
        let bufs: Vec<u64> = (0..RANKS).map(|_| alloc(bytes)).collect();
        let run = || -> Vec<Vec<u16>> {
            for r in 0..RANKS {
                upload(&g, bufs[r], &inputs[r]);
            }
            ck!(cuStreamSynchronize(g.stream));
            for (r, chain) in chains.iter().enumerate() {
                let top = bufs[r] + ((RANKS - 1) * part) as u64;
                assert!(
                    chain
                        .reduce_scatter(top, -(part as i64), part, true, streams[r])
                        .unwrap()
                );
            }
            sync(&streams, &chains);
            (0..RANKS).map(|r| download(&g, bufs[r], n)).collect()
        };
        let (first, second) = (run(), run());
        let (mut bad, mut rel, mut count) = (0usize, 0f64, 0usize);
        for rank in 0..RANKS {
            let from = (RANKS - 1 - rank) * n / RANKS;
            assert_eq!(
                first[rank][from..][..n / RANKS],
                second[rank][from..][..n / RANKS],
                "rank {rank}: a second run of the same inputs differs ({bytes} bytes)"
            );
            for i in from..from + n / RANKS {
                let xs: Vec<f64> = (0..RANKS).map(|r| bf16(inputs[r][i])).collect();
                let exact: f64 = xs.iter().sum();
                let mag: f64 = xs.iter().map(|x| x.abs()).sum();
                let got = bf16(first[rank][i]);
                let err = (got - exact).abs();
                // Three roundings of at most 1/16 of a value no larger than
                // `mag`, the BF16 rounding, and E4M3's floor under a group
                // whose largest value is at most 16 * 4.
                if err > 3.0 * mag / 16.0 + exact.abs() / 128.0 + 64.0 / 448.0 / 512.0 * 3.0 {
                    bad += 1;
                }
                if exact.abs() > 0.5 {
                    rel += err / exact.abs();
                    count += 1;
                }
            }
        }
        let mean = rel / count.max(1) as f64;
        worst = worst.max(mean);
        eprintln!(
            "fp8 reduce-scatter {bytes} B: {bad} values out of bound, mean relative error {mean:.4} over {count}"
        );
        assert_eq!(bad, 0, "{bytes} bytes: values outside the FP8 bound");
    }
    assert!(worst < 0.1, "mean relative error {worst:.4}");
}

/// The fused reduce-scatter (`rs_fused_begin`, `rdma_rs_fused_copy` as the
/// producer, `reduce_scatter_fused_async`): each rank's own segment is the
/// plain reduce-scatter's words, its completion words show the op's epoch,
/// over seams of several sizes with an all-reduce between them (a fused op
/// right after another is refused), the producer and the chain kernel on
/// two streams with nothing ordering them. Needs ATLAS_RDMA_RS_FUSED_SEG
/// (skipped without).
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_reduce_scatter_fused() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let producer = load("rdma_oneshot.cubin", "rdma_rs_fused_copy");
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    let sides = [new_stream(), new_stream(), new_stream(), new_stream()];
    let cap: usize = std::env::var("ATLAS_RDMA_RS_FUSED_SEG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if cap == 0 {
        eprintln!("ATLAS_RDMA_RS_FUSED_SEG unset: skipped");
        return;
    }
    let mut wrong = 0;
    for (k, seg) in [65_536usize, 1 << 20, cap / 2, cap].into_iter().enumerate() {
        let k = k as u64 * 7 + 3;
        let step = Step::new(4 * seg, Kind::ReduceScatter, 0);
        step.prepare(&g, k);
        let fused: Vec<crate::RsFused> = chains
            .iter()
            .enumerate()
            .map(|(r, c)| {
                let top = step.buf[r] + (3 * seg) as u64;
                c.rs_fused_begin(top, -(seg as i64), seg)
                    .unwrap()
                    .expect("fused op refused")
            })
            .collect();
        for (r, chain) in chains.iter().enumerate() {
            let mut plan = fused[r].plan;
            let mut params: [*mut c_void; 1] = [(&raw mut plan).cast()];
            ck!(cuLaunchKernel(
                producer,
                8,
                1,
                1,
                256,
                1,
                1,
                0,
                streams[r],
                params.as_mut_ptr(),
                std::ptr::null_mut()
            ));
            assert!(chain.reduce_scatter_fused(&fused[r], sides[r]).unwrap());
            // A second fused op before an all-reduce is refused.
            let top = step.buf[r] + (3 * seg) as u64;
            assert!(
                chain
                    .rs_fused_begin(top, -(seg as i64), seg)
                    .unwrap()
                    .is_none()
            );
        }
        sync(&sides, &chains);
        sync(&streams, &chains);
        let bad = step.check(&g, k);
        for (r, f) in fused.iter().enumerate() {
            let mut done = vec![0u32; f.pieces];
            ck!(cuMemcpyDtoHAsync_v2(
                done.as_mut_ptr().cast(),
                f.done,
                4 * f.pieces,
                g.stream
            ));
            ck!(cuStreamSynchronize(g.stream));
            assert!(
                done.iter().all(|&d| d == f.epoch),
                "rank {r}: done words {done:?}, epoch {}",
                f.epoch
            );
        }
        eprintln!("fused reduce-scatter of {seg}-byte segments: {bad} wrong");
        wrong += bad;
        // An all-reduce between fused ops.
        let ar = Step::new(65_536, Kind::Sum, 0);
        ar.prepare(&g, k + 1);
        ar.issue(&chains, &streams);
        sync(&streams, &chains);
        wrong += ar.check(&g, k + 1);
    }
    assert_eq!(wrong, 0);
}
