// SPDX-License-Identifier: AGPL-3.0-only

//! Timing with the stand-in peer, and two ranks through the real NIC on one
//! host (GPU + RDMA device; all `#[ignore]`d). The NIC test runs two
//! [`RdmaPair`]s in one process, one stream each on one GPU, bootstrapped
//! over localhost, each rail's two QPs meeting inside the HCA. It exercises
//! the real proxy loop (one-shot ops interleaved with legacy exchanges), the
//! NIC reading copy-engine-staged data, chained flags under
//! `ATLAS_RDMA_PAIR_CHAIN=1`, CUDA-graph replays, and back-to-back eager ops
//! with one rank held up; every result must match `bf16_add_inplace` (add) or
//! the peer's payload (copy) bit for bit.
//!
//!   ATLAS_ONESHOT_CUBIN_DIR=<cubins> ATLAS_RDMA_ONESHOT=1 \
//!   ATLAS_RDMA_RAILS=rocep1s0f0[,roceP2p1s0f0] [ATLAS_RDMA_PAIR_CHAIN=1] \
//!     cargo test -p spark-comm loopback -- --ignored --nocapture
//!
//! (in a container: --device /dev/infiniband --cap-add IPC_LOCK
//! --ulimit memlock=-1:-1)

use super::*;
use crate::nccl_backend::rdma_pair::RdmaPair;

/// The two ranks send each other different data.
fn payload(rank: usize, step: u64, n: usize) -> Vec<u16> {
    if rank == 0 {
        local_payload(step, n)
    } else {
        peer_payload(step, n)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Add,
    Copy,
    /// Legacy-channel all-reduce (host job queue), in place.
    Legacy,
}

/// One collective issued by both ranks, with per-rank buffers.
struct Step {
    bytes: usize,
    kind: Kind,
    src: [u64; 2],
    dst: [u64; 2],
    want: [u64; 2],
}

impl Step {
    fn new(bytes: usize, kind: Kind, misalign: u64) -> Self {
        let dst = [0; 2].map(|_: u64| alloc(bytes + 256) + misalign);
        let src = if kind == Kind::Copy {
            [alloc(bytes), alloc(bytes)]
        } else {
            dst
        };
        Self {
            bytes,
            kind,
            src,
            dst,
            want: [alloc(bytes), alloc(bytes)],
        }
    }

    /// Load both ranks' inputs for payload step `k` (synchronous).
    fn prepare(&self, g: &Gpu, k: u64) {
        let n = self.bytes / 2;
        for rank in 0..2 {
            upload(g, self.src[rank], &payload(rank, k, n));
            if self.kind == Kind::Copy {
                upload(g, self.dst[rank], &vec![0xffff; n]);
            }
        }
        ck!(cuStreamSynchronize(g.stream));
    }

    fn issue(&self, g: &Gpu, pairs: &[RdmaPair], streams: [u64; 2]) {
        for (rank, pair) in pairs.iter().enumerate() {
            let (src, dst, stream) = (self.src[rank], self.dst[rank], streams[rank]);
            let sent = if self.kind == Kind::Legacy {
                pair.exchange(src, dst, self.bytes, stream, |d, s, len| {
                    launch_add(g, stream, d, s, len / 2);
                    Ok(())
                })
            } else {
                let oneshot = pair.oneshot().expect("one-shot channel");
                oneshot.enqueue(src, dst, self.bytes, self.kind != Kind::Copy, stream)
            };
            assert!(sent.unwrap());
        }
    }

    /// Wrong BF16 values across both ranks for payload step `k`.
    fn check(&self, g: &Gpu, k: u64) -> usize {
        let n = self.bytes / 2;
        (0..2)
            .map(|rank| {
                let got = download(g, self.dst[rank], n);
                let want = if self.kind == Kind::Copy {
                    payload(1 - rank, k, n)
                } else {
                    upload(g, self.want[rank], &payload(rank, k, n));
                    upload(g, self.want[1 - rank], &payload(1 - rank, k, n));
                    launch_add(g, g.stream, self.want[rank], self.want[1 - rank], n);
                    download(g, self.want[rank], n)
                };
                got.iter().zip(&want).filter(|(a, b)| a != b).count()
            })
            .sum()
    }
}

fn sync(streams: [u64; 2]) {
    for s in streams {
        ck!(cuStreamSynchronize(s));
    }
}

/// Both ranks' pairs on this GPU, connected through the NIC, or `None` (test
/// skipped) without cubins or `ATLAS_RDMA_ONESHOT=1`.
fn ranks() -> Option<(Gpu, Vec<RdmaPair>)> {
    let g = gpu()?;
    if std::env::var("ATLAS_RDMA_ONESHOT").as_deref() != Ok("1") {
        eprintln!("ATLAS_RDMA_ONESHOT unset: skipped");
        return None;
    }
    let (ctx, port) = (g.ctx, 20_000 + (std::process::id() % 20_000) as u16);
    let pairs: Vec<RdmaPair> = std::thread::scope(|s| {
        let ranks: Vec<_> = (0..2)
            .map(|rank| {
                s.spawn(move || {
                    ck!(cuCtxSetCurrent(ctx));
                    let link = crate::nccl_backend::rdma_pair::bootstrap::Link {
                        at: (std::net::Ipv4Addr::LOCALHOST, port).into(),
                        worker: None,
                        lifeline: None,
                    };
                    RdmaPair::connect(rank, &link, 4 << 20).unwrap()
                })
            })
            .collect();
        ranks.into_iter().map(|r| r.join().unwrap()).collect()
    });
    for pair in &pairs {
        pair.oneshot()
            .expect("one-shot channel")
            .set_kernel(g.oneshot);
    }
    Some((g, pairs))
}

#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn nic_loopback_two_ranks() {
    let Some((g, pairs)) = ranks() else {
        return;
    };
    let streams = [g.stream, new_stream()];
    let steps = [
        Step::new(131_072, Kind::Add, 0),
        Step::new(65_536, Kind::Add, 2),
        Step::new(8192, Kind::Add, 0),
        Step::new(2 << 20, Kind::Legacy, 0),
        Step::new(327_680, Kind::Add, 0),
        Step::new(1 << 20, Kind::Add, 0),
        Step::new(18, Kind::Add, 2),
        Step::new(128, Kind::Copy, 0),
        Step::new(65_536, Kind::Legacy, 0),
        Step::new(65_536, Kind::Copy, 2),
    ];
    // ATLAS_ONESHOT_LOOPBACK_ROUNDS (default 30) scales the run for stress.
    let rounds = std::env::var("ATLAS_ONESHOT_LOOPBACK_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let (mut k, mut wrong) = (0u64, 0usize);
    let eager = |step: &Step, k: &mut u64| {
        *k += 1;
        step.prepare(&g, *k);
        step.issue(&g, &pairs, streams);
        sync(streams);
        step.check(&g, *k)
    };
    // Eager rounds: sizes vary op to op, so a flag carrying another op's
    // size (or stale staged data) shows up as a trap or a wrong value.
    for _ in 0..rounds {
        for step in &steps {
            wrong += eager(step, &mut k);
        }
    }
    let captured: Vec<&Step> = steps.iter().filter(|s| s.kind != Kind::Legacy).collect();
    let execs = [0, 1].map(|rank| {
        capture(
            &Gpu {
                stream: streams[rank],
                ..g
            },
            || {
                for step in &captured {
                    let oneshot = pairs[rank].oneshot().unwrap();
                    let add = step.kind == Kind::Add;
                    let (src, dst) = (step.src[rank], step.dst[rank]);
                    assert!(
                        oneshot
                            .enqueue(src, dst, step.bytes, add, streams[rank])
                            .unwrap()
                    );
                }
            },
        )
    });
    for replay in 0..rounds {
        let base = k;
        for (i, step) in captured.iter().enumerate() {
            step.prepare(&g, base + 1 + i as u64);
        }
        for rank in 0..2 {
            ck!(cuGraphLaunch(execs[rank], streams[rank]));
        }
        sync(streams);
        k += captured.len() as u64;
        for (i, step) in captured.iter().enumerate() {
            wrong += step.check(&g, base + 1 + i as u64);
        }
        wrong += eager(&steps[replay % steps.len()], &mut k);
    }
    // Back-to-back eager rounds, one sync each: before every step one rank's
    // stream is held up by a long add (the rank alternates, the length
    // varies), so the other runs ahead. The early rank waits in its kernel,
    // the late one for `S` to drain, and receive slots are reused with no
    // host sync in between.
    let drag = alloc(8 << 20);
    for round in 0..rounds {
        let base = k;
        for (i, step) in steps.iter().enumerate() {
            step.prepare(&g, base + 1 + i as u64);
        }
        for (i, step) in steps.iter().enumerate() {
            let n = (4 << 20) >> ((i + round) % 8);
            launch_add(&g, streams[(i + round) % 2], drag, drag, n);
            step.issue(&g, &pairs, streams);
        }
        sync(streams);
        k += steps.len() as u64;
        for (i, step) in steps.iter().enumerate() {
            wrong += step.check(&g, base + 1 + i as u64);
        }
    }
    // Back-to-back timing (both ranks share this GPU and HCA).
    let mut timing = String::new();
    for (bytes, kind) in [
        (8192, Kind::Add),
        (65_536, Kind::Add),
        (65_536, Kind::Legacy),
    ] {
        let step = Step::new(bytes, kind, 0);
        step.issue(&g, &pairs, streams);
        sync(streams);
        let t0 = Instant::now();
        for _ in 0..1000 {
            step.issue(&g, &pairs, streams);
        }
        sync(streams);
        let us = t0.elapsed().as_secs_f64() * 1e3;
        timing += &format!(
            " {}{bytes}B {us:.1}us",
            if kind == Kind::Legacy { "legacy " } else { "" }
        );
    }
    let poison: Vec<_> = pairs
        .iter()
        .map(|p| p.oneshot().unwrap().poisoned())
        .collect();
    println!(
        "NIC loopback (chain={}): {k} steps, {wrong} wrong values, poison {poison:?}; per op:{timing}",
        std::env::var("ATLAS_RDMA_PAIR_CHAIN").unwrap_or_default()
    );
    assert_eq!(wrong, 0);
    assert!(poison.iter().all(Option::is_none));
}

/// The peer goes away (its QPs are destroyed), as when its process is killed:
/// our send fails, the proxy stops the channel, and the waiting kernel traps
/// on the poison well inside its time limit instead of hanging.
#[test]
#[ignore = "kills the CUDA context: run alone (GPU, RDMA device, ATLAS_ONESHOT_CUBIN_DIR)"]
fn nic_loopback_peer_loss_traps() {
    let Some((g, mut pairs)) = ranks() else {
        return;
    };
    let streams = [g.stream, new_stream()];
    let step = Step::new(65_536, Kind::Add, 0);
    step.prepare(&g, 1);
    step.issue(&g, &pairs, streams);
    sync(streams);
    assert_eq!(step.check(&g, 1), 0);
    drop(pairs.pop());
    let (t0, os) = (Instant::now(), pairs[0].oneshot().unwrap());
    let (buf, bytes) = (step.dst[0], step.bytes);
    assert!(os.enqueue(buf, buf, bytes, true, g.stream).unwrap());
    let status = unsafe { cuStreamSynchronize(g.stream) };
    let (took, why) = (t0.elapsed(), os.poisoned());
    println!("peer loss: stream status {status} after {took:?}; poison: {why:?}");
    assert_ne!(status, 0, "the kernel must trap");
    assert!(why.expect("poison word set").contains("RDMA proxy failed"));
    assert!(took.as_secs() < 15);
}

/// Microbench (stand-in peer, no NIC): per-op cost of the local path when
/// the peer's data is already there, eager and replayed, next to the legacy
/// local work (stage copy + `bf16_add_inplace`, no memops).
#[test]
#[ignore = "needs a GPU and ATLAS_ONESHOT_CUBIN_DIR"]
fn oneshot_loopback_timing() {
    let Some(g) = gpu() else {
        return;
    };
    let sync = || ck!(cuStreamSynchronize(g.stream));
    for bytes in [8192usize, 32_768, 65_536, 327_680] {
        let os = channel(&g, test_cfg(SPLIT_MIN), 1);
        let stop = Arc::new(AtomicBool::new(false));
        let peer = spawn_peer(&os, Peer::Early(bytes), stop.clone());
        let (buf, scratch) = (alloc(bytes), alloc(bytes));
        let n = 2000;
        let time = |f: &dyn Fn()| {
            f();
            sync();
            let t0 = Instant::now();
            for _ in 0..n {
                f();
            }
            sync();
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        };
        let eager = time(&|| assert!(os.enqueue(buf, buf, bytes, true, g.stream).unwrap()));
        let exec = capture(&g, || {
            for _ in 0..87 {
                assert!(os.enqueue(buf, buf, bytes, true, g.stream).unwrap());
            }
        });
        let replay = time(&|| ck!(cuGraphLaunch(exec, g.stream))) / 87.0;
        let legacy = time(&|| {
            ck!(cuMemcpyAsync(scratch, buf, bytes, g.stream));
            launch_add(&g, g.stream, buf, scratch, bytes / 2);
        });
        stop.store(true, Ordering::Release);
        peer.join().unwrap();
        println!(
            "one-shot {bytes:>7} B, peer already arrived: eager {eager:.2} us/op, graph {replay:.2} us/op; \
             copy+add (no memops) {legacy:.2} us/op"
        );
        assert!(os.poisoned().is_none());
    }
}

/// A comma-separated list of sizes from `key`, else `default`.
fn sizes_from_env(key: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(key)
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn launch_on(f: u64, stream: u64, blocks: u32, params: &mut [*mut c_void]) {
    let null = std::ptr::null_mut();
    ck!(cuLaunchKernel(
        f,
        blocks,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        params.as_mut_ptr(),
        null
    ));
}

/// Stress of what the NIC reads from the staging buffer (two ranks through
/// the NIC, GPU + RDMA device; the staging mode comes from
/// `ATLAS_RDMA_ONESHOT_SM_STAGE`): `ATLAS_ONESHOT_STRESS_OPS` (default
/// 200,000) in-place all-reduces per rank, sizes round-robin from
/// `ATLAS_ONESHOT_STRESS_SIZES` (default 24576,8192,65536,327680). Each op's
/// payload is filled and its result checked on the GPU (`oneshot_stress.cu`
/// in the cubin dir), every element different from the previous op's, with a
/// host sync only every 4096 ops; any stale staged byte is a wrong value.
/// `ATLAS_ONESHOT_STRESS_DRAG_MB` adds an L2-churning add of that many MiB
/// before every 16th op.
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR with oneshot_stress.cubin"]
fn nic_loopback_sm_stage_stress() {
    let Some((g, pairs)) = ranks() else {
        return;
    };
    let fill = load("oneshot_stress.cubin", "stress_fill");
    let check = load("oneshot_stress.cubin", "stress_check");
    let ops: u64 = std::env::var("ATLAS_ONESHOT_STRESS_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let drag_mb: usize = std::env::var("ATLAS_ONESHOT_STRESS_DRAG_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let sizes = sizes_from_env(
        "ATLAS_ONESHOT_STRESS_SIZES",
        &[24_576, 8192, 65_536, 327_680],
    );
    let streams = [g.stream, new_stream()];
    let max = *sizes.iter().max().unwrap();
    let bufs = [alloc(max), alloc(max)];
    let drag = alloc((drag_mb << 20).max(16));
    // Wrong elements, bad blocks, lowest bad op, ops checked.
    let bad = alloc(32);
    let init = [0u64, 0, u64::MAX, 0];
    ck!(cuMemcpyHtoDAsync_v2(
        bad,
        init.as_ptr().cast(),
        32,
        g.stream
    ));
    sync(streams);
    let t0 = Instant::now();
    for k in 1..=ops {
        let bytes = sizes[(k % sizes.len() as u64) as usize];
        let n = (bytes / 2) as u32;
        let blocks = n.div_ceil(256).min(96);
        for rank in 0..2 {
            let stream = streams[rank];
            if drag_mb != 0 && k % 16 == (rank as u64) * 8 {
                launch_add(&g, stream, drag, drag, (drag_mb << 20) / 2);
            }
            let (mut p, mut n, mut r, mut k) = (bufs[rank], n, rank as u32, k);
            let mut args: [*mut c_void; 4] = [
                (&raw mut p).cast(),
                (&raw mut n).cast(),
                (&raw mut r).cast(),
                (&raw mut k).cast(),
            ];
            launch_on(fill, stream, blocks, &mut args);
            let oneshot = pairs[rank].oneshot().unwrap();
            assert!(oneshot.enqueue(p, p, bytes, true, stream).unwrap());
            let (mut add, mut b) = (1u32, bad);
            let mut args: [*mut c_void; 6] = [
                (&raw mut p).cast(),
                (&raw mut n).cast(),
                (&raw mut r).cast(),
                (&raw mut k).cast(),
                (&raw mut add).cast(),
                (&raw mut b).cast(),
            ];
            launch_on(check, stream, blocks, &mut args);
        }
        if k % 4096 == 0 {
            sync(streams);
        }
    }
    sync(streams);
    let took = t0.elapsed();
    let mut got = [0u64; 4];
    ck!(cuMemcpyDtoHAsync_v2(
        got.as_mut_ptr().cast(),
        bad,
        32,
        g.stream
    ));
    sync(streams);
    let poison: Vec<_> = pairs
        .iter()
        .map(|p| p.oneshot().unwrap().poisoned())
        .collect();
    println!(
        "SM-stage stress (sm_stage={}, pdl={}, sizes {sizes:?}, drag {drag_mb} MiB): {} ops checked of {}, \
         {:.2} us/op; {} wrong elements in {} block checks, first bad op {}; poison {poison:?}",
        std::env::var("ATLAS_RDMA_ONESHOT_SM_STAGE").unwrap_or_default(),
        std::env::var("ATLAS_RDMA_ONESHOT_PDL").unwrap_or_default(),
        got[3],
        2 * ops,
        took.as_secs_f64() * 1e6 / ops as f64,
        got[0],
        got[1],
        if got[2] == u64::MAX {
            "none".to_string()
        } else {
            got[2].to_string()
        },
    );
    assert_eq!(got[3], 2 * ops);
    assert_eq!(got[0], 0);
    assert!(poison.iter().all(Option::is_none));
}

/// Launch `bf16_add_inplace` with programmatic dependent launch, as the
/// runtime launches the kernels on its PDL list.
fn launch_add_pdl(g: &Gpu, stream: u64, dst: u64, src: u64, n: usize) {
    let (mut d, mut s, mut c) = (dst, src, n as i32);
    let mut params: [*mut c_void; 3] = [
        (&raw mut d).cast(),
        (&raw mut s).cast(),
        (&raw mut c).cast(),
    ];
    let mut value = [0u8; 64];
    value[..4].copy_from_slice(&1i32.to_ne_bytes());
    let mut attr = CuLaunchAttribute {
        id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value,
    };
    let config = CuLaunchConfig {
        grid: [(n as u32).div_ceil(256), 1, 1],
        block: [256, 1, 1],
        shared_mem_bytes: 0,
        stream,
        attrs: &mut attr,
        num_attrs: 1,
    };
    let null = std::ptr::null_mut();
    ck!(cuLaunchKernelEx(&config, g.add, params.as_mut_ptr(), null));
}

/// Microbench (stand-in peer, no NIC; the peer's data is always there): the
/// all-reduce's added cost inside a PDL chain, producer -> all-reduce ->
/// consumer (`bf16_add_inplace` both, PDL-launched), per staging mode: copy
/// engine, SM staging, SM staging + PDL. Eager and as a replayed graph of 64
/// links; the chain without the all-reduce is the baseline. Each link starts
/// with a PDL-launched add over `ATLAS_ONESHOT_CHAIN_WORK_KB` (default 1024;
/// 0 = none) standing in for the layer's other kernels, so the previous send
/// has drained (as between decode all-reduces) and only the all-reduce's own
/// critical path is added. Sizes from `ATLAS_ONESHOT_CHAIN_SIZES` (default
/// 24576,65536,327680).
#[test]
#[ignore = "needs a GPU and ATLAS_ONESHOT_CUBIN_DIR"]
fn oneshot_chain_timing() {
    let Some(g) = gpu() else {
        return;
    };
    let sync = || ck!(cuStreamSynchronize(g.stream));
    let n = 2000;
    let time = |f: &dyn Fn()| {
        f();
        sync();
        let t0 = Instant::now();
        for _ in 0..n {
            f();
        }
        sync();
        t0.elapsed().as_secs_f64() * 1e6 / n as f64
    };
    let work_kb: usize = std::env::var("ATLAS_ONESHOT_CHAIN_WORK_KB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let work = alloc((work_kb << 10).max(16));
    let layer = || {
        if work_kb != 0 {
            launch_add_pdl(&g, g.stream, work, work, (work_kb << 10) / 2);
        }
    };
    for bytes in sizes_from_env("ATLAS_ONESHOT_CHAIN_SIZES", &[24_576, 65_536, 327_680]) {
        let half = bytes / 2;
        let (x, buf, y) = (alloc(bytes), alloc(bytes), alloc(bytes));
        let base_link = || {
            layer();
            launch_add_pdl(&g, g.stream, buf, x, half);
            launch_add_pdl(&g, g.stream, y, buf, half);
        };
        let links = 64;
        let base_exec = capture(&g, || (0..links).for_each(|_| base_link()));
        let base = time(&base_link);
        let base_graph = time(&|| ck!(cuGraphLaunch(base_exec, g.stream))) / links as f64;
        let mut line = format!(
            "\nchain {bytes:>7} B, work {work_kb} KiB: baseline eager {base:.2} us/link, graph {base_graph:.2}; all-reduce adds"
        );
        for (name, sm_stage, pdl) in [
            ("copy engine", 0, false),
            ("SM", 1, false),
            ("SM+PDL", 1, true),
        ] {
            let cfg = Config {
                sm_stage,
                pdl,
                stage_fence: false,
                ..test_cfg(SPLIT_MIN)
            };
            let os = channel(&g, cfg, 1);
            let stop = Arc::new(AtomicBool::new(false));
            let peer = spawn_peer(&os, Peer::Early(bytes), stop.clone());
            let link = || {
                layer();
                launch_add_pdl(&g, g.stream, buf, x, half);
                assert!(os.enqueue(buf, buf, bytes, true, g.stream).unwrap());
                launch_add_pdl(&g, g.stream, y, buf, half);
            };
            let eager = time(&link);
            let exec = capture(&g, || (0..links).for_each(|_| link()));
            let graph = time(&|| ck!(cuGraphLaunch(exec, g.stream))) / links as f64;
            stop.store(true, Ordering::Release);
            peer.join().unwrap();
            assert!(os.poisoned().is_none());
            line += &format!(
                "; {name}: eager +{:.2} us, graph +{:.2} us",
                eager - base,
                graph - base_graph
            );
        }
        println!("{line}");
    }
}
