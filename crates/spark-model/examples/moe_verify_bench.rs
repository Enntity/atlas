// SPDX-License-Identifier: AGPL-3.0-only
//! Cold-weight DRAM bandwidth of the routed-MoE native-FP4 grouped GEMMs at
//! GLM-5.3 Flash decode/verify shapes under expert TP: 288 experts, all
//! local, H=4096, per-rank expert intermediate I=1024, top-8 routing of T
//! tokens over U distinct experts. Times the kernels production launches,
//! through the production `ops` wrappers:
//!   * compact gate+up  `moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up`
//!     over a `moe_build_tile_worklist` M64 x N128 worklist;
//!   * dense down       `moe_w4a4_grouped_gemm_prequant_t_k128` (N/128, m_tiles, E);
//!   * K128W gate+up+SiLU `moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w`
//!     and K128W down  `moe_w4a4_grouped_gemm_prequant_t_k128w_compact`, both
//!     over the `moe_mtile_prefix` row-tile grid.
//!
//! Each rep is bracketed by CUDA events on one stream (median of the reps).
//! Two cold regimes, both over a ring of routings whose active expert sets
//! are disjoint (consecutive reps never share weights):
//!   * flush:  a 256 MB scratch memset before every rep (evicts L2, but
//!     leaves it dirty, so the kernel pays some write-back);
//!   * rotate: no memset, the disjoint expert ring alone keeps weights cold.
//! GB/s = weight + block-scale bytes of the U active experts / time.
//!
//! Exit: 0 ok, 1 non-finite output, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example moe_verify_bench -- [T:U ...]
//! Default workloads 8:41 16:62 32:90; `MOE_VERIFY_REPS` (default 30).

use anyhow::{Result, bail};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut i32,
        func: u64,
        block: i32,
        smem: usize,
    ) -> i32;
}

const MODULE: &str = "moe_w4a16";
const EXPERTS: usize = 288;
const H: usize = 4096;
const INTER: usize = 1024;
const TOP_K: usize = 8;
const FLUSH_BYTES: usize = 256 << 20;
/// Packed E2M1 + E4M3 block scales of one [1024 x 4096] or [4096 x 1024] matrix.
const PROJ_BYTES: usize = INTER * H / 2 + INTER * H / 16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
    /// Random E2M1 bytes.
    fn packed(&mut self, n: usize) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n.div_ceil(8))
            .flat_map(|_| self.next().to_le_bytes())
            .collect();
        v.truncate(n);
        v
    }
    /// UE4M3 block scales in a sane exponent range (0x30..=0x3F).
    fn scales(&mut self, n: usize) -> Vec<u8> {
        self.packed(n)
            .into_iter()
            .map(|b| 0x30 | (b & 0x0F))
            .collect()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

/// One expert projection: transposed `[K/2, N]` packed + `[K/16, N]` scale
/// pointer tables over all `EXPERTS` (every expert local) and scale2 values.
struct Proj {
    packed: DevicePtr,
    scales: DevicePtr,
    scale2: DevicePtr,
    owned: Vec<DevicePtr>,
}

impl Proj {
    fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize, scale2: f32) -> Result<Self> {
        let (mut pt, mut st, mut owned) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..EXPERTS {
            let (w, s) = (
                up(g, &rng.packed(n * k / 2))?,
                up(g, &rng.scales(n * k / 16))?,
            );
            pt.push(w.0);
            st.push(s.0);
            owned.extend([w, s]);
        }
        let (packed, scales) = (
            up(g, &le(&pt, u64::to_le_bytes))?,
            up(g, &le(&st, u64::to_le_bytes))?,
        );
        let scale2 = up(g, &le(&vec![scale2; EXPERTS], f32::to_le_bytes))?;
        owned.extend([packed, scales, scale2]);
        Ok(Self {
            packed,
            scales,
            scale2,
            owned,
        })
    }
    fn table(&self) -> [DevicePtr; 3] {
        [self.packed, self.scales, self.scale2]
    }
}

/// Top-8 routes of `t` tokens over exactly the `active` experts (each token
/// picks 8 distinct experts; the experts are dealt from reshuffled decks, so
/// every active expert gets a row once `t * 8 >= active + 8`). Returns
/// (`expert_offsets[E + 1]`, `sorted_token_ids`) as `moe_sort_by_expert`.
fn routing(rng: &mut Rng, t: usize, active: &[usize]) -> (Vec<i32>, Vec<i32>) {
    let mut deck: Vec<usize> = Vec::new();
    let mut routes = Vec::with_capacity(t * TOP_K);
    for _ in 0..t {
        let (mut chosen, mut skipped) = (Vec::with_capacity(TOP_K), Vec::new());
        while chosen.len() < TOP_K {
            if deck.is_empty() {
                deck = active.to_vec();
                rng.shuffle(&mut deck);
            }
            let e = deck.pop().unwrap();
            if chosen.contains(&e) {
                skipped.push(e)
            } else {
                chosen.push(e)
            }
        }
        deck.extend(skipped);
        routes.extend(chosen);
    }
    let mut offsets = vec![0i32; EXPERTS + 1];
    let mut sorted = Vec::with_capacity(routes.len());
    for e in 0..EXPERTS {
        for (i, _) in routes.iter().enumerate().filter(|(_, r)| **r == e) {
            sorted.push((i / TOP_K) as i32);
        }
        offsets[e + 1] = sorted.len() as i32;
    }
    (offsets, sorted)
}

/// One routing of the ring, with its device offsets, sorted ids, compact
/// gate/up worklist (`[total, pad, pad, pad, items...]`) and M64 prefix.
struct Route {
    offsets: DevicePtr,
    sorted: DevicePtr,
    work: DevicePtr,
    prefix: DevicePtr,
    distinct: usize,
    /// M64 row tiles over all experts.
    mtiles: usize,
    max_rows: i32,
}

/// Per-rep CUDA-event timer on one stream. `flush` memsets the scratch
/// before each rep; all reps are queued before one sync. Returns median us.
struct Timer {
    stream: u64,
    scratch: DevicePtr,
    reps: usize,
}

impl Timer {
    fn time(
        &self,
        g: &dyn GpuBackend,
        flush: bool,
        f: &dyn Fn(usize) -> Result<()>,
    ) -> Result<f64> {
        f(0)?;
        g.synchronize(self.stream)?;
        let mut ev = vec![0u64; 2 * self.reps];
        for e in ev.iter_mut() {
            if unsafe { cuEventCreate(e, 0) } != 0 {
                bail!("cuEventCreate failed");
            }
        }
        for i in 0..self.reps {
            if flush {
                g.memset_async(self.scratch, i as u8, FLUSH_BYTES, self.stream)?;
            }
            if unsafe { cuEventRecord(ev[2 * i], self.stream) } != 0 {
                bail!("cuEventRecord failed");
            }
            f(i)?;
            if unsafe { cuEventRecord(ev[2 * i + 1], self.stream) } != 0 {
                bail!("cuEventRecord failed");
            }
        }
        if unsafe { cuEventSynchronize(ev[2 * self.reps - 1]) } != 0 {
            bail!("cuEventSynchronize failed");
        }
        let mut us = Vec::with_capacity(self.reps);
        for i in 0..self.reps {
            let mut ms = 0f32;
            if unsafe { cuEventElapsedTime(&mut ms, ev[2 * i], ev[2 * i + 1]) } != 0 {
                bail!("cuEventElapsedTime failed");
            }
            us.push(ms as f64 * 1e3);
        }
        for e in ev {
            unsafe { cuEventDestroy_v2(e) };
        }
        us.sort_by(f64::total_cmp);
        Ok(us[us.len() / 2])
    }
}

struct Kernels {
    gate_up_compact: KernelHandle,
    k128: KernelHandle,
    gate_up_silu: KernelHandle,
    k128w: KernelHandle,
    prefix: KernelHandle,
    worklist: KernelHandle,
}

fn kernels(g: &dyn GpuBackend) -> Option<Kernels> {
    Some(Kernels {
        gate_up_compact: g
            .kernel(
                MODULE,
                "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up",
            )
            .ok()?,
        k128: g
            .kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_t_k128")
            .ok()?,
        gate_up_silu: g
            .kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w")
            .ok()?,
        k128w: g
            .kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_t_k128w_compact")
            .ok()?,
        prefix: g.kernel(MODULE, "moe_mtile_prefix").ok()?,
        worklist: g.kernel("moe", "moe_build_tile_worklist").ok()?,
    })
}

/// Resident CTAs per SM of `kernel` at `threads` (static shared memory only).
fn ctas_per_sm(kernel: KernelHandle, threads: u32) -> Result<u32> {
    let mut n = 0i32;
    if unsafe { cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut n, kernel.0, threads as i32, 0) }
        != 0
    {
        bail!("cuOccupancyMaxActiveBlocksPerMultiprocessor failed");
    }
    Ok(n as u32)
}

/// Count of BF16 values that are Inf/NaN.
fn non_finite_bf16(g: &dyn GpuBackend, p: DevicePtr, values: usize) -> Result<usize> {
    let mut v = vec![0u8; values * 2];
    g.copy_d2h(p, &mut v)?;
    Ok(v.chunks_exact(2)
        .filter(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7F80 == 0x7F80)
        .count())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let Some(k) = kernels(g) else {
        eprintln!("moe_verify_bench: prequant FP4 grouped kernels absent from this target");
        std::process::exit(2);
    };
    let workloads: Vec<(usize, usize)> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let args = if args.is_empty() {
            vec!["8:41".into(), "16:62".into(), "32:90".into()]
        } else {
            args
        };
        args.iter()
            .map(|a| {
                let (t, u) = a
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("workload {a}: want T:U"))?;
                Ok((t.parse()?, u.parse()?))
            })
            .collect::<Result<_>>()?
    };
    let reps = std::env::var("MOE_VERIFY_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30usize)
        .max(1);
    let stream = g.create_stream()?;
    let timer = Timer {
        stream,
        scratch: g.alloc(FLUSH_BYTES)?,
        reps,
    };
    let sms = g.sm_count()?;

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // Output scales that keep the SiLU inputs in range (as moe_fp4_prefill_bench).
    let gate = Proj::new(g, &mut rng, INTER, H, 1.0 / 256.0)?;
    let upp = Proj::new(g, &mut rng, INTER, H, 1.0 / 128.0)?;
    let down = Proj::new(g, &mut rng, H, INTER, 1.0 / 256.0)?;
    let (e, h, inter) = (EXPERTS as u32, H as u32, INTER as u32);
    let n_tiles = inter / 128;
    println!(
        "moe_verify_bench: E={EXPERTS} (all local) H={H} I={INTER} top-{TOP_K}, {sms} SMs, median of {reps} reps, \
         cold via disjoint-expert ring (+{} MB memset for flush)",
        FLUSH_BYTES >> 20
    );
    println!(
        "GB/s = weight+scale bytes of the U active experts / time (streaming-read ceiling ~235 GB/s)"
    );
    println!(
        "{:>3} {:>4} {:>5} {:<22} {:>16} {:>6} {:>7} {:>6} {:>8} | {:>9} {:>6} | {:>9} {:>6}",
        "T",
        "U",
        "rows",
        "kernel",
        "grid",
        "live",
        "CTA/SM",
        "waves",
        "MB",
        "flush us",
        "GB/s",
        "rotate us",
        "GB/s"
    );
    let mut fail = false;
    for (t, u) in workloads {
        if !(TOP_K..=EXPERTS).contains(&u) || t * TOP_K < u {
            bail!("workload T={t} U={u}: need 8 <= U <= {EXPERTS} and T*8 >= U");
        }
        let rows = t * TOP_K;
        // Ring of disjoint active expert sets (one set when U > E/2).
        let mut perm: Vec<usize> = (0..EXPERTS).collect();
        rng.shuffle(&mut perm);
        let ring = (EXPERTS / u).max(1);
        let mut routes = Vec::with_capacity(ring);
        for r in 0..ring {
            let mut active = perm[r * u..(r + 1) * u].to_vec();
            active.sort_unstable();
            let (offsets, sorted) = routing(&mut rng, t, &active);
            let distinct = offsets.windows(2).filter(|w| w[1] > w[0]).count();
            let max_rows = offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
            let mtiles = offsets
                .windows(2)
                .map(|w| ((w[1] - w[0]) as usize).div_ceil(64))
                .sum();
            let offsets = up(g, &le(&offsets, i32::to_le_bytes))?;
            let sorted = up(g, &le(&sorted, i32::to_le_bytes))?;
            // As prequant_fp4::compact_gate_up_worklist_bytes: 16-byte total + items.
            let work = g.alloc(16 + rows * n_tiles as usize * 8)?;
            let prefix = g.alloc((EXPERTS + 1) * 4)?;
            ops::moe_build_tile_worklist(
                g,
                k.worklist,
                offsets,
                gate.packed,
                work.offset(16),
                work,
                e,
                n_tiles,
                64,
                stream,
            )?;
            ops::moe_mtile_prefix(g, k.prefix, offsets, gate.packed, prefix, e, stream)?;
            routes.push(Route {
                offsets,
                sorted,
                work,
                prefix,
                distinct,
                mtiles,
                max_rows,
            });
        }
        g.synchronize(stream)?;
        let (distinct, mtiles) = (routes[0].distinct, routes[0].mtiles);
        if routes.iter().any(|r| r.distinct != u) {
            println!(
                "  warning: routed distinct experts {:?} != U={u}",
                routes.iter().map(|r| r.distinct).collect::<Vec<_>>()
            );
        }
        let max_m_tiles =
            (routes.iter().map(|r| r.max_rows).max().unwrap_or(1).max(1) as u32).div_ceil(64);
        // Activations: token-major gate/up A (T rows x H) and sorted down A (rows x I).
        let a_gu = up(g, &rng.packed(t * H / 2))?;
        let a_gu_s = up(g, &rng.scales(t * H / 16))?;
        let a_dn = up(g, &rng.packed(rows * INTER / 2))?;
        let a_dn_s = up(g, &rng.scales(rows * INTER / 16))?;
        let c_gate = g.alloc(rows * INTER * 2)?;
        let c_up = g.alloc(rows * INTER * 2)?;
        let c_down = g.alloc(rows * H * 2)?;
        let (q_bytes, qs_bytes) = (rows * INTER / 2, rows * INTER / 16);
        let (silu_q, silu_s) = (g.alloc(q_bytes)?, g.alloc(qs_bytes)?);
        for (p, b) in [
            (c_gate, rows * INTER * 2),
            (c_up, rows * INTER * 2),
            (c_down, rows * H * 2),
            (silu_q, q_bytes),
            (silu_s, qs_bytes),
        ] {
            g.memset(p, 0, b)?;
        }
        let bound = (rows as u32).div_ceil(64) + e;
        let max_tiles = rows as u32 * n_tiles;
        let route = |i: usize| &routes[i % ring];
        let gu_bytes = (distinct * 2 * PROJ_BYTES) as f64;
        let dn_bytes = (distinct * PROJ_BYTES) as f64;
        // (label, kernel, threads, grid, live CTAs, bytes, launch)
        let cases: [(
            &str,
            KernelHandle,
            u32,
            [u32; 3],
            usize,
            f64,
            &dyn Fn(usize) -> Result<()>,
        ); 4] = [
            (
                "gate_up k64 compact",
                k.gate_up_compact,
                128,
                [max_tiles, 2, 1],
                mtiles * n_tiles as usize * 2,
                gu_bytes,
                &|i| {
                    let r = route(i);
                    ops::moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
                        g,
                        k.gate_up_compact,
                        a_gu,
                        a_gu_s,
                        gate.packed,
                        gate.scales,
                        gate.scale2,
                        c_gate,
                        upp.packed,
                        upp.scales,
                        upp.scale2,
                        c_up,
                        r.offsets,
                        r.sorted,
                        e,
                        inter,
                        h,
                        r.work.offset(16),
                        r.work,
                        max_tiles,
                        stream,
                    )
                },
            ),
            (
                "down k128 dense",
                k.k128,
                256,
                [h / 128, max_m_tiles, e],
                mtiles * (h / 128) as usize,
                dn_bytes,
                &|i| {
                    let r = route(i);
                    let [pp, sp, s2] = down.table();
                    ops::moe_w4a4_grouped_gemm_prequant_n128(
                        g,
                        k.k128,
                        a_dn,
                        a_dn_s,
                        pp,
                        sp,
                        s2,
                        c_down,
                        r.offsets,
                        DevicePtr::NULL,
                        e,
                        h,
                        inter,
                        max_m_tiles,
                        256,
                        stream,
                    )
                },
            ),
            (
                "gate_up_silu k128w",
                k.gate_up_silu,
                256,
                [inter / 128, bound, 1],
                mtiles * (inter / 128) as usize,
                gu_bytes,
                &|i| {
                    let r = route(i);
                    ops::moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
                        g,
                        k.gate_up_silu,
                        a_gu,
                        a_gu_s,
                        gate.table(),
                        upp.table(),
                        silu_q,
                        silu_s,
                        r.offsets,
                        r.sorted,
                        e,
                        inter,
                        h,
                        r.prefix,
                        bound,
                        stream,
                    )
                },
            ),
            (
                "down k128w compact",
                k.k128w,
                256,
                [h / 256, bound, 1],
                mtiles * (h / 256) as usize,
                dn_bytes,
                &|i| {
                    let r = route(i);
                    let [pp, sp, s2] = down.table();
                    ops::moe_w4a4_grouped_gemm_prequant_k128w(
                        g,
                        k.k128w,
                        a_dn,
                        a_dn_s,
                        pp,
                        sp,
                        s2,
                        c_down,
                        r.offsets,
                        DevicePtr::NULL,
                        e,
                        h,
                        inter,
                        r.prefix,
                        bound,
                        stream,
                    )
                },
            ),
        ];
        for (label, kernel, threads, grid, live, bytes, f) in cases {
            let occ = ctas_per_sm(kernel, threads)?;
            let waves = live as f64 / (occ * sms).max(1) as f64;
            let flush = timer.time(g, true, f)?;
            let rotate = timer.time(g, false, f)?;
            println!(
                "{t:>3} {distinct:>4} {rows:>5} {label:<22} {:>16} {live:>6} {occ:>7} {waves:>6.2} {:>8.1} | {flush:>9.1} {:>6.1} | {rotate:>9.1} {:>6.1}",
                format!("{}x{}x{}", grid[0], grid[1], grid[2]),
                bytes / 1e6,
                bytes / flush / 1e3,
                bytes / rotate / 1e3,
            );
        }
        g.synchronize(stream)?;
        let bad = non_finite_bf16(g, c_gate, rows * INTER)?
            + non_finite_bf16(g, c_up, rows * INTER)?
            + non_finite_bf16(g, c_down, rows * H)?;
        let mut s = vec![0u8; qs_bytes];
        g.copy_d2h(silu_s, &mut s)?;
        let bad_s = s.iter().filter(|&&b| b & 0x7F == 0x7F).count();
        if bad + bad_s != 0 {
            println!("  NON-FINITE: {bad} BF16 outputs, {bad_s} SiLU NVFP4 scales");
            fail = true;
        }
        for p in [
            a_gu, a_gu_s, a_dn, a_dn_s, c_gate, c_up, c_down, silu_q, silu_s,
        ]
        .into_iter()
        .chain(
            routes
                .iter()
                .flat_map(|r| [r.offsets, r.sorted, r.work, r.prefix]),
        ) {
            g.free(p)?;
        }
    }
    for p in gate
        .owned
        .into_iter()
        .chain(upp.owned)
        .chain(down.owned)
        .chain([timer.scratch])
    {
        g.free(p)?;
    }
    std::process::exit(if fail { 1 } else { 0 });
}
