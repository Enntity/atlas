// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): the merge form of a token-sharded pair ([`ShardMerge::run`]
//! on both ranks, two threads with their own streams, the pair's exchanges
//! copied in device memory) against the canonical form an unsharded pair runs
//! (`ops::glm_sparse_canonical`, the whole pool behind a block table whose ids
//! keep no residue), for both ranks' heads: BF16 outputs and merged LSEs must
//! be bitwise equal, owner after owner on the same scratch. BF16 cases are
//! also held to a double-precision softmax attention on the CPU.
//! `canonical_and_shard_are_history_free` holds both forms to the same bits
//! wherever the latents sit (other tables, production-sized pools) and
//! whatever the pools' unmapped blocks, the scratch and the output held before
//! (poison, or an earlier owner's scratch of another shape).
//!
//!   cargo test --release -p spark-model --lib canonical_matches_the_shard -- --ignored --nocapture
//! `a_row_does_not_depend_on_the_rows_beside_it` holds one row to the same
//! bits in owners of 1 to 64 rows (a DFlash verify's width follows the
//! drafter, so it must not reach the target's arithmetic).
//!
//!   cargo test --release -p spark-model --lib canonical_and_shard_are_history_free -- --ignored --nocapture
//!   cargo test --release -p spark-model --lib a_row_does_not_depend_on_the_rows_beside_it -- --ignored --nocapture

use super::*;
use anyhow::{Context, bail};
use spark_runtime::kv_cache::LatentShardSpec;
use std::sync::{Barrier, Mutex};

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
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// The pair's exchange on one GPU: each rank posts its send buffer, both
/// meet, copy the peer's into their receive buffer, and meet again.
struct MemPair<'a> {
    gpu: &'a dyn GpuBackend,
    rank: usize,
    posts: &'a Mutex<[(u64, usize); 2]>,
    meet: &'a Barrier,
}

impl CommBackend for MemPair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, send: u64, recv: u64, n: usize, add: bool, s: u64) -> Result<bool> {
        ensure!(!add, "shard exchanges copy, never add");
        self.gpu.synchronize(s)?;
        self.posts.lock().unwrap()[self.rank] = (send, n);
        self.meet.wait();
        let (peer, bytes) = self.posts.lock().unwrap()[1 - self.rank];
        ensure!(bytes == n, "the ranks exchange {n} and {bytes} bytes");
        self.gpu.copy_d2d(DevicePtr(peer), DevicePtr(recv), n)?;
        self.meet.wait();
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    /// Distinct tokens, `skew` percent of them in even blocks (stored by rank
    /// 0); `holes`: about one in twenty IDs is -1.
    Sparse { skew: usize, holes: bool },
    /// No selection: row `r` attends to tokens `[0, start + r + 1)`.
    Causal(u32),
    /// What the indexer selects for row `r` at position `ctx - rows + r`: up
    /// to 512 random 4-token pools in ascending order, -1 to 2048, then the
    /// partial pool's tokens and -1 to the width.
    Indexer,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    dtype: KvCacheDtype,
    rows: u32,
    ctx: usize,
    selection: Selection,
    owners: usize,
    /// Each owner's row 0 (queries and IDs) is drawn on its own, so it is
    /// the same row whatever `rows` is.
    pinned_row: bool,
}

const BS: usize = 16;

fn block_bytes(dtype: KvCacheDtype) -> usize {
    match dtype {
        KvCacheDtype::Fp8G128 => BS * 528,
        _ => BS * 512 * 2,
    }
}

/// One logical block of latents: random E4M3 codes (no NaN) and positive
/// per-128 scales, or BF16 values in [-1, 1).
fn latent_block(dtype: KvCacheDtype, rng: &mut Rng) -> Vec<u8> {
    match dtype {
        KvCacheDtype::Fp8G128 => {
            let mut b: Vec<u8> = (0..BS * 512)
                .map(|_| match rng.next() as u8 {
                    c if c & 0x7f == 0x7f => c ^ 1,
                    c => c,
                })
                .collect();
            for _ in 0..BS * 4 {
                let scale = (1.0 + rng.below(1000) as f32 / 250.0) / 448.0;
                b.extend(scale.to_le_bytes());
            }
            b
        }
        _ => (0..BS * 512)
            .flat_map(|_| bf16(rng.unit()).to_le_bytes())
            .collect(),
    }
}

/// `n` table entries over physical ids `[0, 2 * half)`: any ids, or with
/// `residues` each on its logical index's residue (a sharded pool's rule).
fn table(n: usize, half: usize, residues: bool, rng: &mut Rng) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..2 * half as u32).collect();
    rng.shuffle(&mut ids);
    if !residues {
        return ids[..n].to_vec();
    }
    let (even, odd): (Vec<u32>, Vec<u32>) = ids.iter().partition(|&&b| b % 2 == 0);
    (0..n)
        .map(|l| if l % 2 == 0 { even[l / 2] } else { odd[l / 2] })
        .collect()
}

fn indexer_row(len: usize, rng: &mut Rng) -> Vec<i32> {
    let pools = len / 4;
    let mut order: Vec<usize> = (0..pools).collect();
    rng.shuffle(&mut order);
    let mut chosen = order[..pools.min(512)].to_vec();
    chosen.sort_unstable();
    let mut row: Vec<i32> = chosen
        .iter()
        .flat_map(|p| (0..4).map(move |i| (p * 4 + i) as i32))
        .collect();
    row.resize(2048, -1);
    row.extend((pools * 4..len).map(|t| t as i32));
    row.resize(WIDTH as usize, -1);
    row
}

fn selection(case: &Case, rng: &mut Rng) -> Option<Vec<i32>> {
    let (skew, holes) = match case.selection {
        Selection::Sparse { skew, holes } => (skew, holes),
        Selection::Causal(_) => return None,
        Selection::Indexer => {
            let first = case.ctx - case.rows as usize;
            let rows = (0..case.rows as usize).flat_map(|r| indexer_row(first + r + 1, rng));
            return Some(rows.collect());
        }
    };
    let width = WIDTH as usize;
    let mut out = Vec::with_capacity(case.rows as usize * width);
    for _ in 0..case.rows {
        let mut by_residue: [Vec<i32>; 2] = [vec![], vec![]];
        for t in 0..case.ctx {
            by_residue[t / BS % 2].push(t as i32);
        }
        by_residue.iter_mut().for_each(|v| rng.shuffle(v));
        let even = (width * skew / 100).min(by_residue[0].len());
        let mut row: Vec<i32> = by_residue[0][..even].to_vec();
        row.extend(&by_residue[1][..width - even]);
        rng.shuffle(&mut row);
        if holes {
            row.iter_mut()
                .filter(|_| rng.below(20) == 0)
                .for_each(|t| *t = -1);
        }
        out.extend(row);
    }
    Some(out)
}

/// One case's device allocations, freed together when it ends.
struct Allocs<'a> {
    gpu: &'a dyn GpuBackend,
    live: Mutex<Vec<DevicePtr>>,
}

impl<'a> Allocs<'a> {
    fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            live: Mutex::new(vec![]),
        }
    }

    /// `bytes` of device memory, every byte `fill`.
    fn alloc(&self, bytes: usize, fill: u8) -> Result<DevicePtr> {
        let bytes = bytes.max(256);
        let p = self.gpu.alloc(bytes)?;
        self.live.lock().unwrap().push(p);
        self.gpu.memset(p, fill, bytes)?;
        Ok(p)
    }

    fn upload<T: Copy>(&self, v: &[T]) -> Result<DevicePtr> {
        let bytes = std::mem::size_of_val(v);
        // SAFETY: plain-old-data slices (u8, u16, u32, i32) viewed as bytes.
        let raw = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), bytes) };
        let p = self.alloc(bytes, 0)?;
        self.gpu.copy_h2d(raw, p)?;
        Ok(p)
    }
}

impl Drop for Allocs<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.synchronize(self.gpu.default_stream());
        for p in self.live.lock().unwrap().drain(..) {
            let _ = self.gpu.free(p);
        }
    }
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c.tp_world_size = 2;
    c
}

/// Double-precision softmax attention of `query` (`[rows, 32, 512]` BF16)
/// over the BF16 latents of each row's tokens.
fn reference(case: &Case, logical: &[Vec<u8>], query: &[u16], ids: &[Vec<i32>]) -> Vec<f64> {
    let latent = |t: usize, d: usize| {
        let b = &logical[t / BS];
        let i = (t % BS * 512 + d) * 2;
        from_bf16(u16::from_le_bytes([b[i], b[i + 1]])) as f64
    };
    let mut out = vec![0f64; query.len()];
    for (r, tokens) in ids.iter().enumerate().take(case.rows as usize) {
        for h in 0..32 {
            let q = &query[(r * 32 + h) * 512..][..512];
            let logits: Vec<f64> = tokens
                .iter()
                .map(|&t| {
                    let dot: f64 = (0..512)
                        .map(|d| from_bf16(q[d]) as f64 * latent(t as usize, d))
                        .sum();
                    dot * 0.0625
                })
                .collect();
            let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for d in 0..512 {
                let v: f64 = tokens
                    .iter()
                    .zip(&w)
                    .map(|(&t, w)| w * latent(t as usize, d))
                    .sum();
                out[(r * 32 + h) * 512 + d] = v / z;
            }
        }
    }
    out
}

/// Where a case's latents sit and what the memory it reads held before; no
/// placement may change a bit. `seed` draws the block tables, `pool_blocks`
/// sizes the unsharded pool (the shard's halves follow; at least the
/// history), and `poison` fills every pool block no table maps, every scratch
/// and the outputs first. `scratch`: the canonical form's region, kept across
/// cases, instead of a fresh one.
#[derive(Clone, Copy, Debug)]
struct Placement {
    seed: u64,
    pool_blocks: usize,
    poison: u8,
    scratch: Option<(DevicePtr, usize)>,
}

impl Placement {
    /// Tight pools, zeroed memory, fresh scratch.
    fn fresh() -> Self {
        Self {
            seed: 1,
            pool_blocks: 0,
            poison: 0,
            scratch: None,
        }
    }
}

/// One case's results: the canonical form's outputs and LSEs per owner and
/// rank, then the shard's, and how many of the forms' bytes differ.
struct CaseOut {
    differing: usize,
    bits: Vec<Vec<u8>>,
}

fn run_case(gpu: &dyn GpuBackend, case: Case, seed: u64, place: Placement) -> Result<CaseOut> {
    let c = config();
    let mem = Allocs::new(gpu);
    // Contents and queries from `seed`, tables from the placement alone.
    let mut rng = Rng(seed | 1);
    let mut placing = Rng(place.seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1);
    let (rows, bb) = (case.rows, block_bytes(case.dtype));
    let blocks = case.ctx.div_ceil(BS);
    let half = blocks.max(place.pool_blocks).div_ceil(2) + 3;
    let logical: Vec<Vec<u8>> = (0..blocks)
        .map(|_| latent_block(case.dtype, &mut rng))
        .collect();
    // Unsharded: the whole history, any ids. Sharded: each rank its own
    // table (the ranks' ids differ, their residues do not) and half pool.
    let table_u = table(blocks, half + 4, false, &mut placing);
    let pool_u = mem.alloc(2 * (half + 4) * bb, place.poison)?;
    for (l, &b) in table_u.iter().enumerate() {
        gpu.copy_h2d(&logical[l], pool_u.offset(b as usize * bb))?;
    }
    let mut ranks = vec![];
    for r in 0..2u32 {
        let t = table(blocks, half, true, &mut placing);
        let pool = mem.alloc(half * bb, place.poison)?;
        for (l, &b) in t.iter().enumerate().filter(|(_, b)| **b % 2 == r) {
            gpu.copy_h2d(&logical[l], pool.offset((b / 2) as usize * bb))?;
        }
        let identity: Vec<u32> = (0..half as u32).collect();
        ranks.push((mem.upload(&t)?, pool, mem.upload(&identity)?));
    }
    let table_u = mem.upload(&table_u)?;

    let qn = rows as usize * 32 * 512;
    let mut owners = vec![];
    for o in 0..case.owners {
        let mut q: [Vec<u16>; 2] =
            [0, 1].map(|_| (0..qn).map(|_| bf16(rng.unit() * 0.6)).collect());
        let mut sel = selection(&case, &mut rng);
        if case.pinned_row {
            let mut pin = Rng(0x5eed_0001 + o as u64);
            for q in &mut q {
                q[..32 * 512]
                    .iter_mut()
                    .for_each(|x| *x = bf16(pin.unit() * 0.6));
            }
            if let Some(s) = sel.as_mut() {
                let one = selection(&Case { rows: 1, ..case }, &mut pin).context("one row")?;
                s[..WIDTH as usize].copy_from_slice(&one);
            }
        }
        let dev_sel = sel.as_ref().map(|s| mem.upload(s)).transpose()?;
        owners.push((
            q.clone(),
            [mem.upload(&q[0])?, mem.upload(&q[1])?],
            sel,
            dev_sel,
        ));
    }
    let causal = match case.selection {
        Selection::Causal(start) => start,
        Selection::Sparse { .. } | Selection::Indexer => 0,
    };
    let splits = shard::MERGE_SPLITS;
    let m = MergeLayout::new(rows, splits);
    let (out_bytes, lse_bytes) = (qn * 2, rows as usize * 32 * 4);

    // The shard: both ranks at once, each owner in turn on one scratch.
    let (posts, meet) = (Mutex::new([(0, 0); 2]), Barrier::new(2));
    let shard_out: Vec<Vec<(Vec<u8>, Vec<u8>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2usize)
            .map(|r| {
                let (owners, ranks, posts, meet, c, mem) =
                    (&owners, &ranks, &posts, &meet, &c, &mem);
                scope.spawn(move || -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
                    gpu.bind_to_thread()?;
                    let stream = gpu.create_stream()?;
                    let comm = MemPair {
                        gpu,
                        rank: r,
                        posts,
                        meet,
                    };
                    let (table, pool, identity) = ranks[r];
                    let scratch = mem.alloc(m.total, place.poison)?;
                    let merge = ShardMerge {
                        gpu,
                        comm: &comm,
                        config: c,
                        shard: LatentShard {
                            spec: LatentShardSpec {
                                rank: r,
                                world: 2,
                                scratch_bytes: m.total,
                                view_blocks: 0,
                                write_rows: 0,
                                lane: false,
                            },
                            local_blocks: half,
                            scratch,
                            identity,
                            lane: None,
                        },
                        work: 0,
                        work_bytes: m.total,
                        dtype: case.dtype,
                        pool,
                        scale: 0.0625,
                        lane: None,
                    };
                    let output = mem.alloc(out_bytes, place.poison)?;
                    let mut got = vec![];
                    for (_, q, _, sel) in owners {
                        let a = ShardRows {
                            query: q[r],
                            selected: *sel,
                            causal_start: causal,
                            block_table: table,
                            rows,
                            end: None,
                            queries_swapped: false,
                        };
                        merge.run(a, output, stream)?;
                        gpu.synchronize(stream)?;
                        got.push((
                            download(gpu, output, out_bytes)?,
                            download(gpu, scratch.offset(m.out_lse), lse_bytes)?,
                        ));
                    }
                    Ok(got)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect::<Result<Vec<_>>>()
    })?;

    // The canonical form, rank by rank on the main thread.
    let (scratch, canonical_bytes) = match place.scratch {
        Some(region) => region,
        None => {
            let bytes = ops::CanonicalLayout::bytes(rows);
            (mem.alloc(bytes, place.poison)?, bytes)
        }
    };
    let layout = ops::CanonicalLayout::new(rows, splits);
    let output = mem.alloc(out_bytes, place.poison)?;
    let stream = gpu.default_stream();
    let mut differing = 0;
    let mut bits = vec![];
    for (o, (q_host, q, sel, dev_sel)) in owners.iter().enumerate() {
        for r in 0..2u32 {
            let a = ops::GlmSparsePrefillTc {
                config: &c,
                dtype: case.dtype,
                identical_kv_latent: true,
                query: q[r as usize],
                k_cache: pool_u,
                v_cache: pool_u,
                indices: dev_sel.unwrap_or(DevicePtr::NULL),
                output,
                block_table: table_u,
                rows,
                heads: 32,
                head_dim: 512,
                index_width: WIDTH,
                block_size: 16,
                scale: 0.0625,
            };
            let region = (scratch, canonical_bytes);
            ops::glm_sparse_canonical(gpu, &a, *dev_sel, causal, r, region, stream)?;
            gpu.synchronize(stream)?;
            let out = download(gpu, output, out_bytes)?;
            let lse = download(gpu, scratch.offset(layout.out_lse), lse_bytes)?;
            let (s_out, s_lse) = &shard_out[r as usize][o];
            let diff = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(a, b)| a != b).count();
            let d = diff(&out, s_out) + diff(&lse, s_lse);
            if d != 0 {
                println!("  rank {r} owner {o}: {d} bytes differ");
            }
            differing += d;
            bits.extend([out.clone(), lse]);
            if case.dtype == KvCacheDtype::Bf16 && rows <= 4 {
                let ids: Vec<Vec<i32>> = (0..rows as usize)
                    .map(|row| match sel {
                        Some(s) => s[row * WIDTH as usize..][..WIDTH as usize]
                            .iter()
                            .copied()
                            .filter(|&t| t >= 0)
                            .collect(),
                        None => (0..=(causal as i32 + row as i32)).collect(),
                    })
                    .collect();
                let want = reference(&case, &logical, &q_host[r as usize], &ids);
                let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
                let max = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                let worst = out
                    .chunks_exact(2)
                    .zip(&want)
                    .map(|(h, w)| (from_bf16(u16::from_le_bytes([h[0], h[1]])) as f64 - w).abs())
                    .fold(0f64, f64::max);
                let tolerance = 0.02 * rms + max / 64.0;
                ensure!(
                    worst <= tolerance,
                    "{case:?} rank {r}: canonical vs CPU attention {worst:e} > {tolerance:e}"
                );
            }
        }
    }
    for (out, lse) in shard_out.into_iter().flatten() {
        bits.extend([out, lse]);
    }
    Ok(CaseOut { differing, bits })
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn canonical_matches_the_shard_bitwise() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    let sparse = |skew, holes| Selection::Sparse { skew, holes };
    let mut cases = vec![];
    for dtype in [KvCacheDtype::Fp8G128, KvCacheDtype::Bf16] {
        let case = |rows, ctx, selection, owners| Case {
            dtype,
            rows,
            ctx,
            selection,
            owners,
            pinned_row: false,
        };
        for rows in 1..=8 {
            cases.push(case(rows, 65536, sparse(50, false), 1));
        }
        cases.extend([
            case(8, 4352, sparse(50, false), 4),
            case(8, 131072, sparse(50, true), 1),
            case(4, 16384, sparse(70, false), 2),
            case(2, 8192, sparse(100, false), 1),
            case(3, 8192, sparse(0, true), 1),
            case(16, 32768, sparse(50, false), 1),
            case(24, 32768, sparse(50, true), 1),
            case(48, 32768, sparse(50, false), 1),
            case(64, 32768, sparse(50, false), 1),
            case(8, 64, Selection::Causal(0), 1),
            case(3, 64, Selection::Causal(2), 1),
            case(8, 2048, Selection::Causal(1000), 2),
            case(8, 2048, Selection::Causal(2040), 1),
            case(1, 2048, Selection::Causal(1500), 1),
            case(64, 2048, Selection::Causal(500), 1),
        ]);
    }
    let mut failed = 0;
    for (i, case) in cases.iter().enumerate() {
        let seed = 0x9e37_79b9_7f4a_7c15 ^ i as u64;
        let differing = run_case(gpu, *case, seed, Placement::fresh())?.differing;
        println!(
            "{} {case:?} splits={}",
            if differing == 0 { "BITWISE" } else { "DIFFERS" },
            shard::MERGE_SPLITS
        );
        failed += (differing != 0) as usize;
    }
    ensure!(failed == 0, "{failed} of {} cases differ", cases.len());
    Ok(())
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn canonical_and_shard_are_history_free() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    // One canonical region for every case: each owner finds the previous
    // owner's partitions, IDs and counts (another row count, another
    // context), or the first time, poison.
    let shared_bytes = ops::CanonicalLayout::bytes(shard::MERGE_MAX_ROWS as u32);
    let shared_base = gpu.alloc(shared_bytes)?;
    gpu.memset(shared_base, 0x7f, shared_bytes)?;
    let shared = Some((shared_base, shared_bytes));
    let sparse = |skew, holes| Selection::Sparse { skew, holes };
    let mut cases = vec![];
    for dtype in [KvCacheDtype::Fp8G128, KvCacheDtype::Bf16] {
        let case = |rows, ctx, selection| Case {
            dtype,
            rows,
            ctx,
            selection,
            owners: 2,
            pinned_row: false,
        };
        for rows in (1..=8).chain([16, 64]) {
            cases.push(case(rows, 25_947, Selection::Indexer));
        }
        cases.extend([
            case(8, 131_072, Selection::Indexer),
            case(2, 131_072, sparse(50, true)),
            case(16, 65_536, sparse(50, false)),
            case(8, 2048, Selection::Causal(1000)),
            case(64, 2048, Selection::Causal(500)),
            case(1, 2048, Selection::Causal(2040)),
        ]);
    }
    let mut failed = 0;
    for (i, case) in cases.iter().enumerate() {
        // Production-sized pools: ~58K blocks resident, 7,500 capped.
        let big = match case.dtype {
            KvCacheDtype::Fp8G128 => 60_000,
            _ => 40_000,
        };
        let placements = [
            Placement {
                seed: 2,
                pool_blocks: big,
                poison: 0xff,
                scratch: None,
            },
            Placement {
                seed: 3,
                pool_blocks: big,
                poison: 0x7f,
                scratch: shared,
            },
            Placement {
                seed: 4,
                pool_blocks: 7_500,
                poison: 0xff,
                scratch: shared,
            },
        ];
        let seed = 0x51ed_270b_27a3_4e8d ^ i as u64;
        let base = run_case(gpu, *case, seed, Placement::fresh())?;
        let mut bad = vec![];
        if base.differing != 0 {
            bad.push(format!("shard vs canonical {} bytes", base.differing));
        }
        for place in placements {
            let got = run_case(gpu, *case, seed, place)?;
            let moved: usize = got
                .bits
                .iter()
                .zip(&base.bits)
                .map(|(x, y)| x.iter().zip(y).filter(|(a, b)| a != b).count())
                .sum();
            if moved != 0 || got.differing != 0 {
                bad.push(format!(
                    "seed {} pool {} poison {:#x} shared {}: {moved} bytes moved, \
                     shard vs canonical {}",
                    place.seed,
                    place.pool_blocks,
                    place.poison,
                    place.scratch.is_some(),
                    got.differing
                ));
            }
        }
        let verdict = if bad.is_empty() {
            "HISTORY-FREE"
        } else {
            "DEPENDS"
        };
        println!("{verdict} {case:?} splits={}", shard::MERGE_SPLITS);
        for b in &bad {
            println!("  {b}");
        }
        failed += usize::from(!bad.is_empty());
    }
    gpu.free(shared_base)?;
    ensure!(
        failed == 0,
        "{failed} of {} cases depend on placement or history",
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn a_row_does_not_depend_on_the_rows_beside_it() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    let mut failed = 0;
    for dtype in [KvCacheDtype::Fp8G128, KvCacheDtype::Bf16] {
        for (ctx, selection) in [
            (25_947, Selection::Indexer),
            (
                65_536,
                Selection::Sparse {
                    skew: 50,
                    holes: true,
                },
            ),
            (2048, Selection::Causal(1000)),
        ] {
            let mut first: Option<Vec<Vec<u8>>> = None;
            for rows in [1, 2, 3, 5, 8, 16, 64] {
                let case = Case {
                    dtype,
                    rows,
                    ctx,
                    selection,
                    owners: 1,
                    pinned_row: true,
                };
                let got = run_case(gpu, case, 0x7e57_c0de, Placement::fresh())?;
                // Row 0 of every output (32 x 512 BF16) and LSE (32 FP32),
                // canonical then shard, both ranks' heads.
                let row0: Vec<Vec<u8>> = got
                    .bits
                    .iter()
                    .enumerate()
                    .map(|(i, b)| b[..if i % 2 == 0 { 32 * 512 * 2 } else { 32 * 4 }].to_vec())
                    .collect();
                let base = first.get_or_insert_with(|| row0.clone());
                let moved: usize = row0
                    .iter()
                    .zip(base.iter())
                    .map(|(x, y)| x.iter().zip(y).filter(|(a, b)| a != b).count())
                    .sum();
                let ok = moved == 0 && got.differing == 0;
                println!(
                    "{} {dtype:?} ctx={ctx} {selection:?} rows={rows} splits={}: row 0 {moved} \
                     bytes from rows=1, shard vs canonical {}",
                    if ok { "SAME" } else { "MOVED" },
                    shard::MERGE_SPLITS,
                    got.differing
                );
                failed += usize::from(!ok);
            }
        }
    }
    ensure!(failed == 0, "{failed} widths change a row's bits");
    Ok(())
}
