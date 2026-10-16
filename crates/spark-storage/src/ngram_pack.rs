// SPDX-License-Identifier: AGPL-3.0-only

//! One-time repack of a SPLIT-ROW n-gram table into one row-major file.
//!
//! Some checkpoints store each table row in more than one tensor. The NVFP4
//! PLE table of Qwen3.8-Flash-Next-FP4-Local is the case this exists for:
//! every shard is `weight` U8 `[R, 80]` (the E2M1 codes) plus `weight_scale`
//! F8_E4M3 `[R, 10]` (the block scales), at unrelated offsets of the file. A
//! row-cache miss read in place would therefore cost TWO NVMe reads, and the
//! PLE stall on Spark is per-IO latency (the APST wake that
//! `ATLAS_PLE_NVME_KEEPALIVE_MS` exists for, and ~74 us per serial fault),
//! not bytes. So the planes are interleaved ONCE into a derived file, with
//! every row `[plane 0 | plane 1 | zero pad]` at a fixed stride, and the row
//! cache opens that file as an ordinary contiguous table: one read per miss.
//!
//! The derived file never goes into the checkpoint directory, which stays
//! byte-identical to the published one; the caller picks the directory.
//!
//! ```text
//! [0, 8)                 MAGIC
//! [8, 16)                u64 LE identity length L
//! [16, 24)               u64 LE rows per checksum chunk V
//! [24, 32)               u64 LE chunk count C = ceil(rows / V)
//! [32, 32 + L)           identity JSON
//! [32 + L, 32 + L + 4C)  per-chunk CRC32s, u32 LE
//! ... 64 KiB             zero pad
//! [64 KiB + r*stride)    row r: plane bytes in plane order, then zeros to stride
//! ```
//!
//! The identity JSON names every source (resolved file name, length, mtime,
//! inode, offset per shard), the geometry and a caller tag. Opening an
//! existing file requires the identity to match exactly, a sample of rows
//! (one per shard) to match the sources byte for byte, and a sample of the
//! chunk checksums to hold, so a re-published checkpoint with the same layout
//! but new values is caught rather than served stale, and so is a packed file
//! damaged after it was written. [`Verify::Full`] checks every chunk.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use atlas_tier::pio::{read_exact_at, write_all_at};

/// Header bytes ahead of row 0: a multiple of the 4 KiB O_DIRECT block, so
/// row 0 starts aligned, and room for the identity and chunk checksums of a
/// 128-shard table (~6 KB + ~5 KB; `identity_of_the_real_table_fits`).
pub const HEADER: u64 = 64 * 1024;
/// Version 2 added the chunk checksums and the sources' mtime and inode. A
/// version-1 file fails the magic and is rebuilt in place: its name, which
/// depends on the layout only, is unchanged.
const MAGIC: &[u8; 8] = b"ATLASPK\x02";
/// Bytes of the fixed header fields ahead of the identity.
const FIXED: u64 = 32;
/// Packed bytes per checksum chunk: 262144 rows of the 96-byte NVFP4 table,
/// 1221 chunks (4.9 KB of checksums) for its 320M rows.
const CHECK_BYTES: u64 = 24 << 20;
/// Chunks [`Verify::Sample`] re-hashes on every open: 16 of the NVFP4
/// table's 1221 (1.3%, 384 MiB), picked at random so successive opens cover
/// different parts of the file.
const SAMPLE_CHUNKS: usize = 16;

/// How much of a packed file's checksummed body an open re-reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// [`SAMPLE_CHUNKS`] random chunks.
    Sample,
    /// Every chunk: the whole table is read once (`ATLAS_PLE_PACKED_VERIFY=full`
    /// for the PLE table).
    Full,
}

impl Verify {
    /// `full` is [`Verify::Full`]; anything else, unset included, samples.
    pub fn parse(v: Option<&str>) -> Self {
        if v == Some("full") {
            Self::Full
        } else {
            Self::Sample
        }
    }
}

/// One plane of a split-row table: where each shard's first row sits, and how
/// many bytes this plane holds per row. Shards may live in different files and
/// need not be laid out in order.
pub struct Plane<'a> {
    pub shards: &'a [(PathBuf, u64)],
    pub row_bytes: usize,
}

/// What to pack. `planes` are concatenated per row in the order given.
pub struct PackSpec<'a> {
    pub planes: Vec<Plane<'a>>,
    pub rows_per_shard: u64,
    /// Bytes per packed row, `>=` the planes' sum; the tail is zero.
    pub stride: usize,
    /// Format label, part of the identity and the file name (e.g. "nvfp4g16").
    pub tag: &'a str,
}

/// A packed table on disk, ready for `NgramRowCache::open_at`.
#[derive(Debug, Clone)]
pub struct PackedTable {
    pub path: PathBuf,
    pub base_offset: u64,
    pub rows_total: u64,
    pub stride: usize,
}

impl PackSpec<'_> {
    fn shards(&self) -> usize {
        self.planes.first().map_or(0, |p| p.shards.len())
    }

    pub fn rows_total(&self) -> u64 {
        self.shards() as u64 * self.rows_per_shard
    }

    fn check(&self) -> Result<()> {
        ensure!(!self.planes.is_empty(), "ngram pack: no planes");
        ensure!(self.rows_per_shard > 0, "ngram pack: zero rows per shard");
        let n = self.shards();
        ensure!(n > 0, "ngram pack: no shards");
        for (i, p) in self.planes.iter().enumerate() {
            ensure!(
                p.shards.len() == n,
                "ngram pack: plane {i} has {} shards, plane 0 has {n}",
                p.shards.len()
            );
            ensure!(p.row_bytes > 0, "ngram pack: plane {i} has zero row bytes");
        }
        let used: usize = self.planes.iter().map(|p| p.row_bytes).sum();
        ensure!(
            used <= self.stride,
            "ngram pack: planes need {used} bytes per row, stride is {}",
            self.stride
        );
        Ok(())
    }

    /// The layout half of the identity: sources by resolved file name and
    /// length, offsets and geometry. It names the file, so a rebuild for a
    /// source that changed in place overwrites the stale table instead of
    /// leaving a 29 GB orphan beside it.
    ///
    /// Sources are named by their RESOLVED file name: in a Hugging Face cache
    /// the snapshot entry is a symlink to a blob named by its content hash, so
    /// a different revision of the same layout gets a different identity
    /// without anything having to read 27 GB.
    fn layout_identity(&self) -> Result<(String, Vec<PathBuf>)> {
        self.check()?;
        let mut files: Vec<&Path> = Vec::new();
        let mut real: Vec<PathBuf> = Vec::new();
        let mut desc = Vec::new();
        let mut planes = Vec::new();
        for p in &self.planes {
            let mut shards = Vec::with_capacity(p.shards.len());
            for (path, off) in p.shards {
                let idx = match files.iter().position(|f| *f == path.as_path()) {
                    Some(i) => i,
                    None => {
                        let r = std::fs::canonicalize(path)
                            .with_context(|| format!("ngram pack: resolve {}", path.display()))?;
                        let len = std::fs::metadata(&r)
                            .with_context(|| format!("ngram pack: stat {}", r.display()))?
                            .len();
                        let name = r.file_name().map(|n| n.to_string_lossy().into_owned());
                        desc.push(serde_json::json!({"name": name, "len": len}));
                        files.push(path.as_path());
                        real.push(r);
                        files.len() - 1
                    }
                };
                shards.push(serde_json::json!([idx, off]));
            }
            planes.push(serde_json::json!({"row_bytes": p.row_bytes, "shards": shards}));
        }
        let layout = serde_json::json!({
            "format": "atlas-ngram-pack/1",
            "tag": self.tag,
            "rows_per_shard": self.rows_per_shard,
            "stride": self.stride,
            "files": desc,
            "planes": planes,
        })
        .to_string();
        Ok((layout, real))
    }

    /// The JSON that a packed file must carry to be reused for this spec: the
    /// layout plus each source's mtime (ns) and inode, so a source rewritten
    /// in place with the same name and length is a different identity.
    pub fn identity(&self) -> Result<String> {
        let (layout, real) = self.layout_identity()?;
        let stat = real
            .iter()
            .map(|r| {
                let m = std::fs::metadata(r)
                    .with_context(|| format!("ngram pack: stat {}", r.display()))?;
                Ok(file_stamp(&m))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(format!(
            "{{\"layout\":{layout},\"stat\":{}}}",
            serde_json::Value::from(stat)
        ))
    }

    /// `ngram-{tag}-{fnv64(layout identity)}.bin`: one file per distinct layout.
    pub fn file_name(&self) -> Result<String> {
        Ok(format!(
            "ngram-{}-{:016x}.bin",
            self.tag,
            fnv1a64(self.layout_identity()?.0.as_bytes())
        ))
    }

    /// Rows per checksum chunk: [`CHECK_BYTES`] of packed rows.
    fn check_rows(&self) -> u64 {
        (CHECK_BYTES / self.stride.max(1) as u64).max(1)
    }

    /// Byte offset of row `row` of plane `plane` in its source file.
    pub fn source_loc(&self, plane: usize, row: u64) -> (&Path, u64) {
        let p = &self.planes[plane];
        let (path, base) = &p.shards[(row / self.rows_per_shard) as usize];
        (
            path,
            base + (row % self.rows_per_shard) * p.row_bytes as u64,
        )
    }
}

/// Byte offset of packed row `row`.
pub fn packed_offset(row: u64, stride: usize) -> u64 {
    HEADER + row * stride as u64
}

/// Interleave `rows` rows: `planes[i]` holds `rows * row_bytes[i]` bytes, and
/// `out` receives `rows * stride` bytes with each row's tail zeroed.
pub fn interleave(planes: &[&[u8]], row_bytes: &[usize], stride: usize, out: &mut [u8]) {
    let rows = out.len() / stride;
    for r in 0..rows {
        let dst = &mut out[r * stride..(r + 1) * stride];
        let mut at = 0;
        for (src, &w) in planes.iter().zip(row_bytes) {
            dst[at..at + w].copy_from_slice(&src[r * w..(r + 1) * w]);
            at += w;
        }
        dst[at..].fill(0);
    }
}

/// Return the packed table for `spec` under `dir`, building it first if no
/// valid one is there. `verify` is how much of an existing file's body the
/// open re-hashes ([`Verify`]).
///
/// A valid file is used without taking the lock, so a table packed at install
/// time can sit on a read-only mount: builds write a temp file and rename it,
/// so the final name only ever holds a complete file. A build holds an
/// exclusive lock on `<file>.lock`, so two processes on one host build once.
pub fn open_or_pack(spec: &PackSpec<'_>, dir: &Path, verify: Verify) -> Result<PackedTable> {
    let name = spec.file_name()?;
    let path = dir.join(&name);
    let table = PackedTable {
        path: path.clone(),
        base_offset: HEADER,
        rows_total: spec.rows_total(),
        stride: spec.stride,
    };
    if path.exists() && validate(spec, &path, verify).is_ok() {
        return Ok(table);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("ngram pack: create {}", dir.display()))?;
    let _lock = DirLock::take(&dir.join(format!("{name}.lock")))?;
    // Re-check under the lock: another process may have just finished.
    if path.exists() {
        match validate(spec, &path, verify) {
            Ok(()) => return Ok(table),
            Err(e) => tracing::warn!("ngram pack: {} is stale, rebuilding: {e:#}", path.display()),
        }
    }
    let need = packed_offset(spec.rows_total(), spec.stride);
    ensure_free_space(dir, need)?;
    let tmp = dir.join(format!("{name}.tmp{}", std::process::id()));
    let t0 = std::time::Instant::now();
    tracing::info!(
        "ngram pack: building {} ({:.1} GB, {} rows x {} B) — one-time",
        path.display(),
        need as f64 / 1e9,
        spec.rows_total(),
        spec.stride
    );
    // Under the lock, a leftover temp file is a crashed build's: drop it.
    remove_stale_temps(dir, &name);
    if let Err(e) = pack(spec, &tmp, CHUNK_ROWS, spec.check_rows()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("ngram pack: rename to {}", path.display()))?;
    validate(spec, &path, Verify::Sample)
        .context("ngram pack: freshly built file failed validation")?;
    tracing::info!(
        "ngram pack: built {} in {:.1}s",
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(table)
}

/// Rows copied per pass. 262144 rows of the NVFP4 table are 21 MB + 2.6 MB in
/// and 25 MB out: big enough for sequential throughput, small enough that the
/// three buffers stay off the radar of a host whose GPU shares its memory.
const CHUNK_ROWS: u64 = 262_144;

/// Write the packed form of `spec` to `out_path`: rows read `io_rows` at a
/// time, checksummed every `check_rows`, and the header LAST, so a file cut
/// short anywhere fails the magic.
fn pack(spec: &PackSpec<'_>, out_path: &Path, io_rows: u64, check_rows: u64) -> Result<()> {
    let out = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_path)
        .with_context(|| format!("ngram pack: create {}", out_path.display()))?;
    let identity = spec.identity()?;
    let rows_total = spec.rows_total();
    let chunks = rows_total.div_ceil(check_rows);
    // Refuse before 29 GB of writes, not after.
    header_fits(identity.len() as u64, chunks)?;

    let widths: Vec<usize> = spec.planes.iter().map(|p| p.row_bytes).collect();
    let chunk = io_rows.max(1);
    let mut srcs: Vec<Vec<u8>> = widths
        .iter()
        .map(|w| vec![0u8; chunk as usize * w])
        .collect();
    let mut buf = vec![0u8; chunk as usize * spec.stride];
    let mut files: Vec<(PathBuf, File)> = Vec::new();
    let mut sums = ChunkSums::new(check_rows, spec.stride);
    for shard in 0..spec.shards() {
        let mut local = 0u64;
        while local < spec.rows_per_shard {
            let n = (spec.rows_per_shard - local).min(chunk) as usize;
            let row = shard as u64 * spec.rows_per_shard + local;
            for (i, src) in srcs.iter_mut().enumerate() {
                let (path, off) = spec.source_loc(i, row);
                let f = cached_open(&mut files, path)?;
                let bytes = &mut src[..n * widths[i]];
                read_exact_at(f, bytes, off).with_context(|| {
                    format!(
                        "ngram pack: read plane {i} shard {shard} @ {off} of {}",
                        path.display()
                    )
                })?;
                drop_cache(f, off, bytes.len() as u64);
            }
            let planes: Vec<&[u8]> = srcs.iter().zip(&widths).map(|(s, w)| &s[..n * w]).collect();
            let dst = &mut buf[..n * spec.stride];
            interleave(&planes, &widths, spec.stride, dst);
            sums.feed(row, dst);
            let at = packed_offset(row, spec.stride);
            write_all_at(&out, dst, at).context("ngram pack: write rows")?;
            // Bound the dirty page cache: on GB10 it is the same memory the
            // GPU allocates from, and a 27 GB burst of dirty pages is not
            // something a loading model should have to compete with.
            out.sync_data().context("ngram pack: sync")?;
            drop_cache(&out, at, dst.len() as u64);
            local += n as u64;
        }
    }
    let sums = sums.finish(rows_total)?;
    let mut header = vec![0u8; HEADER as usize];
    header[..8].copy_from_slice(MAGIC);
    header[8..16].copy_from_slice(&(identity.len() as u64).to_le_bytes());
    header[16..24].copy_from_slice(&check_rows.to_le_bytes());
    header[24..32].copy_from_slice(&chunks.to_le_bytes());
    let id_end = FIXED as usize + identity.len();
    header[FIXED as usize..id_end].copy_from_slice(identity.as_bytes());
    for (i, sum) in sums.iter().enumerate() {
        header[id_end + 4 * i..id_end + 4 * (i + 1)].copy_from_slice(&sum.to_le_bytes());
    }
    write_all_at(&out, &header, 0).context("ngram pack: write header")?;
    out.sync_all().context("ngram pack: final sync")?;
    Ok(())
}

/// Refuse a header whose identity and `chunks` checksums would not fit.
/// Every term is bounded before it is added: these come from a file.
fn header_fits(identity_len: u64, chunks: u64) -> Result<()> {
    ensure!(
        identity_len <= HEADER - FIXED,
        "header length {identity_len} out of range"
    );
    let room = (HEADER - FIXED - identity_len) / 4;
    ensure!(
        chunks <= room,
        "header holds {room} chunk checksums beside a {identity_len}-byte identity, \
         the table needs {chunks}"
    );
    Ok(())
}

/// Running per-chunk CRC32s over rows fed in increasing order.
///
/// CRC32, not a cryptographic hash: this catches a file damaged after it was
/// written (a missed bit has odds of 2^-32), and nothing stored beside the
/// data could stop someone who can rewrite the file anyway. It also has to be
/// cheap enough to run on every open: 7.8 GB/s on the ARMv8 CRC32
/// instructions, against 0.37 GB/s for the tree's software SHA-256.
struct ChunkSums {
    check_rows: u64,
    stride: usize,
    next_row: u64,
    hasher: crc32fast::Hasher,
    sums: Vec<u32>,
}

impl ChunkSums {
    fn new(check_rows: u64, stride: usize) -> Self {
        Self {
            check_rows,
            stride,
            next_row: 0,
            hasher: crc32fast::Hasher::new(),
            sums: Vec::new(),
        }
    }

    /// Rows `row..` of the table, packed (`bytes` is whole rows).
    fn feed(&mut self, row: u64, mut bytes: &[u8]) {
        debug_assert_eq!(row, self.next_row, "rows must arrive in order");
        while !bytes.is_empty() {
            let left_in_chunk = self.check_rows - self.next_row % self.check_rows;
            let take = (bytes.len() as u64 / self.stride as u64).min(left_in_chunk);
            let (now, rest) = bytes.split_at(take as usize * self.stride);
            self.hasher.update(now);
            self.next_row += take;
            if self.next_row.is_multiple_of(self.check_rows) {
                self.sums.push(std::mem::take(&mut self.hasher).finalize());
            }
            bytes = rest;
        }
    }

    fn finish(mut self, rows_total: u64) -> Result<Vec<u32>> {
        ensure!(
            self.next_row == rows_total,
            "ngram pack: checksummed {} of {rows_total} rows",
            self.next_row
        );
        if !rows_total.is_multiple_of(self.check_rows) {
            self.sums.push(self.hasher.finalize());
        }
        Ok(self.sums)
    }
}

/// A packed file's header, bounds-checked.
struct Header {
    identity: Vec<u8>,
    check_rows: u64,
    sums: Vec<u32>,
}

fn read_header(f: &File) -> Result<Header> {
    let mut head = [0u8; FIXED as usize];
    read_exact_at(f, &mut head, 0).context("ngram pack: read header")?;
    ensure!(&head[..8] == MAGIC, "bad magic");
    let word = |i: usize| u64::from_le_bytes(head[i..i + 8].try_into().unwrap());
    let (len, check_rows, chunks) = (word(8), word(16), word(24));
    // Bounded on the u64s, with no addition: `16 + len` wrapped for a length
    // near u64::MAX in a release build, passed, and `vec![0; len]` aborted.
    header_fits(len, chunks)?;
    ensure!(check_rows > 0, "zero rows per checksum chunk");
    let mut rest = vec![0u8; (len + 4 * chunks) as usize];
    read_exact_at(f, &mut rest, FIXED).context("ngram pack: read identity")?;
    let sums = rest[len as usize..]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    rest.truncate(len as usize);
    Ok(Header {
        identity: rest,
        check_rows,
        sums,
    })
}

/// Check that `path` is the packed form of `spec`: header identity, length,
/// one sampled row per shard against its sources, and the chunk checksums
/// `verify` asks for.
pub fn validate(spec: &PackSpec<'_>, path: &Path, verify: Verify) -> Result<()> {
    let f = File::open(path).with_context(|| format!("ngram pack: open {}", path.display()))?;
    let hdr = read_header(&f)?;
    ensure!(
        hdr.identity == spec.identity()?.as_bytes(),
        "identity differs from the sources"
    );
    let rows_total = spec.rows_total();
    let want = packed_offset(rows_total, spec.stride);
    let have = f.metadata()?.len();
    ensure!(have == want, "file is {have} bytes, expected {want}");
    ensure!(
        hdr.sums.len() as u64 == rows_total.div_ceil(hdr.check_rows),
        "{} chunk checksums for {rows_total} rows of {} per chunk",
        hdr.sums.len(),
        hdr.check_rows
    );

    let mut files: Vec<(PathBuf, File)> = Vec::new();
    let mut got = vec![0u8; spec.stride];
    let mut exp = vec![0u8; spec.stride];
    for shard in 0..spec.shards() as u64 {
        let row = shard * spec.rows_per_shard + sample_row(shard, spec.rows_per_shard);
        read_exact_at(&f, &mut got, packed_offset(row, spec.stride))?;
        exp.fill(0);
        let mut at = 0;
        for (i, p) in spec.planes.iter().enumerate() {
            let (src, off) = spec.source_loc(i, row);
            let sf = cached_open(&mut files, src)?;
            read_exact_at(sf, &mut exp[at..at + p.row_bytes], off)?;
            at += p.row_bytes;
        }
        if got != exp {
            bail!("row {row} (shard {shard}) does not match its sources");
        }
    }

    let n = hdr.sums.len();
    let which: Vec<usize> = match verify {
        Verify::Full => (0..n).collect(),
        Verify::Sample => {
            rand::seq::index::sample(&mut rand::thread_rng(), n, SAMPLE_CHUNKS.min(n)).into_vec()
        }
    };
    verify_chunks(&f, &hdr, rows_total, spec.stride, &which)
}

/// Re-hash chunks `which` of the body and compare with the header's sums.
fn verify_chunks(
    f: &File,
    hdr: &Header,
    rows_total: u64,
    stride: usize,
    which: &[usize],
) -> Result<()> {
    // Read in 8 MiB pieces whatever the chunk size.
    let piece_rows = ((8u64 << 20) / stride as u64).max(1);
    let mut buf = vec![0u8; (piece_rows.min(hdr.check_rows)) as usize * stride];
    for &c in which {
        let first = c as u64 * hdr.check_rows;
        let end = (first + hdr.check_rows).min(rows_total);
        let mut h = crc32fast::Hasher::new();
        let mut r = first;
        while r < end {
            let n = (end - r).min(buf.len() as u64 / stride as u64);
            let bytes = &mut buf[..n as usize * stride];
            let at = packed_offset(r, stride);
            read_exact_at(f, bytes, at)?;
            drop_cache(f, at, bytes.len() as u64);
            h.update(&*bytes);
            r += n;
        }
        if h.finalize() != hdr.sums[c] {
            bail!("chunk {c} (rows {first}..{end}) fails its checksum: the packed file is damaged");
        }
    }
    Ok(())
}

/// A row of `shard` to spot-check, spread across the shard by a fixed hash so
/// the same rows are checked every open (first and last rows included).
fn sample_row(shard: u64, rows: u64) -> u64 {
    match shard {
        0 => 0,
        s if s % 2 == 1 => rows - 1,
        s => s.wrapping_mul(0x9E37_79B9_7F4A_7C15) % rows,
    }
}

/// What makes a source file this one: mtime (ns since the epoch) and inode.
fn file_stamp(m: &std::fs::Metadata) -> serde_json::Value {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let ns = m
            .mtime()
            .saturating_mul(1_000_000_000)
            .saturating_add(m.mtime_nsec());
        serde_json::json!({"mtime_ns": ns, "ino": m.ino()})
    }
    #[cfg(not(unix))]
    {
        let ns = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as u64);
        serde_json::json!({"mtime_ns": ns})
    }
}

fn remove_stale_temps(dir: &Path, name: &str) {
    let prefix = format!("{name}.tmp");
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

fn cached_open<'f>(files: &'f mut Vec<(PathBuf, File)>, path: &Path) -> Result<&'f File> {
    if let Some(i) = files.iter().position(|(p, _)| p == path) {
        return Ok(&files[i].1);
    }
    let f = File::open(path).with_context(|| format!("ngram pack: open {}", path.display()))?;
    files.push((path.to_path_buf(), f));
    Ok(&files.last().unwrap().1)
}

/// 64-bit FNV-1a: a stable name for an identity string (std's hasher is not
/// guaranteed stable across releases).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(target_os = "linux")]
fn drop_cache(f: &File, off: u64, len: u64) {
    use std::os::unix::io::AsRawFd;
    // Best effort: the copy is correct either way, only the page cache differs.
    unsafe {
        libc::posix_fadvise(
            f.as_raw_fd(),
            off as i64,
            len as i64,
            libc::POSIX_FADV_DONTNEED,
        )
    };
}

#[cfg(not(target_os = "linux"))]
fn drop_cache(_f: &File, _off: u64, _len: u64) {}

#[cfg(unix)]
fn ensure_free_space(dir: &Path, need: u64) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes())?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return Ok(()); // unknown: let the write itself report ENOSPC
    }
    // u64 on Linux, u32/c_ulong on macOS: the casts are only a no-op on one.
    #[allow(clippy::unnecessary_cast)]
    let free = st.f_bavail as u64 * st.f_frsize as u64;
    let margin = 1 << 30;
    ensure!(
        free >= need + margin,
        "ngram pack: {} has {:.1} GB free, the packed table needs {:.1} GB (+1 GB margin); \
         point ATLAS_PLE_PACKED_DIR at a larger volume",
        dir.display(),
        free as f64 / 1e9,
        need as f64 / 1e9
    );
    Ok(())
}

#[cfg(not(unix))]
fn ensure_free_space(_dir: &Path, _need: u64) -> Result<()> {
    Ok(())
}

/// Exclusive advisory lock on a lock file, released on drop (by close).
struct DirLock(#[allow(dead_code)] File);

impl DirLock {
    fn take(path: &Path) -> Result<Self> {
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .with_context(|| format!("ngram pack: open lock {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
                bail!(
                    "ngram pack: lock {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
        }
        Ok(Self(f))
    }
}

#[cfg(test)]
mod tests;
