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
//! [0, 64 KiB)            header: MAGIC, u64 LE length, identity JSON, zero pad
//! [64 KiB + r*stride)    row r: plane bytes in plane order, then zeros to stride
//! ```
//!
//! The identity JSON names every source (resolved file name, length, offset
//! per shard), the geometry and a caller tag. Opening an existing file
//! requires the identity to match exactly AND a sample of rows (one per shard)
//! to match the sources byte for byte, so a re-published checkpoint with the
//! same layout but new values is caught rather than served stale.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use atlas_tier::pio::{read_exact_at, write_all_at};

/// Header bytes ahead of row 0: a multiple of the 4 KiB O_DIRECT block, so
/// row 0 starts aligned, and room for the identity of a 128-shard table
/// (~5 KB with 10-digit offsets; `identity_of_the_real_table_fits`).
pub const HEADER: u64 = 64 * 1024;
const MAGIC: &[u8; 8] = b"ATLASPK\x01";

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

    /// The JSON that a packed file must carry to be reused for this spec.
    ///
    /// Sources are named by their RESOLVED file name: in a Hugging Face cache
    /// the snapshot entry is a symlink to a blob named by its content hash, so
    /// a different revision of the same layout gets a different identity
    /// without anything having to read 27 GB.
    pub fn identity(&self) -> Result<String> {
        self.check()?;
        let mut files: Vec<&Path> = Vec::new();
        let mut desc = Vec::new();
        let mut planes = Vec::new();
        for p in &self.planes {
            let mut shards = Vec::with_capacity(p.shards.len());
            for (path, off) in p.shards {
                let idx = match files.iter().position(|f| *f == path.as_path()) {
                    Some(i) => i,
                    None => {
                        let real = std::fs::canonicalize(path)
                            .with_context(|| format!("ngram pack: resolve {}", path.display()))?;
                        let len = std::fs::metadata(&real)
                            .with_context(|| format!("ngram pack: stat {}", real.display()))?
                            .len();
                        let name = real.file_name().map(|n| n.to_string_lossy().into_owned());
                        desc.push(serde_json::json!({"name": name, "len": len}));
                        files.push(path.as_path());
                        files.len() - 1
                    }
                };
                shards.push(serde_json::json!([idx, off]));
            }
            planes.push(serde_json::json!({"row_bytes": p.row_bytes, "shards": shards}));
        }
        Ok(serde_json::json!({
            "format": "atlas-ngram-pack/1",
            "tag": self.tag,
            "rows_per_shard": self.rows_per_shard,
            "stride": self.stride,
            "files": desc,
            "planes": planes,
        })
        .to_string())
    }

    /// `ngram-{tag}-{fnv64(identity)}.bin`: one file per distinct source.
    pub fn file_name(&self) -> Result<String> {
        Ok(format!(
            "ngram-{}-{:016x}.bin",
            self.tag,
            fnv1a64(self.identity()?.as_bytes())
        ))
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
/// valid one is there.
///
/// A valid file is used without taking the lock, so a table packed at install
/// time can sit on a read-only mount: builds write a temp file and rename it,
/// so the final name only ever holds a complete file. A build holds an
/// exclusive lock on `<file>.lock`, so two processes on one host build once.
pub fn open_or_pack(spec: &PackSpec<'_>, dir: &Path) -> Result<PackedTable> {
    let name = spec.file_name()?;
    let path = dir.join(&name);
    let table = PackedTable {
        path: path.clone(),
        base_offset: HEADER,
        rows_total: spec.rows_total(),
        stride: spec.stride,
    };
    if path.exists() && validate(spec, &path).is_ok() {
        return Ok(table);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("ngram pack: create {}", dir.display()))?;
    let _lock = DirLock::take(&dir.join(format!("{name}.lock")))?;
    // Re-check under the lock: another process may have just finished.
    if path.exists() {
        match validate(spec, &path) {
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
    if let Err(e) = pack(spec, &tmp, CHUNK_ROWS) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("ngram pack: rename to {}", path.display()))?;
    validate(spec, &path).context("ngram pack: freshly built file failed validation")?;
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

fn pack(spec: &PackSpec<'_>, out_path: &Path, chunk_rows: u64) -> Result<()> {
    let out = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_path)
        .with_context(|| format!("ngram pack: create {}", out_path.display()))?;
    let identity = spec.identity()?;
    ensure!(
        8 + 8 + identity.len() <= HEADER as usize,
        "ngram pack: identity is {} bytes, the header holds {}",
        identity.len(),
        HEADER - 16
    );
    let mut header = vec![0u8; HEADER as usize];
    header[..8].copy_from_slice(MAGIC);
    header[8..16].copy_from_slice(&(identity.len() as u64).to_le_bytes());
    header[16..16 + identity.len()].copy_from_slice(identity.as_bytes());
    write_all_at(&out, &header, 0).context("ngram pack: write header")?;

    let widths: Vec<usize> = spec.planes.iter().map(|p| p.row_bytes).collect();
    let chunk = chunk_rows.max(1);
    let mut srcs: Vec<Vec<u8>> = widths
        .iter()
        .map(|w| vec![0u8; chunk as usize * w])
        .collect();
    let mut buf = vec![0u8; chunk as usize * spec.stride];
    let mut files: Vec<(PathBuf, File)> = Vec::new();
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
    out.sync_all().context("ngram pack: final sync")?;
    Ok(())
}

/// Check that `path` is the packed form of `spec`: header identity, length,
/// and one sampled row per shard against its sources.
pub fn validate(spec: &PackSpec<'_>, path: &Path) -> Result<()> {
    let f = File::open(path).with_context(|| format!("ngram pack: open {}", path.display()))?;
    let mut head = [0u8; 16];
    read_exact_at(&f, &mut head, 0).context("ngram pack: read header")?;
    ensure!(&head[..8] == MAGIC, "bad magic");
    // Bounded on the u64, with no addition: `16 + len` wrapped for a length
    // near u64::MAX in a release build, passed, and `vec![0; len]` aborted.
    let len = u64::from_le_bytes(head[8..16].try_into().unwrap());
    ensure!(len <= HEADER - 16, "header length {len} out of range");
    let mut id = vec![0u8; len as usize];
    read_exact_at(&f, &mut id, 16).context("ngram pack: read identity")?;
    ensure!(
        id == spec.identity()?.as_bytes(),
        "identity differs from the sources"
    );
    let want = packed_offset(spec.rows_total(), spec.stride);
    let have = f.metadata()?.len();
    ensure!(have == want, "file is {have} bytes, expected {want}");

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
