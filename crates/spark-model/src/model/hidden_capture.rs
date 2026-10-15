// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter-distillation hidden-state capture (`ATLAS_HIDDEN_CAPTURE_DIR`).
//!
//! Research only. Writes the target's hidden rows at the drafter layers
//! (`ATLAS_HIDDEN_CAPTURE_LAYERS`, default the DFlash2 set 5,14,24,33,42)
//! plus the final pre-norm hidden for the prefill rows of a request, so a
//! drafter can be trained on exactly the features the engine feeds it. The
//! layer rows use the same mHC stream contraction as the DFlash capture
//! (`dflash_capture_rows`); the final slot is the contracted `hidden` the
//! final norm reads.
//!
//! Capture is armed per request by `<dir>/enable` on each rank's host. The
//! file is empty (tag `untagged`, every row) or JSON
//! `{"tag": "<request>", "from": <first absolute position to keep>}`; rows
//! before `from` are never copied off the device (a drafter only needs its
//! context window before the generated span). It is re-read at the start of
//! every chunk, so the same server can generate (flag absent) and capture.
//!
//! Sequence-parallel prefill (`ATLAS_GLM_PREFILL_SP=1`) is supported: each
//! rank writes the layer rows it owns (rank 0 the upper half of a chunk), and
//! rank 0 writes the final rows, which are all-gathered after the last layer.
//! Without SP, rank 0 writes everything. Each piece is a segment file
//! `<dir>/chunks/<tag>/r<rank>-<L|F>-p<pos0>-n<rows>.bin` (BF16, slot-major
//! `[slots, rows, hidden]`; `L` = the capture layers, `F` = final) with a
//! JSON sidecar holding the absolute start, rows and token ids. A request's
//! segments tile `[from, T)` once per kind; the `.part` file is renamed only
//! when complete.
//!
//! Copies are asynchronous: each layer's rows are contracted into a device
//! staging buffer and copied into a pinned host buffer on the compute stream,
//! the chunk syncs once after its final rows, and a writer thread writes the
//! files while the next chunk computes.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;

struct Config {
    dir: PathBuf,
    layers: Vec<usize>,
    buffers: usize,
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let dir = std::env::var("ATLAS_HIDDEN_CAPTURE_DIR").ok()?;
            if dir.is_empty() {
                return None;
            }
            let layers = std::env::var("ATLAS_HIDDEN_CAPTURE_LAYERS")
                .unwrap_or_else(|_| "5,14,24,33,42".into())
                .split(',')
                .map(|s| s.trim().parse::<usize>().expect("ATLAS_HIDDEN_CAPTURE_LAYERS"))
                .collect::<Vec<_>>();
            assert!(
                !layers.is_empty() && layers.windows(2).all(|w| w[0] < w[1]),
                "ATLAS_HIDDEN_CAPTURE_LAYERS must be strictly increasing"
            );
            let buffers = std::env::var("ATLAS_HIDDEN_CAPTURE_BUFFERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2usize)
                .max(1);
            tracing::warn!(dir = %dir, ?layers, buffers, "hidden capture configured (research only)");
            Some(Config {
                dir: PathBuf::from(dir),
                layers,
                buffers,
            })
        })
        .as_ref()
}

/// Per-request control read from `<dir>/enable`.
struct Control {
    tag: String,
    from: usize,
}

fn read_control(dir: &Path) -> Result<Option<Control>> {
    let raw = match std::fs::read(dir.join("enable")) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).context("read hidden capture enable"),
    };
    if raw.iter().all(u8::is_ascii_whitespace) {
        return Ok(Some(Control {
            tag: "untagged".into(),
            from: 0,
        }));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&raw).context("hidden capture enable JSON")?;
    let tag = v["tag"].as_str().context("enable: tag")?.to_string();
    ensure!(
        !tag.is_empty()
            && tag
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            && !tag.starts_with('.'),
        "enable: bad tag {tag:?}"
    );
    let from = v["from"].as_u64().unwrap_or(0) as usize;
    Ok(Some(Control { tag, from }))
}

/// A page-locked host buffer, owned by exactly one of: the pool, the active
/// chunk, or a writer job.
struct Pinned {
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: the buffer is plain page-locked memory with a single owner at a time
// (moved between the capture thread and the writer thread, never shared).
unsafe impl Send for Pinned {}

/// Free buffers plus how many exist in total (free + in flight).
struct Pool {
    free: Vec<Pinned>,
    total: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    free: Vec::new(),
    total: 0,
});
static POOL_CV: Condvar = Condvar::new();

/// One output segment: `len` bytes at `offset` of the job's buffer.
struct Segment {
    path: PathBuf,
    offset: usize,
    len: usize,
    meta: Vec<u8>,
}

struct Job {
    buf: Pinned,
    segments: Vec<Segment>,
}

fn writer() -> &'static Mutex<mpsc::Sender<Job>> {
    static TX: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("hidden-capture-writer".into())
            .spawn(move || {
                for job in rx {
                    // SAFETY: the producer synchronized the stream after the
                    // last copy into this buffer, and hands over sole ownership.
                    let data = unsafe { std::slice::from_raw_parts(job.buf.ptr, job.buf.bytes) };
                    for seg in &job.segments {
                        if let Err(e) = write_segment(seg, &data[seg.offset..seg.offset + seg.len]) {
                            tracing::error!(path = ?seg.path, error = %e, "hidden capture write failed");
                            let _ = std::fs::write(seg.path.with_extension("err"), e.to_string());
                        }
                    }
                    let mut pool = POOL.lock().unwrap();
                    pool.free.push(job.buf);
                    POOL_CV.notify_all();
                }
            })
            .expect("spawn hidden capture writer");
        Mutex::new(tx)
    })
}

fn write_segment(seg: &Segment, data: &[u8]) -> Result<()> {
    let part = seg.path.with_extension("part");
    let mut file = std::fs::File::create(&part).with_context(|| format!("create {part:?}"))?;
    file.write_all(data)?;
    drop(file);
    std::fs::write(seg.path.with_extension("json"), &seg.meta)?;
    std::fs::rename(&part, &seg.path)?;
    Ok(())
}

/// Absolute half-open row range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rows {
    start: usize,
    end: usize,
}

impl Rows {
    fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// The chunk being captured.
struct Active {
    tag: String,
    from: usize,
    pos0: usize,
    rows: usize,
    next_slot: usize,
    sp: bool,
    /// First absolute position of this rank's local (compacted) rows.
    local0: usize,
    /// Layer rows this rank keeps (empty when none).
    layer_rows: Rows,
    /// Final rows this rank keeps (rank 0 only; empty otherwise).
    final_rows: Rows,
    buf: Pinned,
}

/// Device staging rows (grown on demand) and the active chunk.
struct State {
    staging: DevicePtr,
    staging_bytes: usize,
    active: Option<Active>,
}

static STATE: Mutex<State> = Mutex::new(State {
    staging: DevicePtr::NULL,
    staging_bytes: 0,
    active: None,
});

impl TransformerModel {
    fn hidden_capture_rank(&self) -> usize {
        self.comm.as_ref().map_or(0, |c| c.rank())
    }

    /// A pinned buffer of at least `bytes`, from the pool or newly allocated
    /// (at most `cfg.buffers` exist); waits for the writer when all are busy.
    fn hidden_capture_take_buffer(&self, cfg: &Config, bytes: usize) -> Result<Pinned> {
        let mut pool = POOL.lock().unwrap();
        loop {
            if let Some(i) = pool.free.iter().position(|b| b.bytes >= bytes) {
                return Ok(pool.free.swap_remove(i));
            }
            if pool.total < cfg.buffers {
                pool.total += 1;
                break;
            }
            if let Some(small) = pool.free.pop() {
                // Too small: replace it.
                self.gpu.free_host_pinned(small.ptr, small.bytes)?;
                break;
            }
            pool = POOL_CV.wait(pool).unwrap();
        }
        drop(pool);
        // Round up so a few sizes cover every chunk.
        let alloc = bytes.div_ceil(64 << 20) * (64 << 20);
        let ptr = self.gpu.alloc_host_pinned(alloc).inspect_err(|_| {
            POOL.lock().unwrap().total -= 1;
        })?;
        Ok(Pinned { ptr, bytes: alloc })
    }

    fn hidden_capture_staging(
        &self,
        state: &mut State,
        bytes: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        if state.staging_bytes < bytes {
            if !state.staging.is_null() {
                // Queued copies may still read the old rows.
                self.gpu.synchronize(stream)?;
                self.gpu.free(state.staging)?;
            }
            state.staging = DevicePtr::NULL;
            state.staging_bytes = 0;
            state.staging = self.gpu.alloc(bytes)?;
            state.staging_bytes = bytes;
        }
        Ok(state.staging)
    }

    /// Async copy of `bytes` from `src` into the active buffer at `offset`.
    fn hidden_capture_d2h(
        &self,
        buf: &Pinned,
        offset: usize,
        src: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            offset + bytes <= buf.bytes,
            "hidden capture buffer overflow"
        );
        // SAFETY: in bounds (checked above); the buffer is owned by the active
        // chunk and not read until the stream is synchronized.
        let dst = unsafe { std::slice::from_raw_parts_mut(buf.ptr.add(offset), bytes) };
        self.gpu.copy_d2h_async(src, dst, stream)
    }

    /// After prefill layer `layer_idx` over `rows` chunk rows starting at
    /// absolute position `pos0`: copy this rank's kept rows of that layer.
    pub(super) fn hidden_capture_layer(
        &self,
        layer_idx: usize,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(cfg) = config() else { return Ok(()) };
        let Some(slot) = cfg.layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        if rows == 0 {
            return Ok(());
        }
        let mut state = STATE.lock().unwrap();
        if slot == 0 {
            // A new chunk starts here. A chunk aborted mid-way leaves no file.
            if let Some(stale) = state.active.take() {
                POOL.lock().unwrap().free.push(stale.buf);
                POOL_CV.notify_all();
            }
            let Some(ctl) = read_control(&cfg.dir)? else {
                return Ok(());
            };
            let rank = self.hidden_capture_rank();
            let sp = crate::layers::glm_sp::current();
            if sp.is_none() && rank != 0 {
                return Ok(());
            }
            let (local0, local_rows) = sp.map_or((pos0, rows), |s| (pos0 + s.row0, s.rows));
            let end = pos0 + rows;
            let layer_rows = Rows {
                start: local0.max(ctl.from),
                end: (local0 + local_rows).max(local0.max(ctl.from)),
            };
            let final_rows = if rank == 0 {
                Rows {
                    start: pos0.max(ctl.from).min(end),
                    end,
                }
            } else {
                Rows { start: end, end }
            };
            if layer_rows.len() == 0 && final_rows.len() == 0 {
                return Ok(());
            }
            let h = self.config.hidden_size;
            let bytes = (cfg.layers.len() * layer_rows.len() + final_rows.len()) * h * 2;
            let buf = self.hidden_capture_take_buffer(cfg, bytes)?;
            state.active = Some(Active {
                tag: ctl.tag,
                from: ctl.from,
                pos0,
                rows,
                next_slot: 0,
                sp: sp.is_some(),
                local0,
                layer_rows,
                final_rows,
                buf,
            });
        }
        let Some(active) = state.active.as_ref() else {
            return Ok(());
        };
        ensure!(
            active.next_slot == slot && active.pos0 == pos0 && active.rows == rows,
            "hidden capture out of order: slot {slot} (expected {}), rows {rows}@{pos0}",
            active.next_slot
        );
        let (keep, local0) = (active.layer_rows, active.local0);
        if keep.len() > 0 {
            let h = self.config.hidden_size;
            let bytes = keep.len() * h * 2;
            let staging = self.hidden_capture_staging(&mut state, bytes, stream)?;
            self.dflash_capture_rows(keep.start - local0, keep.len(), staging, h * 2, stream)?;
            let active = state.active.as_ref().unwrap();
            self.hidden_capture_d2h(&active.buf, slot * bytes, staging, bytes, stream)?;
        }
        state.active.as_mut().unwrap().next_slot += 1;
        Ok(())
    }

    /// After the last layer (on every rank): copy the kept final rows of
    /// `hidden` (BF16 `[rows, hidden]`, the final norm's input; all rows of
    /// the chunk, gathered under SP), sync once, and queue the chunk's
    /// segments for the writer.
    pub(super) fn hidden_capture_final(
        &self,
        tokens: &[u32],
        hidden: DevicePtr,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(cfg) = config() else { return Ok(()) };
        if rows == 0 {
            return Ok(());
        }
        let mut state = STATE.lock().unwrap();
        let Some(active) = state.active.take() else {
            return Ok(());
        };
        ensure!(
            active.next_slot == cfg.layers.len() && active.pos0 == pos0 && active.rows == rows,
            "hidden capture final out of order: slot {} rows {rows}@{pos0}",
            active.next_slot
        );
        drop(state);
        let h = self.config.hidden_size;
        let row_bytes = h * 2;
        let nl = cfg.layers.len();
        let layer_bytes = nl * active.layer_rows.len() * row_bytes;
        let fr = active.final_rows;
        if fr.len() > 0 {
            self.hidden_capture_d2h(
                &active.buf,
                layer_bytes,
                hidden.offset((fr.start - pos0) * row_bytes),
                fr.len() * row_bytes,
                stream,
            )?;
        }
        self.gpu.synchronize(stream)?;
        let rank = self.hidden_capture_rank();
        let dir = cfg.dir.join("chunks").join(&active.tag);
        if !dir.exists() {
            std::fs::create_dir_all(&dir).with_context(|| format!("create {dir:?}"))?;
            // The harness (another uid) moves the segments out of it.
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777))?;
        }
        let mut segments = Vec::new();
        for (kind, range, offset, slots) in
            [("L", active.layer_rows, 0, nl), ("F", fr, layer_bytes, 1)]
        {
            if range.len() == 0 {
                continue;
            }
            // Every rank should see the whole prompt; the harness checks the
            // ids against rank 0's, so a short slice is recorded, not fatal.
            let ids = tokens.get(range.start..range.end);
            let meta = serde_json::json!({
                "schema": "atlas.hidden_capture.segment/v2",
                "tag": active.tag,
                "rank": rank,
                "kind": if kind == "L" { "layers" } else { "final" },
                "layers": if kind == "L" { serde_json::json!(cfg.layers) } else { serde_json::json!([]) },
                "pos0": range.start,
                "rows": range.len(),
                "hidden_size": h,
                "dtype": "bf16",
                "layout": "slot-major [slots, rows, hidden]",
                "final_slot": "contracted pre-final-norm hidden",
                "chunk": [pos0, rows],
                "sp": active.sp,
                "from": active.from,
                "seq_tokens": tokens.len(),
                "token_ids": ids,
            });
            segments.push(Segment {
                path: dir.join(format!(
                    "r{rank}-{kind}-p{:09}-n{}.bin",
                    range.start,
                    range.len()
                )),
                offset,
                len: slots * range.len() * row_bytes,
                meta: serde_json::to_vec(&meta)?,
            });
        }
        writer()
            .lock()
            .unwrap()
            .send(Job {
                buf: active.buf,
                segments,
            })
            .map_err(|_| anyhow::anyhow!("hidden capture writer stopped"))?;
        Ok(())
    }
}
