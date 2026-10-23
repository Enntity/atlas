// SPDX-License-Identifier: AGPL-3.0-only
//! Single-shard buffered/direct I/O and upload; extracted without changing order.
use super::{direct_io, header::parse_header};
use crate::gpu::GpuBackend;
use crate::weights::nvfp4_csf::{CsfAssembler, csf_part};
use crate::weights::{WeightDtype, WeightTensor, evict_page_cache, f16_to_bf16_bytes};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;

/// Reader threads per shard; how many tensors, and how many bytes past the
/// next one the upload needs, they may hold ahead of it.
const SHARD_READERS: usize = 8;
const READ_AHEAD: usize = 16;
const READ_AHEAD_BYTES: usize = 1 << 30;

/// Load a single shard with O_DIRECT + pipelined read/copy.
///
/// Pipeline:
///   reader threads: pread tensors into aligned buffers → sync_channel ──▶
///   main thread:    recv, in tensor order → copy_h2d → store tensor
///
/// The readers stay at most [`READ_AHEAD`] tensors, and past the next one
/// [`READ_AHEAD_BYTES`], ahead of the copier.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_shard_fast(
    shard_path: &Path,
    tensor_filter: Option<&[String]>,
    gpu: &dyn GpuBackend,
    skip_fn: &dyn Fn(&str) -> bool,
    defer_fn: &dyn Fn(&str) -> bool,
    try_direct_io: bool,
    direct_io_tensor_cap: usize,
    prefetch_shards: bool,
    out: &mut HashMap<String, WeightTensor>,
    deferred_out: &mut HashMap<String, crate::weights::DeferredTensor>,
    offload_logged: &mut bool,
    csf: &mut CsfAssembler,
) -> Result<()> {
    // Header parsing uses a buffered fd — header is a few KB, cache pollution
    // is negligible and buffered I/O handles short reads cleanly.
    let mut meta_file = File::open(shard_path)
        .with_context(|| format!("Failed to open {}", shard_path.display()))?;
    let mut tensors = parse_header(&mut meta_file)?;
    let file_size = meta_file.metadata()?.len();

    // Filter down to tensors we actually want (index filter + EP filter).
    if let Some(allow) = tensor_filter {
        let allow_set: std::collections::HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
        tensors.retain(|t| allow_set.contains(t.name.as_str()));
    }
    // The n-gram embedding TABLES are never uploaded with the checkpoint —
    // 63 GB (LongCat-Lite) to ~102 GB (Flash-Next) of BF16 would exhaust a
    // 121 GB unified box before any quantization could run, and the fallback
    // on GB10 is managed memory, i.e. Linux swap, i.e. a kernel freeze. They
    // are recorded with their on-disk location and served either by streaming
    // per-table quantize-on-load or straight off NVMe by the row cache.
    //
    // Under expert TP a CSF-compressed scale's COMPONENTS are deferred (see
    // `should_defer_tensor`), to be decoded one matrix at a time by
    // `with_deferred_slice`; the assembler still checks them by shape.
    let mut deferred_here: Vec<(String, crate::weights::DeferredTensor)> = Vec::new();
    let mut csf_deferred: Vec<(String, Vec<usize>, usize)> = Vec::new();
    #[allow(clippy::items_after_statements)]
    tensors.retain(|t| {
        if crate::weights::is_ngram_table(&t.name) || (!skip_fn(&t.name) && defer_fn(&t.name)) {
            if csf_part(&t.name).is_some() {
                csf_deferred.push((t.name.clone(), t.shape.clone(), t.len));
            }
            deferred_here.push((
                t.name.clone(),
                crate::weights::DeferredTensor {
                    path: shard_path.to_path_buf(),
                    offset: t.abs_offset,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                },
            ));
            return false;
        }
        !skip_fn(&t.name)
    });
    for (name, shape, len) in &csf_deferred {
        csf.defer(name, shape, *len)?;
    }
    // A deferred plain scale is still a plain scale: one that is also
    // CSF-compressed is refused whichever form arrives first.
    for (name, _) in deferred_here.iter().filter(|(n, _)| csf_part(n).is_none()) {
        csf.note_plain(name)?;
    }
    if !deferred_here.is_empty() {
        tracing::info!(
            "Deferred {} tensor(s) in {} — served from disk, not uploaded",
            deferred_here.len(),
            shard_path.display()
        );
        deferred_out.extend(deferred_here);
    }

    // Per-shard heuristic: above `direct_io_tensor_cap` tensors, O_DIRECT's
    // per-tensor syscall + 4 KiB alignment overhead costs more than kernel
    // readahead on the buffered path saves. Skip the direct-open attempt
    // entirely in that case — keeps the log clean and avoids a wasted fd.
    let wants_direct = try_direct_io && tensors.len() <= direct_io_tensor_cap;
    if try_direct_io && !wants_direct {
        tracing::info!(
            "  Shard has {} tensors (> {} cap) — using buffered+pipelined path",
            tensors.len(),
            direct_io_tensor_cap
        );
    }

    // File for data reads. Try O_DIRECT; if it fails, fall through to buffered.
    let (direct_file, using_direct) = match wants_direct
        .then(|| direct_io::open_direct(shard_path))
        .transpose()
    {
        Ok(Some(f)) => (Some(f), true),
        Ok(None) => (None, false),
        Err(e) => {
            tracing::warn!(
                "O_DIRECT open failed for {} ({e}); falling back to buffered reads",
                shard_path.display()
            );
            (None, false)
        }
    };
    let buffered_file = File::open(shard_path)?;
    let data_fd = direct_file.as_ref().unwrap_or(&buffered_file);
    if prefetch_shards && !using_direct {
        advise_prefetch_shard(&buffered_file, shard_path, file_size);
    }

    // Pipelined readers: SHARD_READERS threads read tensors at most
    // READ_AHEAD ahead of the copier and send (tensor_index, aligned_buffer,
    // slice_start); the copier uploads them in tensor order. One reader
    // waited on one read at a time (~0.7 GB/s over a GLM-5 TP4 rank's
    // shards).
    type ReadMsg = (usize, direct_io::AlignedBuffer, usize);
    let (tx, rx) = sync_channel::<Result<ReadMsg>>(READ_AHEAD);
    let tensors_for_reader: std::sync::Arc<Vec<(u64, usize)>> =
        std::sync::Arc::new(tensors.iter().map(|t| (t.abs_offset, t.len)).collect());
    let raw_fd = {
        use std::os::unix::io::AsRawFd;
        data_fd.as_raw_fd()
    };
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    let copied = std::sync::Arc::new(AtomicUsize::new(0));
    let held = std::sync::Arc::new(AtomicUsize::new(0));
    // Raised when the copier returns (an upload failed), so no reader waits
    // for it forever.
    struct Stop(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Stop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let stop = Stop(std::sync::Arc::default());

    let _ = file_size; // retained for future use (tail-fragment buffered read)
    let reader_handles: Vec<_> = (0..SHARD_READERS)
        .map(|_| {
            let (tx, list) = (tx.clone(), tensors_for_reader.clone());
            let (next, copied, held) = (next.clone(), copied.clone(), held.clone());
            let stop = stop.0.clone();
            std::thread::spawn(move || {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(abs_offset, len)) = list.get(idx) else {
                        break;
                    };
                    loop {
                        let first = copied.load(Ordering::Acquire);
                        if idx == first
                            || (idx < first + READ_AHEAD
                                && held.load(Ordering::Acquire) + len <= READ_AHEAD_BYTES)
                        {
                            break;
                        }
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                    held.fetch_add(len, Ordering::AcqRel);
                    let msg = direct_io::read_tensor_aligned(raw_fd, abs_offset, len, using_direct)
                        .map(|(buf, slice_start)| (idx, buf, slice_start));
                    if tx.send(msg).is_err() {
                        break; // receiver dropped
                    }
                }
            })
        })
        .collect();
    drop(tx);
    let mut early: std::collections::BTreeMap<usize, (direct_io::AlignedBuffer, usize)> =
        std::collections::BTreeMap::new();
    let mut want = 0usize;
    // The reads in tensor order: each message, held until those before it
    // have arrived.
    let in_order = std::iter::from_fn(|| {
        loop {
            if let Some((buf, slice_start)) = early.remove(&want) {
                held.fetch_sub(tensors_for_reader[want].1, Ordering::AcqRel);
                want += 1;
                copied.store(want, Ordering::Release);
                return Some(Ok((want - 1, buf, slice_start)));
            }
            match rx.recv() {
                Ok(Ok((idx, buf, slice_start))) => {
                    early.insert(idx, (buf, slice_start));
                }
                Ok(Err(e)) => return Some(Err(e)),
                Err(_) => return None,
            }
        }
    });

    // Copier: drains the reads in order, does gpu alloc + copy_h2d, inserts
    // into the map.
    for result in in_order {
        let (idx, buf, slice_start) = result?;
        let meta = &tensors[idx];
        let raw = &buf.as_slice()[slice_start..slice_start + meta.len];
        // CSF-compressed NVFP4 scales (either encoding): the components are
        // gathered (possibly across shards) and only the decoded E4M3
        // `weight_scale` is uploaded.
        if csf_part(&meta.name).is_some() {
            if let Some(scale) = csf.offer(&meta.name, &meta.shape, raw)? {
                let ptr = upload(gpu, &scale.name, &scale.bytes, offload_logged)?;
                out.insert(
                    scale.name,
                    WeightTensor {
                        ptr,
                        shape: scale.shape,
                        dtype: WeightDtype::FP8E4M3,
                    },
                );
            }
            continue;
        }
        csf.note_plain(&meta.name)?;
        // F16 shards: convert bytes to BF16 before upload (same length,
        // different bit layout — meta.dtype is already staged as BF16).
        let converted: Vec<u8>;
        let src: &[u8] = if meta.from_f16 {
            converted = f16_to_bf16_bytes(raw);
            &converted
        } else {
            raw
        };

        let ptr = upload(gpu, &meta.name, src, offload_logged)?;
        out.insert(
            meta.name.clone(),
            WeightTensor {
                ptr,
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            },
        );
    }

    for reader in reader_handles {
        reader
            .join()
            .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;
    }

    // Release file handles, then advise the kernel to drop any pages we did
    // end up caching on the buffered fallback path. O_DIRECT reads never hit
    // the page cache, so the posix_fadvise is a no-op there but cheap.
    drop(direct_file);
    evict_page_cache(&buffered_file);
    drop(buffered_file);
    Ok(())
}

/// GPU-alloc and copy one tensor's bytes, falling back to managed (UVM)
/// memory when the device allocation fails.
fn upload(
    gpu: &dyn GpuBackend,
    name: &str,
    src: &[u8],
    offload_logged: &mut bool,
) -> Result<crate::gpu::DevicePtr> {
    Ok(match gpu.alloc(src.len()) {
        Ok(p) => {
            gpu.copy_h2d(src, p)?;
            p
        }
        Err(_) => {
            if !*offload_logged {
                tracing::warn!(
                    "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory",
                    name,
                    src.len()
                );
                *offload_logged = true;
            }
            let p = gpu.alloc_managed(src.len())?;
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), p.0 as *mut u8, src.len());
            }
            p
        }
    })
}

#[cfg(target_os = "linux")]
fn advise_prefetch_shard(file: &File, shard_path: &Path, file_size: u64) {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    let seq_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
    let willneed_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED) };
    if seq_rc == 0 && willneed_rc == 0 {
        tracing::info!(
            "  NFS/shard prefetch requested for {} ({:.2} GB)",
            shard_path.display(),
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::warn!(
            "  NFS/shard prefetch hint failed for {}: sequential_rc={}, willneed_rc={}",
            shard_path.display(),
            seq_rc,
            willneed_rc
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_prefetch_shard(_file: &File, _shard_path: &Path, _file_size: u64) {}
