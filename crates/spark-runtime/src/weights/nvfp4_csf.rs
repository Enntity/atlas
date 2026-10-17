// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time decoder for CSF-compressed NVFP4 block scales.
//!
//! CSF checkpoints (`quantized_layers.*.weight_scale_encoding = "csf"`, codec
//! `byte-window4-fixed-stream-u24-exceptions/1`) do not ship an NVFP4
//! projection's E4M3 `weight_scale [n, k/16]`. They ship two tensors instead:
//!
//! * `<name>.nvfp4_csf_fixed` — U8 `[rows/16, 16 * (1 + cols/2)]`. Per 16-row
//!   slab: one base byte per row, then each row's 4-bit codes, two per byte,
//!   LOW nibble first. A scale byte is `base + code`.
//! * `<name>.nvfp4_csf_exceptions` — U32, sorted words whose low 24 bits are a
//!   flat `row * cols + col` position and whose high 8 bits REPLACE the scale
//!   byte there (the bytes outside the row's 16-wide window).
//!
//! The decode is lossless: the output is the uncompressed checkpoint's E4M3
//! bytes exactly, so every NVFP4 consumer reads `<name>` unchanged and never
//! learns the checkpoint was compressed. That is why it lives here, at upload,
//! rather than in any model loader: Qwen3.8-Flash-Next-FP4-Local's 73,728
//! routed-expert scales use it, and so does the GLM CSF checkpoint.
//!
//! Format references (both read, neither copied — the arithmetic is the format):
//! * vLLM PR 60646, `vllm/model_executor/model_loader/csf_scales.py`
//!   (Apache-2.0), as carried by `eugr/spark-vllm-docker`'s
//!   `docker/patches/qwen3.8/vllm/14-pr60646.patch`. Its decoder is the golden
//!   reference in `scripts/nvfp4_csf_golden.py`.
//! * flashinfer `flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py`
//!   (Apache-2.0) — the GPU expander. Its `make_nvfp4_csf_batch` is where the
//!   stricter checks below come from: exception positions must be unique,
//!   sorted and in range, and the position field must be able to address the
//!   whole matrix.
//!
//! Only the byte-window codec exists in checkpoints. flashinfer also has two
//! resident-only codecs (`codec` 1/2, 8-row selector/mantissa planes) that are
//! never serialized; a tensor named like a CSF part but shaped like neither is
//! refused by the shape checks rather than guessed at. MXFP4 has no CSF form.

use anyhow::{Context, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Suffix of the fixed stream.
pub const CSF_FIXED_SUFFIX: &str = ".nvfp4_csf_fixed";
/// Suffix of the exception stream.
pub const CSF_EXCEPTIONS_SUFFIX: &str = ".nvfp4_csf_exceptions";

/// Row bases above this would carry into the next byte when a code is added
/// (the reference packs four decoded bytes in one u32 add and relies on it).
/// Shared with `csf_32x2_v1`, whose NVFP4 bases have the same bound.
pub(crate) const MAX_BASE: u8 = 240;
/// Rows per fixed-stream slab.
const SLAB: usize = 16;
/// The exception position field is 24 bits wide.
const POSITION_BITS: u32 = 24;

/// Which half of a CSF pair a tensor name is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsfPart {
    Fixed,
    Exceptions,
}

/// `Some((decoded_name, part))` when `name` is half of a CSF pair. The decoded
/// name is the tensor the pair stands in for (`...weight_scale`).
pub fn csf_part(name: &str) -> Option<(&str, CsfPart)> {
    if let Some(base) = name.strip_suffix(CSF_FIXED_SUFFIX) {
        Some((base, CsfPart::Fixed))
    } else {
        name.strip_suffix(CSF_EXCEPTIONS_SUFFIX)
            .map(|base| (base, CsfPart::Exceptions))
    }
}

/// The decoded size, in bytes, of a fixed stream of this shape — what the
/// pair costs once uploaded. Used by the pre-flight estimate, which would
/// otherwise count the 3.6 GB compressed and miss 3.9 GB of the 7.5 GB that
/// lands on the device.
pub fn decoded_len(fixed_shape: &[usize]) -> Option<usize> {
    geometry(fixed_shape).ok().map(|(_, _, total)| total)
}

/// `(rows, cols, rows * cols)` of the decoded matrix, from the fixed stream's
/// shape. The shape comes from a file header, so every product is checked:
/// unchecked, a hostile header wraps in a release build.
fn geometry(fixed_shape: &[usize]) -> Result<(usize, usize, usize)> {
    ensure!(
        fixed_shape.len() == 2,
        "CSF fixed stream must be 2-D, got {fixed_shape:?}"
    );
    let (slabs, width) = (fixed_shape[0], fixed_shape[1]);
    ensure!(
        slabs > 0 && width % SLAB == 0 && width / SLAB >= 2,
        "CSF fixed stream {fixed_shape:?} is not [rows/16, 16 * (1 + cols/2)]"
    );
    let rows = slabs.checked_mul(SLAB);
    let cols = (width / SLAB - 1).checked_mul(2);
    let (Some(rows), Some(cols)) = (rows, cols) else {
        bail!("CSF fixed stream {fixed_shape:?}: decoded size overflows");
    };
    let total = rows
        .checked_mul(cols)
        .with_context(|| format!("CSF fixed stream {fixed_shape:?}: decoded size overflows"))?;
    Ok((rows, cols, total))
}

/// Refuse a decoded `[rows, cols]` scale that is not the `[n, k/16]` E4M3
/// matrix its packed NVFP4 weight `[n, k/2]` needs.
fn check_against_weight(scale: &str, got: [usize; 2], weight: Option<&[usize]>) -> Result<()> {
    let wname = scale
        .strip_suffix("_scale")
        .filter(|w| w.ends_with(".weight"))
        .with_context(|| format!("CSF scale {scale} is not a `.weight_scale`"))?;
    let Some(w) = weight else {
        bail!("CSF scale {scale} decoded, but there is no `{wname}` for it to scale");
    };
    let want = match *w {
        [n, k_packed] => k_packed
            .checked_mul(2)
            .filter(|k| k % 16 == 0)
            .map(|k| [n, k / 16]),
        _ => None,
    };
    ensure!(
        want == Some(got),
        "CSF scale {scale} decoded to {got:?}, but its weight `{wname}` {w:?} needs \
         {want:?} ([n, k/2] packed NVFP4 -> [n, k/16])"
    );
    Ok(())
}

/// Decode one CSF pair to the E4M3 `[rows, cols]` scale matrix, as raw bytes.
///
/// `exceptions` is the U32 stream's raw little-endian bytes. Every check that
/// can fail does so loudly: a scale that decodes wrong produces an expert that
/// runs and is wrong, which is the failure this codebase most wants to avoid.
pub fn decode_nvfp4_csf(
    fixed: &[u8],
    fixed_shape: &[usize],
    exceptions: &[u8],
) -> Result<(Vec<u8>, [usize; 2])> {
    let (rows, cols, total) = geometry(fixed_shape)?;
    let width = fixed_shape[1];
    // `rows * cols` did not overflow, and the fixed stream is smaller.
    let need = fixed_shape[0] * width;
    ensure!(
        fixed.len() == need,
        "CSF fixed stream holds {} bytes but its shape {fixed_shape:?} needs {need}",
        fixed.len(),
    );
    ensure!(
        exceptions.len().is_multiple_of(4),
        "CSF exception stream is {} bytes, not a whole number of u32 words",
        exceptions.len()
    );
    ensure!(
        total <= 1 << POSITION_BITS,
        "CSF {rows}x{cols} matrix cannot be addressed by a {POSITION_BITS}-bit position"
    );

    let mut out = vec![0u8; total];
    let half = cols / 2;
    for (slab, src) in fixed.chunks_exact(width).enumerate() {
        let (bases, codes) = src.split_at(SLAB);
        for (r, (&base, row_codes)) in bases.iter().zip(codes.chunks_exact(half)).enumerate() {
            ensure!(
                base <= MAX_BASE,
                "CSF row {} base {base:#04x} exceeds {MAX_BASE} — corrupt fixed stream",
                slab * SLAB + r
            );
            let row = slab * SLAB + r;
            let dst = &mut out[row * cols..(row + 1) * cols];
            for (pair, &packed) in dst.chunks_exact_mut(2).zip(row_codes) {
                pair[0] = base + (packed & 0x0F);
                pair[1] = base + (packed >> 4);
            }
        }
    }

    let mut prev: Option<usize> = None;
    for (i, w) in exceptions.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        let pos = (word & ((1 << POSITION_BITS) - 1)) as usize;
        ensure!(
            pos < total,
            "CSF exception {i} addresses {pos}, past the {rows}x{cols} matrix"
        );
        ensure!(
            prev.is_none_or(|p| pos > p),
            "CSF exception {i} at {pos} is out of order (previous {prev:?}) — the \
             stream must be sorted and unique"
        );
        prev = Some(pos);
        out[pos] = (word >> POSITION_BITS) as u8;
    }

    check_decoded_scales(&out, cols, "CSF")?;
    Ok((out, [rows, cols]))
}

/// Refuse a decoded scale matrix holding a byte that is not a non-negative
/// finite E4M3 scale. NVFP4 block scales are amax-derived, so non-negative
/// and finite: an E4M3 byte with the sign bit set or the 0x7F NaN code cannot
/// come from a sound encoder. Checking costs one pass over bytes already in
/// cache and catches a flipped high bit that every structural check would let
/// through. Shared by both encodings.
pub(crate) fn check_decoded_scales(out: &[u8], cols: usize, what: &str) -> Result<()> {
    // A max over chunks vectorizes; `position` alone does not, and was most
    // of the slab decode's time. It runs only to name the bad byte.
    if out
        .chunks(4096)
        .any(|c| c.iter().fold(0u8, |m, &b| m.max(b)) >= 0x7F)
        && let Some(at) = out.iter().position(|&b| b >= 0x7F)
    {
        bail!(
            "{what} decoded scale {:#04x} at row {} col {} is negative or NaN — corrupt stream",
            out[at],
            at / cols,
            at % cols
        );
    }
    Ok(())
}

/// One decoded `weight_scale`, ready to upload as FP8 E4M3.
pub struct DecodedScale {
    pub name: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

struct Part {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Pairs CSF halves as a loader streams tensors past it, in any order and
/// across shards, and decodes each pair once both halves have arrived.
///
/// The disk loaders (`SafetensorsLoader`, `FastSafetensorsLoader`) route CSF
/// parts here instead of uploading them, so the store only ever sees the
/// decoded `weight_scale`, and the U32 exception stream (a dtype the store
/// does not carry) never reaches it. The RDMA weight tier, which copies
/// tensors as they are, refuses CSF parts.
///
/// It is also the one place that knows which scales came from a pair, so the
/// loaders report every plain tensor too ([`Self::note_plain`]): a scale both
/// plain and CSF-compressed is refused whichever arrives first.
#[derive(Default)]
pub struct CsfAssembler {
    pending: HashMap<String, (Option<Part>, Option<Part>)>,
    /// Every scale decoded so far, with its `[rows, cols]`.
    decoded: HashMap<String, [usize; 2]>,
    /// Plain `.weight_scale` tensors the loader uploaded as they were.
    plain_scales: HashSet<String>,
    bytes_in: usize,
    bytes_out: usize,
    time: Duration,
}

impl CsfAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hand over one CSF half. Returns the decoded scale when this completes a
    /// pair, `None` while its partner is still to come.
    pub fn offer(
        &mut self,
        name: &str,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<Option<DecodedScale>> {
        let (base, part) =
            csf_part(name).with_context(|| format!("{name} is not a CSF stream tensor"))?;
        let elem: usize = match part {
            CsfPart::Fixed => 1,
            CsfPart::Exceptions => 4,
        };
        // The byte length against the header shape pins the dtype (U8 vs U32)
        // without trusting the dtype string: a stream with the other width
        // cannot satisfy both.
        let want = shape
            .iter()
            .try_fold(elem, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: shape {shape:?} overflows"))?;
        ensure!(
            want == bytes.len(),
            "{name}: {} bytes for shape {shape:?} — expected {want} ({})",
            bytes.len(),
            if elem == 1 { "U8" } else { "U32" },
        );
        ensure!(
            !self.plain_scales.contains(base),
            "{base} is in the checkpoint both plain and CSF-compressed"
        );
        let slot = self.pending.entry(base.to_string()).or_default();
        let held = match part {
            CsfPart::Fixed => &mut slot.0,
            CsfPart::Exceptions => &mut slot.1,
        };
        ensure!(held.is_none(), "{name} appears twice");
        *held = Some(Part {
            shape: shape.to_vec(),
            bytes: bytes.to_vec(),
        });
        if slot.0.is_none() || slot.1.is_none() {
            return Ok(None);
        }
        let Some((Some(fixed), Some(exc))) = self.pending.remove(base) else {
            unreachable!("both halves were just checked present")
        };
        let t = Instant::now();
        let (out, [rows, cols]) = decode_nvfp4_csf(&fixed.bytes, &fixed.shape, &exc.bytes)
            .with_context(|| format!("decoding CSF scale {base}"))?;
        self.time += t.elapsed();
        self.decoded.insert(base.to_string(), [rows, cols]);
        self.bytes_in += fixed.bytes.len() + exc.bytes.len();
        self.bytes_out += out.len();
        Ok(Some(DecodedScale {
            name: base.to_string(),
            shape: vec![rows, cols],
            bytes: out,
        }))
    }

    /// Report a tensor the loader uploaded as it was (not a CSF half).
    /// Refuses a scale the assembler already decoded from a pair, and
    /// remembers a plain `weight_scale` so a later pair for it is refused.
    pub fn note_plain(&mut self, name: &str) -> Result<()> {
        ensure!(
            !self.decoded.contains_key(name),
            "{name} is in the checkpoint both plain and CSF-compressed"
        );
        if name.ends_with(".weight_scale") {
            self.plain_scales.insert(name.to_string());
        }
        Ok(())
    }

    /// Refuse a load that left a half without its partner, or decoded a scale
    /// that is not the `[n, k/16]` its packed weight `[n, k/2]` needs
    /// (`weight_shape` looks a loaded or deferred tensor's shape up by name),
    /// and log what the decode cost. A lone half means a scale the model will
    /// look for and not find — or worse, find stale — so it is an error, not a
    /// warning.
    ///
    /// The shape check runs here, at the end of the load and before any store
    /// exists, rather than before each upload: a pair and its weight can sit
    /// in different shards, in either order.
    pub fn finish(self, weight_shape: impl Fn(&str) -> Option<Vec<usize>>) -> Result<()> {
        if !self.pending.is_empty() {
            let mut names: Vec<_> = self.pending.keys().take(4).cloned().collect();
            names.sort();
            bail!(
                "{} CSF scale stream(s) without a partner, e.g. {names:?} — the \
                 checkpoint is incomplete or a skip filter split a pair",
                self.pending.len()
            );
        }
        for (name, &got) in &self.decoded {
            let w = name.strip_suffix("_scale").and_then(&weight_shape);
            check_against_weight(name, got, w.as_deref())?;
        }
        if !self.decoded.is_empty() {
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            tracing::info!(
                "Decoded {} CSF NVFP4 scale matrices: {:.2} GiB compressed -> {:.2} GiB E4M3 \
                 in {:.2} s on the load thread",
                self.decoded.len(),
                gib(self.bytes_in),
                gib(self.bytes_out),
                self.time.as_secs_f64(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "nvfp4_csf_tests.rs"]
mod tests;
