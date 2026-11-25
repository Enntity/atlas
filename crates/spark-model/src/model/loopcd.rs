// SPDX-License-Identifier: AGPL-3.0-only

//! LoopCD's guided head (`atlas_core::config::LoopCdMode`, arXiv 2610.02185;
//! oMLX `Model._loopcd_logits`, jundot/omlx#5640, Apache-2.0).
//!
//! The Inkling 2 forward leaves the reference coda's rows beside its hidden
//! rows (`layers::inkling2::loopcd`). Every head path here starts with
//! `final_norm_rows` on some of the forward's hidden rows; that call norms the
//! same reference rows into [`LoopCdHead::normed`] and remembers where it put
//! the model's rows. The head that then reads those normed rows (any of
//! `lm_head`, `lm_head_batched`, `lm_head_project_batched`) projects the
//! reference rows with the same kernels, softcaps them and guides its own
//! logits in place (`inkling2_loopcd_guide`). A head over rows no final norm
//! announced stays unguided, and says so once.
//!
//! Off (`TransformerModel::loopcd` = `None`) nothing here runs.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LOOPCD_FIRST_CONTROL_TOKEN, LOOPCD_MAX_STRENGTH, LoopCdMode};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::types::TransformerModel;
use crate::layers::ops;

/// Rows one guided head may carry: eight sequences of a 16-row verify.
pub(crate) const LOOPCD_HEAD_ROWS: usize = 128;

/// Where the Inkling 2 forward leaves the reference rows.
pub(crate) struct LoopCdRows {
    /// The hidden buffer of the forward that last wrote `rows`.
    pub forward_hidden: Arc<AtomicU64>,
    /// `[max_rows, H]` BF16, row-indexed like that forward's hidden rows.
    pub rows: DevicePtr,
    pub max_rows: usize,
}

pub(crate) struct LoopCdHead {
    src: LoopCdRows,
    keep_control: bool,
    /// `[LOOPCD_HEAD_ROWS, H]` BF16: the last final norm's reference rows.
    normed: DevicePtr,
    /// `[LOOPCD_HEAD_ROWS, V]` BF16 reference logits.
    logits: DevicePtr,
    guide: KernelHandle,
    /// `skip-thinking`: `[src.max_rows]` u8 thinking flags of the forward's
    /// rows (`loopcd_flags.rs`); NULL for `on`.
    thinking: DevicePtr,
    /// An entry stamped `thinking` for the forward the next final norm ends.
    pub(super) stamped: AtomicBool,
    /// The last final norm's announcement, when it normed reference rows.
    announced: Mutex<Option<Announced>>,
    /// A head is projecting the reference rows (its own guide is a no-op).
    projecting: AtomicBool,
    /// Said once each: a head no final norm announced, a forward no entry stamped.
    warned: [AtomicBool; 2],
}

/// What the last final norm left for the heads after it.
#[derive(Clone, Copy)]
struct Announced {
    /// Its output (the normed rows the heads read) and their rows.
    out: u64,
    rows: usize,
    /// The forward row its first row was.
    forward_row: usize,
    /// Whether the forward's rows had their thinking flags stamped.
    stamped: bool,
}

impl LoopCdHead {
    /// `skip-thinking`'s flag buffer and its rows.
    pub(super) fn flags(&self) -> Option<(DevicePtr, usize)> {
        (!self.thinking.is_null()).then_some((self.thinking, self.src.max_rows))
    }
}

/// Row offset of `ptr` inside `[base, base + cap rows)` holding `rows` rows of
/// `row_bytes`, or `None`.
pub(crate) fn row_in(
    ptr: u64,
    base: u64,
    cap: usize,
    rows: usize,
    row_bytes: usize,
) -> Option<usize> {
    let off = usize::try_from(ptr.checked_sub(base)?).ok()?;
    let first = off.is_multiple_of(row_bytes).then_some(off / row_bytes)?;
    (first + rows <= cap).then_some(first)
}

impl TransformerModel {
    /// Turn LoopCD on for an Inkling 2 model whose config carries the
    /// reference visits (`enable_inkling2_loopcd`).
    pub fn attach_loopcd(&mut self) -> Result<()> {
        let mode = self
            .config
            .inkling2
            .as_ref()
            .map_or(LoopCdMode::Off, |i| i.loopcd);
        if mode == LoopCdMode::Off {
            return Ok(());
        }
        ensure!(
            !self.use_fp32_logits && self.lm_head_fp8.is_none() && !self.final_norm_identity,
            "--loopcd guides the BF16 logits of the plain Inkling 2 head"
        );
        let src = self
            .layers
            .last_mut()
            .and_then(|l| l.as_any_mut())
            .and_then(|a| a.downcast_ref::<crate::layers::inkling2::layer::Inkling2Layer>())
            .and_then(|l| l.loopcd_reference())
            .context("--loopcd: the last visit is not LoopCD's reference visit")?;
        let guide = self
            .gpu
            .kernel("inkling2_loopcd", "inkling2_loopcd_guide")
            .context("--loopcd needs the inkling2_loopcd kernel (CUDA targets)")?;
        let (h, v) = (self.config.hidden_size, self.config.vocab_size);
        let normed = self.gpu.alloc(LOOPCD_HEAD_ROWS * h * 2)?;
        let logits = self.gpu.alloc(LOOPCD_HEAD_ROWS * v * 2)?;
        let keep_control = mode == LoopCdMode::SkipThinking;
        let thinking = if keep_control {
            let t = self.gpu.alloc(src.max_rows)?;
            self.gpu.copy_h2d(&vec![0u8; src.max_rows], t)?;
            t
        } else {
            DevicePtr::NULL
        };
        tracing::info!(
            "LoopCD {mode:?}: strength {LOOPCD_MAX_STRENGTH}, reference rows for {} of the \
             forward's rows, {LOOPCD_HEAD_ROWS} a head",
            src.max_rows
        );
        self.loopcd = Some(Box::new(LoopCdHead {
            src,
            keep_control,
            normed,
            logits,
            guide,
            thinking,
            stamped: AtomicBool::new(false),
            announced: Mutex::new(None),
            projecting: AtomicBool::new(false),
            warned: [AtomicBool::new(false), AtomicBool::new(false)],
        }));
        Ok(())
    }

    /// `final_norm_rows` just normed `rows` rows at `input` into `output`:
    /// norm the reference rows beside them too, or announce none.
    pub(super) fn loopcd_norm(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(lc) = self.loopcd.as_deref() else {
            return Ok(());
        };
        let (rows, row_bytes) = (rows as usize, self.config.hidden_size * 2);
        let base = lc.src.forward_hidden.load(Ordering::Relaxed);
        let first = row_in(input.0, base, lc.src.max_rows, rows, row_bytes)
            .filter(|_| rows <= LOOPCD_HEAD_ROWS);
        let mut announced = lc.announced.lock().expect("loopcd announcement");
        *announced = None;
        let stamped = lc.stamped.swap(false, Ordering::Relaxed);
        if let Some(first) = first {
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                lc.src.rows.offset(first * row_bytes),
                &self.final_norm,
                lc.normed,
                rows as u32,
                self.config.hidden_size as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
            *announced = Some(Announced {
                out: output.0,
                rows,
                forward_row: first,
                stamped,
            });
        }
        Ok(())
    }

    /// Guide `rows` rows of softcapped `logits` that a head projected from
    /// `normed`; `head(normed, logits)` is that same head into other buffers.
    pub(super) fn loopcd_guide(
        &self,
        normed: DevicePtr,
        logits: DevicePtr,
        rows: usize,
        head: impl FnOnce(DevicePtr, DevicePtr) -> Result<()>,
        stream: u64,
    ) -> Result<()> {
        let Some(lc) = self.loopcd.as_deref() else {
            return Ok(());
        };
        if lc.projecting.load(Ordering::Relaxed) {
            return Ok(());
        }
        let row_bytes = self.config.hidden_size * 2;
        let announced = *lc.announced.lock().expect("loopcd announcement");
        let found =
            announced.and_then(|a| Some((row_in(normed.0, a.out, a.rows, rows, row_bytes)?, a)));
        let Some((first, a)) = found else {
            if !lc.warned[0].swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "LoopCD: a head over {rows} rows no final norm announced runs unguided \
                     (reported once)"
                );
            }
            return Ok(());
        };
        lc.projecting.store(true, Ordering::Relaxed);
        let projected = head(lc.normed.offset(first * row_bytes), lc.logits);
        lc.projecting.store(false, Ordering::Relaxed);
        projected?;
        let thinking = match lc.flags() {
            Some((flags, _)) if a.stamped => flags.offset(a.forward_row + first),
            Some(_) => {
                if !lc.warned[1].swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "LoopCD skip-thinking: a forward no entry stamped is guided throughout \
                         (reported once)"
                    );
                }
                DevicePtr::NULL
            }
            None => DevicePtr::NULL,
        };
        let (vocab, scored) = (self.config.vocab_size as u32, self.loopcd_scored());
        let ordinary = if lc.keep_control {
            LOOPCD_FIRST_CONTROL_TOKEN.min(scored)
        } else {
            scored
        };
        KernelLaunch::new(self.gpu.as_ref(), lc.guide)
            .grid([rows as u32, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(lc.logits)
            .arg_ptr(thinking)
            .arg_u32(vocab)
            .arg_u32(ordinary)
            .arg_u32(scored)
            .arg_f32(LOOPCD_MAX_STRENGTH)
            .arg_u32(u32::from(lc.keep_control))
            .launch(stream)
    }

    /// The checkpoint's real vocabulary (the head pads past it).
    fn loopcd_scored(&self) -> u32 {
        let v = self.config.vocab_size;
        let real = self
            .config
            .inkling2
            .as_ref()
            .map_or(v, |i| i.unpadded_vocab_size);
        real.min(v) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::row_in;

    #[test]
    fn rows_are_found_only_whole_and_inside() {
        let (base, rb) = (0x1000u64, 8usize);
        assert_eq!(row_in(base, base, 4, 4, rb), Some(0));
        assert_eq!(row_in(base + 24, base, 4, 1, rb), Some(3));
        assert_eq!(row_in(base + 24, base, 4, 2, rb), None, "past the end");
        assert_eq!(row_in(base + 4, base, 4, 1, rb), None, "not a row start");
        assert_eq!(row_in(base - 8, base, 4, 1, rb), None, "before the base");
    }
}
