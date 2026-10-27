// SPDX-License-Identifier: AGPL-3.0-only

//! How the drafter normalises the incoming stream highway with
//! `mtp.pre_fc_norm_hidden` (`[hc_mult * hidden]`).
//!
//! The default is per stream: each stream over its own `hidden` slice with
//! that slice of the weight (`stream_norms`), the convention of the mHC
//! kernels' `hc_norm`. vLLM's `Qwen4ExpMultiTokenPredictor` instead applies
//! ONE `GemmaRMSNorm(hidden * hc_count)` over each token's flattened
//! `[hc_count * hidden]` row (`pre_fc_norm_hidden(hidden_states.flatten(-2))`),
//! and TensorFold's twin does the same: one RMS for all streams, so their
//! relative scales survive into `fc_hidden`. The two agree only when every
//! stream has the same RMS.
//!
//! * `ATLAS_QWEN4EXP_MTP_NORM_DIAG=1`: log, every 64th propose (24 lines at
//!   most), each stream's RMS and the flattened row's for the target's
//!   streams (draft 0) and the drafter's own (draft 1): how far apart the
//!   two norms are.
//! * `ATLAS_QWEN4EXP_MTP_FLAT_NORM=1` (default off): the flattened norm.
//!
//! Drafts only propose, so neither can change an emitted token.

use super::*;

/// `ATLAS_QWEN4EXP_MTP_FLAT_NORM=1`, read once.
pub(crate) fn flat_norm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_FLAT_NORM").as_deref() == Ok("1"))
}

fn diag_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_NORM_DIAG").as_deref() == Ok("1"))
}

/// Each stream's RMS and the flattened row's, of one `[hc, h]` FP32 row.
pub(super) fn stream_rms(row: &[f32], hc: usize) -> (Vec<f32>, f32) {
    let h = row.len() / hc.max(1);
    let ms = |v: &[f32]| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
    let per = row.chunks(h).map(|s| ms(s).sqrt() as f32).collect();
    (per, ms(row).sqrt() as f32)
}

impl Qwen4ExpMtpHead {
    /// `ATLAS_QWEN4EXP_MTP_NORM_DIAG`: the highway's row 0, about to be
    /// normalised for draft `draft` (0 = the target's streams).
    pub(super) fn norm_diag(&self, ctx: &ForwardContext, draft: usize, stream: u64) -> Result<()> {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        use std::sync::atomic::Ordering::Relaxed;
        if !diag_on() || CALLS.fetch_add(1, Relaxed) % 128 > 1 || LINES.load(Relaxed) >= 24 {
            return Ok(());
        }
        LINES.fetch_add(1, Relaxed);
        let (h, hc) = (ctx.config.hidden_size, ctx.config.hc_mult.max(1));
        let mut buf = vec![0u8; hc * h * 4];
        ctx.gpu.synchronize(stream)?;
        ctx.gpu.copy_d2h(ctx.buffers.hc_streams(), &mut buf)?;
        let row: Vec<f32> = buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();
        let (per, flat) = stream_rms(&row, hc);
        let (lo, hi) = per
            .iter()
            .fold((f32::MAX, 0f32), |(l, u), &r| (l.min(r), u.max(r)));
        tracing::info!(
            "MTP_NORM_DIAG draft={draft} stream_rms={per:.3?} flat_rms={flat:.3} max/min={:.3}",
            hi / lo.max(f32::MIN_POSITIVE)
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::stream_rms;

    #[test]
    fn flat_rms_pools_the_streams() {
        // Two streams of RMS 1 and 3: the flattened RMS is sqrt((1 + 9) / 2).
        let row = [1.0, -1.0, 3.0, 3.0];
        let (per, flat) = stream_rms(&row, 2);
        assert_eq!(per, vec![1.0, 3.0]);
        assert!((flat - 5f32.sqrt()).abs() < 1e-6);
    }
}
