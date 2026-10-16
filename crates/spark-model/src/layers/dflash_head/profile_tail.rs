// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter tail timing for the native EXL3 DFlash observer profile
//! (`ATLAS_GLM_EXL3_DFLASH=1`): the lm_head projection and the DFlash2
//! candidate selector, each timed in isolation on lane 0's scratch.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// Diagnostic (EXL3 DFlash observer profile): mean wall milliseconds of
    /// the lm_head projection and of the DFlash2 selector tail (hidden
    /// projection + candidate selector) over the γ rows of lane 0, each timed
    /// over `iters` synchronized repetitions. Runs on the lane's existing
    /// scratch after a proposal has been read back (the selector re-walks
    /// from the anchor that proposal seeded); never call it inside a capture.
    /// The selector time is 0.0 on a head without a DFlash2 selector.
    pub fn profile_tail_ms(&self, gpu: &dyn GpuBackend, iters: usize) -> Result<(f64, f64)> {
        let stream = gpu.default_stream();
        let (_, scratch, _, _) = self.lane(0, stream);
        let iters = iters.max(1);
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            gpu.synchronize(stream)?;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                f()?;
            }
            gpu.synchronize(stream)?;
            Ok(start.elapsed().as_secs_f64() * 1e3 / iters as f64)
        };
        let gamma = self.gamma as u32;
        let head =
            time(&|| self.project_head(gpu, scratch.norm_buf, scratch.logits, gamma, stream))?;
        let Some(selector) = self.candidate_selector.as_ref() else {
            return Ok((head, 0.0));
        };
        // Host copies of the anchor words the last propose seeded, for the
        // selector's host fallback; the GPU kernel reads the device slots.
        let mut words = [0u8; super::MARKOV_PREV_BYTES];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(scratch.markov_prev_dev, &mut words)?;
        let word = |i: usize| u32::from_le_bytes(words[i..i + 4].try_into().unwrap());
        let (last_token, ban_depth) = (word(0), word(super::MARKOV_BAN_DEPTH_OFFSET));
        let selector_ms = time(&|| {
            selector.select_candidates(
                Self::selector_anchor(last_token, ban_depth, scratch),
                scratch.norm_buf,
                scratch.logits,
                scratch.dflash2_projected_hidden,
                scratch.dflash2_selector_scratch,
                scratch.draft_tokens_dev,
                self.gamma,
                gpu,
                &|src, w, dst, m, n, k| self.drafter_dense_gemm(gpu, src, w, dst, m, n, k, stream),
                self.kernels.dflash2_candidate_selector,
                stream,
            )
        })?;
        Ok((head, selector_ms))
    }
}
