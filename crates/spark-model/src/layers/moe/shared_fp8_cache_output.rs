// SPDX-License-Identifier: AGPL-3.0-only
//! Existing M64 FP8 dispatch, with an eager resident-weight K5 diagnostic.
use super::*;
use std::sync::atomic::Ordering;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_fp8_cache(
        &self,
        projection: usize,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = &self.shared_fp8_cache;
        if state.layer.is_none() {
            return ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            );
        }
        let launch = || {
            launch_cached_m64(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            )
        };
        anyhow::ensure!(
            (1..=1024).contains(&rows),
            "shared FP8 cache row capacity exceeded"
        );
        anyhow::ensure!(
            !state.verify || (!ctx.graph_capture && !ctx.gpu.stream_is_capturing(stream)),
            "shared FP8 output VERIFY requires eager stream, including previously checked projections"
        );
        let bit = 1u8 << projection;
        if !state.verify || rows != 5 || state.checked.load(Ordering::Relaxed) & bit != 0 {
            return launch();
        }
        let (pn, pk, owner, capacity, original) = match projection {
            0 => (
                2048,
                4096,
                ctx.buffers.ssm_deinterleaved(),
                ctx.buffers.sizes().ssm_deinterleaved,
                self.shared_gate_t,
            ),
            1 => (
                2048,
                4096,
                ctx.buffers.ssm_qkvz(),
                ctx.buffers.sizes().ssm_qkvz,
                self.shared_up_t,
            ),
            2 => (
                4096,
                2048,
                ctx.buffers.attn_output(),
                ctx.buffers.sizes().attn_output,
                self.shared_down_t,
            ),
            _ => anyhow::bail!("shared FP8 projection ordinal"),
        };
        let bytes = rows as usize * n as usize * 2;
        anyhow::ensure!(
            (n, k) == (pn, pk) && output == owner && capacity >= bytes,
            "shared FP8 output geometry/owner/capacity"
        );
        let original =
            original.ok_or_else(|| anyhow::anyhow!("shared FP8 output oracle missing T weight"))?;
        let out = super::m5_projections::span(output, bytes, 2)?;
        for (ptr, len) in [
            (input, rows as usize * k as usize * 2),
            (weight, n as usize * k as usize),
            (original.weight, n as usize * k as usize / 2),
            (original.weight_scale, n as usize * k as usize / 16),
        ] {
            super::m5_projections::disjoint(&out, &super::m5_projections::span(ptr, len, 16)?)?;
        }
        verify_output(
            ctx.gpu,
            output,
            bytes,
            stream,
            || {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t,
                    input,
                    &original,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            },
            launch,
        )?;
        state.checked.fetch_or(bit, Ordering::Relaxed);
        tracing::info!(
            layer = state.layer,
            projection,
            rows,
            bytes,
            "GLM shared FP8 resident output oracle passed"
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_cached_m64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0
            && (1..=1024).contains(&rows)
            && matches!((n, k), (2048, 4096) | (4096, 2048)),
        "shared FP8 cache M64 handle/geometry contract"
    );
    // Deliberately bypass generic fp8_gemm_n128's default-on LDMAB rewrite:
    // only this exact existing M64 kernel was validated for the cache.
    spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), rows.div_ceil(64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

fn verify_output(
    gpu: &dyn GpuBackend,
    output: DevicePtr,
    bytes: usize,
    stream: u64,
    reference_launch: impl FnOnce() -> Result<()>,
    candidate_launch: impl FnOnce() -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(
        !gpu.stream_is_capturing(stream) && bytes > 0 && bytes <= 40960 && bytes.is_multiple_of(2),
        "shared FP8 output oracle eager/bounded contract"
    );
    super::m5_projections::span(output, bytes, 2)?;
    gpu.memset_async(output, 0xff, bytes, stream)?;
    reference_launch()?;
    let mut reference = vec![0; bytes];
    gpu.copy_d2h_on_stream(output, &mut reference, stream)?;
    gpu.memset_async(output, 0xff, bytes, stream)?;
    candidate_launch()?;
    let mut actual = vec![0; bytes];
    gpu.copy_d2h_on_stream(output, &mut actual, stream)?;
    compare_outputs(&reference, &actual)
}

fn compare_outputs(reference: &[u8], actual: &[u8]) -> Result<()> {
    anyhow::ensure!(
        !reference.is_empty()
            && reference.len().is_multiple_of(2)
            && reference.len() == actual.len(),
        "shared FP8 oracle output length"
    );
    for (i, (r, a)) in reference
        .chunks_exact(2)
        .zip(actual.chunks_exact(2))
        .enumerate()
    {
        let r = u16::from_le_bytes([r[0], r[1]]);
        let a = u16::from_le_bytes([a[0], a[1]]);
        anyhow::ensure!(
            r & 0x7f80 != 0x7f80 && a & 0x7f80 != 0x7f80 && r == a,
            "shared FP8 resident output mismatch/nonfinite at {i}: {r:04x} vs {a:04x}"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "shared_fp8_cache_output_tests.rs"]
mod tests;
