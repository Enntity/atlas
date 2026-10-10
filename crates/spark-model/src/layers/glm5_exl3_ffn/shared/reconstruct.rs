// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

pub(super) fn kernels(gpu: &dyn GpuBackend) -> Result<[KernelHandle; 3]> {
    gpu.require_bf16_gemm_numerics(spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic)?;
    gpu.kernel("exl3", "exl3_silu_mul_dense")?;
    Ok([
        gpu.kernel("exl3_shared", "exl3_shared_decode")?,
        gpu.kernel("exl3_shared", "exl3_shared_left_hadamard")?,
        gpu.kernel("exl3_shared", "exl3_shared_right_hadamard")?,
    ])
}
impl SharedBf16Weights {
    pub(in crate::layers::glm5_exl3_ffn) fn from_global(
        gpu: &dyn GpuBackend,
        mut shards: [Exl3DeviceShard; 3],
        rank: usize,
        world: usize,
        stream: u64,
    ) -> Result<Self> {
        let mut pointers = [DevicePtr::NULL; 3];
        let mut scratch = [DevicePtr::NULL; 3];
        let build = (|| -> Result<(usize, usize)> {
            let h = shards[0].plan().in_features();
            let mid = shards[0].plan().out_features();
            validate_dimensions(h, mid, rank, world)?;
            for (i, s) in shards.iter().enumerate() {
                let p = s.plan();
                let (k, n) = if i == 2 { (mid, h) } else { (h, mid) };
                ensure!(
                    p.axis() == ShardAxis::Replicated
                        && p.rank() == 0
                        && p.world() == 1
                        && p.in_features() == k
                        && p.out_features() == n,
                    "shared reconstruction requires complete global replicated projections"
                );
                s.buffers()?;
            }
            let kernels = kernels(gpu)?;
            let cells = h.checked_mul(mid).context("shared global cells overflow")?;
            for (p, width) in scratch.iter_mut().zip([4, 4, 2]) {
                *p = gpu.alloc(
                    cells
                        .checked_mul(width)
                        .context("shared reconstruction bytes overflow")?,
                )?;
            }
            for (i, shard) in shards.iter().enumerate() {
                let p = shard.plan();
                let [trellis, suh, svh] = shard.buffers()?;
                let (k, n) = (p.in_features() as u32, p.out_features() as u32);
                let launch = |kernel| {
                    KernelLaunch::new(gpu, kernel)
                        .grid([n.div_ceil(128), k, 1])
                        .block([128, 1, 1])
                };
                launch(kernels[0])
                    .arg_ptr(trellis)
                    .arg_ptr(scratch[0])
                    .arg_u32(k)
                    .arg_u32(n)
                    .arg_u32(p.bits() as u32)
                    .launch(stream)?;
                launch(kernels[1])
                    .arg_ptr(scratch[0])
                    .arg_ptr(suh)
                    .arg_ptr(scratch[1])
                    .arg_u32(k)
                    .arg_u32(n)
                    .launch(stream)?;
                launch(kernels[2])
                    .arg_ptr(scratch[1])
                    .arg_ptr(svh)
                    .arg_ptr(scratch[2])
                    .arg_u32(k)
                    .arg_u32(n)
                    .launch(stream)?;
                pointers[i] = copy_global_bf16_shard(
                    gpu,
                    scratch[2],
                    k as usize,
                    n as usize,
                    if i == 2 {
                        ShardAxis::Input
                    } else {
                        ShardAxis::Output
                    },
                    rank,
                    world,
                    stream,
                )?;
            }
            Ok((h, mid / world))
        })();
        let mut cleanup = None;
        for s in &mut shards {
            retain_error(&mut cleanup, s.release(gpu));
        }
        for p in scratch {
            if !p.is_null() {
                retain_error(&mut cleanup, gpu.free(p));
            }
        }
        match (build, cleanup) {
            (Ok((hidden, intermediate)), None) => Ok(Self {
                pointers: merge_local_weights(gpu, pointers, hidden, intermediate, stream)?,
                hidden,
                intermediate,
                rank,
                world,
            }),
            (result, cleanup) => {
                let mut error = result
                    .err()
                    .unwrap_or_else(|| anyhow::anyhow!("shared reconstruction teardown failed"));
                if let Some(e) = cleanup {
                    error = error.context(format!("shared scratch cleanup: {e:#}"));
                }
                let mut local_error = None;
                for p in pointers {
                    if !p.is_null() {
                        retain_error(&mut local_error, gpu.free(p));
                    }
                }
                Err(match local_error {
                    None => error,
                    Some(e) => error.context(format!("shared local rollback: {e:#}")),
                })
            }
        }
    }
}
