// SPDX-License-Identifier: AGPL-3.0-only
//! Global reconstruction precedes TP selection; BF16 rounding must not move
//! across the partition boundary.
use super::*;
use spark_runtime::weights::exl3::tp::TpShardPlan;

mod merge;
mod reconstruct;
pub(super) use merge::{merge_local_weights, split_merged_bf16};

pub(super) struct SharedBf16Weights {
    pub(super) pointers: [DevicePtr; 2],
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) rank: usize,
    pub(super) world: usize,
}
impl SharedBf16Weights {
    pub(super) fn load(
        checkpoint: &Exl3CheckpointRouter,
        prefix: &str,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let mid = config.shared_expert_intermediate_size;
        validate_dimensions(h, mid, config.tp_rank, config.tp_world_size)?;
        // Resolve the full closure before uploading any global packed tensor.
        reconstruct::kernels(gpu)?;
        let mut shards = Vec::new();
        let load = (|| -> Result<()> {
            for (name, k, n) in [
                ("gate_proj", h, mid),
                ("up_proj", h, mid),
                ("down_proj", mid, h),
            ] {
                let view = checkpoint.linear_view(&format!("{prefix}.{name}"), k, n)?;
                let plan = TpShardPlan::new(&view, ShardAxis::Replicated, 0, 1)?;
                shards.push(Exl3DeviceShard::upload(gpu, &view, &plan)?);
            }
            Ok(())
        })();
        if let Err(error) = load {
            let mut cleanup = None;
            for shard in &mut shards {
                retain_error(&mut cleanup, shard.release(gpu));
            }
            return Err(match cleanup {
                None => error,
                Some(e) => error.context(format!("shared upload rollback: {e:#}")),
            });
        }
        Self::from_global(
            gpu,
            shards
                .try_into()
                .map_err(|_| anyhow::anyhow!("shared projection count"))?,
            config.tp_rank,
            config.tp_world_size,
            stream,
        )
    }
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.pointers.iter().all(|p| !p.is_null()),
            "shared BF16 weights released"
        );
        Ok(())
    }
    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut error = None;
        for p in &mut self.pointers {
            if !p.is_null() {
                match gpu.free(*p) {
                    Ok(()) => *p = DevicePtr::NULL,
                    Err(e) => retain_error(&mut error, Err(e)),
                }
            }
        }
        finish(error)
    }
}
fn validate_dimensions(h: usize, mid: usize, rank: usize, world: usize) -> Result<()> {
    ensure!(
        world > 0
            && rank < world
            && mid > 0
            && h > 0
            && mid.is_multiple_of(world)
            && h.is_multiple_of(128)
            && mid.is_multiple_of(128)
            && (mid / world).is_multiple_of(128)
            && h <= 65535
            && mid <= 65535,
        "shared global/TP dimensions must fit H128 reconstruction and CUDA grid"
    );
    Ok(())
}

/// Select from reconstructed row-major BF16 `[n,k]` without altering bytes.
pub(super) fn copy_global_bf16_shard(
    gpu: &dyn GpuBackend,
    source: DevicePtr,
    k: usize,
    n: usize,
    axis: ShardAxis,
    rank: usize,
    world: usize,
    stream: u64,
) -> Result<DevicePtr> {
    ensure!(
        !source.is_null() && k > 0 && n > 0 && world > 0 && rank < world,
        "shared BF16 shard geometry invalid"
    );
    let (local_k, local_n) = match axis {
        ShardAxis::Output => {
            ensure!(n.is_multiple_of(world), "shared output does not divide TP");
            (k, n / world)
        }
        ShardAxis::Input => {
            ensure!(k.is_multiple_of(world), "shared input does not divide TP");
            (k / world, n)
        }
        ShardAxis::Replicated => {
            anyhow::bail!("shared FFN requires explicit output/input partition")
        }
    };
    let bytes = local_k
        .checked_mul(local_n)
        .and_then(|v| v.checked_mul(2))
        .context("shared shard bytes overflow")?;
    let full_bytes = k
        .checked_mul(n)
        .and_then(|v| v.checked_mul(2))
        .context("shared full bytes overflow")?;
    let output = gpu.alloc(bytes)?;
    let copy = if axis == ShardAxis::Output {
        gpu.copy_d2d_async(source.offset(rank * bytes), output, bytes, stream)
    } else {
        let width = local_k * 2;
        ensure!(
            rank * width < full_bytes,
            "shared input partition offset invalid"
        );
        gpu.copy_d2d_2d_async(
            source.offset(rank * width),
            k * 2,
            output,
            width,
            width,
            n,
            stream,
        )
    };
    match copy {
        Ok(()) => Ok(output),
        Err(error) => {
            gpu.free(output).context("shared shard copy rollback")?;
            Err(error)
        }
    }
}
