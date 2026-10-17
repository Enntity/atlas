// SPDX-License-Identifier: AGPL-3.0-only
//! Match the installed merged gate/up matrix geometry without changing bytes.
use super::*;

pub(in crate::layers::glm5_exl3_ffn) fn merge_local_weights(
    gpu: &dyn GpuBackend,
    mut local: [DevicePtr; 3],
    hidden: usize,
    intermediate: usize,
    stream: u64,
) -> Result<[DevicePtr; 2]> {
    let mut merged = DevicePtr::NULL;
    let build = (|| -> Result<()> {
        ensure!(
            hidden > 0 && intermediate > 0 && local.iter().all(|p| !p.is_null()),
            "shared merge requires live positive matrices"
        );
        let bytes = hidden
            .checked_mul(intermediate)
            .and_then(|v| v.checked_mul(2))
            .context("shared local matrix bytes overflow")?;
        merged = gpu.alloc(
            bytes
                .checked_mul(2)
                .context("shared merged weight bytes overflow")?,
        )?;
        gpu.copy_d2d_async(local[0], merged, bytes, stream)?;
        gpu.copy_d2d_async(local[1], merged.offset(bytes), bytes, stream)?;
        Ok(())
    })();
    let mut cleanup = None;
    for p in &mut local[..2] {
        if !p.is_null() {
            match gpu.free(*p) {
                Ok(()) => *p = DevicePtr::NULL,
                Err(e) => retain_error(&mut cleanup, Err(e)),
            }
        }
    }
    match (build, cleanup) {
        (Ok(()), None) => Ok([merged, local[2]]),
        (result, cleanup) => {
            let mut error = result
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("shared merge cleanup failed"));
            if let Some(e) = cleanup {
                error = error.context(format!("shared gate/up cleanup: {e:#}"));
            }
            let mut rollback = None;
            for p in local.into_iter().chain([merged]) {
                if !p.is_null() {
                    retain_error(&mut rollback, gpu.free(p));
                }
            }
            Err(match rollback {
                None => error,
                Some(e) => error.context(format!("shared merge rollback: {e:#}")),
            })
        }
    }
}

/// Split each merged BF16 row `[gate,up]`; a base offset alone is wrong for rows >1.
pub(in crate::layers::glm5_exl3_ffn) fn split_merged_bf16(
    gpu: &dyn GpuBackend,
    merged: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    rows: usize,
    intermediate: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > 0
            && intermediate > 0
            && [merged, gate, up].iter().all(|p| !p.is_null())
            && merged != gate
            && merged != up
            && gate != up,
        "shared activation split requires distinct positive live spans"
    );
    let width = intermediate
        .checked_mul(2)
        .context("shared split width overflow")?;
    let pitch = width
        .checked_mul(2)
        .context("shared split pitch overflow")?;
    rows.checked_mul(pitch)
        .context("shared split span overflow")?;
    gpu.copy_d2d_2d_async(merged, pitch, gate, width, width, rows, stream)?;
    gpu.copy_d2d_2d_async(merged.offset(width), pitch, up, width, width, rows, stream)
}
