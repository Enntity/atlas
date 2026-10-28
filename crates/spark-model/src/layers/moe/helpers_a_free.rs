// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Free one original weight and remember it for the store (see
/// `transpose_for_prefill_unified`).
pub(super) fn free_original(
    gpu: &dyn GpuBackend,
    freed: &mut Vec<DevicePtr>,
    ptr: DevicePtr,
) -> Result<()> {
    gpu.free(ptr)?;
    freed.push(ptr);
    Ok(())
}
