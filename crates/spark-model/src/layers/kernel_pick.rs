// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::{GpuBackend, KernelHandle};

#[track_caller]
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(kernel) => kernel,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}
