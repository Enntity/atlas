// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::weights::exl3::ffn_io::DiagnosticTensor;
pub(super) use spark_runtime::weights::exl3::ffn_io::{DiagnosticDtype, Exl3FfnRouter};

type TensorSpec<'a> = (&'static str, DiagnosticDtype, DevicePtr, &'a [usize]);
pub(super) fn capture(
    router: &Option<Exl3FfnRouter>,
    invocation: Option<usize>,
    gpu: &dyn GpuBackend,
    stage: &str,
    tensors: &[TensorSpec<'_>],
    stream: u64,
) -> Result<()> {
    let (Some(router), Some(invocation)) = (router, invocation) else {
        return Ok(());
    };
    let tensors: Vec<_> = tensors
        .iter()
        .map(|(name, dtype, ptr, shape)| DiagnosticTensor {
            name,
            dtype: *dtype,
            ptr: *ptr,
            shape: shape.to_vec(),
        })
        .collect();
    router.capture(gpu, invocation, stage, &tensors, stream)
}
