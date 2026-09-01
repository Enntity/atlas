// SPDX-License-Identifier: AGPL-3.0-only

//! Launch contract for the appliance-fixed merged KDA input projection.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM53_KDA_PROJECTION_MODULE: &str = "glm53_kda_projection";
pub const GLM53_KDA_SPLIT_MERGED_ENTRY: &str = "glm53_kda_split_merged_bf16";

pub struct Glm53KdaSplitMergedArgs {
    pub merged: DevicePtr,
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub beta: DevicePtr,
    pub forget_a: DevicePtr,
    pub gate_a: DevicePtr,
    pub rows: u32,
}

impl Glm53KdaSplitMergedArgs {
    fn validate(&self) -> Result<()> {
        ensure!(self.rows > 0, "GLM KDA merged split requires rows");
        for (name, pointer) in [
            ("merged", self.merged),
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("beta", self.beta),
            ("forget-a", self.forget_a),
            ("gate-a", self.gate_a),
        ] {
            ensure!(!pointer.is_null(), "GLM KDA merged split {name} is null");
        }
        Ok(())
    }
}

pub fn glm53_kda_split_merged_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Glm53KdaSplitMergedArgs,
    stream: u64,
) -> Result<()> {
    args.validate()?;
    KernelLaunch::new(gpu, kernel)
        .grid([args.rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(args.merged)
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.beta)
        .arg_ptr(args.forget_a)
        .arg_ptr(args.gate_a)
        .arg_u32(args.rows)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/glm53_kda_projection.cu");
    const REGISTRY: &str =
        include_str!("../../../../../kernels/gb10/glm-5.3-flash/exl3/KERNEL.toml");

    #[test]
    fn merged_projection_symbol_and_fixed_tp2_geometry_are_pinned() {
        assert!(SOURCE.contains(&format!("void {GLM53_KDA_SPLIT_MERGED_ENTRY}(")));
        assert!(REGISTRY.contains("glm53_kda_projection = \"glm53_kda_projection\""));
        assert!(SOURCE.contains("KDA_LOCAL_WIDTH = 4096"));
        assert!(SOURCE.contains("KDA_LOCAL_HEADS = 32"));
        assert!(SOURCE.contains("KDA_LOW_RANK = 128"));
    }

    #[test]
    fn zero_rows_fail_closed() {
        let args = Glm53KdaSplitMergedArgs {
            merged: DevicePtr(1),
            query: DevicePtr(2),
            key: DevicePtr(3),
            value: DevicePtr(4),
            beta: DevicePtr(5),
            forget_a: DevicePtr(6),
            gate_a: DevicePtr(7),
            rows: 0,
        };
        assert!(args.validate().unwrap_err().to_string().contains("rows"));
    }
}
