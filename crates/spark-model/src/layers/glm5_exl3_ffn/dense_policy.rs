// SPDX-License-Identifier: AGPL-3.0-only

//! Dense/shared arithmetic roles, distinct from the routed expert activation.

use super::*;
use spark_runtime::weights::exl3::native::Exl3InputPrecision;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDenseFfnRole {
    /// Reference packed dense projections consume FP16 input transforms.
    DensePacked,
    /// The reference excludes shared experts from its packed dense policy.
    SharedExperts,
}
impl NativeDenseFfnRole {
    pub fn packed_input_precision(self) -> Result<Exl3InputPrecision> {
        match self {
            Self::DensePacked => Ok(Exl3InputPrecision::F16),
            Self::SharedExperts => anyhow::bail!("shared experts require reconstructed BF16 GEMM"),
        }
    }
}

/// SiLU is rounded to BF16 before its product with up, then rounded again.
/// Clamp, when configured, acts on gate BEFORE SiLU and symmetrically on up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeDenseActivation {
    SiLuBf16,
    PreSiLuClampedBf16 { limit: f32 },
}
impl NativeDenseActivation {
    pub fn from_text_policy(policy: Exl3FfnActivation) -> Result<Self> {
        policy.validate()?;
        Ok(match policy {
            Exl3FfnActivation::SiLu => Self::SiLuBf16,
            Exl3FfnActivation::ClampedSiLu { limit } => Self::PreSiLuClampedBf16 { limit },
        })
    }
    pub(super) fn validate(self) -> Result<()> {
        if let Self::PreSiLuClampedBf16 { limit } = self {
            ensure!(
                limit.is_finite() && limit > 0.0,
                "native dense activation requires finite positive clamp"
            );
        }
        Ok(())
    }
    pub(super) fn launch(
        self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        gate: DevicePtr,
        up: DevicePtr,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        let (limit, clamp) = match self {
            Self::SiLuBf16 => (0.0, 0),
            Self::PreSiLuClampedBf16 { limit } => (limit, 1),
        };
        KernelLaunch::new(gpu, kernel)
            .grid([
                u32::try_from(count.div_ceil(256))
                    .context("native dense activation grid overflow")?,
                1,
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(gate)
            .arg_u64(count as u64)
            .arg_f32(limit)
            .arg_u32(clamp)
            .launch(stream)
    }
}
