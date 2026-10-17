// SPDX-License-Identifier: AGPL-3.0-only
use super::DevicePtr;
use anyhow::{Result, ensure};

/// Explicit numerical implementation; the shared EXL3 profile is separately qualified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16GemmNumerics {
    CublasLt,
    Glm53Classic,
    Glm53AuxiliaryClassic,
}
impl Bf16GemmNumerics {
    pub const fn provider_version(self) -> Option<i32> {
        match self {
            Self::CublasLt => None,
            Self::Glm53Classic | Self::Glm53AuxiliaryClassic => Some(130101),
        }
    }
    /// Classic compute, algorithm, math mode and owned per-stream workspace bytes.
    pub const fn profile(self) -> Option<(i32, i32, i32, usize)> {
        match self {
            Self::CublasLt => None,
            Self::Glm53Classic => Some((68, 99, 16, 32 * 1024 * 1024)),
            Self::Glm53AuxiliaryClassic => Some((68, 99, 0, 32 * 1024 * 1024)),
        }
    }
}

/// Row-major BF16 `output[rows, columns] = input[rows, inner] * weight[columns, inner]^T`.
/// Backend owns the numerical implementation; callers cannot silently substitute F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16Gemm {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub columns: u32,
    pub inner: u32,
    pub stream: u64,
    pub numerics: Bf16GemmNumerics,
}
impl Bf16Gemm {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.rows > 0 && self.columns > 0 && self.inner > 0,
            "BF16 GEMM dimensions must be positive"
        );
        ensure!(
            self.rows <= i32::MAX as u32
                && self.columns <= i32::MAX as u32
                && self.inner <= i32::MAX as u32,
            "BF16 GEMM dimensions exceed i32 library ABI"
        );
        ensure!(
            !self.input.is_null() && !self.weight.is_null() && !self.output.is_null(),
            "BF16 GEMM requires live device pointers"
        );
        ensure!(
            self.input != self.output && self.weight != self.output,
            "BF16 GEMM output must not alias inputs"
        );
        for (a, b) in [
            (self.rows, self.columns),
            (self.rows, self.inner),
            (self.columns, self.inner),
        ] {
            ensure!(
                (a as usize)
                    .checked_mul(b as usize)
                    .and_then(|n| n.checked_mul(2))
                    .is_some(),
                "BF16 GEMM operand byte size overflow"
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_profile_requires_explicit_classic_math() {
        assert_eq!(
            Bf16GemmNumerics::Glm53Classic.profile(),
            Some((68, 99, 16, 32 * 1024 * 1024))
        );
        assert_eq!(
            Bf16GemmNumerics::Glm53AuxiliaryClassic.profile(),
            Some((68, 99, 0, 32 * 1024 * 1024))
        );
        assert_eq!(
            Bf16GemmNumerics::Glm53Classic.provider_version(),
            Some(130101)
        );
        assert_eq!(
            Bf16GemmNumerics::Glm53AuxiliaryClassic.provider_version(),
            Some(130101)
        );
        assert_eq!(Bf16GemmNumerics::CublasLt.provider_version(), None);
        assert_eq!(Bf16GemmNumerics::CublasLt.profile(), None);
    }
    #[test]
    fn typed_gemm_rejects_missing_dimensions_and_aliases() {
        let valid = Bf16Gemm {
            input: DevicePtr(1),
            weight: DevicePtr(2),
            output: DevicePtr(3),
            rows: 10,
            columns: 288,
            inner: 4096,
            stream: 77,
            numerics: Bf16GemmNumerics::CublasLt,
        };
        valid.validate().unwrap();
        for invalid in [
            Bf16Gemm { rows: 0, ..valid },
            Bf16Gemm {
                rows: i32::MAX as u32 + 1,
                ..valid
            },
            Bf16Gemm {
                columns: i32::MAX as u32 + 1,
                ..valid
            },
            Bf16Gemm {
                inner: i32::MAX as u32 + 1,
                ..valid
            },
            Bf16Gemm {
                weight: DevicePtr::NULL,
                ..valid
            },
            Bf16Gemm {
                output: valid.input,
                ..valid
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
}
