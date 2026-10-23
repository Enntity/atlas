// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit GLM TP2/TP4 serving policies. Other model launches retain their behavior.

use anyhow::{Result, bail, ensure};

use super::NativePolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServingMode {
    Strict,
    Spark,
}

impl ServingMode {
    pub fn parse(value: Option<&str>) -> Result<Option<Self>> {
        match value {
            None => Ok(None),
            Some("strict") => Ok(Some(Self::Strict)),
            Some("spark") => Ok(Some(Self::Spark)),
            Some(value) => bail!("ATLAS_GLM_SERVING_MODE must be strict or spark, got {value:?}"),
        }
    }

    pub fn validate(
        self,
        policy: NativePolicy,
        input_scales: bool,
        decode_a16: bool,
    ) -> Result<()> {
        // Spark permits the separately qualified shared/dense FP4 paths. MLA
        // and KDA prefill remain native; KDA decode remains corrected FP4.
        let (mask, expected) = match self {
            Self::Strict => (0b11111, 0b11111),
            Self::Spark => (0b11001, 0b10001),
        };
        ensure!(
            policy.bits() & mask == expected,
            "{self:?} mode rejects native group policy {:05b}; expected {expected:05b} under mask {mask:05b}",
            policy.bits()
        );
        ensure!(
            input_scales,
            "{self:?} mode requires checkpoint input scales"
        );
        ensure!(
            decode_a16 == (self == Self::Strict),
            "{self:?} mode has an incompatible routed decode activation policy"
        );
        Ok(())
    }
}

pub fn selected() -> Result<Option<ServingMode>> {
    static MODE: std::sync::OnceLock<std::result::Result<Option<ServingMode>, String>> =
        std::sync::OnceLock::new();
    MODE.get_or_init(|| {
        ServingMode::parse(std::env::var("ATLAS_GLM_SERVING_MODE").ok().as_deref())
            .map_err(|error| error.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// This includes Spark: unapproved fallback requantization is forbidden in both modes.
pub fn require_format(present: bool, description: &str) -> Result<()> {
    ensure!(
        selected()?.is_none() || present,
        "GLM serving mode requires {description}; refusing fallback weight conversion"
    );
    Ok(())
}

/// Native MTP remains an explicit, bounded strict-mode option until qualified.
pub fn strict_native_mtp() -> bool {
    std::env::var("ATLAS_GLM_STRICT_NATIVE_MTP").as_deref() == Ok("1")
}

pub fn strict_serial() -> bool {
    matches!(selected(), Ok(Some(ServingMode::Strict))) && !strict_native_mtp()
}

/// Drafter weight conversions share the GLM-only RNE entry. Activations do not.
pub fn weight_quantizer() -> Result<&'static str> {
    Ok(if selected()?.is_some() {
        "quantize_bf16_to_nvfp4_weight_rne"
    } else {
        "quantize_bf16_to_nvfp4"
    })
}

/// A calibrated expert is either local with all scales, or an empty EP placeholder.
pub fn validate_calibration(required: bool, live: [bool; 3], scaled: [bool; 3]) -> Result<()> {
    ensure!(
        !required || live == [false; 3] || (live == [true; 3] && scaled == [true; 3]),
        "GLM serving mode requires gate/up/down input scales on every live expert"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native() -> NativePolicy {
        NativePolicy::from_lookup(|name| (name == NativePolicy::UMBRELLA).then(|| "1".into()))
    }

    #[test]
    fn modes_reject_unknown_values_and_unreviewed_conversions() {
        assert_eq!(ServingMode::parse(None).unwrap(), None);
        assert_eq!(
            ServingMode::parse(Some("strict")).unwrap(),
            Some(ServingMode::Strict)
        );
        assert_eq!(
            ServingMode::parse(Some("spark")).unwrap(),
            Some(ServingMode::Spark)
        );
        for value in ["", "fast", "STRICT"] {
            assert!(ServingMode::parse(Some(value)).is_err());
        }
        assert!(ServingMode::Strict.validate(native(), true, true).is_ok());
        let mut spark = native();
        spark.kda_decode = false;
        assert!(ServingMode::Spark.validate(spark, true, false).is_ok());
        assert!(ServingMode::Strict.validate(spark, true, true).is_err());
        assert!(ServingMode::Spark.validate(spark, true, true).is_err());
        assert!(ServingMode::Spark.validate(spark, false, false).is_err());
        spark.mla_fp8 = false;
        assert!(ServingMode::Spark.validate(spark, true, false).is_err());
    }

    #[test]
    fn spark_shared_and_dense_are_independent_but_strict_requires_native() {
        for shared_fp8 in [false, true] {
            for dense_mxfp8 in [false, true] {
                let mut policy = native();
                policy.kda_decode = false;
                policy.shared_fp8 = shared_fp8;
                policy.dense_mxfp8 = dense_mxfp8;
                assert!(ServingMode::Spark.validate(policy, true, false).is_ok());
                assert!(ServingMode::Strict.validate(policy, true, true).is_err());
                policy.kda_decode = true;
                assert!(ServingMode::Spark.validate(policy, true, false).is_err());
                assert_eq!(
                    ServingMode::Strict.validate(policy, true, true).is_ok(),
                    shared_fp8 && dense_mxfp8
                );
                policy.kda_decode = false;
                policy.kda_w8a8 = false;
                assert!(ServingMode::Spark.validate(policy, true, false).is_err());
            }
        }
    }

    #[test]
    fn missing_local_scales_cannot_masquerade_as_remote_experts() {
        assert!(validate_calibration(true, [true; 3], [true; 3]).is_ok());
        assert!(validate_calibration(true, [false; 3], [false; 3]).is_ok());
        assert!(validate_calibration(true, [true; 3], [false; 3]).is_err());
        assert!(validate_calibration(true, [true; 3], [true, true, false]).is_err());
        assert!(validate_calibration(true, [true, false, true], [true; 3]).is_err());
        assert!(validate_calibration(false, [true; 3], [false; 3]).is_ok());
    }
}
