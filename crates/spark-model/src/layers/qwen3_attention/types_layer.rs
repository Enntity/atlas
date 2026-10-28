// SPDX-License-Identifier: AGPL-3.0-only

//! The `Qwen3AttentionLayer` struct, split from `types.rs` (500-LoC cap).

use crate::layers::FfnComponent;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantWeight, QuantizedWeight};

use super::types::HeadGateActivation;
use super::types_weights::{HcWeights, MlaWeights};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

#[path = "types_layer_attention.rs"]
mod attention;
pub use attention::Qwen3AttentionLayer;
