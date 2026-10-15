// SPDX-License-Identifier: AGPL-3.0-only

use crate::weight_loader::Glm5Exl3Projection;
use crate::weight_map::QuantizedWeight;
use anyhow::{Result, bail};

/// Checkpoint format determines the compute path; packed EXL3 is never
/// relabeled or requantized to NVFP4.
pub enum Glm5Projection {
    Nvfp4 {
        nvfp4: QuantizedWeight,
        prefill_nvfp4_t: Option<QuantizedWeight>,
    },
    Exl3(Box<Glm5Exl3Projection>),
}

impl Glm5Projection {
    pub fn as_nvfp4(&self) -> Result<(&QuantizedWeight, Option<&QuantizedWeight>)> {
        match self {
            Self::Nvfp4 {
                nvfp4,
                prefill_nvfp4_t,
            } => Ok((nvfp4, prefill_nvfp4_t.as_ref())),
            Self::Exl3(_) => bail!("NVFP4-only GLM projection path cannot consume EXL3"),
        }
    }

    pub fn is_exl3(&self) -> bool {
        matches!(self, Self::Exl3(_))
    }
}
