// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit cold wider-compute selection; capacity alone never enables it.
use super::{TransformerModel, glm_owner_wire::Mode};
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use anyhow::{Result, bail, ensure};

pub(in crate::model) fn requested() -> Result<Option<Mode>> {
    match std::env::var("ATLAS_GLM_OWNER_VERIFY") {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(value) if value == "0" => Ok(None),
        Ok(value) if value == "joint" => Ok(Some(Mode::OwnersJoint)),
        _ => bail!("ATLAS_GLM_OWNER_VERIFY requires literal0|joint"),
    }
}

impl TransformerModel {
    pub(in crate::model) fn initialize_glm_owner_verification(&mut self, mode: Mode) -> Result<()> {
        ensure!(
            self.glm_owner_verify_mode.is_none(),
            "wider compute already selected"
        );
        self.paired_wire_profile(self.config.ep_rank)?;
        let shape = GlmOwnerBatchShape::new(self.paired_owner_capacity()?)?;
        self.validate_glm_owner_compute(shape)?;
        self.glm_owner_verify_mode = Some(mode);
        Ok(())
    }
}
