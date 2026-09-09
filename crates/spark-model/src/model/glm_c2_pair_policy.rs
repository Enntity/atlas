// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit default-off selection; no allocation or per-token environment lookup.
use super::TransformerModel;
use crate::layer::glm_pair_verify::{GlmPairFfn, GlmPairWorkspace};
use anyhow::{Context, Result, bail, ensure};

pub fn requested() -> Result<Option<GlmPairFfn>> {
    let enabled = match std::env::var("ATLAS_GLM_C2_PAIRED_VERIFY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        _ => bail!("ATLAS_GLM_C2_PAIRED_VERIFY requires literal0/1"),
    };
    let mode = match std::env::var("ATLAS_GLM_C2_PAIR_FFN") {
        Err(std::env::VarError::NotPresent) if !enabled => return Ok(None),
        Ok(value) if value == "two-k5" => GlmPairFfn::TwoK5,
        Ok(value) if value == "joint" => GlmPairFfn::Joint,
        Ok(value) if value == "joint-shared-m10" => GlmPairFfn::JointSharedM10,
        _ => bail!(
            "paired verification requires explicit ATLAS_GLM_C2_PAIR_FFN=two-k5|joint|joint-shared-m10"
        ),
    };
    Ok(enabled.then_some(mode))
}

impl TransformerModel {
    /// Called while the actual selected factory still owns the cold model.
    pub(crate) fn configure_glm_pair_verification(&mut self) -> Result<()> {
        if let Some(mode) = requested()? {
            self.initialize_glm_pair_verification(mode)?;
        }
        if let Some(mode) = super::glm_owner_policy::requested()? {
            self.initialize_glm_owner_verification(mode)?;
        }
        Ok(())
    }

    pub(in crate::model) fn initialize_glm_pair_verification(
        &mut self,
        mode: GlmPairFfn,
    ) -> Result<()> {
        ensure!(
            self.glm_pair_verify_mode.is_none(),
            "paired compute already selected"
        );
        self.paired_wire_profile(self.config.ep_rank)?;
        self.paired_handoff()
            .context("paired compute needs actual head")?
            .validate_session(self.gpu.as_ref())?;
        self.validate_glm_pair_compute(mode)?;
        self.glm_pair_verify_mode = Some(mode);
        Ok(())
    }

    pub(in crate::model) fn validate_glm_pair_compute(&self, mode: GlmPairFfn) -> Result<()> {
        ensure!(
            self.config.model_type == "glm5_next"
                && self.config.hidden_size == 4096
                && self.config.hc_mult == 4
                && self.config.tp_world_size == 2
                && self.config.ep_world_size == 2
                && self.config.adapter_max_rank == 0
                && self.config.dflash_capture_layers.is_empty()
                && self.config.vision.is_none()
                && self.lora.is_none()
                && self.layers.len() == self.config.num_hidden_layers,
            "paired compute requires complete base GLM TP2/EP2 hc4 layers"
        );
        let ctx = self.glm_repair_context();
        // No new allocations: the original prefill arena owns all working/tail rows.
        GlmPairWorkspace::new(&ctx, mode)?;
        ensure!(
            self.buffers.sizes().logits
                >= self
                    .config
                    .vocab_size
                    .checked_mul(20)
                    .context("paired logits capacity overflow")?,
            "paired target requires ten actual BF16 logit rows"
        );
        let metadata_bytes = 768usize
            .checked_add(
                (self.max_blocks_per_seq as usize)
                    .checked_mul(20)
                    .context("paired metadata size overflow")?,
            )
            .context("paired metadata size overflow")?;
        let stride = metadata_bytes
            .checked_add(255)
            .context("paired metadata alignment overflow")?
            & !255;
        ensure!(
            self.max_blocks_per_seq > 0
                && self.max_blocks_per_seq <= 128
                && 32768 + 2 * stride <= 49152
                && self.buffers.sizes().scratch >= 49152,
            "paired temporal metadata exceeds reserved scratch window"
        );
        for (index, layer) in self.layers.iter().enumerate() {
            ensure!(
                layer.supports_glm_pair_verify(),
                "unsupported paired target layer"
            );
            layer
                .validate_glm_pair_verify(&ctx, mode, self.gpu.default_stream())
                .with_context(|| format!("paired target layer {index} mode {mode:?}"))?;
        }
        Ok(())
    }
}
