// SPDX-License-Identifier: AGPL-3.0-only
//! Outer Model::decode owns all successful eager/profile/graph return paths.
use super::*;
use spark_runtime::gpu::DevicePtr;

impl TransformerModel {
    pub(in crate::model) fn paired_before_decode(
        &self,
        seq: &mut SequenceState,
        token: u32,
    ) -> Result<bool> {
        let Some(capability) = self.paired_handoff() else {
            return Ok(false);
        };
        let mut state = seq
            .proposer_state
            .take()
            .context("paired decode state missing")?;
        let result = (|| {
            let input = self.paired_input(seq)?;
            capability.begin_decode(&input, token, state.as_mut(), &self.glm_repair_context())
        })();
        seq.proposer_state = Some(state);
        result?;
        Ok(true)
    }

    pub(in crate::model) fn paired_after_decode(
        &self,
        seq: &mut SequenceState,
        token: u32,
        decoded: Result<DevicePtr>,
    ) -> Result<DevicePtr> {
        let capability = self
            .paired_handoff()
            .context("paired decode capability disappeared")?;
        let mut state = seq
            .proposer_state
            .take()
            .context("paired decoded state missing")?;
        let result: Result<DevicePtr> = (|| {
            let logits = decoded?;
            let input = self.paired_input(seq)?;
            capability.publish_decode(
                &input,
                token,
                state.as_mut(),
                &self.glm_repair_context(),
                self.gpu.default_stream(),
            )?;
            Ok(logits)
        })();
        let result = match result {
            Ok(ptr) => Ok(ptr),
            Err(error) => {
                let quarantined = capability.quarantine(state.as_mut(), self.gpu.as_ref());
                Err(error).context(format!("paired target failed; quarantine={quarantined:?}"))
            }
        };
        seq.proposer_state = Some(state);
        result
    }
}
