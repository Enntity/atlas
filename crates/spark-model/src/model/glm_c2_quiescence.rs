// SPDX-License-Identifier: AGPL-3.0-only
//! Local selected completion checks, never a two-rank release certificate.
use super::*;

impl TransformerModel {
    pub(super) fn paired_validate_session_rank(&self, expected_rank: u8) -> Result<()> {
        self.paired_completion_preflight()?;
        ensure!(
            self.config.ep_rank == usize::from(expected_rank),
            "paired Model rank differs from inherited session rank"
        );
        Ok(())
    }
    fn paired_completion_preflight(&self) -> Result<()> {
        ensure!(
            self.config.model_type == "glm5_next"
                && self.config.tp_world_size == 2
                && self.config.ep_world_size == 2
                && self.config.ep_rank < 2
                && self.config.tp_rank == self.config.ep_rank,
            "paired completion requires actual GLM TP2/EP2 local rank"
        );
        self.paired_wire_profile(self.config.ep_rank)?;
        self.paired_handoff()
            .context("paired completion pool missing")?
            .validate_session(self.gpu.as_ref())?;
        let default = self.gpu.default_stream();
        ensure!(
            !self.gpu.stream_is_capturing(default)
                && (self.secondary_stream == default
                    || !self.gpu.stream_is_capturing(self.secondary_stream)),
            "paired completion refuses active stream capture"
        );
        Ok(())
    }
    fn paired_probe_communication(&self) -> Result<()> {
        let comm = self
            .comm
            .as_ref()
            .context("paired completion communicator missing")?;
        ensure!(comm.is_healthy(), "paired communicator is unhealthy");
        Ok(())
    }
    pub(super) fn paired_check_communication_health(&self) -> Result<()> {
        self.paired_completion_preflight()?;
        self.paired_probe_communication()
            .map_err(|error| self.paired_transport_error(error))
    }
    pub(super) fn paired_quiesce(&self) -> Result<()> {
        self.paired_completion_preflight()?;
        (|| {
            self.paired_probe_communication()?;
            let default = self.gpu.default_stream();
            self.gpu.synchronize(default)?;
            self.paired_probe_communication()?;
            if self.secondary_stream != default {
                self.gpu.synchronize(self.secondary_stream)?;
                self.paired_probe_communication()?;
            }
            Ok(())
        })()
        .map_err(|error| self.paired_transport_error(error))
    }
}
