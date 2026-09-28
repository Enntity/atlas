// SPDX-License-Identifier: AGPL-3.0-only
//! Selected worker: arm before bind, allocation and the entire command receive.

use super::SelectedModel;
use spark_model::traits::SequenceState;

impl SelectedModel {
    pub(crate) fn run_worker(mut self) -> ! {
        if self.rank != 1 {
            crate::glm_terminal_session::terminate();
        }
        self.bind_execution_thread();
        let operation = self.begin();
        operation.require(self.model().bind_gpu_to_thread());
        operation.complete();

        // Publish each successful allocation into the retained vector before the
        // completion check. Never return a local SequenceState through Err.
        let mut slots: Vec<Option<SequenceState>> =
            (0..self.owner_capacity()).map(|_| None).collect();
        for (index, slot) in slots.iter_mut().enumerate() {
            let operation = self.begin();
            self.check_health();
            *slot = Some(operation.require(self.model().alloc_sequence()));
            if slot.as_ref().is_none_or(|seq| seq.slot_idx != index) {
                crate::glm_terminal_session::terminate();
            }
            operation.complete();
        }
        tracing::info!(
            "Selected EP worker ready (rank 1, {} retained slots)",
            slots.len()
        );
        loop {
            let operation = self.begin();
            self.check_health();
            let running = operation.require(self.model().ep_worker_step(&mut slots));
            operation.complete();
            if !running {
                // Keep all physical/private slots, including F1 replacements,
                // alive while joining and waiting for the paired release.
                self.shutdown_worker();
            }
        }
    }
}
