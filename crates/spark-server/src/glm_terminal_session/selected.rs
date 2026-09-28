// SPDX-License-Identifier: AGPL-3.0-only
//! Actual selected Model ownership. Ordinary return/Drop is never a release.

use super::core::{InFlight, SessionKey};
use super::inherited::InheritedSession;
use super::inherited_startup::ReceivedStartup;
use anyhow::{Context, Result, ensure};
use spark_model::speculative::glm_paired_execution::GlmPairedExecution;
use spark_model::traits::Model;

pub(crate) struct SelectedModel {
    model: Box<dyn Model>,
    session: Option<InheritedSession>,
    key: SessionKey<'static>,
    rank: u8,
    poll_ms: u64,
    owner_capacity: usize,
}

/// An accidental owner drop terminates before Rust drops any owned fields.
impl Drop for SelectedModel {
    fn drop(&mut self) {
        super::terminate()
    }
}

fn require<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(_error) => super::terminate(),
    }
}

fn capability(model: &dyn Model) -> Result<&dyn GlmPairedExecution> {
    model
        .glm_paired_execution()
        .context("selected owner requires actual paired Model capability")
}

impl SelectedModel {
    /// The caller has checked parsed CLI/model settings against this exact
    /// recipe before construction. Recheck the actual capability/rank here;
    /// receipt possession alone never selects a legacy Model.
    pub(crate) fn register(model: Box<dyn Model>, startup: ReceivedStartup, rank: u8) -> Self {
        // Arm while both incoming owners remain on this stack. Every subsequent
        // error/panic is nonreturning, including failed actual health validation.
        let key = require(super::CORE.activate());
        let validation = (|| {
            startup.recipe.profile.validate()?;
            ensure!(
                rank < 2
                    && startup.recipe.world == 2
                    && startup.recipe.rank == rank
                    && startup.session.rank() == rank,
                "selected Model/ticket/recipe rank mismatch"
            );
            let paired = capability(model.as_ref())?;
            paired.validate_session_rank(rank)?;
            let owner_capacity = paired.owner_capacity()?;
            ensure!(
                owner_capacity == usize::from(startup.recipe.profile.max_sequences),
                "selected Model/recipe owner capacity mismatch"
            );
            paired.check_communication_health()?;
            Ok(owner_capacity)
        })();
        let owner_capacity = require(validation);
        let poll_ms = startup.session.poll_ms();
        Self {
            model,
            session: Some(startup.session),
            key,
            rank,
            poll_ms,
            owner_capacity,
        }
    }

    pub(crate) fn model(&self) -> &(dyn Model + 'static) {
        self.model.as_ref()
    }

    pub(crate) fn owner_capacity(&self) -> usize {
        self.owner_capacity
    }

    /// Establish the new thread's protection before bind/alloc/receive. No GPU
    /// query or per-token prctl: the registered core already makes errors fatal.
    pub(crate) fn bind_execution_thread(&self) {
        require(
            self.session
                .as_ref()
                .context("selected execution thread requires retained session")
                .and_then(InheritedSession::bind_execution_thread),
        );
    }

    pub(crate) fn begin(&self) -> SelectedOperation<'_> {
        SelectedOperation::begin(self.model(), &self.key)
    }

    pub(crate) fn check_health(&self) {
        require(capability(self.model()).and_then(|paired| paired.check_communication_health()));
    }

    pub(crate) fn poll_ms(&self) -> u64 {
        self.poll_ms
    }

    pub(crate) fn shutdown_head(mut self) -> ! {
        if self.rank != 0 {
            super::terminate();
        }
        let operation = self.begin();
        self.check_health();
        crate::ep_peer_lifeline::expect_peer_exit();
        operation.require(self.model().ep_broadcast_cmd_for_seq(0, u32::MAX));
        operation.complete();
        self.exit_quiescent()
    }

    /// Worker reaches this only after the real ep_worker_step consumed shutdown.
    pub(super) fn shutdown_worker(&mut self) -> ! {
        if self.rank != 1 {
            super::terminate();
        }
        crate::ep_peer_lifeline::expect_peer_exit();
        self.exit_quiescent()
    }

    fn exit_quiescent(&mut self) -> ! {
        let operation = self.begin();
        operation.require(capability(self.model()).and_then(|paired| paired.quiesce()));
        operation.complete();
        // No key close, Model teardown, sequence free or subsequent GPU work.
        // The caller's sequences and this boxed Model stay live through _exit.
        let session = match self.session.take() {
            Some(session) => session,
            None => super::terminate(),
        };
        session.exit_after_quiescence()
    }
}

pub(crate) struct SelectedOperation<'a> {
    model: &'a dyn Model,
    flight: InFlight<'a>,
}

impl<'a> SelectedOperation<'a> {
    fn begin(model: &'a dyn Model, key: &'a SessionKey<'_>) -> Self {
        Self {
            model,
            flight: require(key.begin()),
        }
    }

    pub(crate) fn require<T>(&self, result: Result<T>) -> T {
        self.flight.require(result)
    }

    pub(crate) fn complete(self) {
        // Never expose idle/completed state after an asynchronous communicator
        // failure, even when the preceding Model operation returned Ok.
        self.flight
            .require(capability(self.model).and_then(|paired| paired.check_communication_health()));
        self.flight.complete();
    }
}

#[path = "selected_worker.rs"]
mod worker;

#[cfg(test)]
#[path = "selected_tests.rs"]
mod tests;
