// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot production state, without Model/protocol/admission assertions.
use anyhow::{Result, bail};
use std::sync::atomic::{AtomicU8, Ordering};

const NEVER_SELECTED: u8 = 0;
const IDLE: u8 = 1;
const IN_FLIGHT: u8 = 2;
const CLOSED: u8 = 3;

pub(super) struct TerminalCore {
    state: AtomicU8,
}

impl TerminalCore {
    pub(super) const fn new() -> Self {
        Self {
            state: AtomicU8::new(NEVER_SELECTED),
        }
    }

    /// Registration authority is deliberately absent from T1. Only this module
    /// and its later checked registration sibling can activate the core.
    pub(super) fn activate(&self) -> Result<SessionKey<'_>> {
        if self
            .state
            .compare_exchange(NEVER_SELECTED, IDLE, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            bail!("terminal core cannot be activated twice");
        }
        Ok(SessionKey {
            core: self,
            closed: false,
        })
    }

    pub(super) fn panic_if_live(&self) {
        if matches!(self.state.load(Ordering::Acquire), IDLE | IN_FLIGHT) {
            self.fatal();
        }
    }

    pub(super) fn fatal(&self) -> ! {
        super::terminate()
    }
}

/// A borrow binds this key to its actual core without address-based lookup.
pub(super) struct SessionKey<'a> {
    core: &'a TerminalCore,
    closed: bool,
}

impl SessionKey<'_> {
    pub(super) fn begin(&self) -> Result<InFlight<'_>> {
        if self
            .core
            .state
            .compare_exchange(IDLE, IN_FLIGHT, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            bail!("terminal operation requires an idle selected core");
        }
        Ok(InFlight {
            core: self.core,
            completed: false,
        })
    }

    /// Core close only: T2 must supply actual both-rank clean-drain authority.
    /// The consumed key cannot close while an operation still borrows it.
    pub(super) fn close(mut self) -> Result<()> {
        if self
            .core
            .state
            .compare_exchange(IDLE, CLOSED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            bail!("terminal session cannot close an incomplete operation");
        }
        self.closed = true;
        Ok(())
    }
}

impl Drop for SessionKey<'_> {
    fn drop(&mut self) {
        if !self.closed {
            self.core.fatal();
        }
    }
}

pub(super) struct InFlight<'a> {
    core: &'a TerminalCore,
    completed: bool,
}

impl InFlight<'_> {
    /// Exit before this Result/error payload and caller-owned resources drop.
    /// This does not retroactively prevent cleanup already done inside a callee.
    pub(super) fn require<T>(&self, result: Result<T>) -> T {
        match result {
            Ok(value) => value,
            Err(_error) => self.core.fatal(),
        }
    }

    pub(super) fn complete(mut self) {
        if self
            .core
            .state
            .compare_exchange(IN_FLIGHT, IDLE, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.core.fatal();
        }
        self.completed = true;
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.core.fatal();
        }
    }
}
