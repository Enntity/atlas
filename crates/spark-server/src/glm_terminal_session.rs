// SPDX-License-Identifier: AGPL-3.0-only

//! Inert, server-local terminal core for future selected paired registration.
//! No live Model, supervisor ticket, serving flag, or activation caller exists.
mod core;
mod panic;

#[cfg(test)]
mod tests;

const EXIT_GLM_PAIRED_UNCERTAIN: i32 = 74;
static CORE: core::TerminalCore = core::TerminalCore::new();

/// Install before server work; inactive behavior chains to the prior hook.
pub(crate) fn install_panic_ingress() {
    panic::install(&CORE);
}

/// The late TUI hook must call this before restoration/logging/previous hooks.
pub(crate) fn panic_if_selected_live() {
    CORE.panic_if_live();
}

fn terminate() -> ! {
    // SAFETY: immediate process termination deliberately skips Rust destructors
    // and C atexit handlers. Do not add logging, cleanup, or device operations.
    unsafe { libc::_exit(EXIT_GLM_PAIRED_UNCERTAIN) }
}
