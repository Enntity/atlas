// SPDX-License-Identifier: AGPL-3.0-only

//! Supervised inherited startup, actual Model ownership and terminal operations.
//! Selected serving requires the guard channel; a CLI flag alone is insufficient.
mod core;
#[cfg(target_os = "linux")]
pub(crate) mod inherited;
#[cfg(target_os = "linux")]
pub(crate) mod inherited_startup;
mod panic;
mod runtime;
use runtime::CORE;
#[cfg(test)]
use runtime::EXIT_GLM_PAIRED_UNCERTAIN;
pub(crate) use runtime::{install_panic_ingress, panic_if_selected_live, terminate};
#[cfg(target_os = "linux")]
mod selected;
#[cfg(target_os = "linux")]
pub(crate) use selected::{SelectedModel, SelectedOperation};
pub(crate) mod startup;

#[cfg(test)]
mod tests;
