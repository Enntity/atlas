// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded Linux transport for the paired GLM guard and its inherited child.
//! This crate supplies I/O, never Model registration or successful release.
#![cfg(target_os = "linux")]
#![deny(warnings)]

mod channel;
pub mod identity;
mod startup_files;
pub use channel::{Channel, Credentials};
pub use startup_files::PrivateDirectory;
