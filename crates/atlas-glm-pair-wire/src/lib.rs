// SPDX-License-Identifier: AGPL-3.0-only

//! Untrusted wire data, not process identity evidence or serving authority.
//! Callers own credentials, phase/replay/deadline checks and actual Model binding.
#![forbid(unsafe_code)]

mod codec;
mod digest;
mod frame;
mod recipe;
mod records;
mod validate;

pub use digest::{
    manifest_digest, policy_digest, quiescent_digest, recipe_digest, release_digest, ticket_digest,
};
pub use frame::{Body, Direction, Encoded, Frame, control_frame_len};
pub use recipe::{
    DeviceMapping, DeviceRequest, MAX_RECIPE_BYTES, Mount, Profile, Recipe, Resources, Ulimit,
};
pub use records::*;

pub const MAX_FRAME: usize = 4096;
pub const MAX_ENCODED: usize = 832;
pub const HEADER_LEN: usize = 16;
pub const SHUTDOWN_COMMAND: u32 = 0xffff_ffff;
pub const DRAIN_EPOCH: u64 = 1;
pub type Digest = [u8; 32];
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

#[cfg(test)]
mod tests;
