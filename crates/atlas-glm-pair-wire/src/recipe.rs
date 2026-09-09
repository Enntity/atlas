// SPDX-License-Identifier: AGPL-3.0-only

//! Canonical recipe v1 data only. No resource admission, environment defaults,
//! Docker observations or process/Model authority are supplied by this codec.
//!
//! Wire order is declaration order below, preceded by Recipe version u16=1.
//! Vectors use u16 counts, strings u32 UTF-8 byte lengths, booleans one 0/1 byte.
//! Integers are big-endian; Docker count/ulimit -1 is explicit signed i64.
//! Environment/options sort by key, mounts by destination, devices by driver,
//! ulimits by name, and set-like string lists lexicographically. Argv preserves
//! order, including meaningful duplicates. No field is omitted/defaulted.

use crate::{Digest, Result, recipe_digest};

#[path = "recipe_codec.rs"]
mod codec;
#[path = "recipe_validate.rs"]
mod validate;

pub const MAX_RECIPE_BYTES: usize = 65536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub argv: Vec<String>,
    pub environment: Vec<(String, String)>,
    pub mounts: Vec<Mount>,
    pub image_digest: Digest,
    pub guard_elf_digest: Digest,
    pub server_elf_digest: Digest,
    pub rank: u8,
    pub world: u8,
    pub profile: Profile,
    pub resources: Resources,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mount {
    pub source: String,
    pub destination: String,
    pub read_only: bool,
    pub propagation: String,
}

/// Fixed selected profile, exactly 24 bytes; BF16 kv_format is wire code1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub tp: u8,
    pub ep: u8,
    pub ep_protocol: u8,
    pub max_sequences: u16,
    pub context: u32,
    pub prefill: u32,
    pub drafts: u8,
    pub eager: bool,
    pub kv_format: u8,
    pub cold_min: u32,
    pub cold_max: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resources {
    pub memory: u64,
    pub swap: u64,
    pub cpuset: String,
    pub shm: u64,
    pub device_requests: Vec<DeviceRequest>,
    pub ulimits: Vec<Ulimit>,
    pub cap_add: Vec<String>,
    pub cap_drop: Vec<String>,
    pub uid: u32,
    pub gid: u32,
    pub network_mode: String,
    pub pid_mode: String,
    pub restart_policy: String,
    pub init: bool,
    pub no_new_privileges: bool,
    pub ipc_mode: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRequest {
    pub driver: String,
    pub count: i64,
    pub device_ids: Vec<String>,
    pub capabilities: Vec<Vec<String>>,
    pub options: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ulimit {
    pub name: String,
    pub soft: i64,
    pub hard: i64,
}

impl Recipe {
    pub fn encode(&self) -> Result<Vec<u8>> {
        validate::recipe(self)?;
        codec::encode(self)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let value = codec::decode(bytes)?;
        validate::recipe(&value)?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<()> {
        self.encode().map(|_| ())
    }
    pub fn digest(&self) -> Result<Digest> {
        recipe_digest(&self.encode()?)
    }
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
