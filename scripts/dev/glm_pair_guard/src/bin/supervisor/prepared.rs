// SPDX-License-Identifier: AGPL-3.0-only

//! Immutable-by-digest, one-shot local inputs. No Docker/Model authority.
use atlas_glm_pair_wire as wire;
use sha2::{Digest as _, Sha256};
use std::io;
use std::path::{Path, PathBuf};

#[path = "prepared_files.rs"]
mod files;
#[path = "prepared_types.rs"]
mod types;
pub use types::*;

const MAX_JSON: usize = 200 * 1024;
const MAX_INPUT: usize = 1024 * 1024;
const NAMES: [&str; 5] = [
    "launch.json",
    "session.bin",
    "rank0.recipe.bin",
    "rank1.recipe.bin",
    "workload.input",
];
fn error(s: &'static str) -> io::Error {
    io::Error::other(s)
}
fn raw_hash(bytes: &[u8]) -> wire::Digest {
    Sha256::digest(bytes).into()
}
fn verify_hash(bytes: &[u8], hex: &str) -> io::Result<()> {
    if raw_hash(bytes) != super::docker::parse_id(hex)? {
        return Err(error("literal input hash mismatch"));
    }
    Ok(())
}
fn bundle_digest(parts: &[Vec<u8>; 5]) -> wire::Digest {
    let mut hash = Sha256::new();
    hash.update(b"atlas.glm.pair.prepared.v1\0");
    for (name, bytes) in NAMES.iter().zip(parts) {
        hash.update((name.len() as u32).to_be_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}
pub struct Prepared {
    pub launch: Launch,
    pub session: wire::Digest,
    pub recipes: [wire::Recipe; 2],
    pub directory: PathBuf,
    pub digest: wire::Digest,
    pub workload_input: Vec<u8>,
    pinned: files::Directory,
}
impl Prepared {
    pub fn prepare(input: &Path, out: &Path) -> io::Result<Self> {
        let mut launch: Launch = serde_json::from_slice(&files::read_input(input, MAX_JSON)?)?;
        launch.validate(false)?;
        let mut bytes = [
            files::read_input(&launch.nodes[0].recipe_file, wire::MAX_RECIPE_BYTES)?,
            files::read_input(&launch.nodes[1].recipe_file, wire::MAX_RECIPE_BYTES)?,
        ];
        for (r, bytes) in bytes.iter().enumerate() {
            verify_hash(bytes, &launch.nodes[r].recipe_sha256)?;
        }
        let mut recipes = [
            wire::Recipe::decode(&bytes[0]).map_err(io::Error::other)?,
            wire::Recipe::decode(&bytes[1]).map_err(io::Error::other)?,
        ];
        let workload_input = files::read_input(&launch.workload.input_file, MAX_INPUT)?;
        verify_hash(&workload_input, &launch.workload.input_sha256)?;
        if workload_input.len() > launch.workload.limits.stdin_bytes {
            return Err(error("workload exceeds explicit stdin cap"));
        }
        let session = atlas_glm_pair_io::identity::fresh_nonce()?;
        for r in 0..2 {
            if recipes[r].rank != r as u8 || recipes[r].world != 2 {
                return Err(error("recipe rank/world mismatch"));
            }
            let mut binds = recipes[r]
                .mounts
                .iter_mut()
                .filter(|m| m.destination == "/run/atlas-pair");
            let bind = binds.next().ok_or_else(|| error("missing session bind"))?;
            if bind.read_only || bind.propagation != "rprivate" {
                return Err(error("session bind must be private writable"));
            }
            bind.source = format!(
                "/run/atlas-glm-pairs/{}/rank{r}",
                super::docker::hex(&session)
            );
            if binds.next().is_some() {
                return Err(error("multiple session binds"));
            }
            bytes[r] = recipes[r].encode().map_err(io::Error::other)?;
            launch.nodes[r].recipe_file = PathBuf::from(NAMES[r + 2]);
            launch.nodes[r].recipe_sha256 = super::docker::hex(&raw_hash(&bytes[r]));
        }
        launch.workload.input_file = PathBuf::from("workload.input");
        launch.validate(true)?;
        let parts = [
            serde_json::to_vec(&launch)?,
            session.to_vec(),
            bytes[0].clone(),
            bytes[1].clone(),
            workload_input.clone(),
        ];
        if parts[0].len() > MAX_JSON {
            return Err(error("canonical launch JSON exceeds bound"));
        }
        let digest = bundle_digest(&parts);
        let pinned = files::Directory::create(out)?;
        for (name, part) in NAMES.iter().zip(&parts) {
            pinned.write(name, part)?;
        }
        Ok(Self {
            launch,
            session,
            recipes,
            directory: out.to_owned(),
            digest,
            workload_input,
            pinned,
        })
    }
    pub fn load(dir: &Path, expected: wire::Digest) -> io::Result<Self> {
        let pinned = files::Directory::open(dir)?;
        let parts = [
            pinned.read(NAMES[0], MAX_JSON)?,
            pinned.read(NAMES[1], 32)?,
            pinned.read(NAMES[2], wire::MAX_RECIPE_BYTES)?,
            pinned.read(NAMES[3], wire::MAX_RECIPE_BYTES)?,
            pinned.read(NAMES[4], MAX_INPUT)?,
        ];
        let digest = bundle_digest(&parts);
        if expected == [0; 32] || digest != expected {
            return Err(error("prepared digest mismatch"));
        }
        let launch: Launch = serde_json::from_slice(&parts[0])?;
        launch.validate(true)?;
        if serde_json::to_vec(&launch)? != parts[0] {
            return Err(error("noncanonical prepared launch JSON"));
        }
        let session: wire::Digest = parts[1]
            .as_slice()
            .try_into()
            .map_err(|_| error("prepared session length"))?;
        if session == [0; 32] {
            return Err(error("zero prepared session"));
        }
        let recipes = [
            wire::Recipe::decode(&parts[2]).map_err(io::Error::other)?,
            wire::Recipe::decode(&parts[3]).map_err(io::Error::other)?,
        ];
        for r in 0..2 {
            verify_hash(&parts[r + 2], &launch.nodes[r].recipe_sha256)?;
            if recipes[r].rank != r as u8
                || recipes[r].world != 2
                || recipes[r].encode().map_err(io::Error::other)? != parts[r + 2]
            {
                return Err(error("noncanonical bundled recipe rank/world"));
            }
            let mut binds = recipes[r]
                .mounts
                .iter()
                .filter(|m| m.destination == "/run/atlas-pair");
            let bind = binds
                .next()
                .ok_or_else(|| error("missing bundled session bind"))?;
            if bind.read_only
                || bind.propagation != "rprivate"
                || binds.next().is_some()
                || bind.source
                    != format!(
                        "/run/atlas-glm-pairs/{}/rank{r}",
                        super::docker::hex(&session)
                    )
            {
                return Err(error("bundled session bind mismatch"));
            }
        }
        verify_hash(&parts[4], &launch.workload.input_sha256)?;
        if parts[4].len() > launch.workload.limits.stdin_bytes {
            return Err(error("bundled workload exceeds explicit stdin cap"));
        }
        pinned.revalidate()?;
        let [_, _, _, _, workload_input] = parts;
        Ok(Self {
            launch,
            session,
            recipes,
            directory: dir.to_owned(),
            digest,
            workload_input,
            pinned,
        })
    }
    pub fn consume(&self) -> io::Result<()> {
        let mut receipt = self.session.to_vec();
        receipt.extend(self.digest);
        self.pinned.write("consumed", &receipt)
    }
    /// The caller additionally enforces the campaign's aggregate evidence cap.
    pub fn record(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        if name.len() > 128
            || name.strip_prefix("evidence-").is_none_or(|s| s.is_empty())
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || bytes.len() > MAX_INPUT
        {
            return Err(error("invalid bounded evidence record"));
        }
        self.pinned.write(name, bytes)
    }
}

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod tests;
