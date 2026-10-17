// SPDX-License-Identifier: AGPL-3.0-only
//! CPU/file boundary of the actual production-dispatch scratch replay.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub file: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputCase {
    pub id: String,
    pub rows: usize,
    pub input_a: FileBinding,
    pub input_b: FileBinding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub checkpoint: PathBuf,
    pub config_sha256: String,
    pub index_sha256: String,
    pub rank: usize,
    pub world: usize,
    pub capacity: usize,
    pub device: usize,
    pub stream: String,
    pub layers: [usize; 2],
    pub cases: Vec<InputCase>,
    pub parent_evidence: Vec<FileBinding>,
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
impl Plan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.world == 2 && self.rank < 2 && self.device == 0,
            "explicit isolated-device TP2 rank required"
        );
        ensure!(
            self.capacity == 129 && self.stream == "default",
            "explicit capacity129/default-stream ownership required"
        );
        ensure!(
            self.layers[0] != self.layers[1],
            "two distinct checkpoint layers required"
        );
        ensure!(
            !self.cases.is_empty() && !self.parent_evidence.is_empty(),
            "explicit input cases/checkpoint evidence required"
        );
        for c in &self.cases {
            ensure!(
                !c.id.is_empty()
                    && c.id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
                "safe explicit case ID required"
            );
            ensure!(
                c.rows > 0 && c.rows <= self.capacity,
                "input rows exceed declared capacity"
            );
        }
        for h in self
            .parent_evidence
            .iter()
            .map(|v| &v.sha256)
            .chain([&self.config_sha256, &self.index_sha256])
            .chain(
                self.cases
                    .iter()
                    .flat_map(|c| [&c.input_a.sha256, &c.input_b.sha256]),
            )
        {
            ensure!(
                h.len() == 64
                    && h.bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "lowercase SHA256 binding required"
            );
        }
        Ok(())
    }
}
pub fn validate_bf16(bytes: &[u8], rows: usize, hidden: usize) -> Result<()> {
    ensure!(
        rows.checked_mul(hidden).and_then(|v| v.checked_mul(2)) == Some(bytes.len()),
        "BF16 input shape/bytes mismatch"
    );
    ensure!(
        bytes
            .chunks_exact(2)
            .all(|v| f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16).is_finite()),
        "BF16 input is not finite"
    );
    Ok(())
}
pub struct ReplayIORouter {
    pub plan: Plan,
    pub plan_sha256: String,
    pub out: PathBuf,
}
impl ReplayIORouter {
    pub fn open(plan: &Path, out: &Path) -> Result<Self> {
        let bytes = std::fs::read(plan).context("read explicit replay plan")?;
        let parsed: Plan = serde_json::from_slice(&bytes)?;
        parsed.validate()?;
        ensure!(!out.exists(), "output directory must be new");
        for e in &parsed.parent_evidence {
            Self::read_bound(e)?;
        }
        Ok(Self {
            plan: parsed,
            plan_sha256: digest(&bytes),
            out: out.to_owned(),
        })
    }
    pub fn read_bound(binding: &FileBinding) -> Result<Vec<u8>> {
        let bytes = std::fs::read(&binding.file)
            .with_context(|| format!("read bound file {}", binding.file.display()))?;
        ensure!(
            digest(&bytes) == binding.sha256,
            "source/input binding mismatch: {}",
            binding.file.display()
        );
        Ok(bytes)
    }
    pub fn config(&self) -> Result<String> {
        let bytes = Self::read_bound(&FileBinding {
            file: self.plan.checkpoint.join("config.json"),
            sha256: self.plan.config_sha256.clone(),
        })?;
        Self::read_bound(&FileBinding {
            file: self.plan.checkpoint.join("model.safetensors.index.json"),
            sha256: self.plan.index_sha256.clone(),
        })?;
        String::from_utf8(bytes).context("checkpoint config UTF8")
    }
    pub fn write(&self, name: &str, bytes: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.out)?;
        let path = self.out.join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        std::io::Write::write_all(&mut file, bytes)?;
        Ok(())
    }
    pub fn loaded_blas() -> Result<std::collections::BTreeMap<String, String>> {
        let maps = std::fs::read_to_string("/proc/self/maps")?;
        let mut bindings = std::collections::BTreeMap::new();
        for line in maps.lines() {
            let Some(name) = line.split_whitespace().last() else {
                continue;
            };
            if name.starts_with('/') && name.contains("/libcublas") {
                let path = Path::new(name);
                let canonical = std::fs::canonicalize(path)?;
                let basename = canonical
                    .file_name()
                    .context("BLAS library basename")?
                    .to_string_lossy()
                    .to_string();
                bindings.insert(basename, digest(&std::fs::read(canonical)?));
            }
        }
        ensure!(
            bindings.len() == 2,
            "actual sole classic/Lt library pair required"
        );
        Ok(bindings)
    }
    pub fn executable_hash() -> Result<String> {
        Ok(digest(&std::fs::read(std::env::current_exe()?)?))
    }
}
#[cfg(test)]
#[path = "protocol_tests.rs"]
mod tests;
