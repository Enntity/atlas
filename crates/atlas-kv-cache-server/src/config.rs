// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Server config — per plan §5.1 (fail-fast validation, PCND).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ServerConfig {
    pub bind_addr: String,            // e.g. "0.0.0.0:50052"
    pub max_memory_bytes: usize,      // cap; must be > 0
    pub nvme_path: Option<PathBuf>,   // None = pure mem; Some = spill path
    pub block_size: usize,            // tokens per block (matches kv_cache)
    pub enable_rdma: bool,            // requires atlas_rdma_verbs + cuda
    pub compression: CompressionMode, // zstd / lz4 / none (plan §5.6)
    pub dedup_enabled: bool,          // sha2 content-hash index (plan §5.6)
}

#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialEq, Eq)]
pub enum CompressionMode {
    #[default]
    None,
    Zstd,
    Lz4,
}

impl ServerConfig {
    /// Fail-fast at startup (PCND — no implicit defaults).
    pub fn validate(&self) -> Result<(), crate::error::ServerError> {
        if self.max_memory_bytes == 0 {
            return Err(crate::error::ServerError::ConfigInvalid(
                "max_memory_bytes must be > 0".into(),
            ));
        }
        if self.block_size == 0 {
            return Err(crate::error::ServerError::ConfigInvalid(
                "block_size must be > 0".into(),
            ));
        }
        if let Some(ref p) = self.nvme_path {
            if !p.exists() {
                return Err(crate::error::ServerError::NvmePathInvalid(format!(
                    "nvme_path does not exist: {}",
                    p.display()
                )));
            }
            // B4 improved: attempt actual write rather than readonly bit alone.
            if std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(p.join(".atlas_write_test"))
                .and_then(|f| f.set_len(0))
                .is_err()
            {
                return Err(crate::error::ServerError::NvmePathInvalid(format!(
                    "nvme_path not writable: {}",
                    p.display()
                )));
            }
            let _ = std::fs::remove_file(p.join(".atlas_write_test"));
        }
        if self.bind_addr.is_empty() {
            return Err(crate::error::ServerError::ConfigInvalid(
                "bind_addr empty".into(),
            ));
        }
        Ok(())
    }
}
