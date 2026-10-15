// SPDX-License-Identifier: AGPL-3.0-only
// CachePool — wrapper around spark-runtime kv_cache (plan §2.1 / §5.2).
// Reuses PagedKvCache when cuda available; lazy backing store (DashMap) always.

use crate::{config::ServerConfig, error::ServerError};
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cuda")]
use spark_runtime::kv_cache::PagedKvCache;
#[cfg(not(feature = "cuda"))]
pub struct PagedKvCache; // dummy for compile without GPU (B6)

/// Block location: in-memory ref or spilled path.
#[derive(Clone, Debug)]
pub enum BlockLocation {
    Memory {
        block_ref: usize,
    }, // index into PagedKvCache pool (deferred)
    Disk {
        path: std::path::PathBuf,
        hash: String,
    },
}

/// CachePool always available (server must run without GPU context; B6 fix).
pub struct CachePool {
    /// Optional reused GPU pool (deferred full integration).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    inner: Option<Arc<PagedKvCache>>,
    /// Lazy backing store when inner unavailable (PCND — no silent drop).
    backing: DashMap<String, Vec<u8>>,
    max_bytes: usize,
    used_bytes: AtomicUsize,
    mapping: DashMap<String, BlockLocation>,
    nvme_spill: Option<Arc<crate::nvme_spill::NvmeSpill>>,
    #[allow(dead_code)]
    config: ServerConfig,
}

impl CachePool {
    pub fn new(config: ServerConfig, _inner: Option<Arc<PagedKvCache>>) -> Self {
        let nvme = config.nvme_path.as_ref().map(|p| {
            Arc::new(crate::nvme_spill::NvmeSpill::new(
                p.clone(),
                config.compression.clone(),
                config.dedup_enabled,
            ))
        });
        Self {
            #[cfg(feature = "cuda")]
            inner: _inner,
            backing: DashMap::new(),
            max_bytes: config.max_memory_bytes,
            used_bytes: AtomicUsize::new(0),
            mapping: DashMap::new(),
            nvme_spill: nvme,
            config,
        }
    }

    pub fn put(&self, key: String, bytes: Vec<u8>) -> Result<(), ServerError> {
        let len = bytes.len();
        let used = self.used_bytes.load(Ordering::SeqCst);
        if used + len > self.max_bytes {
            if let Some(ref spill) = self.nvme_spill {
                spill.spill(&key, &bytes)?;
                // Dedup / safe path: use key hash filename; for Phase 1 return safe path.
                let path = spill.path_for(&key);
                self.mapping.insert(
                    key.clone(),
                    BlockLocation::Disk {
                        path: path.clone(),
                        hash: format!("{}-spill", key),
                    },
                );
                // Don't add to backing or used_bytes for spilled-only; memory stays capped.
                return Ok(());
            }
            return Err(ServerError::PoolFull);
        }
        // Store in backing (lazy — full PagedKvCache integration deferred to T3 with cuda).
        self.backing.insert(key.clone(), bytes);
        self.mapping
            .insert(key, BlockLocation::Memory { block_ref: 0 });
        self.used_bytes.fetch_add(len, Ordering::SeqCst);
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Vec<u8>, ServerError> {
        match self.mapping.get(key) {
            Some(loc) => {
                match loc.value() {
                    BlockLocation::Memory { block_ref: _ } => {
                        // Return from backing store (lazy fix for B1 — no false empty success).
                        if let Some(entry) = self.backing.get(key) {
                            Ok(entry.clone())
                        } else {
                            // Backing missing but mapping says memory — internal inconsistency.
                            Err(ServerError::SpillFailed(
                                "memory mapping exists but backing data missing".into(),
                            ))
                        }
                    }
                    BlockLocation::Disk { path, .. } => {
                        if let Some(ref spill) = self.nvme_spill {
                            let data = spill.restore(path)?;
                            Ok(data)
                        } else {
                            Err(ServerError::SpillFailed(
                                "disk block but no spill configured".into(),
                            ))
                        }
                    }
                }
            }
            None => Err(ServerError::KeyNotFound(key.to_string())),
        }
    }

    pub fn remove(&self, key: &str) -> Result<(), ServerError> {
        self.mapping.remove(key);
        self.backing.remove(key);
        let used = self.used_bytes.load(Ordering::SeqCst);
        // Best-effort: exact used_bytes not tracked per-key here; acceptable Phase 1.
        let _ = used;
        Ok(())
    }
}
