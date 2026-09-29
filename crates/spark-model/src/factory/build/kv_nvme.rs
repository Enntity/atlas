// SPDX-License-Identifier: AGPL-3.0-only

//! Construction of the prefix cache's NVMe spill tier (default OFF).
//!
//! * `ATLAS_KV_NVME_DIR=<dir>` — enable: evicted prefix-cache blocks are
//!   written to an O_DIRECT record file in `<dir>` instead of being dropped.
//! * `ATLAS_KV_NVME_GB=<GiB>` — REQUIRED with the dir (no implicit default):
//!   the per-rank disk budget; the coldest on-disk blocks are dropped past it.
//!
//! The record file is unlinked right after creation (unix), so its space is
//! returned when the process exits for ANY reason — nothing to clean up after
//! a crash, and no stale file can ever be read by a later process.

use std::path::PathBuf;

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

pub(super) const DIR_VAR: &str = "ATLAS_KV_NVME_DIR";
pub(super) const GB_VAR: &str = "ATLAS_KV_NVME_GB";

#[derive(Debug, PartialEq)]
pub(super) struct NvmeKvConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
}

/// Env-free parse (strict, PCND): `None` when the tier is off.
pub(super) fn config_from(dir: Option<&str>, gb: Option<&str>) -> Result<Option<NvmeKvConfig>> {
    let dir = dir.map(str::trim).filter(|s| !s.is_empty());
    let gb = gb.map(str::trim).filter(|s| !s.is_empty());
    let Some(dir) = dir else {
        if let Some(gb) = gb {
            bail!("{GB_VAR}={gb:?} is set but {DIR_VAR} is not — the budget would be inert");
        }
        return Ok(None);
    };
    let Some(gb) = gb else {
        bail!("{DIR_VAR} is set: {GB_VAR}=<GiB> (per-rank disk budget) is required");
    };
    let v: f64 = gb
        .parse()
        .map_err(|e| anyhow::anyhow!("{GB_VAR}={gb:?} is not a number: {e}"))?;
    ensure!(v.is_finite() && v > 0.0, "{GB_VAR}={gb:?} must be > 0");
    Ok(Some(NvmeKvConfig {
        dir: PathBuf::from(dir),
        budget_bytes: (v * (1u64 << 30) as f64) as u64,
    }))
}

/// Records that fit the budget (refuses a budget below one record).
pub(super) fn max_slots(budget_bytes: u64, record_bytes: usize) -> Result<u32> {
    let n = budget_bytes / record_bytes.max(1) as u64;
    ensure!(
        n >= 1,
        "{GB_VAR}: {budget_bytes} B cannot hold one {record_bytes} B KV block record"
    );
    Ok(n.min(u32::MAX as u64 - 1) as u32)
}

/// What every rank must agree on before serving: the KV tier's geometry and
/// budget plus the SSM tier switch (both gate the rank-agreement collectives
/// in `model/kv_nvme.rs`, so a mismatch would pair them wrongly → deadlock).
fn rank_fingerprint(slots: u32, record_bytes: usize, ssm_tier: bool) -> u64 {
    atlas_tier::hash::mix64(
        atlas_tier::hash::mix64(slots as u64, record_bytes as u64),
        ssm_tier as u64 + 1,
    )
}

/// Attach the tier to a freshly built KV cache (after the sparse index) and
/// enable spill-on-evict in the prefix cache. On a multi-rank world with
/// prefix caching this also verifies — collectively, on EVERY rank, tier on
/// or off — that all ranks run the same spill-tier config.
pub(super) fn attach(
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    comm: Option<&dyn spark_comm::CommBackend>,
) -> Result<()> {
    let cfg = config_from(
        std::env::var(DIR_VAR).ok().as_deref(),
        std::env::var(GB_VAR).ok().as_deref(),
    )?;
    let ssm_tier = std::env::var_os("ATLAS_SSM_TIER").is_some();
    attach_with(cfg, ssm_tier, kv_cache, prefix_cache, gpu, comm)
}

/// Env-free body of [`attach`].
fn attach_with(
    cfg: Option<NvmeKvConfig>,
    ssm_tier: bool,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    comm: Option<&dyn spark_comm::CommBackend>,
) -> Result<()> {
    let record = kv_cache.nvme_record_bytes();
    let slots = match &cfg {
        Some(c) => max_slots(c.budget_bytes, record)?,
        None => 0,
    };
    let rank = comm.map_or(0, |c| c.rank());
    if prefix_cache.is_active()
        && let Some(comm) = comm.filter(|c| c.world_size() > 1)
    {
        let fp = rank_fingerprint(slots, record, ssm_tier);
        let all = super::glm::gather_u64(comm, gpu, fp)?;
        ensure!(
            all.iter().all(|&v| v == fp),
            "spill-tier config differs across ranks (fingerprints {all:x?}); set identical \
             {DIR_VAR}/{GB_VAR}/ATLAS_SSM_TIER on every rank"
        );
    }
    let Some(cfg) = cfg else {
        return Ok(());
    };
    ensure!(
        prefix_cache.is_active(),
        "{DIR_VAR} requires --enable-prefix-caching (it spills evicted prefix-cache blocks)"
    );
    let tier = prefix_cache
        .nvme()
        .ok_or_else(|| anyhow::anyhow!("this prefix cache has no NVMe spill tier"))?;
    std::fs::create_dir_all(&cfg.dir)?;
    let path = cfg.dir.join(format!(
        "atlas-kv-prefix.{}.r{rank}.swap",
        std::process::id()
    ));
    // Records hold prompt-derived KV: owner-only, and never through a
    // pre-planted file or symlink (remove_file drops a link, not its target;
    // create_new refuses anything that reappears).
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let _ = std::fs::remove_file(&path);
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", path.display()))?;
    }
    let store = atlas_tier::DirectSwapFile::create(&path, record)?;
    #[cfg(unix)]
    std::fs::remove_file(&path)?; // anonymous from here on: freed at exit
    kv_cache.attach_nvme_spill(Box::new(store), gpu)?;
    ensure!(tier.enable(slots), "prefix cache NVMe tier already enabled");
    let per_token = record as f64 / kv_cache.block_size() as f64;
    tracing::info!(
        "prefix cache NVMe spill tier ON (rank {rank}): {} ({DIR_VAR}, O_DIRECT, unlinked), \
         {record} B/block record ({per_token:.0} B/token: latent + pooled index, no raw \
         tails), budget {:.1} GiB = {slots} blocks = {} tokens",
        cfg.dir.display(),
        cfg.budget_bytes as f64 / (1u64 << 30) as f64,
        slots as u64 * kv_cache.block_size() as u64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_dir_is_set() {
        assert_eq!(config_from(None, None).unwrap(), None);
        assert_eq!(config_from(Some(" "), None).unwrap(), None);
    }

    #[test]
    fn budget_is_required_and_strict() {
        assert!(config_from(Some("/nvme"), None).is_err());
        assert!(config_from(Some("/nvme"), Some("lots")).is_err());
        assert!(config_from(Some("/nvme"), Some("0")).is_err());
        assert!(config_from(Some("/nvme"), Some("-1")).is_err());
        assert!(config_from(Some("/nvme"), Some("inf")).is_err());
        assert!(
            config_from(None, Some("100")).is_err(),
            "inert budget is an error"
        );
        let c = config_from(Some("/nvme/kv"), Some("1.5")).unwrap().unwrap();
        assert_eq!(c.dir, PathBuf::from("/nvme/kv"));
        assert_eq!(c.budget_bytes, 3 << 29);
    }

    #[test]
    fn slots_floor_the_budget() {
        assert_eq!(max_slots(106_496 * 10 + 5, 106_496).unwrap(), 10);
        assert!(max_slots(4095, 4096).is_err());
    }

    #[test]
    fn fingerprint_separates_every_field() {
        let base = rank_fingerprint(10, 4096, false);
        assert_ne!(base, rank_fingerprint(11, 4096, false));
        assert_ne!(base, rank_fingerprint(10, 8192, false));
        assert_ne!(base, rank_fingerprint(10, 4096, true));
        assert_ne!(
            rank_fingerprint(0, 4096, false),
            rank_fingerprint(0, 4096, true)
        );
    }

    fn glm_kv(gpu: &spark_runtime::gpu::mock::MockGpuBackend) -> PagedKvCache {
        use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, SparseIndexCacheConfig};
        let cfg = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 512,
            num_layers: 2,
            dtype: KvCacheDtype::Fp8G128,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut kv = PagedKvCache::new_with_v_alias(cfg, 4, gpu, true).unwrap();
        kv.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        kv
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("atlas-kv-nvme-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn attach_enables_tier_and_leaves_no_file_behind() {
        use spark_runtime::prefix_cache::NvmePrefixTier;
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let mut kv = glm_kv(&gpu);
        let tree = spark_runtime::radix_tree::RadixTree::new();
        let dir = scratch_dir("on");
        let cfg = NvmeKvConfig {
            dir: dir.clone(),
            budget_bytes: 1 << 30,
        };
        attach_with(Some(cfg), false, &mut kv, &tree, &gpu, None).unwrap();
        assert!(kv.nvme_attached());
        assert!(tree.is_enabled());
        assert_eq!(
            tree.nvme_stats().max_slots as usize,
            (1usize << 30) / kv.nvme_record_bytes()
        );
        #[cfg(unix)]
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "record file unlinked"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn off_is_a_no_op_and_prefix_caching_is_required() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let mut kv = glm_kv(&gpu);
        let tree = spark_runtime::radix_tree::RadixTree::new();
        attach_with(None, true, &mut kv, &tree, &gpu, None).unwrap();
        assert!(!kv.nvme_attached());
        let none = spark_runtime::prefix_cache::NoPrefixCaching;
        let cfg = NvmeKvConfig {
            dir: scratch_dir("nopc"),
            budget_bytes: 1 << 30,
        };
        assert!(attach_with(Some(cfg), false, &mut kv, &none, &gpu, None).is_err());
    }
}
