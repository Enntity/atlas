// SPDX-License-Identifier: AGPL-3.0-only

//! Construction of the prefix cache's NVMe spill tier (default OFF).
//!
//! * `ATLAS_KV_NVME_DIR=<dir>` — enable: evicted prefix-cache blocks are
//!   written to an O_DIRECT record file in `<dir>` instead of being dropped.
//! * `ATLAS_KV_NVME_GB=<GiB>` — REQUIRED with the dir (no implicit default):
//!   the per-rank disk budget; the coldest on-disk blocks are dropped past it.
//! * `ATLAS_GLM_NVME_FAST=1` — move the same records through the fast path
//!   (`kv_cache/nvme_fast.rs`: pitched copies, run-sized I/O, write-behind).
//! * `ATLAS_GLM_NVME_KEEP=1` — a restored block keeps its record, so evicting
//!   it again writes nothing (`NvmePrefixTier::set_keep_restored`).
//!
//! The record file is unlinked right after creation (unix), so its space is
//! returned when the process exits for ANY reason — nothing to clean up after
//! a crash, and no stale file can ever be read by a later process. A file a
//! process left behind by dying before that unlink is swept at the next start.

use std::path::PathBuf;

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

pub(super) const DIR_VAR: &str = "ATLAS_KV_NVME_DIR";
pub(super) const GB_VAR: &str = "ATLAS_KV_NVME_GB";
pub(super) const FAST_VAR: &str = "ATLAS_GLM_NVME_FAST";
pub(super) const KEEP_VAR: &str = "ATLAS_GLM_NVME_KEEP";
/// Record files are `<prefix><pid>.r<rank>.swap`.
const FILE_PREFIX: &str = "atlas-kv-prefix.";

#[derive(Debug, PartialEq)]
pub(super) struct NvmeKvConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub fast: bool,
    pub keep: bool,
}

/// The tier's configuration from the environment.
fn config_from_env() -> Result<Option<NvmeKvConfig>> {
    let var = |name| std::env::var(name).ok();
    config_from(
        var(DIR_VAR).as_deref(),
        var(GB_VAR).as_deref(),
        [(FAST_VAR, var(FAST_VAR)), (KEEP_VAR, var(KEEP_VAR))],
    )
}

/// A strict `=1` / `=0` switch (unset or empty = off).
fn switch((name, value): &(&str, Option<String>)) -> Result<bool> {
    match value.as_deref().map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name}={other:?}: expected 1 (on) or 0 (off)"),
    }
}

/// Env-free parse (strict, PCND): `None` when the tier is off.
pub(super) fn config_from(
    dir: Option<&str>,
    gb: Option<&str>,
    [fast, keep]: [(&str, Option<String>); 2],
) -> Result<Option<NvmeKvConfig>> {
    let dir = dir.map(str::trim).filter(|s| !s.is_empty());
    let gb = gb.map(str::trim).filter(|s| !s.is_empty());
    let (fast, keep) = (switch(&fast)?, switch(&keep)?);
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
        fast,
        keep,
    }))
}

/// Host memory the spill tiers take AFTER the KV pool is sized, which
/// `build_model` therefore leaves out of the pool (0 with the KV tier off):
///
/// * the pinned staging (`PagedKvCache::nvme_staging_bytes`);
/// * the tree's index for a FULL disk budget — every on-disk block keeps its
///   radix node (`NVME_HOST_BYTES_PER_BLOCK`), so the budget in GiB is also a
///   budget in host RAM;
/// * `ssm_lazy_bytes`: what the SSM snapshot tier commits on first use.
///
/// Never fails: a bad configuration is reported by [`attach`], after the rank
/// exchange (an early error here would leave the peers in a collective).
pub(super) fn host_reserve_bytes(record_bytes: usize, ssm_lazy_bytes: usize) -> usize {
    let cfg = config_from_env().ok().flatten();
    let reserve = reserve_for(cfg.as_ref(), record_bytes, ssm_lazy_bytes);
    if reserve > 0 {
        tracing::info!(
            "NVMe spill tiers: reserving {:.1} MiB of host memory out of the KV budget \
             (staging + on-disk index at a full {GB_VAR} + {:.1} MiB SSM tier arena/staging)",
            reserve as f64 / (1u64 << 20) as f64,
            ssm_lazy_bytes as f64 / (1u64 << 20) as f64,
        );
    }
    reserve
}

fn reserve_for(cfg: Option<&NvmeKvConfig>, record_bytes: usize, ssm_lazy_bytes: usize) -> usize {
    let Some(cfg) = cfg else {
        return 0;
    };
    let Ok(slots) = max_slots(cfg.budget_bytes, record_bytes) else {
        return 0;
    };
    PagedKvCache::nvme_staging_bytes(record_bytes, cfg.fast)
        + slots as usize * spark_runtime::prefix_cache::NVME_HOST_BYTES_PER_BLOCK
        + ssm_lazy_bytes
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
/// in `model/kv_nvme.rs`, so a mismatch would pair them wrongly → deadlock),
/// and whether restored records are kept (it changes which blocks the budget
/// drops, so the ranks' trees would drift apart).
fn rank_fingerprint(slots: u32, record_bytes: usize, ssm_tier: bool, keep: bool) -> u64 {
    atlas_tier::hash::mix64(
        atlas_tier::hash::mix64(slots as u64, record_bytes as u64),
        ssm_tier as u64 + 1 + 2 * keep as u64,
    )
}

/// Fingerprint a rank sends when its OWN setup failed: never equal to a real
/// one, so every peer fails the check too instead of waiting on this rank.
const FAILED_RANK: u64 = u64::MAX;

/// Attach the tier to a freshly built KV cache (after the sparse index) and
/// enable spill-on-evict in the prefix cache. On a multi-rank world this also
/// verifies — collectively, on EVERY rank, tier on or off — that all ranks run
/// the same spill-tier config and that none failed to set it up.
pub(super) fn attach(
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    comm: Option<&dyn spark_comm::CommBackend>,
) -> Result<()> {
    let cfg = config_from_env();
    let ssm_tier = std::env::var_os("ATLAS_SSM_TIER").is_some();
    attach_with(cfg, ssm_tier, kv_cache, prefix_cache, gpu, comm)
}

/// Env-free body of [`attach`]. Everything rank-local that can fail runs
/// FIRST, then the ranks exchange fingerprints (a failed rank sends
/// [`FAILED_RANK`]), and only then does any rank return an error — so one
/// misconfigured rank (bad env, unwritable dir, full disk) fails every rank
/// at startup instead of leaving its peers blocked in a later collective.
fn attach_with(
    cfg: Result<Option<NvmeKvConfig>>,
    ssm_tier: bool,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    comm: Option<&dyn spark_comm::CommBackend>,
) -> Result<()> {
    let rank = comm.map_or(0, |c| c.rank());
    let record = kv_cache.nvme_record_bytes();
    let keep = matches!(&cfg, Ok(Some(c)) if c.keep);
    let local = cfg.and_then(|cfg| setup_local(cfg, rank, record, kv_cache, prefix_cache, gpu));
    // Every multi-rank world exchanges (one 8-byte all-gather at startup),
    // prefix caching or not: the condition must not depend on anything that
    // can differ per rank, or the collective itself would pair up wrongly.
    if let Some(comm) = comm.filter(|c| c.world_size() > 1) {
        let fp = match &local {
            Ok(slots) => rank_fingerprint(*slots, record, ssm_tier, keep),
            Err(_) => FAILED_RANK,
        };
        let all = super::glm::gather_u64(comm, gpu, fp)?;
        return verify_ranks(local, fp, &all);
    }
    local.map(|_| ())
}

/// After the fingerprint exchange: this rank's own error first, then any
/// peer that failed, then any config mismatch.
fn verify_ranks(local: Result<u32>, fp: u64, all: &[u64]) -> Result<()> {
    local?;
    if let Some(bad) = all.iter().position(|&v| v == FAILED_RANK) {
        bail!("spill-tier setup failed on rank {bad} (see its log); every rank stops");
    }
    ensure!(
        all.iter().all(|&v| v == fp),
        "spill-tier config differs across ranks (fingerprints {all:x?}); set identical \
         {DIR_VAR}/{GB_VAR}/ATLAS_SSM_TIER on every rank"
    );
    Ok(())
}

/// The snapshot tier's swap directory, when it is configured to use one.
fn ssm_swap_dir(dir: Option<String>) -> Option<PathBuf> {
    let on =
        std::env::var_os("ATLAS_SSM_TIER").is_some() && crate::model::ssm_tier::ssm_tier_unified();
    dir.filter(|d| on && !d.is_empty()).map(PathBuf::from)
}

/// `dir` (created if missing) is on a real disk and this process can create
/// files in it — checked with a throwaway file, so a read-only or foreign
/// mount fails here, by name, instead of inside a tier.
fn usable_disk_dir(var: &str, dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("{var}={}: cannot create: {e}", dir.display()))?;
    if let Some(kind) = atlas_tier::unsuitable_swap_fs(dir) {
        bail!(
            "{var}={} is on {kind}; bind-mount a directory of the node's NVMe instead",
            dir.display()
        );
    }
    let probe = dir.join(format!("atlas-dir-probe.{}.swap", std::process::id()));
    let _ = std::fs::remove_file(&probe);
    let made = atlas_tier::SharedRecordFile::create(&probe, 4096);
    let _ = std::fs::remove_file(&probe);
    made.map(|_| ())
        .map_err(|e| e.context(format!("{var}={} is not usable", dir.display())))
}

/// The rank-local part of [`attach_with`]: validate, create the record file,
/// attach it and enable the tree side. Returns the slot budget (0 = off).
fn setup_local(
    cfg: Option<NvmeKvConfig>,
    rank: usize,
    record: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
) -> Result<u32> {
    let Some(cfg) = cfg else {
        return Ok(0);
    };
    let slots = max_slots(cfg.budget_bytes, record)?;
    ensure!(
        prefix_cache.is_active(),
        "{DIR_VAR} requires --enable-prefix-caching (it spills evicted prefix-cache blocks)"
    );
    let tier = prefix_cache
        .nvme()
        .ok_or_else(|| anyhow::anyhow!("this prefix cache has no NVMe spill tier"))?;
    usable_disk_dir(DIR_VAR, &cfg.dir)?;
    if let Some(dir) = ssm_swap_dir(std::env::var("ATLAS_SSM_TIER_SWAP_DIR").ok()) {
        // The snapshot tier falls back to HOST RAM when its directory is
        // unusable; with the KV tier on, that is a misconfigured install.
        usable_disk_dir("ATLAS_SSM_TIER_SWAP_DIR", &dir)?;
    }
    let stale = atlas_tier::remove_stale_swap_files(&cfg.dir, FILE_PREFIX);
    if stale > 0 {
        tracing::info!(
            "prefix cache NVMe spill tier: removed {stale} stale record file(s) from {}",
            cfg.dir.display()
        );
    }
    let path = cfg
        .dir
        .join(format!("{FILE_PREFIX}{}.r{rank}.swap", std::process::id()));
    // Records hold prompt-derived KV: owner-only, and never through a
    // pre-planted file or symlink (remove_file drops a link, not its target;
    // an exclusive create refuses anything that reappears). ONE open per
    // path: nothing re-opens the name after another process could have
    // swept it.
    let _ = std::fs::remove_file(&path);
    if cfg.fast {
        let store = atlas_tier::SharedRecordFile::create(&path, record)?;
        // The whole budget, now: a disk that cannot hold it fails the start
        // (every rank stops) instead of the spills later, and writers never
        // serialise on block allocation.
        let reserved = store
            .reserve(slots as u64 * record as u64)
            .map_err(|e| e.context(format!("{GB_VAR}: the disk budget does not fit {DIR_VAR}")));
        #[cfg(unix)]
        let _ = std::fs::remove_file(&path);
        if !reserved? {
            tracing::warn!(
                "prefix cache NVMe spill tier: this filesystem cannot reserve space; the record \
                 file grows on demand (slower concurrent writes, and it can run out of disk)"
            );
        }
        kv_cache.attach_nvme_fast(std::sync::Arc::new(store), gpu)?;
    } else {
        let store = atlas_tier::DirectSwapFile::create_new(&path, record)?;
        kv_cache.attach_nvme_spill(Box::new(store), gpu)?;
    }
    // Anonymous from here on: freed at exit. (Another process's stale sweep
    // may have unlinked it first — the descriptor is what matters.)
    #[cfg(unix)]
    let _ = std::fs::remove_file(&path);
    ensure!(tier.enable(slots), "prefix cache NVMe tier already enabled");
    tier.set_keep_restored(cfg.keep);
    let per_token = record as f64 / kv_cache.block_size() as f64;
    let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
    tracing::info!(
        "prefix cache NVMe spill tier ON (rank {rank}): {} ({DIR_VAR}, O_DIRECT, unlinked), \
         {record} B/block record ({per_token:.0} B/token: latent + pooled index, no raw \
         tails), budget {:.1} GiB = {slots} blocks = {} tokens; {} I/O ({FAST_VAR}), \
         restored records {} ({KEEP_VAR}), {:.1} MiB pinned staging, up to {:.1} MiB host index",
        cfg.dir.display(),
        cfg.budget_bytes as f64 / (1u64 << 30) as f64,
        slots as u64 * kv_cache.block_size() as u64,
        if cfg.fast { "fast" } else { "synchronous" },
        if cfg.keep { "kept" } else { "released" },
        mib(PagedKvCache::nvme_staging_bytes(record, cfg.fast)),
        mib(slots as usize * spark_runtime::prefix_cache::NVME_HOST_BYTES_PER_BLOCK),
    );
    Ok(slots)
}

#[cfg(test)]
#[path = "kv_nvme_tests.rs"]
mod tests;
