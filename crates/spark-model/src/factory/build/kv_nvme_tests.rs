// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the NVMe spill tier's construction (`kv_nvme.rs`): env parsing,
//! the rank exchange, attach on both I/O paths and the host-memory reserve.

use super::*;

fn switches(fast: Option<&str>, keep: Option<&str>) -> [(&'static str, Option<String>); 2] {
    [
        (FAST_VAR, fast.map(str::to_owned)),
        (KEEP_VAR, keep.map(str::to_owned)),
    ]
}

#[test]
fn off_unless_dir_is_set() {
    assert_eq!(config_from(None, None, switches(None, None)).unwrap(), None);
    assert_eq!(
        config_from(Some(" "), None, switches(None, None)).unwrap(),
        None
    );
    // The switches alone are inert, not an error (arms share a profile) —
    // and with the tier off they are not read at all, whatever they hold.
    for (fast, keep) in [
        (Some("1"), Some("1")),
        (Some("true"), None),
        (None, Some("x")),
    ] {
        assert_eq!(config_from(None, None, switches(fast, keep)).unwrap(), None);
    }
    // … while an inert BUDGET stays an error, switches or not.
    assert!(config_from(None, Some("40"), switches(Some("1"), None)).is_err());
}

#[test]
fn budget_is_required_and_strict() {
    assert!(config_from(Some("/nvme"), None, switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("lots"), switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("0"), switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("-1"), switches(None, None)).is_err());
    assert!(config_from(Some("/nvme"), Some("inf"), switches(None, None)).is_err());
    assert!(
        config_from(None, Some("100"), switches(None, None)).is_err(),
        "inert budget is an error"
    );
    let c = config_from(Some("/nvme/kv"), Some("1.5"), switches(None, None))
        .unwrap()
        .unwrap();
    assert_eq!(c.dir, PathBuf::from("/nvme/kv"));
    assert_eq!(c.budget_bytes, 3 << 29);
    assert!(!c.fast && !c.keep);
    let on = |fast, keep| config_from(Some("/nvme"), Some("1"), switches(fast, keep));
    let c = on(Some("1"), Some("0")).unwrap().unwrap();
    assert!(c.fast && !c.keep);
    let c = on(None, Some("1")).unwrap().unwrap();
    assert!(!c.fast && c.keep);
    assert!(on(Some("yes"), None).is_err(), "the switches are strict");
    assert!(on(None, Some("true")).is_err());
}

#[test]
fn slots_floor_the_budget() {
    assert_eq!(max_slots(106_496 * 10 + 5, 106_496).unwrap(), 10);
    assert!(max_slots(4095, 4096).is_err());
}

/// `(fast, keep)` for [`rank_fingerprint`].
const OFF: (bool, bool) = (false, false);
const FAST: (bool, bool) = (true, false);
const KEEP: (bool, bool) = (false, true);

#[test]
fn fingerprint_separates_every_field() {
    let base = rank_fingerprint(10, 4096, false, OFF);
    // A pair started with the fast path on one rank only must not start.
    let fast = rank_fingerprint(10, 4096, false, FAST);
    assert_ne!(base, fast);
    assert_ne!(fast, rank_fingerprint(10, 4096, false, KEEP));
    assert_ne!(fast, rank_fingerprint(10, 4096, true, OFF));
    assert_ne!(fast, rank_fingerprint(10, 4096, false, (true, true)));
    assert_ne!(base, rank_fingerprint(11, 4096, false, OFF));
    assert_ne!(base, rank_fingerprint(10, 8192, false, OFF));
    assert_ne!(base, rank_fingerprint(10, 4096, true, OFF));
    assert_ne!(base, rank_fingerprint(10, 4096, false, KEEP));
    assert_ne!(
        rank_fingerprint(10, 4096, true, OFF),
        rank_fingerprint(10, 4096, false, KEEP)
    );
    assert_ne!(
        rank_fingerprint(0, 4096, false, OFF),
        rank_fingerprint(0, 4096, true, OFF)
    );
}

#[test]
fn a_failed_rank_fails_every_rank_after_the_exchange() {
    let fp = rank_fingerprint(10, 4096, false, OFF);
    assert!(verify_ranks(Ok(10), fp, &[fp, fp]).is_ok());
    let own = verify_ranks(
        Err(anyhow::anyhow!("bad env")),
        FAILED_RANK,
        &[FAILED_RANK, fp],
    );
    assert!(format!("{:#}", own.unwrap_err()).contains("bad env"));
    let peer = verify_ranks(Ok(10), fp, &[fp, FAILED_RANK]).unwrap_err();
    assert!(format!("{peer:#}").contains("rank 1"));
    let other = rank_fingerprint(11, 4096, false, OFF);
    assert!(verify_ranks(Ok(10), fp, &[fp, other]).is_err());
    assert_ne!(fp, FAILED_RANK);
}

#[test]
fn config_errors_reach_the_exchange_instead_of_bailing_early() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = glm_kv(&gpu);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    let bad = config_from(Some("/nvme"), None, switches(None, None));
    assert!(bad.is_err());
    // Single rank: the error still surfaces (after the no-op exchange).
    assert!(attach_with(bad, false, &mut kv, &tree, &gpu, None).is_err());
    assert!(!kv.nvme_attached());
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

/// A scratch directory under the cargo target directory this test binary was
/// built into — `<target>/<profile>/deps/<binary>`, so `CARGO_TARGET_DIR` and
/// `build.target-dir` are honoured (a container's `/tmp` is an overlay or
/// tmpfs, which the tier refuses by design).
fn scratch_dir(tag: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let d = exe
        .ancestors()
        .nth(3)
        .expect("test binary under <target>/<profile>/deps")
        .join("atlas-kv-nvme-tests")
        .join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// `None` when even the build tree sits on a filesystem the tier refuses —
/// nothing to test there. Never silently: the skip is printed, and a run
/// that must cover the record file sets `ATLAS_TIER_REQUIRE_O_DIRECT` (as
/// `atlas-tier`'s own tests do), which turns the skip into a failure.
fn disk_scratch_dir(tag: &str) -> Option<PathBuf> {
    let d = scratch_dir(tag);
    std::fs::create_dir_all(&d).unwrap();
    match atlas_tier::unsuitable_swap_fs(&d) {
        None => Some(d),
        Some(kind) => {
            assert!(
                std::env::var_os("ATLAS_TIER_REQUIRE_O_DIRECT").is_none(),
                "ATLAS_TIER_REQUIRE_O_DIRECT set but {} is on {kind}",
                d.display()
            );
            eprintln!(
                "SKIPPED {tag}: the build tree ({}) is on {kind}",
                d.display()
            );
            None
        }
    }
}

#[test]
fn attach_enables_tier_and_leaves_no_file_behind() {
    use spark_runtime::prefix_cache::NvmePrefixTier;
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = glm_kv(&gpu);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    let Some(dir) = disk_scratch_dir("on") else {
        return;
    };
    // A record file left behind by a process that died before unlinking.
    std::fs::write(dir.join("atlas-kv-prefix.4242.r0.swap"), b"stale").unwrap();
    let cfg = NvmeKvConfig {
        dir: dir.clone(),
        budget_bytes: 1 << 30,
        fast: false,
        keep: false,
    };
    attach_with(Ok(Some(cfg)), false, &mut kv, &tree, &gpu, None).unwrap();
    assert!(kv.nvme_attached());
    assert!(!kv.nvme_io_stats().fast);
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
    attach_with(Ok(None), true, &mut kv, &tree, &gpu, None).unwrap();
    assert!(!kv.nvme_attached());
    let none = spark_runtime::prefix_cache::NoPrefixCaching;
    let cfg = NvmeKvConfig {
        dir: scratch_dir("nopc"),
        budget_bytes: 1 << 30,
        fast: false,
        keep: false,
    };
    assert!(attach_with(Ok(Some(cfg)), false, &mut kv, &none, &gpu, None).is_err());
}

#[test]
fn fast_attach_spills_and_restores_through_the_record_file() {
    use spark_runtime::prefix_cache::{DiskRef, SpillOrder};
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = glm_kv(&gpu);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    let Some(dir) = disk_scratch_dir("fast") else {
        return;
    };
    let cfg = NvmeKvConfig {
        dir: dir.clone(),
        budget_bytes: 64 << 20,
        fast: true,
        keep: true,
    };
    attach_with(Ok(Some(cfg)), false, &mut kv, &tree, &gpu, None).unwrap();
    assert!(kv.nvme_io_stats().fast);
    #[cfg(unix)]
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "unlinked");
    // Blocks 2, 1 leave for slots 0, 1 (a leaf-first chain) and come back.
    let pattern: Vec<u8> = (0..kv.k_block_stride_bytes_for_layer(0))
        .map(|i| (i % 253) as u8)
        .collect();
    gpu.copy_h2d(&pattern, kv.k_cache_ptr(0, 1)).unwrap();
    let order = |block, slot| SpillOrder {
        block,
        slot,
        tag: 7 + slot as u64,
    };
    assert!(
        kv.nvme_write(&[order(2, 0), order(1, 1)], &gpu, 0)
            .is_empty()
    );
    let disk = [DiskRef { slot: 1, tag: 8 }, DiskRef { slot: 0, tag: 7 }];
    let mut blocks = [3, 0];
    assert_eq!(kv.nvme_read(&disk, &mut blocks, &gpu, 0), (2, false));
    assert_eq!(blocks, [0, 3]);
    let mut back = vec![0u8; pattern.len()];
    gpu.copy_d2h(kv.k_cache_ptr(0, 0), &mut back).unwrap();
    assert_eq!(back, pattern, "block 1's bytes restored into block 0");
    assert!(kv.nvme_take_failed().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn host_reserve_covers_staging_index_and_the_ssm_tier() {
    use spark_runtime::prefix_cache::NVME_HOST_BYTES_PER_BLOCK;
    let record = 106_496; // GLM-5.3, per rank
    assert_eq!(reserve_for(None, record, 1 << 30), 0, "tier off: nothing");
    let cfg = |fast| NvmeKvConfig {
        dir: PathBuf::from("/nvme"),
        budget_bytes: 24 << 30,
        fast,
        keep: false,
    };
    let slots = (24usize << 30) / record;
    let index = slots * NVME_HOST_BYTES_PER_BLOCK;
    assert_eq!(
        reserve_for(Some(&cfg(false)), record, 0),
        32 * record + index
    );
    assert_eq!(
        reserve_for(Some(&cfg(true)), record, 5000),
        128 * record + index + 5000
    );
    // 24 GiB of records costs about 148 MiB of host RAM for the index alone.
    assert!((140 << 20..150 << 20).contains(&index), "{index}");
    // A budget below one record is refused at attach; the reserve stays 0.
    let tiny = NvmeKvConfig {
        budget_bytes: 100,
        ..cfg(true)
    };
    assert_eq!(reserve_for(Some(&tiny), record, 5000), 0);
}

#[test]
fn an_unusable_directory_is_refused_by_name() {
    let Some(dir) = disk_scratch_dir("usable") else {
        return;
    };
    usable_disk_dir(DIR_VAR, &dir).unwrap();
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "the probe file is removed"
    );
    // A path that cannot be a directory (its parent is a file).
    let file = dir.join("plain-file");
    std::fs::write(&file, b"x").unwrap();
    let e = usable_disk_dir("ATLAS_SSM_TIER_SWAP_DIR", &file.join("sub")).unwrap_err();
    assert!(
        format!("{e:#}").contains("ATLAS_SSM_TIER_SWAP_DIR"),
        "{e:#}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let ro = dir.join("read-only");
        std::fs::create_dir_all(&ro).unwrap();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o500)).unwrap();
        // (root ignores directory modes — the sandbox runs as root.)
        if std::fs::write(ro.join("w"), b"x").is_err() {
            let e = usable_disk_dir(DIR_VAR, &ro).unwrap_err();
            assert!(format!("{e:#}").contains("not usable"), "{e:#}");
        }
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
    // The snapshot tier's directory only counts when that tier would use it.
    assert_eq!(ssm_swap_dir(None), None);
    assert_eq!(ssm_swap_dir(Some(String::new())), None);
}
