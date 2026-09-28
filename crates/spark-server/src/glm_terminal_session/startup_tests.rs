// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_glm_pair_wire::{Profile, Recipe, Resources};
use clap::Parser;

fn args() -> ServeArgs {
    ServeArgs::parse_from([
        "spark",
        "--model-from-path",
        "/model",
        "--glm-paired-mtp",
        "--speculative",
        "--num-drafts",
        "4",
        "--max-batch-size",
        "2",
        "--max-num-seqs",
        "2",
        "--max-seq-len",
        "2044",
        "--max-prefill-tokens",
        "1024",
        "--world-size",
        "2",
        "--tp-size",
        "2",
        "--ep-size",
        "2",
        "--kv-cache-dtype",
        "bf16",
        "--mtp-vocab",
        "0",
        "--no-tui",
        "--no-auto-swap",
        "--ssm-h-dtype",
        "f32",
        "--oom-guard-mb",
        "4096",
        "--swap-space-gb",
        "0",
        "--disable-tool-grammar",
        "true",
    ])
}
fn recipe() -> Recipe {
    Recipe {
        argv: vec!["/spark".into()],
        environment: [
            ("ATLAS_GLM_PAIR_FD", "3"),
            ("ATLAS_EP_PROTOCOL", "v2"),
            ("ATLAS_GLM_MTP_DISTRIBUTED", "1"),
            ("ATLAS_GLM_INDEPENDENT_DECODE", "0"),
            ("ATLAS_DFLASH_DEBUG_NO_GRAPH", "1"),
            ("ATLAS_GLM_MTP_REPAIR", "0"),
            ("ATLAS_DFLASH_ADAPTIVE", "0"),
            ("ATLAS_KV_OVERCOMMIT", "0"),
            ("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect(),
        mounts: Vec::new(),
        image_digest: [1; 32],
        guard_elf_digest: [2; 32],
        server_elf_digest: [3; 32],
        rank: 0,
        world: 2,
        profile: Profile {
            tp: 2,
            ep: 2,
            ep_protocol: 2,
            max_sequences: 2,
            context: 2044,
            prefill: 1024,
            drafts: 4,
            eager: true,
            kv_format: 1,
            cold_min: 2,
            cold_max: 1024,
        },
        resources: Resources {
            devices: vec![],
            security_options: vec![],
            memory: 114 << 30,
            swap: 114 << 30,
            cpuset: "0-19".into(),
            shm: 1 << 30,
            device_requests: Vec::new(),
            ulimits: Vec::new(),
            cap_add: Vec::new(),
            cap_drop: Vec::new(),
            uid: 0,
            gid: 0,
            network_mode: "host".into(),
            pid_mode: "private".into(),
            restart_policy: "no".into(),
            init: false,
            no_new_privileges: true,
            ipc_mode: "private".into(),
        },
    }
}

#[test]
fn actual_cli_profile_requires_fixed_pair_and_excluded_routes() {
    let good = args();
    let recipe = recipe();
    validate_args(&good, &recipe).unwrap();
    for mutate in [
        |a: &mut ServeArgs| a.rank = 1,
        |a: &mut ServeArgs| a.max_batch_size = 4,
        |a: &mut ServeArgs| a.num_drafts = Some(3),
        |a: &mut ServeArgs| a.kv_cache_dtype = Some("fp8".into()),
        |a: &mut ServeArgs| a.enable_prefix_caching = true,
        |a: &mut ServeArgs| a.high_speed_swap = true,
        |a: &mut ServeArgs| a.no_tui = false,
        |a: &mut ServeArgs| a.adaptive_sampling = true,
        |a: &mut ServeArgs| a.oom_guard_mb = 1,
    ] {
        let mut bad = good.clone();
        mutate(&mut bad);
        assert!(validate_args(&bad, &recipe).is_err());
    }
    let mut bad = recipe;
    bad.environment
        .retain(|(k, _)| k != "ATLAS_DFLASH_DEBUG_NO_GRAPH");
    assert!(validate_args(&good, &bad).is_err());
}

#[test]
fn actual_profile_requires_prefill_only_and_eager_drafter() {
    let good = args();
    let recipe = recipe();
    validate_args(&good, &recipe).unwrap();
    let mut missing = recipe.clone();
    missing
        .environment
        .retain(|(key, _)| key != "ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE");
    assert!(
        validate_args(&good, &missing).is_err(),
        "default carry mode was admitted"
    );
    for value in ["0", "1"] {
        let mut disabled = recipe.clone();
        disabled
            .environment
            .push(("ATLAS_NO_MTP_EAGER_DRAFTER".into(), value.into()));
        assert!(
            validate_args(&good, &disabled).is_err(),
            "presence-style eager kill switch was admitted"
        );
    }
}
