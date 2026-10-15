// SPDX-License-Identifier: AGPL-3.0-only
// F4 stub self-check — runs without GPU; proves interface + config contract load.
// ponytail: one runnable check, smallest thing that fails if stub breaks.

#[test]
fn f4_contract_stubs_load() {
    // Config contract (in-tree, CPU)
    use atlas_core::config::dsflash_f4::DsflashF4Contract;
    let c = DsflashF4Contract::default();
    assert!(!c.dspark_enabled);
    c.check(32000).expect("default F4 contract must validate");

    let full = DsflashF4Contract {
        dspark_enabled: true, mtp_enabled: true, gamma: 4, verify_k: 3,
        draft_pool_max_seqs: 4, shared_kv_pool: true,
    };
    full.check(32000).expect("F4 DSpark+MTP contract must validate");
}

#[test]
fn f4_server_stub_builds() {
    // Server stub interface only — no loop, no GPU.
    use spark_server::dsflash_f4_stub::DsflashStub;
    let s = DsflashStub::new(4, 3, false);
    assert_eq!(s.verify_step(&[], &[]), 0); // stub returns 0
}
