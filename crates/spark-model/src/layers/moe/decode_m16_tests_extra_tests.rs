// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn verify_ffn_launches_the_m16_twins_only_with_the_flag() {
    const SENTINEL: &str = "ATLAS_TEST_DECODE_M16";
    const MISSING: &str = "ATLAS_TEST_DECODE_M16_MISSING";
    if std::env::var_os(SENTINEL).is_none() {
        let name = concat!(
            module_path!(),
            "::verify_ffn_launches_the_m16_twins_only_with_the_flag"
        );
        let name = name.split_once("::").unwrap().1;
        // Flags M16, ZSKIP, K128W, STREAM, MISSING (a lacking kernel), L2PF,
        // PERSIST. Off; M16 with and without the skip, the skip alone, the
        // K128W control, both; the stream twins alone, with M16, the skip,
        // K128W, and on targets lacking a stream kernel; the prefetch twins
        // (1, 2: gate/up only) with and without the skip or stream twins and
        // lacking a prefetching down; the persistent twins with L2PF=1 and the
        // skip, without either or the prefetch twins, alone, and lacking a
        // persistent down. Ignored flags keep what runs without them.
        for mode in [
            "0000000", "1000000", "1100000", "0100000", "0010000", "1010000", "0001000", "1001000",
            "1101000", "1011000", "1001100", "1101200", "1001010", "1101010", "1101020", "1001020",
            "1000010", "0000010", "1101310", "1101011", "1001011", "1101021", "1101001", "0000001",
            "1101411",
        ] {
            let flag = |i: usize| &mode[i..=i];
            let mut cmd = ffn_child(name, SENTINEL, mode);
            cmd.env("ATLAS_GLM_MOE_DECODE_M16", flag(0))
                .env("ATLAS_GLM_MOE_DOWN_ZSKIP", flag(1))
                .env("ATLAS_GLM_MOE_DECODE_K128W", flag(2))
                .env("ATLAS_GLM_MOE_DECODE_STREAM", flag(3))
                .env(MISSING, flag(4))
                .env("ATLAS_GLM_MOE_DECODE_L2PF", flag(5))
                .env("ATLAS_GLM_MOE_DECODE_PERSIST", flag(6))
                .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
                .env("ATLAS_GLM_C3_GROUPED_MOE", "1")
                .env("ATLAS_MOE_PREQUANT_K128", "1")
                // Batches of 9 rows or more otherwise read the expert offsets
                // back for exact M64 tiles; the recording backend serves none.
                .env("ATLAS_MOE_PREFILL_EXACT_TILES", "0");
            assert_child_passed(mode, cmd);
        }
        return;
    }
    // One verify block of 2..=8 rows, 3-row blocks of four owners, and
    // prefill batches of 12, 24 and 40 rows, which only the stream twins take
    // (up to 32 rows; 40 the four-slab persistent pair).
    let batches = (2..=8).map(|rows| (Entry::Independent, rows)).chain([
        (Entry::Owner, 12),
        (Entry::Prefill, 12),
        (Entry::Prefill, 24),
        (Entry::Prefill, 40),
    ]);
    for (entry, rows) in batches {
        for (rank, vector, expert_tp) in [
            (0, false, false),
            (1, true, false),
            (0, true, true),
            (1, false, true),
        ] {
            let gpu = Gpu::new();
            let missing = match std::env::var(MISSING).as_deref() {
                Ok("1") => Some("glm_moe_decode_m32s_gate_up_silu_k128w"),
                Ok("2") => Some("glm_moe_decode_m16s_k128w_zskip"),
                Ok("3") => Some("glm_moe_decode_m32s_k128w_zskip_l2pf"),
                Ok("4") => Some("glm_moe_decode_m32s_k128w_zskip_l2pf_p"),
                _ => None,
            };
            gpu.missing
                .lock()
                .unwrap()
                .extend(missing.map(String::from));
            let trace = verify_ffn(&gpu, rank, vector, expert_tp, entry, rows);
            // Graph-safe either way: no allocation, free or host read.
            assert!(
                !trace
                    .iter()
                    .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..) | Event::Read(..))),
                "{rows} rows"
            );
        }
    }
}
