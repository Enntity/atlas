// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn repaired_long_mtp_reserves_tp_local_arena_and_all_dummy_slots() {
    if isolated(
        "long_mtp::repaired_long_mtp_reserves_tp_local_arena_and_all_dummy_slots",
        &[
            ("ATLAS_GLM_INDEPENDENT_DECODE", "0"),
            ("ATLAS_GLM_MTP_REPAIR", "1"),
            ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
            ("ATLAS_GLM_MTP_DISTRIBUTED", "1"),
            ("ATLAS_GLM_MTP_BATCHED_PREFILL", "1"),
            ("ATLAS_MTP_SPEC_THINK", "1"),
            ("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1"),
            ("ATLAS_DISABLE_CUDA_GRAPHS", "1"),
        ],
    ) {
        return;
    }
    for capacity in [1, 4] {
        for rank in 0..2 {
            for chunk in [1024, 2048, 4096] {
                let mut a = args(capacity, rank);
                a.speculative = true;
                a.num_drafts = Some(2);
                a.mtp_gate = Some("force".into());
                a.max_seq_len = 32768;
                a.max_prefill_tokens = chunk;
                a.ssm_cache_slots = 0;
                a.enable_prefix_caching = false;
                let mut cfg = config();
                let (topology, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30)
                    .unwrap_or_else(|e| panic!("long MTP preparation failed: {e:#}"));
                let topology = topology.expect("resolved topology must be retained for handoff");
                assert_eq!((topology.tp_size, topology.ep_size), (2, 2));
                assert_eq!((topology.tp_rank, topology.ep_rank), (rank, rank));
                assert_eq!(
                    (cfg.linear_num_key_heads, cfg.linear_num_value_heads),
                    (32, 32)
                );
                assert_eq!((cfg.num_attention_heads, cfg.num_key_value_heads), (32, 32));
                let rows = chunk + capacity;
                assert_eq!(reserve.max_batch_tokens_pre, rows);
                assert_eq!(
                    reserve.resolved_prefill.as_ref().unwrap().max_batch_tokens,
                    rows
                );
                let h = 34 * 32 * 128 * 128 * 4;
                let conv = 34 * 3 * 32 * 128 * 4 * 4;
                // Actual base dummy plus actual MTP dummy, matching SsmStatePool::new:
                // K3 has two H snapshots, three conv snapshots and one checkpoint.
                let pool = (capacity + 1) * ((h + conv) + (2 * h + 3 * conv + h + conv));
                let gdn = rows * (12288 * 2 + 32 * 2 * 4 + 4096 * 2 * 2);
                let private_and_capture = 8196 * 41984 + 4 * 3 * 4096 * 2 + 32768 * 4096 * 2;
                assert_eq!(reserve.gdn_two_phase_bytes, gdn);
                assert_eq!(
                    reserve.inference_reserve,
                    pool + gdn + private_and_capture + (4usize << 30)
                );
                assert_eq!(
                    reserve.buffer_arena_bytes,
                    spark_runtime::buffers::BufferSizes::from_config(
                        &cfg, rows, 32768, 16, capacity
                    )
                    .total_bytes()
                );
                if capacity == 4 && chunk == 4096 {
                    assert_eq!(pool, 1_593_180_160);
                    assert_eq!(reserve.inference_reserve, 6_669_767_680);
                    // Pre-tokenizer vocab154880 is conservatively24 rows wider than
                    // the actual154856 cap; its96-row logits region adds4608 bytes.
                    assert_eq!(reserve.buffer_arena_bytes, 2_439_854_672);
                }
                let total = reserve.inference_reserve + reserve.buffer_arena_bytes;
                assert!(prepare_reserve(&a, &mut config(), total).is_ok());
                assert!(prepare_reserve(&a, &mut config(), total - 1).is_err());
            }
        }
    }
}
