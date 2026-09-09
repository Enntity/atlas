// SPDX-License-Identifier: AGPL-3.0-only
//! Real Model/owned slots and wire replay; target body is a host sentinel,
//! not KDA numerics (the separate real KDA entry test covers dispatch).
use super::*;
use crate::model::glm_c2_test_support::{
    fixture::{CALLER, Event, Fixture},
    wire::Wire,
};
use crate::traits::{Model, SequenceState};
use atlas_core::scope::ModelResource;
use spark_runtime::buffers::BufferArena;
use spark_runtime::kv_cache::SparseIndexCacheConfig;
use std::sync::Arc;

fn setup(rank: usize) -> (Fixture, Vec<Option<SequenceState>>) {
    let mut f = Fixture::new_legacy(rank);
    f.model.proposer = None;
    f.model.levers.max_decode_seqs = 8;
    let c = &mut f.model.config;
    c.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
    c.linear_num_key_heads = 32;
    c.linear_num_value_heads = 32;
    c.linear_key_head_dim = 128;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.num_attention_heads = 32;
    c.q_lora_rank = 1536;
    c.kv_lora_rank = 512;
    c.head_dim = 256;
    c.num_experts = 288;
    c.num_experts_per_tok = 8;
    c.qk_nope_head_dim = 256;
    c.qk_rope_head_dim = 0;
    c.v_head_dim = 256;
    c.index_topk = 2048;
    c.hc_mult = 4;
    f.model.buffers.release(f.model.gpu.as_ref()).unwrap();
    f.model.buffers = BufferArena::new(c, 8, 2048, 16, 8, f.model.gpu.as_ref()).unwrap();
    f.model.ssm_pool = Arc::new(
        crate::model::ssm_pool::SsmStatePool::new(
            c,
            8,
            false,
            4,
            0,
            false,
            crate::ssm_reserve::SsmRollbackMode::Snapshot,
            f.model.gpu.as_ref(),
        )
        .unwrap(),
    );
    let slots = (0..8)
        .map(|slot| {
            let seq = f.model.alloc_sequence().unwrap();
            assert_eq!(seq.slot_idx, slot);
            Some(seq)
        })
        .collect();
    (f, slots)
}

#[test]
fn actual_environment_exclusions() {
    const CHILD: &str = "ATLAS_INDEPENDENT_ENV_TEST";
    if std::env::var_os(CHILD).is_some() {
        assert!(validate_environment().is_err());
        return;
    }
    for (flag, value) in [
        ("ATLAS_GLM_TARGET_SHARED_FP8", "1"),
        ("ATLAS_GLM_TARGET_SHARED_FP8_VERIFY", "1"),
        ("ATLAS_NVFP4_MMQ_MOE", "TRUE"),
        ("ATLAS_MOE_GROUPED_CUTLASS", "TrUe"),
        ("ATLAS_HOLO_MOE_GROUPED_CUTLASS", "true"),
        ("ATLAS_HOLO_MOE_GROUPED_DOWN", "1"),
    ] {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "model::glm_independent::transport_tests::actual_environment_exclusions",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env(flag, value)
            .env_remove("ATLAS_SSM_H_FP16")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{flag}: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn actual_e0_owned_slots_and_all_drains() {
    const CHILD: &str = "ATLAS_INDEPENDENT_TRANSPORT_TEST";
    if std::env::var_os(CHILD).is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "model::glm_independent::transport_tests::actual_e0_owned_slots_and_all_drains",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_NO_DECODE_GRAPHS_MULTISEQ", "1")
            .env("ATLAS_NO_DECODE_GRAPHS", "1")
            .env("ATLAS_GLM_K5_HC_CUBLAS", "0")
            .env_remove("ATLAS_SSM_H_FP16")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    let (mut head, mut hs) = setup(0);
    let tx = Wire::install(&mut head, 0);
    let (mut worker, mut ws) = setup(1);
    let rx = Wire::install(&mut worker, 1);
    // The missing semantic index refuses before header or slot-ID upload.
    head.gpu.clear();
    let mut refs: Vec<_> = hs.iter_mut().map(|s| s.as_mut().unwrap()).collect();
    let error = head
        .model
        .decode_batch(&[1; 8], &mut refs, CALLER)
        .unwrap_err();
    assert!(error.to_string().contains("semantic index"), "{error:#}");
    assert!(tx.packets().is_empty());
    // Existing outer error cleanup is preserved; no metadata upload/target work.
    assert_eq!(head.gpu.trace(), vec![Event::AbortCapture(7)]);
    drop(refs);
    for f in [&head, &worker] {
        f.model
            .kv_cache
            .lock()
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), f.model.gpu.as_ref())
            .unwrap();
    }
    for rows in (1..=8).rev() {
        tx.clear();
        head.gpu.clear();
        worker.gpu.clear();
        let mut refs: Vec<_> = hs.iter_mut().filter_map(Option::as_mut).collect();
        assert_eq!(refs.len(), rows);
        head.model
            .decode_batch(&vec![1; rows], &mut refs, CALLER)
            .unwrap();
        let packets = tx.packets();
        if rows > 1 {
            assert_eq!(packets[1], vec![0xffffffe0]);
            assert_eq!(packets[2], vec![rows as u32]);
            assert_eq!(
                packets[3],
                ((8 - rows)..8).map(|x| x as u32).collect::<Vec<_>>()
            );
        } else {
            assert_eq!(packets[0], vec![7]);
            assert_eq!(packets[1], vec![1]);
        }
        rx.queue(&packets);
        worker.model.ep_worker_step(&mut ws).unwrap();
        rx.done();
        for slot in (8 - rows)..8 {
            assert_eq!(
                hs[slot].as_ref().unwrap().seq_len,
                ws[slot].as_ref().unwrap().seq_len
            );
            assert_eq!(hs[slot].as_ref().unwrap().slot_idx, slot);
        }
        assert!(
            head.gpu
                .trace()
                .iter()
                .any(|e| matches!(e, Event::Target(..)))
        );
        if rows > 1 {
            let slot = 8 - rows;
            head.model
                .free_sequence(hs[slot].as_mut().unwrap())
                .unwrap();
            hs[slot] = None;
            worker
                .model
                .free_sequence(ws[slot].as_mut().unwrap())
                .unwrap();
            ws[slot] = None;
        }
    }
    assert_eq!(hs[7].as_ref().unwrap().seq_len, 8);
    // Worker checks width before allocating or receiving variable payloads.
    worker.gpu.clear();
    rx.queue(&[vec![0], vec![0xffffffe0], vec![9]]);
    assert!(worker.model.ep_worker_step(&mut ws).is_err());
    rx.done();
    assert!(
        !worker
            .gpu
            .trace()
            .iter()
            .any(|e| matches!(e, Event::Alloc(..) | Event::Target(..)))
    );
}
