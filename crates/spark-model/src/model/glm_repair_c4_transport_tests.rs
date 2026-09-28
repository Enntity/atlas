// SPDX-License-Identifier: AGPL-3.0-only
//! Actual F0/E1/F5 request adapters and retained private rows on both ranks.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::traits::Model;
use std::sync::atomic::Ordering;

#[test]
fn two_repaired_owners_replay_prefill_proposal_and_k3_verdict() {
    if std::env::var("ATLAS_REPAIR_C4_TEST_CHILD").as_deref() != Ok("1") {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "model::glm_c2_handoff_tests::repair_c4_transport_tests::two_repaired_owners_replay_prefill_proposal_and_k3_verdict", "--nocapture"])
            .env("ATLAS_REPAIR_C4_TEST_CHILD", "1")
            .env("ATLAS_GLM_MTP_REPAIR", "1")
            .env("ATLAS_GLM_MTP_LONG_CONTEXT", "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
            .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
            .env("ATLAS_MTP_SPEC_THINK", "1")
            .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1")
            .env("ATLAS_GLM_MTP_ALL_GATHER", "1")
            .env("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0")
            .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
            .env_remove("ATLAS_NO_MTP_EAGER_DRAFTER")
            .env_remove("ATLAS_GLM_MTP_FUSED_EH_NORM")
            .status().unwrap();
        assert!(status.success());
        return;
    }
    let mut head = Fixture::new_repaired_c4(0);
    let mut peer = Fixture::new_repaired_c4(1);
    for f in [&mut head, &mut peer] {
        f.gpu.deterministic_logits.store(true, Ordering::Relaxed);
    }
    let tx = wire::Wire::install(&mut head, 0);
    let rx = wire::Wire::install(&mut peer, 1);
    tx.enable_cold_prefix();
    rx.enable_cold_prefix();
    for (owner, prompt) in [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]]
        .iter()
        .enumerate()
    {
        head.seqs[owner].prompt_len = prompt.len();
        tx.clear();
        head.model
            .ep_broadcast_cmd_for_seq(owner as u32, 0xfffffff0)
            .unwrap();
        for word in [prompt.len() as u32, 0, prompt.len() as u32] {
            head.model.ep_broadcast_cmd(word).unwrap();
        }
        head.model.ep_broadcast_tokens(prompt).unwrap();
        head.model
            .prefill_chunk(prompt, &mut head.seqs[owner], 0, prompt.len(), true, CALLER)
            .unwrap();
        rx.queue(&tx.packets());
        assert!(wire::worker(&mut peer).unwrap());
        rx.done();
        assert_eq!(flow::private(&head.seqs[owner]).seq_len, prompt.len() - 1);
        assert_eq!(flow::private(&peer.seqs[owner]).seq_len, prompt.len() - 1);
    }
    for owner in 0..2 {
        tx.clear();
        head.model
            .ep_broadcast_cmd_for_seq(owner as u32, 5)
            .unwrap();
        head.model.decode(5, &mut head.seqs[owner], CALLER).unwrap();
        rx.queue(&tx.packets());
        assert!(wire::worker(&mut peer).unwrap());
        rx.done();
        tx.clear();
        head.model.save_hidden_for_mtp(0, CALLER).unwrap();
        let position = head.seqs[owner].seq_len;
        let drafts = head
            .model
            .run_mtp_propose_multi(6, position, 2, &mut head.seqs[owner], CALLER, None)
            .unwrap();
        rx.queue(&tx.packets());
        assert!(wire::worker(&mut peer).unwrap());
        rx.done();
        let mut tokens = vec![6];
        tokens.extend(drafts);
        assert_eq!(tokens.len(), 3);
        tx.clear();
        head.model
            .ep_broadcast_cmd_for_seq(owner as u32, 0xfffffff5)
            .unwrap();
        head.model.ep_broadcast_cmd(3).unwrap();
        head.model.ep_broadcast_tokens(&tokens).unwrap();
        head.model
            .decode_verify_dflash(&tokens, &mut head.seqs[owner], CALLER)
            .unwrap();
        head.model.ep_broadcast_cmd(2).unwrap();
        head.model
            .record_glm_mtp_verified(&mut head.seqs[owner], position, &tokens, 2)
            .unwrap();
        head.model
            .trim_proposer_state(&mut head.seqs[owner], 2, CALLER)
            .unwrap();
        head.model
            .commit_accepted_prefix(&mut head.seqs[owner], 3, 3)
            .unwrap();
        rx.queue(&tx.packets());
        assert!(wire::worker(&mut peer).unwrap());
        rx.done();
        assert_eq!(head.seqs[owner].tokens, peer.seqs[owner].tokens);
        tx.clear();
        head.model.save_hidden_for_mtp(2, CALLER).unwrap();
        let position = head.seqs[owner].seq_len;
        head.model
            .run_mtp_propose_multi(7, position, 2, &mut head.seqs[owner], CALLER, None)
            .unwrap();
        rx.queue(&tx.packets());
        assert!(wire::worker(&mut peer).unwrap());
        rx.done();
        assert_eq!(
            flow::private(&head.seqs[owner]).seq_len,
            flow::private(&peer.seqs[owner]).seq_len
        );
        assert_eq!(
            flow::private(&head.seqs[owner]).block_table,
            flow::private(&peer.seqs[owner]).block_table
        );
    }
    for f in [&mut head, &mut peer] {
        for seq in &mut f.seqs {
            f.model.free_sequence(seq).unwrap();
        }
    }
}
