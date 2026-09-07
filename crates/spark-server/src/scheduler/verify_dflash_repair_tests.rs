// SPDX-License-Identifier: AGPL-3.0-only

//! Execute the real K5 scheduler entrypoint with a host-only target/proposer.
use super::*;
use spark_model::traits::SequenceState;
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;

struct Target {
    accepted: usize,
    record_error: bool,
    calls: Mutex<Vec<&'static str>>,
}
impl Model for Target {
    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        _: usize,
        _: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!()
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        unreachable!()
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        unreachable!()
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        unreachable!()
    }
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _: u64,
    ) -> Result<Vec<u32>> {
        assert_eq!(tokens, &[7, 10, 11, 12, 13]);
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += tokens.len();
        let mut result = vec![99; 5];
        result[..self.accepted].copy_from_slice(&tokens[1..1 + self.accepted]);
        self.calls.lock().unwrap().push("verify");
        Ok(result)
    }
    fn record_glm_mtp_verified(
        &self,
        seq: &mut SequenceState,
        base: usize,
        tokens: &[u32],
        accepted: usize,
    ) -> Result<()> {
        assert_eq!(base, 3);
        assert_eq!(accepted, self.accepted);
        assert_eq!(seq.seq_len, 4 + accepted);
        assert_eq!(
            seq.tokens,
            [vec![0, 1, 2], tokens[..accepted + 1].to_vec()].concat()
        );
        self.calls.lock().unwrap().push("record");
        if self.record_error {
            anyhow::bail!("injected stale verdict");
        }
        Ok(())
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        unreachable!()
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        position: usize,
        n: usize,
        seq: &mut SequenceState,
        _: u64,
        grammar: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        assert_eq!(position, seq.seq_len);
        assert_eq!(n, 4);
        assert!(grammar.is_none());
        self.calls.lock().unwrap().push("propose");
        Ok(vec![20, 21, 22, 23])
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, a: usize, _: u64) -> Result<()> {
        assert_eq!(a, self.accepted);
        self.calls.lock().unwrap().push("trim");
        Ok(())
    }
    fn save_hidden_for_mtp(&self, row: usize, _: u64) -> Result<()> {
        assert_eq!(row, self.accepted);
        self.calls.lock().unwrap().push("save");
        Ok(())
    }
    fn commit_accepted_prefix(
        &self,
        _: &mut SequenceState,
        accepted: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!((accepted, k), (self.accepted + 1, 5));
        self.calls.lock().unwrap().push("commit");
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        2048
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn copy_logits_to_host(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        unreachable!()
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        unreachable!()
    }
    fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn cache_sequence(&self, _: &SequenceState) {}
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _: &mut SequenceState) {}
}

fn run(accepted: usize, remaining: usize, record_error: bool) -> (ActiveSeq, Vec<&'static str>) {
    let target = Target {
        accepted,
        record_error,
        calls: Mutex::new(Vec::new()),
    };
    let (mut a, _) = crate::scheduler::test_support::test_seq(vec![7], remaining, None, 3);
    a.finished = false;
    a.seq.tokens = vec![0, 1, 2];
    a.min_tokens = 0;
    let mut sched = crate::scheduler::sched_ctx::SchedCtx::for_test();
    sched.limits.max_seq_len = 100;
    let verify_ctx = crate::scheduler::logit_processors::LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        watchdog: sched.watchdog,
        boundary_mask: None,
        mid_word_mask: None,
        sampling: Default::default(),
        timing: sched.timing.clone(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    };
    step_verify_dflash(
        &target,
        &mut a,
        &sched,
        &[10, 11, 12, 13],
        4,
        &verify_ctx,
        true,
    );
    let calls = target.calls.into_inner().unwrap();
    (a, calls)
}

#[test]
fn actual_k5_records_all_verdicts_before_save_trim_and_next_proposal() {
    for accepted in 0..=4 {
        let (a, calls) = run(accepted, 20, false);
        assert!(!a.finished);
        assert_eq!(
            calls,
            ["verify", "record", "commit", "save", "trim", "propose"]
        );
        assert_eq!(a.pending_drafts, [20, 21, 22, 23]);
    }
}

#[test]
fn terminal_or_invalid_verdict_never_reaches_next_proposal() {
    let (a, calls) = run(4, 1, false);
    assert!(a.finished);
    assert_eq!(
        calls,
        ["verify", "record"],
        "record precedes terminal token emission"
    );
    let (a, calls) = run(2, 20, true);
    assert!(a.finished);
    assert_eq!(
        a.output_tokens,
        [7],
        "invalid verdict cannot emit accepted tokens"
    );
    assert_eq!(calls, ["verify", "record"]);
}
