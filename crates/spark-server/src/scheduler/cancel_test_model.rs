// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only logits provider; every compute entry point is deliberately rejected.
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(super) struct TestModel {
    pub tokens: Vec<u32>,
    pub host_logits: bool,
    pub cancel_after_sampling: Option<Arc<AtomicBool>>,
    pub cancel_after_row_commit: Option<Arc<AtomicBool>>,
}

impl Model for TestModel {
    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
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
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
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
    fn decode_logits_fp32(&self) -> bool {
        self.host_logits
    }
    fn copy_logits_to_host(&self, _: DevicePtr, dst: &mut [u8]) -> Result<()> {
        assert!(self.host_logits);
        for (i, bytes) in dst.chunks_exact_mut(4).enumerate() {
            let value = if i % 2048 == 101 { 5.0f32 } else { 0.0f32 };
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        anyhow::bail!("unexpected scalar argmax in cancellation test")
    }
    fn argmax_batch(&self, _: DevicePtr, n: usize, _: u64) -> Result<Vec<u32>> {
        assert_eq!(n, self.tokens.len());
        if let Some(flag) = &self.cancel_after_sampling {
            flag.store(true, Ordering::Release);
        }
        Ok(self.tokens.clone())
    }
    fn decode_marconi_checkpoint(&self, _: &mut SequenceState) {
        if let Some(flag) = &self.cancel_after_row_commit {
            flag.store(true, Ordering::Release);
        }
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
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn cache_sequence(&self, _: &SequenceState) {}
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
}
