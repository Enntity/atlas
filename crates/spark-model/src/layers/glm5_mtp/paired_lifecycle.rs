// SPDX-License-Identifier: AGPL-3.0-only
//! Allocation authority stays in the pool; public state block tables are views.
use super::*;

impl Pool {
    pub(super) fn begin_retire(
        &mut self,
        state: &Glm5MtpProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<usize>> {
        self.backend(gpu)?;
        let Some(lease) = state.paired.as_ref() else {
            ensure!(
                state.block_table.is_empty(),
                "retired paired state retains block view"
            );
            return Ok(None);
        };
        ensure!(
            std::sync::Arc::ptr_eq(&lease.owner, &self.identity)
                && lease.slab == self.slab
                && self.capacity.contains(lease.slot)
                && lease.generation == self.slots[lease.slot].generation,
            "foreign paired cleanup lease"
        );
        let index = lease.slot;
        self.slots[index].retiring = true;
        if self.verification.as_ref().is_some_and(|v| v.owns(index)) {
            self.producer_failed = true;
            self.slots[index].failed = true;
        }
        if let Err(error) = self.validate(state, gpu) {
            self.slots[index].failed = true;
            return Err(error);
        }
        Ok(Some(index))
    }
    pub(super) fn backend(&self, gpu: &dyn GpuBackend) -> Result<()> {
        ensure!(
            !self.closed && self.backend == gpu as *const dyn GpuBackend as *const () as usize,
            "paired pool closed or foreign backend"
        );
        Ok(())
    }

    pub(super) fn claim_candidate(&self, gpu: &dyn GpuBackend) -> Result<(usize, u64)> {
        self.backend(gpu)?;
        ensure!(!self.producer_failed, "paired session is terminal");
        let index = self
            .slots
            .iter()
            .position(|s| !s.active && !s.failed)
            .context("paired pool has no reusable request slot")?;
        let slot = &self.slots[index];
        ensure!(slot.blocks.is_empty(), "unreleased paired block reserve");
        let generation = slot
            .generation
            .checked_add(1)
            .context("paired generation exhausted")?;
        Ok((index, generation))
    }

    fn begin_claim(&mut self, gpu: &dyn GpuBackend) -> Result<usize> {
        let (index, generation) = self.claim_candidate(gpu)?;
        let slot = &mut self.slots[index];
        *slot = Slot {
            generation,
            active: true,
            issued_prefix: Vec::with_capacity(self.context),
            ..Slot::default()
        };
        Ok(index)
    }

    pub(super) fn validate(
        &self,
        state: &Glm5MtpProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.backend(gpu)?;
        let lease = state
            .paired
            .as_ref()
            .context("paired state lease missing")?;
        let slot = self.slots.get(lease.slot).context("invalid paired slot")?;
        ensure!(
            std::sync::Arc::ptr_eq(&lease.owner, &self.identity)
                && lease.slab == self.slab
                && lease.generation == slot.generation
                && slot.active
                && !slot.failed,
            "stale or failed paired lease"
        );
        ensure!(
            state.block_table == slot.blocks && slot.blocks.len() == self.blocks_per_slot,
            "paired block view differs from owned reserve"
        );
        Ok(())
    }

    fn reserve(&mut self, index: usize, cache: &mut PagedKvCache) -> Result<Lease> {
        ensure!(!self.closed, "paired pool closed during state allocation");
        let slot = &mut self.slots[index];
        let mut blocks = Vec::with_capacity(self.blocks_per_slot);
        for _ in 0..self.blocks_per_slot {
            match cache.alloc_block() {
                Ok(block) => blocks.push(block),
                Err(error) => {
                    cache.free_blocks(&blocks);
                    slot.failed = true;
                    return Err(error).context("paired full-prefix reserve failed");
                }
            }
        }
        slot.blocks = blocks;
        Ok(Lease {
            owner: self.identity.clone(),
            slot: index,
            generation: slot.generation,
            slab: self.slab,
        })
    }
}

impl Glm5MtpHead {
    pub(in crate::layers::glm5_mtp) fn validate_paired_live(
        &self,
        state: &Glm5MtpProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let Some(owner) = &self.paired else {
            ensure!(state.paired.is_none(), "paired state passed to legacy head");
            return Ok(());
        };
        let pool = owner.lock();
        pool.validate(state, gpu)?;
        let slot = state.paired.as_ref().expect("validated lease").slot;
        ensure!(
            !pool.slots[slot].retiring && !pool.slots[slot].writing,
            "paired lease is retiring or has an incomplete writer"
        );
        Ok(())
    }

    pub(in crate::layers::glm5_mtp) fn free_paired_state(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut Glm5MtpProposerState,
    ) -> Result<()> {
        let owner = self.paired.as_ref().context("paired capability absent")?;
        let mut pool = owner.lock();
        let Some(index) = pool.begin_retire(state, gpu)? else {
            return Ok(());
        };
        drop(pool);
        let completed = gpu.synchronize(gpu.default_stream());
        let mut pool = owner.lock();
        if let Err(error) = completed {
            pool.slots[index].failed = true;
            return Err(error).context("paired cleanup completion failed; slot quarantined");
        }
        pool.validate(state, gpu)?;
        self.kv_cache.lock().free_blocks(&pool.slots[index].blocks);
        let generation = pool.slots[index].generation;
        pool.slots[index] = Slot {
            generation,
            ..Slot::default()
        };
        state.paired = None;
        state.block_table.clear();
        state.seq_len = 0;
        state.last_num_drafted = 0;
        state.repair = repair_state::RepairPhase::Failed;
        state.hidden_trace.reset();
        Ok(())
    }

    pub(in crate::layers::glm5_mtp) fn alloc_paired_state(
        &self,
        gpu: &dyn GpuBackend,
    ) -> Result<Glm5MtpProposerState> {
        let owner = self.paired.as_ref().context("paired capability absent")?;
        let index = owner.lock().begin_claim(gpu)?;
        // Do not expose a token until both body state and the complete reserve
        // exist. A failed body allocation may have enqueued work; quarantine.
        let body_state = match self.module.body.alloc_state(gpu) {
            Ok(state) => state,
            Err(error) => {
                owner.lock().slots[index].failed = true;
                return Err(error).context("paired body state allocation failed");
            }
        };
        let mut pool = owner.lock();
        let lease = pool.reserve(index, &mut self.kv_cache.lock())?;
        let state = Glm5MtpProposerState {
            paired: Some(lease),
            hidden_trace: hidden_trace::HiddenTrace::new(self.hidden_trace_enabled),
            repair: repair_state::RepairPhase::Capture,
            repair_owned: None,
            block_table: pool.slots[index].blocks.clone(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state,
        };
        pool.validate(&state, gpu)?;
        Ok(state)
    }
}
