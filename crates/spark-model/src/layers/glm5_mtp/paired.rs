// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit staged paired-head construction; no factory/admission caller.
use super::*;
use anyhow::{Context, ensure};
#[cfg(any(test, feature = "glm-c2-test-utils"))]
#[path = "paired_test_inspection.rs"]
mod inspection;

pub(super) const ROW_BYTES: usize = 8192;
pub(super) const SLOT_BYTES: usize = 6 * ROW_BYTES;
#[cfg(test)]
pub(super) const SLAB_BYTES: usize = 2 * SLOT_BYTES;

/// Checked once before private cache/slab allocation; not serving admission.
#[derive(Clone, Copy)]
pub(super) struct OwnerCapacity(usize);

impl OwnerCapacity {
    pub(super) fn new(owners: usize) -> Result<Self> {
        ensure!(
            (2..=4).contains(&owners),
            "paired owner capacity must be2..4"
        );
        Ok(Self(owners))
    }

    pub(super) fn owners(self) -> usize {
        self.0
    }

    pub(super) fn contains(self, slot: usize) -> bool {
        slot < self.0
    }

    pub(super) fn slab_bytes(self) -> usize {
        // The private constructor bounds owners at four; SLOT_BYTES is fixed.
        self.0 * SLOT_BYTES
    }

    pub(super) fn cache_blocks(self, context: usize) -> Result<usize> {
        blocks_per_slot(context)?
            .checked_mul(self.0)
            .context("paired KV block count overflow")
    }
}

#[path = "paired_bootstrap.rs"]
mod bootstrap;
#[path = "paired_close.rs"]
mod close;
#[path = "paired_joint.rs"]
mod joint;
#[path = "paired_lifecycle.rs"]
mod lifecycle;
#[path = "paired_predispatch.rs"]
mod predispatch;
#[path = "paired_prime.rs"]
mod prime;
#[path = "paired_repair.rs"]
mod repair;
#[path = "paired_target.rs"]
mod target;
#[path = "paired_verdict.rs"]
mod verdict;
#[path = "paired_verify.rs"]
mod verify;

#[derive(Clone, Copy, PartialEq, Eq)]
struct IssuedProposal {
    attempt: u64,
    base: usize,
    tokens: [u32; 5],
}
#[derive(Clone, Copy)]
struct Verification {
    slot: usize,
    generation: u64,
    issued: IssuedProposal,
    normalized: DevicePtr,
    produced: bool,
}

enum Producer {
    Single(Verification),
    Pair(PairVerification),
}
impl Producer {
    fn owns(&self, index: usize) -> bool {
        match self {
            Self::Single(record) => record.slot == index,
            Self::Pair(pair) => pair.records.iter().any(|record| record.slot == index),
        }
    }
}
#[derive(Clone, Copy)]
struct PairVerification {
    records: [Verification; 2],
    normalized_bytes: usize,
    detached: [bool; 2],
    committed: [bool; 2],
}

impl PairVerification {
    fn ordinal(&self, physical_slot: usize) -> Result<usize> {
        self.records
            .iter()
            .position(|record| record.slot == physical_slot)
            .context("physical owner does not belong to active Pair producer")
    }
}

struct Binding {
    sequence_slot: usize,
    capture_generation: u64,
    prompt: usize,
    capture: DevicePtr,
    normalized: DevicePtr,
}
struct HiddenView {
    generation: u64,
    position: usize,
    row: usize,
    rows: usize,
}

/// Issued only by the actual pool allocator; never inferred from a row pointer.
pub(super) struct Lease {
    owner: std::sync::Arc<()>,
    slot: usize,
    generation: u64,
    slab: DevicePtr,
}

#[derive(Default)]
struct Slot {
    generation: u64,
    active: bool,
    failed: bool,
    retiring: bool,
    writing: bool,
    binding: Option<Binding>,
    tail: Option<HiddenView>,
    bonus: Option<HiddenView>,
    pending_target: Option<u32>,
    proposing: bool,
    attempt: u64,
    issued: Option<IssuedProposal>,
    issued_prefix: Vec<u32>,
    commit_queued: bool,
    blocks: Vec<u32>,
}

pub(super) fn blocks_per_slot(context: usize) -> Result<usize> {
    let rows = context.checked_add(4).context("paired context overflow")?;
    ensure!(
        context > 1 && rows <= 2048,
        "paired context plus four drafts exceeds2048"
    );
    Ok(rows.div_ceil(16))
}

/// Model-owned storage. Lease/read lifecycle is private to the paired capability.
pub(super) struct Pool {
    identity: std::sync::Arc<()>,
    slab: DevicePtr,
    backend: usize,
    rank: usize,
    context: usize,
    blocks_per_slot: usize,
    capacity: OwnerCapacity,
    slots: Box<[Slot]>,
    closed: bool,
    close_failed: bool,
    verification: Option<Producer>,
    producer_failed: bool,
}
impl Pool {
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        context: usize,
        cache: &PagedKvCache,
        rank: usize,
        capacity: OwnerCapacity,
    ) -> Result<Self> {
        let blocks = blocks_per_slot(context)?;
        ensure!(
            cache.num_blocks() == capacity.cache_blocks(context)? && cache.block_size() == 16,
            "paired cache must own one complete reserve per admitted owner"
        );
        let cache_spans = kv_rows::cache_spans(cache)?;
        let slab = gpu.alloc(capacity.slab_bytes())?;
        let span = kv_rows_plan::DeviceSpan {
            ptr: slab,
            bytes: capacity.slab_bytes(),
        };
        // A backend alias is not a newly owned allocation, even when the
        // returned interior pointer is malformed. Never free that old owner.
        for owner in cache_spans {
            let ptr = owner.ptr;
            let end = owner.end()?;
            let contains_start = slab.0 >= ptr.0 && slab.0 < end;
            let intersects = span
                .end()
                .is_ok_and(|slab_end| slab.0 < end && ptr.0 < slab_end);
            ensure!(
                !contains_start && !intersects,
                "paired slab aliases actual KV; existing owner is not cleanup authority"
            );
        }
        let valid = (|| {
            ensure!(
                !slab.is_null() && slab.0.is_multiple_of(2),
                "invalid paired slab address"
            );
            span.end()?;
            Ok(())
        })();
        if let Err(error) = valid {
            let cleanup = gpu.free(slab);
            return Err(error).context(format!("paired slab rejected; cleanup={cleanup:?}"));
        }
        Ok(Self {
            identity: std::sync::Arc::new(()),
            slab,
            backend: gpu as *const dyn GpuBackend as *const () as usize,
            rank,
            context,
            blocks_per_slot: blocks,
            capacity,
            slots: (0..capacity.owners())
                .map(|_| Slot::default())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            closed: false,
            close_failed: false,
            verification: None,
            producer_failed: false,
        })
    }
}

impl Glm5MtpHead {
    /// Membership only: callers must also validate actual live leases and capacity.
    pub(crate) fn validate_fixed_pair_slots(slots: [usize; 2], capacity: usize) -> Result<()> {
        let capacity = OwnerCapacity::new(capacity)?;
        ensure!(
            matches!(slots, [0, 1] | [2, 3])
                && slots.into_iter().all(|slot| capacity.contains(slot)),
            "fixed pair requires a complete canonical physical owner group"
        );
        Ok(())
    }

    /// Explicit cold owner capacity; serving still calls the fixed-two wrapper.
    /// Capacity expansion is staged separately from any pair mapping or E6 admission.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn new_paired_with_owner_capacity(
        module: Glm5MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        context_tokens: usize,
        owners: usize,
    ) -> Result<Self> {
        let capacity = OwnerCapacity::new(owners)?;
        blocks_per_slot(context_tokens)?;
        ensure!(
            config.model_type == "glm5_next"
                && config.hidden_size == 4096
                && config.kv_lora_rank == 512
                && config.qk_rope_head_dim == 0
                && config.tp_world_size == 2
                && config.ep_world_size == 2
                && config.tp_rank == config.ep_rank
                && config.ep_rank < 2
                && config.adapter_max_rank == 0,
            "paired head requires base GLM4096 NoPE512 TP2/EP2"
        );
        Self::new_with_capacity(
            module,
            embed_tokens,
            lm_head,
            lm_head_nvfp4,
            config,
            gpu,
            mtp_vocab_size,
            context_tokens,
            Some((context_tokens, capacity)),
        )
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn new_paired(
        module: Glm5MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        context_tokens: usize,
    ) -> Result<Self> {
        Self::new_paired_with_owner_capacity(
            module,
            embed_tokens,
            lm_head,
            lm_head_nvfp4,
            config,
            gpu,
            mtp_vocab_size,
            context_tokens,
            2,
        )
    }
}

#[cfg(test)]
#[path = "paired_capacity_tests.rs"]
mod capacity_tests;
