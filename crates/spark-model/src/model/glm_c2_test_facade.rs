// SPDX-License-Identifier: AGPL-3.0-only
//! Feature-only construction and observation; callers execute the real Model API.
use super::{fixture as inner, wire};
use crate::layers::glm5_mtp::{Glm5MtpHead, Glm5MtpProposerState};
use crate::model::TransformerModel;
use crate::traits::SequenceState;
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::{Arc, atomic::Ordering};

#[path = "glm_c2_test_events.rs"]
mod events;
pub use events::Event;

/// Fixed bounded model/byte backend. Does not implement Model or mint capabilities.
pub struct Fixture(inner::Fixture);
impl Fixture {
    pub fn paired(rank: usize) -> Self {
        Self(inner::Fixture::new(rank))
    }
    /// Actual fixed-pair target configuration over the bounded byte backend.
    /// This selects the real constructor validation, not a fabricated capability.
    pub fn paired_compute(rank: usize) -> Self {
        let mut fixture = inner::Fixture::new_pair_compute(rank);
        fixture
            .model
            .initialize_glm_pair_verification(crate::layer::glm_pair_verify::GlmPairFfn::TwoK5)
            .expect("actual fixture pair-compute configuration");
        Self(fixture)
    }
    /// Real cold-selected wider compute plus the existing pair fallback.
    pub fn owner_compute(rank: usize) -> Self {
        Self::owner_compute_with_owner_capacity(rank, 4)
    }
    /// Actual explicit-capacity construction; no serving admission is granted.
    pub fn owner_compute_with_owner_capacity(rank: usize, owners: usize) -> Self {
        let mut fixture = inner::Fixture::new_owner_compute_with_owner_capacity(rank, owners);
        fixture
            .model
            .initialize_glm_pair_verification(crate::layer::glm_pair_verify::GlmPairFfn::TwoK5)
            .expect("actual fixture pair fallback configuration");
        fixture
            .model
            .initialize_glm_owner_verification(super::super::glm_owner_wire::Mode::OwnersJoint)
            .expect("actual fixture owner-compute configuration");
        Self(fixture)
    }
    /// Explicit test-only owner residency; no serving admission is granted.
    pub fn paired_compute_with_owner_capacity(rank: usize, owners: usize) -> Self {
        let mut fixture = inner::Fixture::new_pair_compute_with_owner_capacity(rank, owners);
        fixture
            .model
            .initialize_glm_pair_verification(crate::layer::glm_pair_verify::GlmPairFfn::TwoK5)
            .expect("actual fixture bounded-owner pair configuration");
        Self(fixture)
    }
    pub fn legacy(rank: usize) -> Self {
        Self(inner::Fixture::new_legacy(rank))
    }
    pub fn deterministic_logits(&self, enabled: bool) {
        self.0
            .gpu
            .deterministic_logits
            .store(enabled, Ordering::Relaxed);
    }
    /// Borrow actual objects for caller-driven cold setup before E1/F5 wire installation.
    /// This does not execute a producer or qualify worker F0 transport.
    pub fn parts_mut(&mut self) -> (&TransformerModel, &mut [SequenceState; 2]) {
        (&self.0.model, &mut self.0.seqs)
    }
    pub fn install_wire(&mut self) -> Wire {
        let rank = self.0.model.config.tp_rank;
        Wire(wire::Wire::install(&mut self.0, rank))
    }
    /// Moves every genuine owner once. Retirement/teardown remain caller responsibilities.
    pub fn into_parts(self) -> (TransformerModel, [SequenceState; 2], Observer) {
        let inner::Fixture {
            model,
            seqs,
            head,
            gpu,
        } = self.0;
        let backend = backend_id(model.gpu.as_ref());
        (model, seqs, Observer { head, gpu, backend })
    }
}

/// Local packet recorder/replay, not concurrent collective agreement.
pub struct Wire(Arc<wire::Wire>);
impl Wire {
    pub fn enable_cold_prefix(&self) {
        self.0.enable_cold_prefix();
    }
    pub fn roots(&self) -> Vec<usize> {
        self.0.roots()
    }
    pub fn packets(&self) -> Vec<Vec<u32>> {
        self.0.packets()
    }
    pub fn clear(&self) {
        self.0.clear();
    }
    pub fn queue(&self, packets: &[Vec<u32>]) {
        self.0.queue(packets);
    }
    pub fn assert_drained(&self) {
        self.0.done();
    }
    pub fn fail_at(&self, ordinal: usize) {
        assert!(ordinal > 0);
        self.0.fail.store(ordinal, Ordering::Relaxed);
    }
}

/// Opaque saved spans: never constructible from caller-supplied raw pointers.
/// They convey byte observation only, not readiness or permission to resume a lease.
pub struct Snapshot {
    gpu: Arc<inner::Recorder>,
    spans: Vec<(DevicePtr, usize)>,
    initial: Vec<Vec<u8>>,
}
impl Snapshot {
    pub fn initial(&self) -> &[Vec<u8>] {
        &self.initial
    }
}

/// Retaining this handle never grants cleanup authority after Model teardown.
pub struct Observer {
    head: Arc<Glm5MtpHead>,
    gpu: Arc<inner::Recorder>,
    backend: usize,
}
fn backend_id(gpu: &dyn GpuBackend) -> usize {
    gpu as *const dyn GpuBackend as *const () as usize
}
impl Observer {
    pub fn clear(&self) {
        self.gpu.clear();
    }
    pub fn events(&self) -> Vec<Event> {
        self.gpu.trace().into_iter().map(Event::from).collect()
    }
    /// Recorded device-to-host reads only; no memory access or ownership authority.
    pub fn read_spans(&self) -> Vec<(DevicePtr, usize, u64)> {
        self.gpu
            .trace()
            .into_iter()
            .filter_map(|event| match event {
                inner::Event::Read(ptr, bytes, stream) => Some((ptr, bytes, stream)),
                _ => None,
            })
            .collect()
    }
    pub fn fail_at(&self, ordinal: usize) {
        assert!(ordinal > 0);
        self.gpu.fail.store(ordinal, Ordering::Relaxed);
    }
    pub fn private_free_blocks(&self) -> usize {
        self.head.paired_test_free_blocks()
    }
    fn live_rows(
        &self,
        model: &TransformerModel,
        seq: &SequenceState,
        rows: usize,
    ) -> Result<Vec<(DevicePtr, DevicePtr)>> {
        ensure!(
            backend_id(model.gpu.as_ref()) == self.backend,
            "foreign fixture backend"
        );
        self.head.paired_test_kv_rows(
            seq.proposer_state
                .as_deref()
                .context("missing actual private owner")?,
            model.gpu.as_ref(),
            rows,
        )
    }
    pub fn private_cursor(&self, model: &TransformerModel, seq: &SequenceState) -> Result<usize> {
        self.live_rows(model, seq, 0)?;
        Ok(seq
            .proposer_state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("not actual GLM state")?
            .seq_len)
    }
    /// Canonical K/V row pairs, then the fixed whole six-row-per-owner slab.
    /// The final slab span is whole-pool observation, not caller-selected row authority.
    pub fn snapshot(
        &self,
        model: &TransformerModel,
        seq: &SequenceState,
        rows: usize,
    ) -> Result<Snapshot> {
        let mut spans: Vec<_> = self
            .live_rows(model, seq, rows)?
            .into_iter()
            .flat_map(|(k, v)| [(k, 1024), (v, 1024)])
            .collect();
        let owners = crate::speculative::glm_repair::GlmPairedHandoff::owner_capacity(
            self.head.as_ref(),
            model.gpu.as_ref(),
        )?;
        spans.push((
            self.gpu.slab_for_owners(owners),
            owners * 6 * inner::ROW_BYTES,
        ));
        let initial = spans
            .iter()
            .map(|(p, n)| self.gpu.read_live_span(*p, *n))
            .collect::<Result<_>>()?;
        Ok(Snapshot {
            gpu: self.gpu.clone(),
            spans,
            initial,
        })
    }
    /// Post-revocation comparison uses saved spans only; no ownership rediscovery.
    /// Refuses backing already released. Callers drop observers before Model teardown.
    pub fn read_snapshot(&self, saved: &Snapshot) -> Result<Vec<Vec<u8>>> {
        ensure!(
            Arc::ptr_eq(&self.gpu, &saved.gpu),
            "foreign fixture snapshot"
        );
        saved
            .spans
            .iter()
            .map(|(p, n)| self.gpu.read_live_span(*p, *n))
            .collect()
    }
}
