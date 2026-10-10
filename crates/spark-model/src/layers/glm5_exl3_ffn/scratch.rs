// SPDX-License-Identifier: AGPL-3.0-only
//! One model/stream-owned routed workspace; layers borrow without reclaiming it.
use super::*;
use parking_lot::{Mutex, MutexGuard};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeScratchGeometry {
    pub(super) hidden: usize,
    intermediate: usize,
    experts: usize,
    topk: usize,
    capacity: usize,
    stream: u64,
    rank: usize,
    world: usize,
}
impl MoeScratchGeometry {
    pub fn new(c: &ModelConfig, capacity: usize, stream: u64) -> Result<Self> {
        ensure!(
            c.tp_world_size > 0
                && c.tp_rank < c.tp_world_size
                && c.moe_intermediate_size.is_multiple_of(c.tp_world_size),
            "EXL3 scratch requires valid TP geometry"
        );
        u32::try_from(c.hidden_size).context("EXL3 scratch hidden exceeds kernel ABI")?;
        u32::try_from(c.moe_intermediate_size)
            .context("EXL3 scratch global intermediate exceeds kernel ABI")?;
        let g = Self {
            hidden: c.hidden_size,
            intermediate: c.moe_intermediate_size / c.tp_world_size,
            experts: c.num_experts,
            topk: c.num_experts_per_tok,
            capacity,
            stream,
            rank: c.tp_rank,
            world: c.tp_world_size,
        };
        ensure!(
            g.hidden > 0
                && g.intermediate > 0
                && g.hidden.is_multiple_of(128)
                && g.intermediate.is_multiple_of(128)
                && g.experts > 0
                && g.experts <= 512
                && g.topk > 0
                && g.topk <= 32
                && g.topk <= g.experts
                && capacity > 0
                && capacity.checked_mul(g.topk).is_some_and(|n| n <= 65535),
            "EXL3 scratch geometry exceeds kernel contract"
        );
        g.total_bytes()?;
        Ok(g)
    }
    pub fn sizes(self) -> Result<[usize; 11]> {
        let size = |a: usize, b: usize, c: usize| {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(c))
                .context("EXL3 MoE scratch bytes overflow")
        };
        let slots = self
            .capacity
            .checked_mul(self.topk)
            .context("EXL3 scratch slots overflow")?;
        Ok([
            size(self.capacity, self.experts, 4)?,
            size(self.capacity, self.topk, 4)?,
            size(self.capacity, self.topk, 4)?,
            size(slots, self.intermediate, 4)?,
            size(slots, self.intermediate, 4)?,
            size(slots, self.hidden, 4)?,
            size(self.capacity, self.hidden, 2)?,
            size(slots, self.hidden.max(self.intermediate), 4)?,
            size(slots, self.hidden.max(self.intermediate), 4)?,
            size(self.experts, 1, 4)?,
            size(self.capacity, self.experts, 2)?,
        ])
    }
    pub fn total_bytes(self) -> Result<usize> {
        self.sizes()?.into_iter().try_fold(0usize, |sum, n| {
            sum.checked_add(n)
                .context("EXL3 total scratch bytes overflow")
        })
    }
}
struct ScratchState {
    pointers: [DevicePtr; 11],
    active: bool,
}
#[derive(Clone)]
pub struct MoeScratchBorrow {
    geometry: MoeScratchGeometry,
    state: Arc<Mutex<ScratchState>>,
    enqueue: Arc<Mutex<()>>,
}
impl MoeScratchBorrow {
    pub(super) fn acquire(&self) -> Result<MutexGuard<'_, ()>> {
        let lease = self.enqueue.lock();
        self.pointers()?;
        Ok(lease)
    }
    pub(super) fn geometry(&self) -> MoeScratchGeometry {
        self.geometry
    }
    pub fn validate(&self, geometry: MoeScratchGeometry, rows: usize, stream: u64) -> Result<()> {
        ensure!(
            self.geometry == geometry && rows <= geometry.capacity && stream == geometry.stream,
            "EXL3 shared scratch geometry/capacity/stream mismatch"
        );
        self.pointers()?;
        Ok(())
    }
    pub fn pointers(&self) -> Result<[DevicePtr; 11]> {
        let state = self.state.lock();
        ensure!(
            state.active && state.pointers.iter().all(|p| !p.is_null()),
            "EXL3 model scratch has been released"
        );
        Ok(state.pointers)
    }
}
pub struct MoeScratch {
    borrowed: MoeScratchBorrow,
}
impl MoeScratch {
    pub fn allocate(gpu: &dyn GpuBackend, geometry: MoeScratchGeometry) -> Result<Self> {
        let mut owner = Self {
            borrowed: MoeScratchBorrow {
                geometry,
                state: Arc::new(Mutex::new(ScratchState {
                    pointers: [DevicePtr::NULL; 11],
                    active: true,
                })),
                enqueue: Arc::new(Mutex::new(())),
            },
        };
        for (index, bytes) in geometry.sizes()?.into_iter().enumerate() {
            match gpu.alloc(bytes) {
                Ok(p) => owner.borrowed.state.lock().pointers[index] = p,
                Err(e) => {
                    return match owner.release(gpu) {
                        Ok(()) => Err(e),
                        Err(cleanup) => {
                            Err(e.context(format!("EXL3 scratch rollback failed: {cleanup:#}")))
                        }
                    };
                }
            }
        }
        tracing::info!(
            scratch_bytes = geometry.total_bytes()?,
            capacity = geometry.capacity,
            stream = geometry.stream,
            "EXL3 routed scratch allocated once"
        );
        Ok(owner)
    }
    pub fn borrow(&self) -> MoeScratchBorrow {
        self.borrowed.clone()
    }
    /// Close new enqueue leases and establish completion before any model operand is freed.
    pub fn quiesce(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let _lease = self.borrowed.enqueue.lock();
        let pending = {
            let mut state = self.borrowed.state.lock();
            state.active = false;
            state.pointers.iter().any(|p| !p.is_null())
        };
        if pending {
            gpu.synchronize(self.borrowed.geometry.stream)?;
        }
        Ok(())
    }
    pub fn total_bytes(&self) -> Result<usize> {
        self.borrowed.geometry.total_bytes()
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for MoeScratch {
    fn label(&self) -> &'static str {
        "GLM EXL3 model routed scratch"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        self.quiesce(gpu)?;
        let _lease = self.borrowed.enqueue.lock();
        let mut state = self.borrowed.state.lock();
        state.active = false;
        let mut error = None;
        for p in &mut state.pointers {
            if !p.is_null() {
                match gpu.free(*p) {
                    Ok(()) => *p = DevicePtr::NULL,
                    Err(e) => retain_error(&mut error, Err(e)),
                }
            }
        }
        finish(error)
    }
}
