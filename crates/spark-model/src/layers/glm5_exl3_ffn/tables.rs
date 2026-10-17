// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side EXL3 descriptor upload, with checkpoint-native expert ownership.

use super::*;

pub(super) struct ExpertTable {
    pub shards: Vec<Exl3DeviceShard>,
    pub ptr: DevicePtr,
}
impl ExpertTable {
    pub fn upload(gpu: &dyn GpuBackend, mut shards: Vec<Exl3DeviceShard>) -> Result<Self> {
        let result = (|| -> Result<DevicePtr> {
            ensure!(!shards.is_empty(), "EXL3 expert table is empty");
            let mut data = Vec::with_capacity(shards.len() * 32);
            for shard in &shards {
                for ptr in shard.buffers()? {
                    data.extend_from_slice(&ptr.0.to_le_bytes());
                }
                data.extend_from_slice(&u32::from(shard.plan().bits()).to_le_bytes());
                data.extend_from_slice(&0u32.to_le_bytes());
            }
            let ptr = gpu.alloc(data.len())?;
            if let Err(error) = gpu.copy_h2d(&data, ptr) {
                gpu.free(ptr)?;
                return Err(error);
            }
            Ok(ptr)
        })();
        match result {
            Ok(ptr) => Ok(Self { shards, ptr }),
            Err(error) => {
                for shard in &mut shards {
                    shard.release(gpu)?;
                }
                Err(error)
            }
        }
    }
    pub fn validate(
        &self,
        experts: usize,
        rank: usize,
        world: usize,
        k: usize,
        n: usize,
        axis: ShardAxis,
    ) -> Result<()> {
        ensure!(
            !self.ptr.is_null() && self.shards.len() == experts,
            "EXL3 expert table released/count mismatch"
        );
        for shard in &self.shards {
            let plan = shard.plan();
            ensure!(
                (
                    plan.rank(),
                    plan.world(),
                    plan.in_features(),
                    plan.out_features(),
                    plan.axis()
                ) == (rank, world, k, n, axis),
                "EXL3 expert table TP geometry mismatch"
            );
            shard.buffers()?;
        }
        Ok(())
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for ExpertTable {
    fn label(&self) -> &'static str {
        "EXL3 expert pointer table"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut error = None;
        if !self.ptr.is_null() {
            match gpu.free(self.ptr) {
                Ok(()) => self.ptr = DevicePtr::NULL,
                Err(e) => retain_error(&mut error, Err(e)),
            }
        }
        for shard in &mut self.shards {
            retain_error(&mut error, shard.release(gpu));
        }
        finish(error)
    }
}
