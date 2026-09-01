// SPDX-License-Identifier: AGPL-3.0-only

//! Non-POSIX fail-closed counterpart to the GLM FlashKDA bridge.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

pub struct Glm53FlashKdaPrefillArgs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub forget: DevicePtr,
    pub beta_ht: DevicePtr,
    pub recurrent_state: DevicePtr,
    pub output: DevicePtr,
    pub workspace: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub cu_seqlens: DevicePtr,
    pub state_slot_ids: DevicePtr,
    pub total_tokens: u32,
    pub heads: u32,
    pub sequences: u32,
    pub state_capacity: u32,
    pub query_scale: f32,
    pub lower_bound: f32,
}

pub fn available() -> bool {
    false
}

pub fn glm53_flash_kda_workspace_size(
    _total_tokens: u32,
    _heads: u32,
    _sequences: u32,
) -> Result<usize> {
    bail!("GLM FlashKDA prefill requires a POSIX shared-library loader")
}

pub fn glm53_flash_kda_prefill(_args: &Glm53FlashKdaPrefillArgs, _stream: u64) -> Result<()> {
    bail!("GLM FlashKDA prefill requires a POSIX shared-library loader")
}
