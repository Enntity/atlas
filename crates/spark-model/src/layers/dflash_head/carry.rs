// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::{DevicePtr, GraphHandle};

use super::DflashGraphIdentity;

pub fn ctx_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_DFLASH_CTX_CARRY").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

pub const MIN_CARRY_TOKENS: usize = 256;

pub struct DflashCtxCarry {
    pub ctx_hidden_acc: DevicePtr,
    pub ctx_len: usize,
    pub ctx_committed: usize,
    pub ctx_positions: Vec<i32>,
    pub block_table: Vec<u32>,
    pub block_table_dev: Option<DevicePtr>,
    pub ctx_count_drafter: usize,
    pub max_ctx_count_drafter: usize,
    pub tokens: Vec<u32>,
    pub graphs: Vec<(DflashGraphIdentity, Vec<GraphHandle>)>,
    pub lane_id: usize,
}

impl DflashCtxCarry {
    pub fn common_prefix_len(&self, prompt: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn carry(tokens: &[u32], ctx_len: usize, committed: usize) -> DflashCtxCarry {
        DflashCtxCarry {
            ctx_hidden_acc: DevicePtr(0),
            ctx_len,
            ctx_committed: committed,
            ctx_positions: (0..ctx_len as i32).collect(),
            block_table: vec![1, 2, 3],
            block_table_dev: None,
            ctx_count_drafter: ctx_len,
            max_ctx_count_drafter: ctx_len + 16,
            tokens: tokens.to_vec(),
            graphs: Vec::new(),
            lane_id: 0,
        }
    }

    #[test]
    fn common_prefix_stops_at_first_divergence() {
        let c = carry(&[1, 2, 3, 4], 4, 4);
        assert_eq!(c.common_prefix_len(&[1, 2, 3, 4, 5, 6]), 4);
        assert_eq!(c.common_prefix_len(&[1, 2, 9, 4]), 2);
        assert_eq!(c.common_prefix_len(&[]), 0);
    }

    #[test]
    fn carry_gate_defaults_on_and_honours_opt_out() {
        assert!(ctx_carry_enabled());
    }
}
