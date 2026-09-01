// SPDX-License-Identifier: AGPL-3.0-only

//! Shared four-GEMM KDA projection path for decode, verify, and prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::{KDA_HEADS, KDA_WIDTH};
use super::types::{Glm5Layer, KdaWeights};
use crate::layer::ForwardContext;
use crate::layers::ops::{self, Glm53KdaSplitMergedArgs};

const KDA_LOW_RANK: usize = 128;
const KDA_MERGED_WIDTH: usize = 3 * KDA_WIDTH + KDA_HEADS + 2 * KDA_LOW_RANK;

pub(super) struct KdaProjectedInputs {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub forget: DevicePtr,
    pub output_gate: DevicePtr,
    pub beta: DevicePtr,
}

impl Glm5Layer {
    pub(super) fn kda_project_inputs(
        &self,
        weights: &KdaWeights,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<KdaProjectedInputs> {
        ensure!(rows > 0, "GLM KDA projection requires rows");
        ensure!(
            KDA_WIDTH == 4096 && KDA_HEADS == 32,
            "GLM merged KDA projection requires official TP2 geometry"
        );

        // The generic SSM deinterleave arena is 32,768 BF16 values per GLM
        // row, comfortably above this appliance-specific 12,576-wide result.
        let merged = ctx.buffers.ssm_deinterleaved();
        self.project(
            normed,
            &weights.input_merged,
            merged,
            rows,
            KDA_MERGED_WIDTH,
            ctx.config.hidden_size,
            ctx,
            stream,
        )?;

        let matrix_bytes = rows * KDA_WIDTH * size_of::<u16>();
        let query = ctx.buffers.ssm_qkvz();
        let key = query.offset(matrix_bytes);
        let value = query.offset(matrix_bytes * 2);
        let forget = query.offset(matrix_bytes * 3);
        let gates = ctx.buffers.qkv_output();
        let output_gate = gates;
        let beta = gates.offset(matrix_bytes);
        let beta_bytes = rows * KDA_HEADS * size_of::<u16>();
        let gate_a = beta.offset(beta_bytes);
        let forget_a = ctx.buffers.ssm_ba();

        ops::glm53_kda_split_merged_bf16(
            ctx.gpu,
            self.kernels.kda_split_merged,
            &Glm53KdaSplitMergedArgs {
                merged,
                query,
                key,
                value,
                beta,
                forget_a,
                gate_a,
                rows: rows as u32,
            },
            stream,
        )?;
        self.project(
            forget_a,
            &weights.forget_b,
            forget,
            rows,
            KDA_WIDTH,
            KDA_LOW_RANK,
            ctx,
            stream,
        )?;
        self.project(
            gate_a,
            &weights.gate_b,
            output_gate,
            rows,
            KDA_WIDTH,
            KDA_LOW_RANK,
            ctx,
            stream,
        )?;

        Ok(KdaProjectedInputs {
            query,
            key,
            value,
            forget,
            output_gate,
            beta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_tp2_merged_width_matches_checkpoint_rows() {
        assert_eq!(KDA_MERGED_WIDTH, 12_576);
        assert_eq!(KDA_MERGED_WIDTH, 3 * 4096 + 32 + 128 + 128);
    }
}
