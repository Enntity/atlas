// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3-only reference-policy KDA core (`ATLAS_KDA_FLA_REFERENCE=1`).
//!
//! Selects the byte-exact port of the pinned FLA/vLLM reference
//! (`kernels/gb10/common/kda_fla.cu`, docs/design/exl3-kda-reference-core.md)
//! instead of Atlas's correctly rounded recurrence. Unset or `0` leaves every
//! kernel handle and launch exactly as before. Setting it on an unsupported
//! checkpoint or together with another prefill override is a startup error.
//! Prefill intermediates borrow the model-owned routed MoE scratch, which is
//! idle during attention on the same stream.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5KdaLayer;
use crate::layer::{ForwardContext, SsmLayerState};
use crate::layers::glm5_exl3_ffn::MoeScratchBorrow;
use crate::layers::ops::{self, KdaFlaIo, KdaFlaKernels, KdaFlaScratchLayout};

/// Index of the routed scratch buffer the prefill intermediates occupy
/// (`[capacity * topk, max(hidden, intermediate)]` F32, the largest one).
const SCRATCH_BUFFER: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KdaArithmeticPolicy {
    CorrectlyRounded,
    FlaReference,
}

pub(super) fn parse_policy(value: Option<&str>) -> Result<KdaArithmeticPolicy> {
    match value {
        None | Some("0") => Ok(KdaArithmeticPolicy::CorrectlyRounded),
        Some("1") => Ok(KdaArithmeticPolicy::FlaReference),
        Some(other) => bail!("ATLAS_KDA_FLA_REFERENCE must be 0 or 1, got {other:?}"),
    }
}

/// Geometry and exclusivity facts the policy requires, gathered at assembly.
pub(super) struct PolicyRequirements {
    pub exl3_projections: bool,
    pub heads: usize,
    pub dim: usize,
    pub conv_width: usize,
    pub lower_bound: f32,
    pub register_resident_prefill: bool,
}

pub(super) fn validate(req: &PolicyRequirements) -> Result<()> {
    ensure!(
        req.exl3_projections,
        "ATLAS_KDA_FLA_REFERENCE=1 requires native EXL3 KDA projections"
    );
    ensure!(
        req.heads == 32
            && req.dim == ops::KDA_FLA_DIM
            && req.conv_width == 4
            && req.lower_bound == -5.0,
        "ATLAS_KDA_FLA_REFERENCE=1 is qualified only for 32 rank-local heads, head_dim 128, \
         conv width 4 and lower_bound -5 (got {} / {} / {} / {})",
        req.heads,
        req.dim,
        req.conv_width,
        req.lower_bound
    );
    ensure!(
        !req.register_resident_prefill,
        "ATLAS_KDA_FLA_REFERENCE=1 excludes ATLAS_KDA_REGRESIDENT_PREFILL"
    );
    Ok(())
}

pub(super) struct FlaReference {
    kernels: KdaFlaKernels,
    scratch: MoeScratchBorrow,
}

impl FlaReference {
    /// `None` unless the policy is selected; then every requirement and the
    /// borrowed scratch capacity are checked before any forward runs.
    pub(super) fn load(
        gpu: &dyn GpuBackend,
        policy: KdaArithmeticPolicy,
        req: &PolicyRequirements,
        scratch: Option<MoeScratchBorrow>,
    ) -> Result<Option<Self>> {
        if policy == KdaArithmeticPolicy::CorrectlyRounded {
            return Ok(None);
        }
        validate(req)?;
        let Some(scratch) = scratch else {
            bail!("ATLAS_KDA_FLA_REFERENCE=1 requires the model-owned routed scratch");
        };
        let geometry = scratch.geometry();
        let capacity = geometry.capacity();
        ensure!(
            capacity > 1,
            "ATLAS_KDA_FLA_REFERENCE=1 needs prefill capacity > 1"
        );
        let need = KdaFlaScratchLayout::new(capacity, req.heads)?.total_bytes;
        let have = geometry.sizes()?[SCRATCH_BUFFER];
        ensure!(
            need <= have,
            "KDA FLA prefill scratch {need} B exceeds routed scratch buffer {have} B at capacity {capacity}"
        );
        scratch.validate(geometry, capacity, geometry.stream())?;
        Ok(Some(Self {
            kernels: KdaFlaKernels::resolve(gpu)?,
            scratch,
        }))
    }
}

impl Glm5KdaLayer {
    /// Convolution + recurrence of one sequence's rows under the reference
    /// policy. One row takes the decode kernels, as vLLM classifies any
    /// `query_len == 1` request as a decode.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fla_sequence_rows(
        &self,
        fla: &FlaReference,
        packed: DevicePtr,
        convolved: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        core_out: DevicePtr,
        state: &mut SsmLayerState,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(tokens > 0, "KDA FLA core needs at least one row");
        ensure!(
            !state.h_is_f16,
            "KDA FLA core requires FP32 recurrent state"
        );
        // ATLAS_KDA_FLA_GRAPH=1 admits capture of the single-row decode
        // kernels (fixed per-sequence state pointers); multi-row stays eager.
        let decode_graph_ok =
            tokens == 1 && std::env::var("ATLAS_KDA_FLA_GRAPH").as_deref() == Ok("1");
        ensure!(
            !ctx.graph_capture || decode_graph_ok,
            "KDA FLA core is not qualified for graph capture"
        );
        let io = KdaFlaIo {
            qkv: packed,
            raw_gate: g1,
            raw_beta: beta,
            conv_weight: self.weights.conv.weight,
            a_log: self.weights.a_log.weight,
            dt_bias: self.weights.dt_bias.weight,
            conv_state: state.conv_state,
            state: state.h_state,
            conv_out: convolved,
            output: core_out,
            heads: self.heads as u32,
            lower_bound: self.lower_bound,
        };
        if tokens == 1 {
            return ops::kda_fla_decode(ctx.gpu, &fla.kernels, &io, stream);
        }
        let geometry = fla.scratch.geometry();
        fla.scratch.validate(geometry, tokens, stream)?;
        let _lease = fla.scratch.acquire()?;
        let buffer = fla.scratch.pointers()?[SCRATCH_BUFFER];
        let bytes = geometry.sizes()?[SCRATCH_BUFFER];
        ops::kda_fla_prefill(
            ctx.gpu,
            &fla.kernels,
            &io,
            tokens as u32,
            buffer,
            bytes,
            stream,
        )
    }

    /// Speculative verify under the reference policy: each of the `tokens`
    /// rows takes the same single-row decode kernel the serial decode uses
    /// (conv + recurrence on the live state), and the state after row `t` is
    /// snapshotted into intermediate `t` (h for all but the last row, conv
    /// for every row) for the accepted-prefix commit / rollback. Row-for-row
    /// the recurrence is the serial arithmetic; only the projections around
    /// it see `tokens` rows. Records mode is not supported.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fla_verify_rows(
        &self,
        fla: &FlaReference,
        packed: DevicePtr,
        convolved: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        core_out: DevicePtr,
        state: &mut SsmLayerState,
        row0: usize,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            state.kda_records.is_null(),
            "ATLAS_KDA_FLA_REFERENCE=1 verify does not support --ssm-rollback-mode records"
        );
        ensure!(
            !state.h_is_f16,
            "KDA FLA core requires FP32 recurrent state"
        );
        ensure!(
            !ctx.graph_capture,
            "KDA FLA verify is not qualified for graph capture"
        );
        ensure!(
            tokens >= 1
                && state.h_state_intermediates.len() + 1 >= tokens
                && state.conv_state_intermediates.len() >= tokens,
            "GLM-5 KDA FLA verify needs K-1 h and K conv intermediates (h={}, conv={}, K={tokens})",
            state.h_state_intermediates.len(),
            state.conv_state_intermediates.len(),
        );
        let p = self.heads * self.dim;
        let bf16 = 2usize;
        let packed_row = 3 * p * bf16;
        let gate_row = p * bf16;
        let beta_row = self.heads * bf16;
        // One conv + one recurrence launch for all rows when the snapshot
        // slabs are contiguous (pool layout); same per-row arithmetic.
        let contiguous = |ptrs: &[DevicePtr], n: usize, stride: usize| {
            n == 0
                || ptrs.first().is_some_and(|base| {
                    !base.is_null()
                        && ptrs[..n]
                            .iter()
                            .enumerate()
                            .all(|(t, ptr)| ptr.0 == base.0 + (t * stride) as u64)
                })
        };
        if std::env::var("ATLAS_GLM_EXL3_DFLASH_KDA_PER_ROW").as_deref() != Ok("1")
            && contiguous(
                &state.conv_state_intermediates,
                tokens,
                self.conv_state_bytes,
            )
            && contiguous(&state.h_state_intermediates, tokens - 1, self.h_state_bytes)
        {
            let io = KdaFlaIo {
                qkv: packed.offset(row0 * packed_row),
                raw_gate: g1.offset(row0 * gate_row),
                raw_beta: beta.offset(row0 * beta_row),
                conv_weight: self.weights.conv.weight,
                a_log: self.weights.a_log.weight,
                dt_bias: self.weights.dt_bias.weight,
                conv_state: state.conv_state,
                state: state.h_state,
                conv_out: convolved.offset(row0 * packed_row),
                output: core_out.offset(row0 * gate_row),
                heads: self.heads as u32,
                lower_bound: self.lower_bound,
            };
            let h_snaps = state
                .h_state_intermediates
                .first()
                .copied()
                .unwrap_or(DevicePtr::NULL);
            if ops::kda_fla_verify_rows(
                ctx.gpu,
                &fla.kernels,
                &io,
                tokens as u32,
                state.conv_state_intermediates[0],
                self.conv_state_bytes,
                h_snaps,
                self.h_state_bytes,
                stream,
            )? {
                return Ok(());
            }
        }
        for t in 0..tokens {
            let r = row0 + t;
            let io = KdaFlaIo {
                qkv: packed.offset(r * packed_row),
                raw_gate: g1.offset(r * gate_row),
                raw_beta: beta.offset(r * beta_row),
                conv_weight: self.weights.conv.weight,
                a_log: self.weights.a_log.weight,
                dt_bias: self.weights.dt_bias.weight,
                conv_state: state.conv_state,
                state: state.h_state,
                conv_out: convolved.offset(r * packed_row),
                output: core_out.offset(r * gate_row),
                heads: self.heads as u32,
                lower_bound: self.lower_bound,
            };
            ops::kda_fla_decode(ctx.gpu, &fla.kernels, &io, stream)?;
            if t + 1 < tokens {
                ctx.gpu.copy_d2d_async(
                    state.h_state,
                    state.h_state_intermediates[t],
                    self.h_state_bytes,
                    stream,
                )?;
            }
            ctx.gpu.copy_d2d_async(
                state.conv_state,
                state.conv_state_intermediates[t],
                self.conv_state_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    pub(super) fn ensure_policy_off(&self, path: &str) -> Result<()> {
        ensure!(
            self.fla_reference.is_none(),
            "ATLAS_KDA_FLA_REFERENCE=1 does not support the {path} KDA path"
        );
        Ok(())
    }

    pub(super) fn exl3_projections(&self) -> bool {
        [
            &self.weights.q_proj,
            &self.weights.k_proj,
            &self.weights.v_proj,
            &self.weights.o_proj,
        ]
        .iter()
        .all(|p| p.is_exl3())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> PolicyRequirements {
        PolicyRequirements {
            exl3_projections: true,
            heads: 32,
            dim: 128,
            conv_width: 4,
            lower_bound: -5.0,
            register_resident_prefill: false,
        }
    }

    #[test]
    fn policy_flag_is_strict_and_defaults_to_existing_path() {
        assert_eq!(
            parse_policy(None).unwrap(),
            KdaArithmeticPolicy::CorrectlyRounded
        );
        assert_eq!(
            parse_policy(Some("0")).unwrap(),
            KdaArithmeticPolicy::CorrectlyRounded
        );
        assert_eq!(
            parse_policy(Some("1")).unwrap(),
            KdaArithmeticPolicy::FlaReference
        );
        assert!(parse_policy(Some("true")).is_err());
        assert!(parse_policy(Some("")).is_err());
    }

    #[test]
    fn requirements_accept_only_the_qualified_geometry() {
        assert!(validate(&req()).is_ok());
        for bad in [
            PolicyRequirements {
                exl3_projections: false,
                ..req()
            },
            PolicyRequirements { heads: 64, ..req() },
            PolicyRequirements { dim: 64, ..req() },
            PolicyRequirements {
                conv_width: 3,
                ..req()
            },
            PolicyRequirements {
                lower_bound: -4.0,
                ..req()
            },
            PolicyRequirements {
                register_resident_prefill: true,
                ..req()
            },
        ] {
            assert!(validate(&bad).is_err());
        }
    }

    #[test]
    fn unset_policy_loads_nothing_without_touching_the_gpu() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let bad = PolicyRequirements { heads: 7, ..req() };
        let loaded =
            FlaReference::load(&gpu, KdaArithmeticPolicy::CorrectlyRounded, &bad, None).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn selected_policy_requires_scratch() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        assert!(FlaReference::load(&gpu, KdaArithmeticPolicy::FlaReference, &req(), None).is_err());
    }
}
