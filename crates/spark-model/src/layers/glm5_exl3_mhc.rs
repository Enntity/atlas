// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 mHC arithmetic over the existing four-byte highway arena.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::qwen3_attention::HcSiteWeights;
use crate::exl3_gate_perturbation::Exl3GatePerturbation;

/// Explicit native serving policy, bound to the installed GLM-5.3 TP=2
/// reference. The pinned checkpoint omits mhc_post_mult_value; the selected
/// policy supplies it explicitly and validates any declared checkpoint value.
/// Rounded projection geometry is bound to the reference's 48-SM GB10 host.
#[derive(Clone, Copy)]
pub(crate) struct MhcNumericalPolicy {
    post_multiplier: f32,
    rounded_sms: u32,
    /// Statistical-gate control replacing the consumed post multiplier.
    /// Declared checkpoint values still validate against the reference.
    perturbed_post_multiplier: Option<f32>,
}
impl MhcNumericalPolicy {
    pub(crate) const GLM53_TP2_REFERENCE: Self = Self {
        post_multiplier: 2.0,
        rounded_sms: 48,
        perturbed_post_multiplier: None,
    };

    /// Apply the `mhc_post_1p9` gate control; every other value is a no-op.
    pub(crate) fn with_perturbation(mut self, p: Option<Exl3GatePerturbation>) -> Self {
        if p == Some(Exl3GatePerturbation::MhcPost19) {
            self.perturbed_post_multiplier = Some(1.9);
        }
        self
    }

    /// The installed GB10 reference partitions whole K64 blocks and selects
    /// splits from its explicit 48-SM profile and ceil(rows/64) launch grid.
    fn rounded_splits(&self, hidden: usize, rows: usize) -> Result<u32> {
        let width = hidden
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("mHC width overflow"))?;
        ensure!(
            hidden > 0 && hidden.is_multiple_of(128) && width <= u32::MAX as usize,
            "rounded mHC projection requires an aligned positive hidden size"
        );
        ensure!(
            rows > 0 && u32::try_from(rows).is_ok() && (1..=64).contains(&self.rounded_sms),
            "rounded mHC projection requires explicit positive rows and bounded SM policy"
        );
        let grid = rows.div_ceil(64);
        Ok((self.rounded_sms as usize / grid).clamp(1, width / 64 / 4) as u32)
    }

    pub(crate) fn post_multiplier(&self, checkpoint: Option<f32>) -> Result<f32> {
        if let Some(value) = checkpoint {
            ensure!(
                value == self.post_multiplier,
                "checkpoint mhc_post_mult_value conflicts with explicit GLM53 TP2 reference policy"
            );
        }
        Ok(self
            .perturbed_post_multiplier
            .unwrap_or(self.post_multiplier))
    }
}

/// The installed reference fuses the unrounded post residual into its next
/// mixing projection for batches of at most sixteen rows. Larger batches
/// materialize BF16 residuals before that projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MhcResidualPrecision {
    Rounded,
    UnroundedPost,
}
impl MhcResidualPrecision {
    pub(crate) fn for_fused_rows(rows: usize) -> Self {
        if rows <= 16 {
            Self::UnroundedPost
        } else {
            Self::Rounded
        }
    }
}

pub(crate) struct NativeExl3Mhc {
    pre: KernelHandle,
    /// `ATLAS_EXL3_SPLIT_MHC=1`: (partials, finish, rounded partials) split of the pre site.
    pre_split: Option<(KernelHandle, KernelHandle, KernelHandle)>,
    post: KernelHandle,
    /// `ATLAS_EXL3_PAR_SMALL=1`: feature x row grid variant of `post`.
    post_wide: Option<KernelHandle>,
    contract: KernelHandle,
    hidden: u32,
    iterations: u32,
    rms_eps: f32,
    hc_eps: f32,
    post_multiplier: f32,
    numerical_policy: MhcNumericalPolicy,
}

#[derive(Clone, Copy)]
pub(crate) struct NativeMhcBuffers {
    pub highway: DevicePtr,
    pub normed: DevicePtr,
    pub post: DevicePtr,
    pub comb: DevicePtr,
}

impl NativeExl3Mhc {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        gpu: &dyn GpuBackend,
        hidden: usize,
        streams: usize,
        iterations: usize,
        rms_eps: f32,
        hc_eps: f32,
        numerical_policy: MhcNumericalPolicy,
        checkpoint_post_multiplier: Option<f32>,
    ) -> Result<Self> {
        let post_multiplier = numerical_policy.post_multiplier(checkpoint_post_multiplier)?;
        numerical_policy.rounded_splits(hidden, 1)?;
        ensure!(
            hidden > 0
                && hidden.is_multiple_of(128)
                && hidden <= u32::MAX as usize / 4
                && streams == 4,
            "native EXL3 mHC requires an aligned positive hidden size and four streams"
        );
        ensure!(
            iterations > 0 && u32::try_from(iterations).is_ok(),
            "native EXL3 mHC requires explicit positive Sinkhorn iterations"
        );
        ensure!(
            [rms_eps, hc_eps, post_multiplier]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0),
            "native EXL3 mHC requires finite positive numerical policy"
        );
        Ok(Self {
            pre: gpu.kernel("exl3_mhc", "exl3_mhc_pre_norm")?,
            pre_split: if std::env::var("ATLAS_EXL3_SPLIT_MHC").as_deref() == Ok("1") {
                Some((
                    gpu.kernel("exl3_mhc", "exl3_mhc_pre_partials")?,
                    gpu.kernel("exl3_mhc", "exl3_mhc_pre_norm_split")?,
                    gpu.kernel("exl3_mhc", "exl3_mhc_pre_partials_rounded")?,
                ))
            } else {
                None
            },
            post: gpu.kernel("exl3_mhc", "exl3_mhc_post_unrounded")?,
            post_wide: if std::env::var("ATLAS_EXL3_PAR_SMALL").as_deref() == Ok("1") {
                Some(gpu.kernel("exl3_mhc", "exl3_mhc_post_unrounded_wide")?)
            } else {
                None
            },
            contract: gpu.kernel("exl3_mhc", "exl3_mhc_contract")?,
            hidden: hidden as u32,
            iterations: iterations as u32,
            rms_eps,
            hc_eps,
            post_multiplier,
            numerical_policy,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pre_norm(
        &self,
        gpu: &dyn GpuBackend,
        site: &HcSiteWeights,
        norm_weight: DevicePtr,
        buffers: NativeMhcBuffers,
        rows: usize,
        precision: MhcResidualPrecision,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            u32::try_from(rows).is_ok(),
            "native EXL3 mHC row count overflow"
        );
        ensure!(
            site.lowrank.is_none(),
            "native EXL3 mHC does not support low-rank mixing weights"
        );
        if rows == 0 {
            return Ok(());
        }
        let pointers = [
            buffers.highway,
            buffers.normed,
            buffers.post,
            buffers.comb,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            norm_weight,
        ];
        ensure!(
            pointers.iter().all(|p| !p.is_null()),
            "native EXL3 mHC requires live tensors"
        );
        for (index, pointer) in pointers.iter().enumerate() {
            ensure!(
                !pointers[..index].contains(pointer),
                "native EXL3 mHC tensor bases must be distinct"
            );
        }
        let rounded_splits = self
            .numerical_policy
            .rounded_splits(self.hidden as usize, rows)?;
        // Rounded sites (layer-0 attention) run the same TF32 splits one warp
        // per block for decode-sized batches; the finish folds them in the
        // rounded branch's order.
        let rounded_split = precision == MhcResidualPrecision::Rounded
            && rows < 8
            && self.hidden <= 4096
            && 25 * rounded_splits * 4 <= self.hidden * 2;
        if let (true, Some((partials, finish, partials_rounded))) = (
            precision == MhcResidualPrecision::UnroundedPost || rounded_split,
            self.pre_split,
        ) {
            let splits = if rounded_split {
                KernelLaunch::new(gpu, partials_rounded)
                    .grid([rounded_splits, rows as u32, 1])
                    .block([32, 1, 1])
                    .arg_ptr(buffers.highway)
                    .arg_ptr(site.hc_fn)
                    .arg_ptr(buffers.normed)
                    .arg_u32(self.hidden)
                    .arg_u32(rounded_splits)
                    .launch(stream)?;
                rounded_splits
            } else {
                // Same split count exl3_mhc_pre_norm derives from gridDim.x.
                let splits: u32 = if rows < 8 && self.hidden <= 4096 {
                    8
                } else {
                    4
                };
                ensure!(
                    self.hidden.is_multiple_of(splits)
                        && 25 * splits * 4 <= self.hidden * 2
                        && self.hidden <= 4096,
                    "split mHC partials must fit the normed row"
                );
                KernelLaunch::new(gpu, partials)
                    .grid([25, splits, rows as u32])
                    .block([256, 1, 1])
                    .arg_ptr(buffers.highway)
                    .arg_ptr(site.hc_fn)
                    .arg_ptr(buffers.normed)
                    .arg_u32(self.hidden)
                    .arg_u32(splits)
                    .launch(stream)?;
                splits
            };
            return KernelLaunch::new(gpu, finish)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(buffers.highway)
                .arg_ptr(site.hc_fn)
                .arg_ptr(site.hc_scale)
                .arg_ptr(site.hc_base)
                .arg_ptr(norm_weight)
                .arg_ptr(buffers.normed)
                .arg_ptr(buffers.post)
                .arg_ptr(buffers.comb)
                .arg_u32(self.hidden)
                .arg_u32(self.iterations)
                .arg_f32(self.rms_eps)
                .arg_f32(self.rms_eps)
                .arg_f32(self.hc_eps)
                .arg_f32(self.post_multiplier)
                .arg_u32(splits)
                .arg_u32(rounded_splits)
                .launch(stream);
        }
        KernelLaunch::new(gpu, self.pre)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buffers.highway)
            .arg_ptr(site.hc_fn)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(norm_weight)
            .arg_ptr(buffers.normed)
            .arg_ptr(buffers.post)
            .arg_ptr(buffers.comb)
            .arg_u32(self.hidden)
            .arg_u32(self.iterations)
            .arg_f32(self.rms_eps)
            .arg_f32(self.rms_eps)
            .arg_f32(self.hc_eps)
            .arg_f32(self.post_multiplier)
            .arg_u32(u32::from(precision == MhcResidualPrecision::UnroundedPost))
            .arg_u32(rounded_splits)
            .launch(stream)
    }

    /// Keep the post result unrounded until the next site chooses its explicit
    /// small/large batch precision. The output may alias the old highway.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn post(
        &self,
        gpu: &dyn GpuBackend,
        block_output: DevicePtr,
        highway: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            u32::try_from(rows).is_ok(),
            "native EXL3 mHC row count overflow"
        );
        if rows == 0 {
            return Ok(());
        }
        let pointers = [block_output, highway, post, comb];
        ensure!(
            !output.is_null() && pointers.iter().all(|p| !p.is_null()),
            "native EXL3 mHC post requires live tensors"
        );
        for (index, pointer) in pointers.iter().enumerate() {
            ensure!(
                !pointers[..index].contains(pointer),
                "native EXL3 mHC post inputs must be distinct"
            );
        }
        ensure!(
            output == highway || !pointers.contains(&output),
            "native EXL3 mHC post only permits highway aliasing"
        );
        let (post_kernel, post_grid) = match self.post_wide {
            Some(wide) => (wide, [self.hidden.div_ceil(256), rows as u32, 1]),
            None => (self.post, [rows as u32, 1, 1]),
        };
        KernelLaunch::new(gpu, post_kernel)
            .grid(post_grid)
            .block([256, 1, 1])
            .arg_ptr(block_output)
            .arg_ptr(highway)
            .arg_ptr(post)
            .arg_ptr(comb)
            .arg_ptr(output)
            .arg_u32(self.hidden)
            .launch(stream)
    }

    pub(crate) fn contract(
        &self,
        gpu: &dyn GpuBackend,
        highway: DevicePtr,
        output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            u32::try_from(rows).is_ok(),
            "native EXL3 mHC row count overflow"
        );
        if rows == 0 {
            return Ok(());
        }
        ensure!(
            !highway.is_null() && !output.is_null() && highway != output,
            "native EXL3 mHC contraction requires distinct live tensors"
        );
        KernelLaunch::new(gpu, self.contract)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(highway)
            .arg_ptr(output)
            .arg_u32(self.hidden)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn mhc_post_perturbation_replaces_consumed_multiplier_only() {
        let reference = MhcNumericalPolicy::GLM53_TP2_REFERENCE;
        let perturbed = reference.with_perturbation(Some(Exl3GatePerturbation::MhcPost19));
        assert_eq!(perturbed.post_multiplier(None).unwrap(), 1.9);
        // A checkpoint declaring the reference 2.0 must not reject the control.
        assert_eq!(perturbed.post_multiplier(Some(2.0)).unwrap(), 1.9);
        assert!(perturbed.post_multiplier(Some(1.9)).is_err());
        for other in [None, Some(Exl3GatePerturbation::RouterScale24)] {
            let policy = reference.with_perturbation(other);
            assert_eq!(policy.post_multiplier(Some(2.0)).unwrap(), 2.0);
        }
    }

    #[test]
    fn perturbed_mhc_load_stores_the_perturbed_multiplier() {
        let gpu = MockGpuBackend::new();
        let policy = MhcNumericalPolicy::GLM53_TP2_REFERENCE
            .with_perturbation(Some(Exl3GatePerturbation::MhcPost19));
        let mhc = NativeExl3Mhc::load(&gpu, 4096, 4, 20, 1e-6, 1e-6, policy, Some(2.0)).unwrap();
        assert_eq!(mhc.post_multiplier, 1.9);
    }

    #[test]
    fn named_reference_policy_validates_declared_checkpoint_values() {
        let policy = MhcNumericalPolicy::GLM53_TP2_REFERENCE;
        assert_eq!(policy.post_multiplier(None).unwrap(), 2.0);
        assert_eq!(policy.post_multiplier(Some(2.0)).unwrap(), 2.0);
        for value in [0.0, 2.5, f32::NAN] {
            assert!(policy.post_multiplier(Some(value)).is_err());
        }
    }

    #[test]
    fn rounded_projection_uses_explicit_balanced_reference_split_policy() {
        let policy = MhcNumericalPolicy::GLM53_TP2_REFERENCE;
        for (rows, splits) in [
            (1, 48),
            (10, 48),
            (17, 48),
            (64, 48),
            (65, 24),
            (128, 24),
            (129, 16),
            (4096, 1),
        ] {
            assert_eq!(policy.rounded_splits(4096, rows).unwrap(), splits);
        }
        assert_eq!(policy.rounded_splits(128, 1).unwrap(), 2);
        for (hidden, rows) in [(0, 1), (4096, 0), (127, 1), (usize::MAX, 1)] {
            assert!(policy.rounded_splits(hidden, rows).is_err());
        }
    }

    #[test]
    fn closure_is_required_and_precision_boundary_is_explicit() {
        let gpu = MockGpuBackend::new();
        let mhc = NativeExl3Mhc::load(
            &gpu,
            4096,
            4,
            20,
            1e-5,
            1e-6,
            MhcNumericalPolicy::GLM53_TP2_REFERENCE,
            None,
        )
        .unwrap();
        assert_eq!(
            gpu.kernel_requests_snapshot(),
            vec![
                ("exl3_mhc".into(), "exl3_mhc_pre_norm".into()),
                ("exl3_mhc".into(), "exl3_mhc_post_unrounded".into()),
                ("exl3_mhc".into(), "exl3_mhc_contract".into())
            ]
        );
        assert_eq!(
            MhcResidualPrecision::for_fused_rows(16),
            MhcResidualPrecision::UnroundedPost
        );
        assert_eq!(
            MhcResidualPrecision::for_fused_rows(17),
            MhcResidualPrecision::Rounded
        );
        gpu.fail_next_kernel();
        assert!(
            NativeExl3Mhc::load(
                &gpu,
                4096,
                4,
                20,
                1e-5,
                1e-6,
                MhcNumericalPolicy::GLM53_TP2_REFERENCE,
                None
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        mhc.contract(&gpu, DevicePtr::NULL, DevicePtr::NULL, 0, 7)
            .unwrap();
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn post_launch_permits_only_the_reference_highway_alias() {
        let gpu = MockGpuBackend::new();
        let mhc = NativeExl3Mhc::load(
            &gpu,
            4096,
            4,
            20,
            1e-5,
            1e-6,
            MhcNumericalPolicy::GLM53_TP2_REFERENCE,
            None,
        )
        .unwrap();
        let p: Vec<_> = (0..4).map(|_| gpu.alloc(16).unwrap()).collect();
        mhc.post(&gpu, p[0], p[1], p[2], p[3], p[1], 5, 9).unwrap();
        assert_eq!(gpu.launches_snapshot()[0].grid, [5, 1, 1]);
        assert!(mhc.post(&gpu, p[0], p[1], p[2], p[3], p[0], 5, 9).is_err());
        assert!(mhc.post(&gpu, p[0], p[0], p[2], p[3], p[1], 5, 9).is_err());
        assert_eq!(gpu.launch_count(), 1);
        for pointer in p {
            gpu.free(pointer).unwrap();
        }
    }

    #[test]
    fn preflight_rejects_alias_and_invalid_policy_before_launch() {
        let gpu = MockGpuBackend::new();
        for (hidden, streams, iters, eps) in [
            (0, 4, 20, 1e-5),
            (4096, 3, 20, 1e-5),
            (4096, 4, 0, 1e-5),
            (4096, 4, 20, f32::NAN),
        ] {
            assert!(
                NativeExl3Mhc::load(
                    &gpu,
                    hidden,
                    streams,
                    iters,
                    eps,
                    1e-6,
                    MhcNumericalPolicy::GLM53_TP2_REFERENCE,
                    None
                )
                .is_err()
            );
        }
        assert!(gpu.kernel_requests_snapshot().is_empty());
        let mhc = NativeExl3Mhc::load(
            &gpu,
            4096,
            4,
            20,
            1e-5,
            1e-6,
            MhcNumericalPolicy::GLM53_TP2_REFERENCE,
            None,
        )
        .unwrap();
        let pointers: Vec<_> = (0..8).map(|_| gpu.alloc(16).unwrap()).collect();
        let site = HcSiteWeights {
            hc_fn: pointers[4],
            hc_scale: pointers[5],
            hc_base: pointers[6],
            hc_fn_bf16: DevicePtr::NULL,
            lowrank: None,
        };
        let mut buffers = NativeMhcBuffers {
            highway: pointers[0],
            normed: pointers[1],
            post: pointers[2],
            comb: pointers[3],
        };
        mhc.pre_norm(
            &gpu,
            &site,
            pointers[7],
            buffers,
            2,
            MhcResidualPrecision::Rounded,
            9,
        )
        .unwrap();
        assert_eq!(gpu.launches_snapshot()[0].grid, [2, 1, 1]);
        assert_eq!(gpu.launches_snapshot()[0].block, [256, 1, 1]);
        buffers.normed = buffers.highway;
        assert!(
            mhc.pre_norm(
                &gpu,
                &site,
                pointers[7],
                buffers,
                2,
                MhcResidualPrecision::Rounded,
                9
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 1);
        for pointer in pointers {
            gpu.free(pointer).unwrap();
        }
    }
}
