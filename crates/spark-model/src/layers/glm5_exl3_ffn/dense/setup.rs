// SPDX-License-Identifier: AGPL-3.0-only
//! Packed dense constructor and failure rollback.
use super::*;
use spark_runtime::weights::exl3::native::Exl3LinearNumerics;

impl NativeDenseFfn {
    pub fn new(
        gpu: &dyn GpuBackend,
        weights: [Exl3DeviceShard; 3],
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        role: NativeDenseFfnRole,
    ) -> Result<Self> {
        Self::setup(
            gpu,
            Box::new(weights),
            capacity,
            stream,
            activation,
            role,
            None,
        )
    }

    pub fn new_with_numerics(
        gpu: &dyn GpuBackend,
        weights: [Exl3DeviceShard; 3],
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        numerics: [Exl3LinearNumerics; 2],
    ) -> Result<Self> {
        Self::setup(
            gpu,
            Box::new(weights),
            capacity,
            stream,
            activation,
            NativeDenseFfnRole::DensePacked,
            Some(numerics),
        )
    }

    /// ATLAS_EXL3_SHARED_PACKED=1 packed shared expert whose gate and up carry
    /// the fused [gate | up] numerics (offsets 0 / n of logical 2n): up gets
    /// its own workspace and gate|up one cooperative launch when available.
    pub fn new_shared_packed(
        gpu: &dyn GpuBackend,
        weights: [Exl3DeviceShard; 3],
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        numerics: [Exl3LinearNumerics; 3],
    ) -> Result<Self> {
        let [gate, up, down] = numerics;
        let mut ffn = Self::setup(
            gpu,
            Box::new(weights),
            capacity,
            stream,
            activation,
            NativeDenseFfnRole::DensePacked,
            Some([gate, down]),
        )?;
        let DenseProjections::Packed {
            weights,
            workspaces,
            up_workspace,
            fused_gate_up,
        } = &mut ffn.projections
        else {
            unreachable!("setup builds packed projections")
        };
        let plan = weights[1].plan();
        ensure!(
            plan.bits() == up.bits,
            "EXL3 shared up numerical bitrate mismatch"
        );
        let up_ws = match Exl3LinearWorkspace::new_with_numerics(
            gpu,
            plan.in_features(),
            plan.out_features(),
            capacity,
            stream,
            up,
        ) {
            Ok(ws) => ws,
            Err(error) => {
                let cleanup = ffn.release(gpu);
                return Err(match cleanup {
                    Ok(()) => error,
                    Err(e) => error.context(format!("EXL3 shared FFN cleanup failed: {e:#}")),
                });
            }
        };
        *fused_gate_up = spark_runtime::weights::exl3::native::Exl3FusedCoop::resolve(
            gpu,
            &[(&workspaces[0], &weights[0]), (&up_ws, &weights[1])],
        );
        *up_workspace = Some(Box::new(up_ws));
        Ok(ffn)
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        gpu: &dyn GpuBackend,
        mut weights: Box<[Exl3DeviceShard; 3]>,
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        role: NativeDenseFfnRole,
        numerics: Option<[Exl3LinearNumerics; 2]>,
    ) -> Result<Self> {
        let result = (|| -> Result<_> {
            ensure!(
                role == NativeDenseFfnRole::DensePacked,
                "shared FFN requires global BF16 reconstruction before TP"
            );
            let [g, u, d] = (*weights).each_ref().map(|w| w.plan());
            ensure!(
                g.axis() == ShardAxis::Output
                    && u.axis() == ShardAxis::Output
                    && d.axis() == ShardAxis::Input,
                "EXL3 native FFN requires output gate/up and input down TP axes"
            );
            ensure!(
                g.in_features() == u.in_features()
                    && g.out_features() == u.out_features()
                    && g.in_features() == d.out_features()
                    && g.out_features() == d.in_features(),
                "EXL3 native FFN local shape mismatch"
            );
            ensure!(
                g.rank() == u.rank()
                    && g.rank() == d.rank()
                    && g.world() == u.world()
                    && g.world() == d.world()
                    && g.output_ranges() == u.output_ranges()
                    && g.output_ranges() == [d.input_range()],
                "EXL3 native FFN TP ownership mismatch"
            );
            activation.validate()?;
            let hidden = g.in_features();
            let intermediate = g.out_features();
            let rank = g.rank();
            let world = g.world();
            let silu = gpu.kernel("exl3", "exl3_silu_mul_dense")?;
            let bytes = capacity
                .checked_mul(intermediate)
                .and_then(|n| n.checked_mul(2))
                .context("EXL3 FFN activation bytes overflow")?;
            if let Some([gate, down]) = numerics {
                ensure!(
                    g.bits() == gate.bits && u.bits() == gate.bits && d.bits() == down.bits,
                    "EXL3 dense numerical bitrate mismatch"
                );
                Exl3LinearWorkspace::preflight_numerics(gpu, hidden, intermediate, capacity, gate)?;
                Exl3LinearWorkspace::preflight_numerics(gpu, intermediate, hidden, capacity, down)?;
            }
            let workspace = |k, n, index| match numerics {
                Some(policies) => Exl3LinearWorkspace::new_with_numerics(
                    gpu,
                    k,
                    n,
                    capacity,
                    stream,
                    policies[index],
                ),
                None => Exl3LinearWorkspace::new_with_input_precision(
                    gpu,
                    k,
                    n,
                    capacity,
                    stream,
                    role.packed_input_precision()?,
                ),
            };
            let gate = workspace(hidden, intermediate, 0)?;
            let down = match workspace(intermediate, hidden, 1) {
                Ok(w) => w,
                Err(error) => {
                    let mut gate = gate;
                    gate.release(gpu)?;
                    return Err(error);
                }
            };
            let mut activations = [DevicePtr::NULL; 2];
            for ptr in &mut activations {
                match gpu.alloc(bytes) {
                    Ok(p) => *ptr = p,
                    Err(error) => {
                        for &p in &activations {
                            if !p.is_null() {
                                gpu.free(p)?;
                            }
                        }
                        let mut gate = gate;
                        let mut down = down;
                        gate.release(gpu)?;
                        down.release(gpu)?;
                        return Err(error);
                    }
                }
            }
            Ok((
                gate,
                down,
                activations,
                silu,
                hidden,
                intermediate,
                rank,
                world,
            ))
        })();
        match result {
            Ok((gate, down, activations, silu, hidden, intermediate, rank, world)) => Ok(Self {
                layer: None,
                capture: None,
                projections: DenseProjections::Packed {
                    weights,
                    workspaces: Box::new([gate, down]),
                    up_workspace: None,
                    fused_gate_up: None,
                },
                activations,
                merged_activation: DevicePtr::NULL,
                silu,
                activation,
                role,
                hidden,
                intermediate,
                rank,
                world,
                capacity,
                stream,
                active: true,
            }),
            Err(error) => {
                let mut cleanup = None;
                for w in weights.iter_mut() {
                    retain_error(&mut cleanup, w.release(gpu));
                }
                Err(match cleanup {
                    None => error,
                    Some(e) => error.context(format!("EXL3 FFN cleanup failed: {e:#}")),
                })
            }
        }
    }
}
