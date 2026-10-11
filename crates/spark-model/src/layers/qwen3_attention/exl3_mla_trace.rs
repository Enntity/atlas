// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in native EXL3 MLA boundary capture for reference comparison.
//!
//! `ATLAS_EXL3_MLA_DUMP_DIR` + `ATLAS_EXL3_MLA_DUMP_LAYERS` select layers; a
//! `capture-enabled` marker in the directory arms one capture per layer
//! invocation (same router contract as the FFN/upstream dumps). Stages mirror
//! the reference mHC hooks: `attn_pre` (normed attention input, post, comb),
//! `attn_post` (attention block output), `ffn_pre` (normed FFN input), and
//! each native projection's input/output (`proj_<field>`; O is pre-reduction).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::ffn_io::{DiagnosticDtype, DiagnosticTensor, Exl3FfnRouter};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Site {
    Attn,
    Ffn,
}

pub(crate) struct MlaTrace {
    router: Exl3FfnRouter,
    hidden: usize,
    /// Invocation armed at the attention pre site and the next post site.
    state: Mutex<Option<(usize, Site)>>,
}

impl MlaTrace {
    pub(crate) fn from_env(config: &ModelConfig, layer: usize) -> Result<Option<Self>> {
        Ok(Exl3FfnRouter::from_env_named(
            layer,
            config.num_hidden_layers,
            config.tp_rank,
            config.tp_world_size,
            "ATLAS_EXL3_MLA_DUMP_DIR",
            "ATLAS_EXL3_MLA_DUMP_LAYERS",
        )?
        .map(|router| Self {
            router,
            hidden: config.hidden_size,
            state: Mutex::new(None),
        }))
    }

    fn write(
        &self,
        gpu: &dyn GpuBackend,
        invocation: usize,
        stage: &str,
        tensors: &[(&'static str, DiagnosticDtype, DevicePtr, Vec<usize>)],
        stream: u64,
    ) -> Result<()> {
        let tensors: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, ptr, shape)| DiagnosticTensor {
                name,
                dtype: *dtype,
                ptr: *ptr,
                shape: shape.clone(),
            })
            .collect();
        self.router
            .capture(gpu, invocation, stage, &tensors, stream)
    }

    /// After a native pre-norm. `attn` selects the attention site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pre(
        &self,
        gpu: &dyn GpuBackend,
        attn: bool,
        normed: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let mut state = self.state.lock();
        let invocation = if attn {
            let Some(invocation) = self.router.begin()? else {
                *state = None;
                return Ok(());
            };
            *state = Some((invocation, Site::Attn));
            invocation
        } else {
            let Some((invocation, _)) = *state else {
                return Ok(());
            };
            *state = Some((invocation, Site::Ffn));
            invocation
        };
        let (h, bf, f) = (self.hidden, DiagnosticDtype::Bf16, DiagnosticDtype::F32);
        let stage = if attn { "attn_pre" } else { "ffn_pre" };
        self.write(
            gpu,
            invocation,
            stage,
            &[
                ("normed", bf, normed, vec![rows, h]),
                ("post", f, post, vec![rows, 4]),
                ("comb", f, comb, vec![rows, 4, 4]),
            ],
            stream,
        )
    }

    /// At the attention post site (block output before mixing).
    pub(crate) fn post(
        &self,
        gpu: &dyn GpuBackend,
        block_output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let state = *self.state.lock();
        let Some((invocation, Site::Attn)) = state else {
            return Ok(());
        };
        self.write(
            gpu,
            invocation,
            "attn_post",
            &[(
                "block_output",
                DiagnosticDtype::Bf16,
                block_output,
                vec![rows, self.hidden],
            )],
            stream,
        )
    }

    /// Native projection boundary inside the armed attention invocation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn projection(
        &self,
        gpu: &dyn GpuBackend,
        name: &'static str,
        input: DevicePtr,
        k: usize,
        output: DevicePtr,
        n: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let state = *self.state.lock();
        let Some((invocation, Site::Attn)) = state else {
            return Ok(());
        };
        let bf = DiagnosticDtype::Bf16;
        self.write(
            gpu,
            invocation,
            name,
            &[
                ("input", bf, input, vec![rows, k]),
                ("output", bf, output, vec![rows, n]),
            ],
            stream,
        )
    }
}
