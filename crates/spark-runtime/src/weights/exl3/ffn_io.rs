// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit diagnostic I/O boundary for native FFN numerical localization.
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeSet,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy)]
pub enum DiagnosticDtype {
    Bf16,
    F32,
    I32,
}
impl DiagnosticDtype {
    fn width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16-le",
            Self::F32 => "f32-le",
            Self::I32 => "i32-le",
        }
    }
}
pub struct DiagnosticTensor {
    pub name: &'static str,
    pub dtype: DiagnosticDtype,
    pub shape: Vec<usize>,
    pub ptr: DevicePtr,
}
pub struct Exl3FfnRouter {
    directory: PathBuf,
    layer: usize,
    rank: usize,
    world: usize,
    sequence: AtomicUsize,
}
impl Exl3FfnRouter {
    pub fn new(directory: PathBuf, layer: usize, rank: usize, world: usize) -> Result<Self> {
        ensure!(
            directory.is_dir(),
            "EXL3 FFN dump requires an existing directory"
        );
        ensure!(world > 0 && rank < world, "EXL3 FFN dump topology invalid");
        Ok(Self {
            directory,
            layer,
            rank,
            world,
            sequence: AtomicUsize::new(0),
        })
    }
    pub fn from_env(
        layer: usize,
        layers: usize,
        rank: usize,
        world: usize,
    ) -> Result<Option<Self>> {
        Self::from_env_named(
            layer,
            layers,
            rank,
            world,
            "ATLAS_EXL3_FFN_DUMP_DIR",
            "ATLAS_EXL3_FFN_DUMP_LAYERS",
        )
    }
    pub fn from_env_named(
        layer: usize,
        layers: usize,
        rank: usize,
        world: usize,
        directory_key: &str,
        layers_key: &str,
    ) -> Result<Option<Self>> {
        let Some(directory) = std::env::var_os(directory_key) else {
            return Ok(None);
        };
        let selected = parse_layers(
            &std::env::var(layers_key)
                .context("FFN dump requires explicit ATLAS_EXL3_FFN_DUMP_LAYERS")?,
            layers,
        )?;
        for flag in [
            "ATLAS_NO_DECODE_GRAPHS",
            "ATLAS_NO_DECODE_GRAPHS_MULTISEQ",
            "ATLAS_DEBUG_NO_GRAPH",
        ] {
            ensure!(
                std::env::var(flag).as_deref() == Ok("1"),
                "FFN dump requires {flag}=1"
            );
        }
        if !selected.contains(&layer) {
            return Ok(None);
        }
        Self::new(directory.into(), layer, rank, world).map(Some)
    }
    pub fn begin(&self) -> Result<Option<usize>> {
        if !self.directory.join("capture-enabled").is_file() {
            return Ok(None);
        }
        Ok(Some(self.sequence.fetch_add(1, Ordering::SeqCst)))
    }
    pub fn capture(
        &self,
        gpu: &dyn GpuBackend,
        invocation: usize,
        stage: &str,
        tensors: &[DiagnosticTensor],
        stream: u64,
    ) -> Result<()> {
        self.capture_record(gpu, invocation, stage, tensors, stream, None)
    }
    /// Bind explicit host-side launch constants alongside resident tensors.
    pub fn capture_with_metadata(
        &self,
        gpu: &dyn GpuBackend,
        invocation: usize,
        stage: &str,
        tensors: &[DiagnosticTensor],
        stream: u64,
        binding: &serde_json::Value,
    ) -> Result<()> {
        ensure!(
            binding.is_object(),
            "FFN diagnostic binding must be an object"
        );
        self.capture_record(gpu, invocation, stage, tensors, stream, Some(binding))
    }
    #[allow(clippy::too_many_arguments)]
    fn capture_record(
        &self,
        gpu: &dyn GpuBackend,
        invocation: usize,
        stage: &str,
        tensors: &[DiagnosticTensor],
        stream: u64,
        binding: Option<&serde_json::Value>,
    ) -> Result<()> {
        let safe = |name: &str| {
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        };
        ensure!(
            safe(stage) && !tensors.is_empty(),
            "FFN diagnostic stage invalid"
        );
        let mut names = BTreeSet::new();
        let stem = format!(
            "ffn-L{}-rank{}-{invocation:06}-{stage}",
            self.layer, self.rank
        );
        let mut records = Vec::new();
        for tensor in tensors {
            ensure!(
                safe(tensor.name) && names.insert(tensor.name),
                "FFN diagnostic tensor name invalid/duplicate"
            );
            ensure!(
                !tensor.ptr.is_null()
                    && !tensor.shape.is_empty()
                    && tensor.shape.iter().all(|n| *n > 0),
                "FFN diagnostic shape/pointer invalid"
            );
            let bytes = tensor
                .shape
                .iter()
                .try_fold(tensor.dtype.width(), |size, n| size.checked_mul(*n))
                .context("FFN diagnostic bytes overflow")?;
            let mut host = vec![0; bytes];
            gpu.copy_d2h_on_stream(tensor.ptr, &mut host, stream)?;
            let filename = format!("{stem}-{}.bin", tensor.name);
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(&filename))?
                .write_all(&host)?;
            records.push(
                serde_json::json!({"name":tensor.name,"dtype":tensor.dtype.label(),
                "shape":tensor.shape,"bytes":bytes,"file":filename}),
            );
        }
        let metadata = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(format!("{stem}.json")))?;
        let mut record = serde_json::json!({"schema":"atlas-exl3-ffn-v1",
            "layer":self.layer,"tp_rank":self.rank,"tp_world":self.world,
            "invocation":invocation,"stage":stage,"stream":stream,"tensors":records});
        if let Some(binding) = binding {
            record["binding"] = binding.clone();
        }
        serde_json::to_writer(metadata, &record)?;
        Ok(())
    }
}

fn parse_layers(value: &str, layers: usize) -> Result<BTreeSet<usize>> {
    let mut selected = BTreeSet::new();
    for token in value.split(',') {
        let layer: usize = token
            .trim()
            .parse()
            .context("FFN dump layers must be explicit integers")?;
        ensure!(
            layer < layers && selected.insert(layer),
            "FFN dump layers invalid/duplicate"
        );
    }
    Ok(selected)
}

#[cfg(test)]
#[path = "ffn_io_tests.rs"]
mod tests;
