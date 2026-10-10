// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in diagnostic I/O boundary for full vocabulary validation artifacts.
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Context, Result, ensure};
use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Exl3LogitsRouter {
    directory: PathBuf,
    sequence: AtomicUsize,
}
impl Exl3LogitsRouter {
    /// Only TP rank zero writes. The caller explicitly disables graph capture
    /// when enabling this synchronous diagnostic sink.
    pub fn from_env(rank: usize) -> Result<Option<Self>> {
        let Some(directory) = std::env::var_os("ATLAS_EXL3_LOGITS_DIR") else {
            return Ok(None);
        };
        for flag in [
            "ATLAS_NO_DECODE_GRAPHS",
            "ATLAS_NO_DECODE_GRAPHS_MULTISEQ",
            "ATLAS_DEBUG_NO_GRAPH",
        ] {
            ensure!(
                std::env::var(flag).as_deref() == Ok("1"),
                "EXL3 logits capture requires {flag}=1 for eager execution"
            );
        }
        if rank != 0 {
            return Ok(None);
        }
        let directory = PathBuf::from(directory);
        ensure!(
            directory.is_dir(),
            "ATLAS_EXL3_LOGITS_DIR must name an existing run directory"
        );
        Ok(Some(Self {
            directory,
            sequence: AtomicUsize::new(0),
        }))
    }
    pub fn capture(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        rows: usize,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.directory.join("capture-enabled").is_file() {
            return Ok(());
        }
        ensure!(
            rows > 0 && vocab > 0,
            "EXL3 logit capture requires positive dimensions"
        );
        let bytes = rows
            .checked_mul(vocab)
            .and_then(|n| n.checked_mul(4))
            .context("EXL3 logits size overflow")?;
        let mut host = vec![0u8; bytes];
        gpu.copy_d2h_on_stream(logits, &mut host, stream)?;
        ensure!(
            host.chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().expect("four bytes")).is_finite()),
            "EXL3 logits contain nonfinite values"
        );
        let index = self.sequence.fetch_add(1, Ordering::SeqCst);
        let stem = format!("logits-{index:06}");
        let mut data = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(format!("{stem}.f32")))?;
        data.write_all(&host)?;
        let metadata = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(format!("{stem}.json")))?;
        serde_json::to_writer(
            metadata,
            &serde_json::json!({"dtype":"f32-le", "rows":rows, "vocab":vocab, "file":format!("{stem}.f32")}),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;
    #[test]
    fn captures_exact_full_rows_and_rejects_nonfinite_and_directory_reuse() {
        let directory = std::env::temp_dir().join(format!(
            "atlas-logits-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("capture-enabled"), b"").unwrap();
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(24).unwrap();
        let values = [1f32, -2., 3., 4., 5., -6.];
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        gpu.copy_h2d(&bytes, ptr).unwrap();
        let router = Exl3LogitsRouter {
            directory: directory.clone(),
            sequence: AtomicUsize::new(0),
        };
        router.capture(&gpu, ptr, 2, 3, 17).unwrap();
        assert_eq!(
            std::fs::read(directory.join("logits-000000.f32")).unwrap(),
            bytes
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("logits-000000.json")).unwrap())
                .unwrap();
        assert_eq!(metadata["rows"], 2);
        assert_eq!(metadata["vocab"], 3);
        let reused = Exl3LogitsRouter {
            directory: directory.clone(),
            sequence: AtomicUsize::new(0),
        };
        assert!(reused.capture(&gpu, ptr, 2, 3, 17).is_err());
        gpu.copy_h2d(&f32::NAN.to_le_bytes(), ptr).unwrap();
        assert!(router.capture(&gpu, ptr, 2, 3, 17).is_err());
        assert!(!directory.join("logits-000001.f32").exists());
        gpu.free(ptr).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }
}
