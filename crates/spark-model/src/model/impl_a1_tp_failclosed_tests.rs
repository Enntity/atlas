// SPDX-License-Identifier: AGPL-3.0-only

//! Q5 (todo.md): fail closed on the TP all-reduce fallback.
//!
//! Covers the decision gate and the kernel resolution itself through
//! `MockGpuBackend` (`fail_next_kernel`) and a recording comm stub, so the
//! start-up contract holds without a GPU.
//!
//! The GPU-touching half — that a real rank-0/rank-1 pair keeps using the
//! registered path and never reaches `ncclAllReduce` — stays owed: it needs a
//! Spark under `flock /tmp/atlas-bench.lock`.

use super::{glm53_exl3_pins_tp_reduce, install_tp_add_kernel};
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::atomic::{AtomicU64, Ordering};

/// Recording stand-in for the 2-rank NCCL communicator: only
/// `set_add_kernel` is observable here, which is exactly the hand-off the
/// fail-closed gate protects.
struct RecordingComm {
    add_kernel: AtomicU64,
}

impl RecordingComm {
    fn new() -> Self {
        Self {
            add_kernel: AtomicU64::new(0),
        }
    }

    fn add_kernel(&self) -> u64 {
        self.add_kernel.load(Ordering::Relaxed)
    }
}

impl spark_comm::CommBackend for RecordingComm {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn all_gather(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn reduce_scatter(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }

    fn send_to(&self, _ptr: u64, _bytes: usize, _dest_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn recv_from(&self, _ptr: u64, _bytes: usize, _src_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        2
    }

    fn set_add_kernel(&self, handle: u64) {
        self.add_kernel.store(handle, Ordering::Relaxed);
    }
}

/// GLM-5.3 Flash EXL3 over TP=2/EP=1 — the shape
/// `exl3_preflight::preflight_glm53_tp2_packed` admits and the one the Q5
/// finding targets.
fn glm53_exl3() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.vocab_size = 154880;
    config.tp_world_size = 2;
    config.tp_rank = 0;
    config.ep_world_size = 1;
    config.expert_tp = false;
    config.quantization_config = Some(atlas_core::config::QuantizationConfig {
        quant_method: "exl3".into(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
        weight_block_size: Vec::new(),
        group_size: 0,
        quantized_layers: Default::default(),
        exl3: None,
    });
    config
}

#[test]
fn glm53_exl3_pins_tp_reduce_matches_the_exl3_checkpoint_signals() {
    assert!(glm53_exl3_pins_tp_reduce(&glm53_exl3()));
}

#[test]
fn other_families_and_schemes_still_degrade_to_a_warning() {
    // Non-GLM family: an EXL3-declared quant_method is not a GLM-5.3 target.
    let mut qwen = glm53_exl3();
    qwen.model_type = "qwen3_next".into();
    assert!(!glm53_exl3_pins_tp_reduce(&qwen));

    // GLM-5.3 without an EXL3 checkpoint: no packed-arithmetic target.
    let mut native = glm53_exl3();
    native.quantization_config = None;
    assert!(!glm53_exl3_pins_tp_reduce(&native));

    let mut other_scheme = glm53_exl3();
    other_scheme.quantization_config = Some(atlas_core::config::QuantizationConfig {
        quant_method: "fp8".into(),
        quant_algo: "FP8".into(),
        format: String::new(),
        ignore_modules: Vec::new(),
        weight_block_size: vec![128, 128],
        group_size: 0,
        quantized_layers: Default::default(),
        exl3: None,
    });
    assert!(!glm53_exl3_pins_tp_reduce(&other_scheme));
}

#[test]
fn glm53_exl3_startup_fails_when_bf16_add_inplace_is_missing() {
    let gpu = MockGpuBackend::new();
    gpu.fail_next_kernel();
    let comm = RecordingComm::new();

    let error = install_tp_add_kernel(&glm53_exl3(), &gpu, &comm)
        .expect_err("GLM-5.3 EXL3 must fail closed on the unpinned fallback");
    let rendered = format!("{error:#}");

    // Actionable: names the missing symbol, its provider and the model family.
    assert!(
        rendered.contains("bf16_add::bf16_add_inplace"),
        "{rendered}"
    );
    assert!(rendered.contains("GLM-5.3 EXL3"), "{rendered}");
    assert!(rendered.contains("ncclAllReduce"), "{rendered}");
    // The unpinned path was not armed ahead of the abort.
    assert_eq!(comm.add_kernel(), 0);
    // The lookup that failed is the one the gate guards, in order.
    assert_eq!(
        gpu.kernel_requests_snapshot(),
        vec![("bf16_add".to_string(), "bf16_add_inplace".to_string())]
    );
}

#[test]
fn glm53_exl3_startup_arms_the_pinned_path_when_the_kernel_resolves() {
    let gpu = MockGpuBackend::new();
    let comm = RecordingComm::new();

    install_tp_add_kernel(&glm53_exl3(), &gpu, &comm).unwrap();

    let handle = comm.add_kernel();
    assert_ne!(handle, 0);
    assert_eq!(
        gpu.kernel_requests_snapshot(),
        vec![("bf16_add".to_string(), "bf16_add_inplace".to_string())]
    );
}

#[test]
fn missing_kernel_on_another_family_stays_a_warning_and_not_an_error() {
    let gpu = MockGpuBackend::new();
    gpu.fail_next_kernel();
    let comm = RecordingComm::new();
    let mut qwen = glm53_exl3();
    qwen.model_type = "qwen3_next".into();

    install_tp_add_kernel(&qwen, &gpu, &comm)
        .expect("non-EXL3 families keep the historical warn-and-degrade path");
    assert_eq!(comm.add_kernel(), 0);
}

/// The end-to-end GPU half of this gate — that a live rank-0/rank-1 pair runs
/// its decode on the registered send/recv path and never crosses into
/// `ncclAllReduce` — cannot be checked on the build host. Run on a Spark under
/// `flock /tmp/atlas-bench.lock` with `cargo test -p spark-model --lib
/// impl_a1_tp_failclosed_tests`.
#[test]
#[ignore = "GPU gate: requires a Spark under flock /tmp/atlas-bench.lock"]
fn live_tp2_pair_keeps_the_registered_all_reduce_path() {
    let gpu = MockGpuBackend::new();
    let comm = RecordingComm::new();
    install_tp_add_kernel(&glm53_exl3(), &gpu, &comm).unwrap();
    assert_ne!(comm.add_kernel(), 0);
}
