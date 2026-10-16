// SPDX-License-Identifier: AGPL-3.0-only

//! Both safetensors loaders turn CSF pairs into decoded `weight_scale` tensors.
//!
//! The checkpoint built here is shaped like FP4-Local where it matters: the
//! index routes to shards under a `tensors/` SUBDIRECTORY, and each CSF pair is
//! split ACROSS the two shards (so a loader that decodes per shard fails). The
//! bytes are the real golden pairs, so "decoded" is checked against vLLM's
//! reference by sha256, not against this crate's own decoder.
#![cfg(unix)]

use super::*;
use crate::fast_weights::FastSafetensorsLoader;
use crate::gpu::mock::MockGpuBackend;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

const GATE: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale";
const DOWN: &str = "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale";
const PLAIN: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("atlas-csf-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(p.join("tensors")).unwrap();
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/nvfp4_csf")
            .join(name),
    )
    .unwrap()
}

fn golden_sha(tag: &str) -> String {
    let doc: serde_json::Value = serde_json::from_slice(&fixture("golden.json")).unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["tag"] == tag)
        .unwrap()["decoded_sha256"]
        .as_str()
        .unwrap()
        .to_string()
}

type T<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

fn write_shard(path: &Path, tensors: &[T<'_>]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}),
        );
        off += bytes.len();
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&json).unwrap();
    for (_, _, _, bytes) in tensors {
        f.write_all(bytes).unwrap();
    }
}

/// Two shards under `tensors/`, each CSF pair split across them.
fn checkpoint(dir: &Path, with_gate_exceptions: bool) {
    let gate_f = fixture("l0_e0_gate.fixed.bin");
    let gate_e = fixture("l0_e0_gate.exceptions.bin");
    let down_f = fixture("l0_e0_down.fixed.bin");
    let down_e = fixture("l0_e0_down.exceptions.bin");
    let fx = |n: &str| format!("{n}.nvfp4_csf_fixed");
    let ex = |n: &str| format!("{n}.nvfp4_csf_exceptions");
    let (gf, ge, df, de) = (fx(GATE), ex(GATE), fx(DOWN), ex(DOWN));
    write_shard(
        &dir.join("tensors/a.safetensors"),
        &[
            (&gf, "U8", vec![40, 1296], gate_f),
            (&de, "U32", vec![down_e.len() / 4], down_e),
            (PLAIN, "U8", vec![4, 2], vec![7; 8]),
        ],
    );
    let mut b: Vec<T<'_>> = vec![(&df, "U8", vec![160, 336], down_f)];
    if with_gate_exceptions {
        b.push((&ge, "U32", vec![gate_e.len() / 4], gate_e));
    }
    write_shard(&dir.join("tensors/b.safetensors"), &b);
    let mut map = serde_json::Map::new();
    for (n, s) in [
        (&gf, "a"),
        (&de, "a"),
        (&PLAIN.to_string(), "a"),
        (&df, "b"),
    ] {
        map.insert(n.clone(), format!("tensors/{s}.safetensors").into());
    }
    if with_gate_exceptions {
        map.insert(ge.clone(), "tensors/b.safetensors".into());
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
}

fn check(store: &WeightStore, gpu: &MockGpuBackend, which: &str) {
    for (name, tag, shape) in [
        (GATE, "l0_e0_gate", [640, 160]),
        (DOWN, "l0_e0_down", [2560, 40]),
    ] {
        let t = store.get(name).unwrap_or_else(|e| panic!("{which}: {e}"));
        assert_eq!(t.dtype, WeightDtype::FP8E4M3, "{which}: {name}");
        assert_eq!(t.shape, shape.to_vec(), "{which}: {name}");
        let bytes = gpu.read_alloc(t.ptr).unwrap();
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha, golden_sha(tag), "{which}: {name} decoded wrong");
    }
    assert!(store.contains(PLAIN), "{which}: plain tensor lost");
    assert!(
        store.names().all(|n| nvfp4_csf::csf_part(n).is_none()),
        "{which}: a CSF stream reached the store"
    );
    assert_eq!(store.len(), 3, "{which}");
}

#[test]
fn both_loaders_decode_cross_shard_pairs_from_a_subdirectory_index() {
    let dir = Dir::new("ok");
    checkpoint(&dir.0, true);

    let gpu = MockGpuBackend::new();
    let store = SafetensorsLoader::new().load(&dir.0, &gpu, 0).unwrap();
    check(&store, &gpu, "mmap");

    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    check(&store, &gpu, "fast");

    // The pre-flight counts each pair at its decoded size.
    let shards = [
        dir.0.join("tensors/a.safetensors"),
        dir.0.join("tensors/b.safetensors"),
    ];
    assert_eq!(
        estimate_load_bytes(&shards, &|_| false).unwrap(),
        640 * 160 + 2560 * 40 + 8
    );
}

#[test]
fn a_lone_csf_half_refuses_the_load() {
    let dir = Dir::new("lone");
    checkpoint(&dir.0, false);
    let gpu = MockGpuBackend::new();
    let e = format!(
        "{:#}",
        SafetensorsLoader::new()
            .load(&dir.0, &gpu, 0)
            .err()
            .unwrap()
    );
    assert!(e.contains("without a partner"), "{e}");
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let e = format!("{:#}", fast.load(&dir.0, &gpu, 0).err().unwrap());
    assert!(e.contains("without a partner"), "{e}");
}

/// Under EP each rank keeps its own experts: both halves of a remote expert's
/// pair are skipped together, so nothing is left unpaired.
#[test]
fn ep_skips_both_halves_of_a_remote_pair() {
    let dir = Dir::new("ep");
    checkpoint(&dir.0, true);
    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::with_ep(1, 2, 2);
    fast.try_direct_io = false;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    assert!(store.contains(DOWN) && !store.contains(GATE) && !store.contains(PLAIN));
    let store = SafetensorsLoader::with_ep(1, 2, 2)
        .load(&dir.0, &gpu, 0)
        .unwrap();
    assert!(store.contains(DOWN) && !store.contains(GATE));
}
