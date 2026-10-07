// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP=<file>` (diagnostic): the expert ids of
//! every rows-pair MoE launch (`forward_rows.rs`), one line of `rows * top_k`
//! space-separated ids per launch, appended to `<file>`. The unique local
//! experts per layer per step -- what a C8 step's routed MoE streams -- come
//! from these; `scripts/dev/qwen4exp_moe_c8_bench.cu` replays them
//! (`ROUTES=<file> ROUTE_ROWS=<rows>`).
//!
//! The first `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP_MAX` launches (default 384: eight
//! steps of 48 layers) also append a record to `<file>.bin` with the launch's
//! MoE input rows, for `scripts/dev/qwen4exp_moe_fidelity.cu`. Record, little
//! endian: u32 magic 0x31444D51 ("QMD1"), u32 layer (order of the layer's
//! first launch), u32 rows, u32 top_k, u32 hidden, then u32 ids
//! [rows * top_k], f32 routing weights [rows * top_k], BF16 input
//! [rows * hidden].
//!
//! Synchronizes the stream per launch: a measurement run, not a timing one.
//! Launches recorded into a CUDA graph cannot be read at capture and are
//! skipped (run with the decode graphs off to see every step).

use std::io::Write;
use std::sync::Mutex;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

struct Dump {
    text: Option<std::fs::File>,
    bin: Option<std::fs::File>,
    layers: Vec<u64>,
    records: usize,
}

static DUMP: Mutex<Dump> = Mutex::new(Dump {
    text: None,
    bin: None,
    layers: Vec::new(),
    records: 0,
});

fn open(path: &str) -> Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("ATLAS_QWEN4EXP_MOE_ROUTE_DUMP: open {path}"))
}

fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut v)?;
    Ok(v)
}

/// The routing of one rows-pair launch: `rows` rows of `top_k` ids at
/// `indices` and weights at `weights`, its `[rows, hidden]` BF16 `input`;
/// `layer` identifies the layer (its expert table address).
pub(super) struct Launch {
    pub indices: DevicePtr,
    pub weights: DevicePtr,
    pub input: DevicePtr,
    pub rows: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub layer: u64,
}

/// Appends `l` (after `stream`'s work) to `path` and, within the record cap,
/// to `path.bin`.
pub(super) fn append(gpu: &dyn GpuBackend, l: &Launch, path: &str, stream: u64) -> Result<()> {
    if gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let slots = l.rows * l.top_k;
    let ids = read(gpu, l.indices, slots * 4)?;
    let mut line = String::with_capacity(slots * 4);
    for (i, id) in ids.chunks_exact(4).enumerate() {
        let id = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
        line.push_str(if i == 0 { "" } else { " " });
        line.push_str(&id.to_string());
    }
    line.push('\n');
    let mut d = DUMP.lock().unwrap_or_else(|e| e.into_inner());
    if d.text.is_none() {
        d.text = Some(open(path)?);
    }
    if let Some(f) = d.text.as_mut() {
        f.write_all(line.as_bytes())?;
    }
    let cap = std::env::var("ATLAS_QWEN4EXP_MOE_ROUTE_DUMP_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(384usize);
    if d.records >= cap {
        return Ok(());
    }
    d.records += 1;
    let layer = match d.layers.iter().position(|&k| k == l.layer) {
        Some(i) => i,
        None => {
            d.layers.push(l.layer);
            d.layers.len() - 1
        }
    };
    let mut rec = Vec::with_capacity(20 + slots * 8 + l.rows * l.hidden * 2);
    for w in [
        0x3144_4D51u32,
        layer as u32,
        l.rows as u32,
        l.top_k as u32,
        l.hidden as u32,
    ] {
        rec.extend_from_slice(&w.to_le_bytes());
    }
    rec.extend_from_slice(&ids);
    rec.extend_from_slice(&read(gpu, l.weights, slots * 4)?);
    rec.extend_from_slice(&read(gpu, l.input, l.rows * l.hidden * 2)?);
    if d.bin.is_none() {
        d.bin = Some(open(&format!("{path}.bin"))?);
    }
    if let Some(f) = d.bin.as_mut() {
        f.write_all(&rec)?;
    }
    Ok(())
}
