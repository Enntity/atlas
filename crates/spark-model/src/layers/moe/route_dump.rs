// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP=<file>` (diagnostic): the expert ids of
//! every rows-pair MoE launch (`forward_rows.rs`), one line of `rows * top_k`
//! space-separated ids per launch, appended to `<file>`. The unique local
//! experts per layer per step -- what a C8 step's routed MoE streams -- come
//! from these; `scripts/dev/qwen4exp_moe_c8_bench.cu` replays them
//! (`ROUTES=<file> ROUTE_ROWS=<rows>`). Synchronizes the stream per launch:
//! a measurement run, not a timing one. Launches recorded into a CUDA graph
//! cannot be read at capture and are skipped (run with the decode graphs off
//! to see every step).

use std::io::Write;
use std::sync::Mutex;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// Appends the `slots` u32 ids at `indices` (after `stream`'s work) to `path`.
pub(super) fn append(
    gpu: &dyn GpuBackend,
    indices: DevicePtr,
    slots: usize,
    path: &str,
    stream: u64,
) -> Result<()> {
    if gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let mut raw = vec![0u8; slots * 4];
    gpu.copy_d2h(indices, &mut raw)?;
    let mut line = String::with_capacity(slots * 4);
    for (i, id) in raw.chunks_exact(4).enumerate() {
        let id = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
        line.push_str(if i == 0 { "" } else { " " });
        line.push_str(&id.to_string());
    }
    line.push('\n');
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if file.is_none() {
        *file = Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("ATLAS_QWEN4EXP_MOE_ROUTE_DUMP: open {path}"))?,
        );
    }
    if let Some(f) = file.as_mut() {
        f.write_all(line.as_bytes())?;
    }
    Ok(())
}
