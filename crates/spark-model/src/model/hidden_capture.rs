// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter-distillation hidden-state capture (`ATLAS_HIDDEN_CAPTURE_DIR`).
//!
//! Research only. Writes the target's hidden rows at the drafter layers
//! (`ATLAS_HIDDEN_CAPTURE_LAYERS`, default the DFlash2 set 5,14,24,33,42)
//! plus the final pre-norm hidden for every prefill chunk of a request, so a
//! drafter can be trained on exactly the features the engine feeds it. The
//! layer rows use the same mHC stream contraction as the DFlash capture
//! (`dflash_capture_rows`); the final slot is the contracted `hidden` the
//! final norm reads.
//!
//! Capture runs on rank 0 only, and only while `<dir>/enable` exists, so the
//! same server can generate (flag absent) and then capture (flag present).
//! Sequence-parallel prefill must be off (`ATLAS_GLM_PREFILL_SP=0`): rank 0
//! would otherwise hold only half of a chunk's rows.
//!
//! One file per chunk, `<dir>/chunks/p<pos0>-n<rows>.bin`: BF16, slot-major
//! `[slots, rows, hidden]` (capture layers in order, then final), with a JSON
//! sidecar carrying the absolute start position, row count, layers and the
//! chunk's token ids. The `.part` file is renamed only when complete.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;

struct Config {
    dir: PathBuf,
    layers: Vec<usize>,
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let dir = std::env::var("ATLAS_HIDDEN_CAPTURE_DIR").ok()?;
            if dir.is_empty() {
                return None;
            }
            let layers = std::env::var("ATLAS_HIDDEN_CAPTURE_LAYERS")
                .unwrap_or_else(|_| "5,14,24,33,42".into())
                .split(',')
                .map(|s| s.trim().parse::<usize>().expect("ATLAS_HIDDEN_CAPTURE_LAYERS"))
                .collect::<Vec<_>>();
            assert!(
                !layers.is_empty() && layers.windows(2).all(|w| w[0] < w[1]),
                "ATLAS_HIDDEN_CAPTURE_LAYERS must be strictly increasing"
            );
            tracing::warn!(dir = %dir, ?layers, "hidden capture configured (research only)");
            Some(Config {
                dir: PathBuf::from(dir),
                layers,
            })
        })
        .as_ref()
}

/// The chunk being captured: its `.part` path, expected slot, and geometry.
struct Active {
    part: PathBuf,
    next_slot: usize,
    pos0: usize,
    rows: usize,
}

/// Device staging rows (grown on demand) and the active chunk.
struct State {
    staging: DevicePtr,
    staging_bytes: usize,
    active: Option<Active>,
}

static STATE: Mutex<State> = Mutex::new(State {
    staging: DevicePtr::NULL,
    staging_bytes: 0,
    active: None,
});

impl TransformerModel {
    fn hidden_capture_rank0(&self) -> bool {
        self.comm.as_ref().is_none_or(|c| c.rank() == 0)
    }

    /// After prefill layer `layer_idx` over `rows` chunk rows starting at
    /// absolute position `pos0`: append that layer's contracted rows.
    pub(super) fn hidden_capture_layer(
        &self,
        layer_idx: usize,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(cfg) = config() else { return Ok(()) };
        let Some(slot) = cfg.layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        if rows == 0 || !self.hidden_capture_rank0() {
            return Ok(());
        }
        let mut state = STATE.lock().unwrap();
        if slot == 0 {
            // A new chunk starts here; a stale `.part` from an aborted chunk
            // is left for the harness to reject.
            state.active = None;
            if !cfg.dir.join("enable").exists() {
                return Ok(());
            }
            ensure!(
                crate::layers::glm_sp::current().is_none(),
                "hidden capture needs ATLAS_GLM_PREFILL_SP=0"
            );
            let chunks = cfg.dir.join("chunks");
            std::fs::create_dir_all(&chunks)?;
            let part = chunks.join(format!("p{pos0:09}-n{rows}.part"));
            std::fs::File::create(&part).with_context(|| format!("create {part:?}"))?;
            state.active = Some(Active {
                part,
                next_slot: 0,
                pos0,
                rows,
            });
        }
        let Some(active) = state.active.as_ref() else {
            return Ok(());
        };
        ensure!(
            active.next_slot == slot && active.pos0 == pos0 && active.rows == rows,
            "hidden capture out of order: slot {slot} (expected {}), rows {rows}@{pos0}",
            active.next_slot
        );
        let h = self.config.hidden_size;
        let bytes = rows * h * 2;
        let staging = self.hidden_capture_staging(&mut state, bytes)?;
        self.dflash_capture_rows(0, rows, staging, h * 2, stream)?;
        self.hidden_capture_append(&mut state, staging, bytes, stream)
    }

    /// After the last layer: append the contracted final hidden (`hidden`,
    /// BF16 `[rows, hidden]`, the final norm's input) and publish the chunk.
    pub(super) fn hidden_capture_final(
        &self,
        tokens: &[u32],
        hidden: DevicePtr,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(cfg) = config() else { return Ok(()) };
        if rows == 0 || !self.hidden_capture_rank0() {
            return Ok(());
        }
        let mut state = STATE.lock().unwrap();
        let Some(active) = state.active.as_ref() else {
            return Ok(());
        };
        ensure!(
            active.next_slot == cfg.layers.len() && active.pos0 == pos0 && active.rows == rows,
            "hidden capture final out of order: slot {} rows {rows}@{pos0}",
            active.next_slot
        );
        let h = self.config.hidden_size;
        self.hidden_capture_append(&mut state, hidden, rows * h * 2, stream)?;
        let active = state.active.take().unwrap();
        let ids = tokens
            .get(pos0..pos0 + rows)
            .with_context(|| format!("hidden capture tokens {pos0}+{rows} of {}", tokens.len()))?;
        let meta = serde_json::json!({
            "schema": "atlas.hidden_capture.chunk/v1",
            "pos0": pos0,
            "rows": rows,
            "hidden_size": h,
            "dtype": "bf16",
            "layout": "slot-major [slots, rows, hidden]",
            "layers": cfg.layers,
            "final_slot": "contracted pre-final-norm hidden",
            "seq_tokens": tokens.len(),
            "token_ids": ids,
        });
        let done = active.part.with_extension("bin");
        std::fs::write(done.with_extension("json"), serde_json::to_vec(&meta)?)?;
        std::fs::rename(&active.part, &done)?;
        Ok(())
    }

    fn hidden_capture_staging(&self, state: &mut State, bytes: usize) -> Result<DevicePtr> {
        if state.staging_bytes < bytes {
            if !state.staging.is_null() {
                self.gpu.free(state.staging)?;
            }
            state.staging = DevicePtr::NULL;
            state.staging_bytes = 0;
            state.staging = self.gpu.alloc(bytes)?;
            state.staging_bytes = bytes;
        }
        Ok(state.staging)
    }

    fn hidden_capture_append(
        &self,
        state: &mut State,
        src: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let mut host = vec![0u8; bytes];
        self.gpu.copy_d2h_on_stream(src, &mut host, stream)?;
        let active = state.active.as_mut().unwrap();
        let mut file = std::fs::OpenOptions::new().append(true).open(&active.part)?;
        file.write_all(&host)?;
        active.next_slot += 1;
        Ok(())
    }
}
