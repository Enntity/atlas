// SPDX-License-Identifier: AGPL-3.0-only
//! Periodic info-level logging of model-loading progress.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, spawn};
use std::time::{Duration, Instant};

#[derive(Clone)]
struct State {
    total_bytes: u64,
    loaded_bytes: u64,
    current_file: String,
    start: Instant,
    running: bool,
}

pub struct LoadProgress {
    inner: Arc<Mutex<State>>,
    handle: Option<JoinHandle<()>>,
    interval_secs: u64,
}

impl LoadProgress {
    pub fn new(total_bytes: u64, interval_secs: u64) -> Self {
        if interval_secs == 0 {
            return Self {
                inner: Arc::new(Mutex::new(State {
                    total_bytes,
                    loaded_bytes: 0,
                    current_file: "loading".to_string(),
                    start: Instant::now(),
                    running: false,
                })),
                handle: None,
                interval_secs: 0,
            };
        }
        let inner = Arc::new(Mutex::new(State {
            total_bytes,
            loaded_bytes: 0,
            current_file: "loading".to_string(),
            start: Instant::now(),
            running: true,
        }));
        let inner_c = inner.clone();
        let iv = interval_secs;
        let handle = spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(iv));
                let running = {
                    let s = inner_c.lock().unwrap();
                    s.running
                };
                if !running {
                    break;
                }
                let (loaded, total, file, elapsed) = {
                    let s = inner_c.lock().unwrap();
                    (
                        s.loaded_bytes,
                        s.total_bytes,
                        s.current_file.clone(),
                        s.start.elapsed().as_secs_f64(),
                    )
                };
                let bw = if elapsed > 0.0 {
                    loaded as f64 / elapsed
                } else {
                    0.0
                };
                let pct = if total > 0 {
                    (loaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                tracing::info!(
                    target: "atlas.load_progress",
                    loaded_bytes = loaded,
                    total_bytes = total,
                    file = %file,
                    bandwidth_bytes_per_sec = bw,
                    progress_pct = pct,
                    "load_progress"
                );
            }
        });
        Self {
            inner,
            handle: Some(handle),
            interval_secs,
        }
    }

    pub fn update(&self, loaded_bytes: u64, current_file: &str) {
        if self.interval_secs == 0 {
            return;
        }
        let mut s = self.inner.lock().unwrap();
        s.loaded_bytes = loaded_bytes;
        s.current_file = current_file.to_string();
    }

    pub fn stop(mut self) {
        {
            let mut s = self.inner.lock().unwrap();
            s.running = false;
        }
        if let Some(h) = self.handle.take() {
            let _ = h; // detached; loop exits on next sleep
        }
    }
}

impl Drop for LoadProgress {
    fn drop(&mut self) {
        {
            let mut s = self.inner.lock().unwrap();
            s.running = false;
        }
        // Detached; loop exits on next sleep because running=false.
        drop(self.handle.take());
    }
}
