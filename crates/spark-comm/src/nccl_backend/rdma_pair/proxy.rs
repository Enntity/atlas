// SPDX-License-Identifier: AGPL-3.0-only

//! The RDMA pair proxy thread: posts each staged segment as RDMA WRITEs and
//! then signals the peer's `arrived` word.

use super::{ARRIVED, FLAG_SRC, Job, Peer, READY, SPLIT_MIN, flag_off, recv_off, send_off};
use anyhow::Result;
use atlas_rdma::Verbs;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(super) fn proxy_loop(
    mut rails: Vec<Verbs>,
    lkeys: &[u32],
    peer: &Peer,
    host: usize,
    capacity: usize,
    jobs: &Mutex<VecDeque<Job>>,
    stop: &AtomicBool,
) -> Result<()> {
    let flags = host + flag_off(capacity);
    // SAFETY: the flag page lives in the pinned region for the pair's lifetime;
    // `ready` is written by the GPU (stream memop) and read only here.
    let ready = unsafe { &*((flags + READY) as *const AtomicU64) };
    let mut idle = 0u32;
    let stats = std::env::var("ATLAS_RDMA_PAIR_STATS").as_deref() == Ok("1");
    // [ready wait, data, flag] microseconds and small/large job counts.
    let (mut acc, mut jobs_small, mut jobs_large) = ([0f64; 3], 0u64, 0u64);
    loop {
        let Some(job) = jobs.lock().pop_front() else {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            idle = idle.saturating_add(1);
            if idle > 20_000 {
                std::thread::sleep(std::time::Duration::from_micros(50));
            } else {
                std::hint::spin_loop();
            }
            continue;
        };
        idle = 0;
        let slot = (job.seq & 1) as usize;
        let bytes: usize = job.parts.iter().map(|p| p.1).sum();
        for (i, &(seg_off, seg_len)) in job.parts.iter().enumerate() {
            let count = job.first + i as u64 + 1;
            let t0 = std::time::Instant::now();
            while ready.load(Ordering::Acquire) < count {
                if stop.load(Ordering::Acquire) {
                    return Ok(());
                }
                std::hint::spin_loop();
            }
            let src = host + send_off(capacity, slot) + seg_off;
            let dst = peer.base + (recv_off(capacity, slot) + seg_off) as u64;
            let used = if seg_len >= SPLIT_MIN { rails.len() } else { 1 };
            let part = seg_len.div_ceil(used).next_multiple_of(64);
            let mut posted = 0;
            for (r, rail) in rails.iter_mut().enumerate().take(used) {
                let off = r * part;
                if off >= seg_len {
                    break;
                }
                let len = part.min(seg_len - off);
                // SAFETY: src..+len lies in the registered region (send slot);
                // the slot is not rewritten until this seq's flag is consumed.
                unsafe {
                    rail.post_write(
                        (src + off) as *mut c_void,
                        lkeys[r],
                        dst + off as u64,
                        peer.rkeys[r],
                        u32::try_from(len)?,
                        count,
                    )
                }?;
                posted += 1;
            }
            let t1 = std::time::Instant::now();
            for rail in rails.iter_mut().take(posted) {
                rail.poll()?;
            }
            let t2 = std::time::Instant::now();
            let src_flag = flags + FLAG_SRC + slot * 8;
            // SAFETY: the flag source word is ours; the previous WRITE from this
            // word completed before that segment's poll returned.
            unsafe { (src_flag as *mut u64).write_volatile(count) };
            unsafe {
                rails[0].post_write(
                    src_flag as *mut c_void,
                    lkeys[0],
                    peer.base + (flag_off(capacity) + ARRIVED) as u64,
                    peer.rkeys[0],
                    8,
                    count,
                )
            }?;
            rails[0].poll()?;
            if stats {
                let t3 = std::time::Instant::now();
                acc[0] += (t1 - t0).as_secs_f64() * 1e6;
                acc[1] += (t2 - t1).as_secs_f64() * 1e6;
                acc[2] += (t3 - t2).as_secs_f64() * 1e6;
                if bytes >= SPLIT_MIN {
                    jobs_large += 1;
                } else {
                    jobs_small += 1;
                }
                let n = jobs_small + jobs_large;
                if n % 4096 == 0 {
                    tracing::info!(
                        "RDMA pair stats: {n} jobs ({jobs_small} small, {jobs_large} large); mean us: ready-wait {:.1} data {:.1} flag {:.1}",
                        acc[0] / n as f64,
                        acc[1] / n as f64,
                        acc[2] / n as f64
                    );
                }
            }
        }
    }
}
