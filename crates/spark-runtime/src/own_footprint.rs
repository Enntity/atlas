// SPDX-License-Identifier: AGPL-3.0-only

//! This process's own memory, measured without reference to free memory.
//!
//! KV sizing takes "what Atlas holds" as free-at-init minus free-now. On
//! unified memory that difference moves whenever ANOTHER process frees or
//! allocates during the load: memory released by a co-tenant is credited to
//! Atlas as memory it never took, the pool is sized into it, and the host
//! runs out (a GB10 then hangs). The counters here belong to this process
//! alone, so nothing a co-tenant does can lower them.
//!
//! Measured on GB10 (driver 580.178, 2026-09-30):
//!
//! * `cuMemAlloc` is NOT in the process's RSS (`RssAnon` +0 for 1 GiB) and is
//!   committed at allocation, before first touch. The driver's per-process
//!   accounting (NVML `usedGpuMemory`) counts it exactly, including its 64 KiB
//!   rounding, and answers unprivileged inside a container with the PID
//!   translated to the container's namespace.
//! * `cuMemAllocHost` (page-locked host memory) is in `RssShmem`, not in
//!   `RssAnon` and not in the driver's figure.
//! * `RssFile` is page cache (the mmapped safetensors), which `MemAvailable`
//!   still counts as available, so it is left out.
//!
//! So own = device (driver accounting, or the backend's allocation ledger
//! where that is unavailable or lower) + host (`RssAnon` + `RssShmem`), each
//! as growth since the backend was created, which is where the free-memory
//! baseline is taken too.

/// Where [`OwnFootprint::device`] came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceSource {
    /// The driver's per-process accounting: every device allocation in the
    /// process, whoever made it.
    Driver,
    /// Requested bytes of the backend's live allocations. A lower bound: it
    /// misses driver rounding and the workspaces that call `cuMemAlloc`
    /// directly (CUTLASS, cuBLASLt, FlashInfer, NCCL).
    Ledger,
}

impl DeviceSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Driver => "driver accounting",
            Self::Ledger => "allocation ledger",
        }
    }
}

/// Memory this process took since its GPU backend was created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnFootprint {
    pub device: usize,
    pub device_source: DeviceSource,
    pub host: usize,
}

impl OwnFootprint {
    pub fn total(&self) -> usize {
        self.device.saturating_add(self.host)
    }
}

/// One reading of the process-wide counters. `None` = not readable here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    /// Driver-accounted device bytes of this process.
    pub driver: Option<usize>,
    /// `RssAnon` + `RssShmem`.
    pub host: Option<usize>,
}

/// Growth from `base` to `now`, with `ledger` as the device floor.
///
/// A counter missing from either sample contributes nothing rather than a
/// guess, so the result can only under-state — and the caller takes the
/// larger of this and its free-memory delta.
pub fn since(base: Sample, now: Sample, ledger: usize) -> OwnFootprint {
    let grown = |b: Option<usize>, n: Option<usize>| Some(n?.saturating_sub(b?));
    let (device, device_source) = match grown(base.driver, now.driver) {
        Some(driver) if driver >= ledger => (driver, DeviceSource::Driver),
        _ => (ledger, DeviceSource::Ledger),
    };
    OwnFootprint {
        device,
        device_source,
        host: grown(base.host, now.host).unwrap_or(0),
    }
}

/// `RssAnon` + `RssShmem` out of a `/proc/<pid>/status` text.
pub fn parse_host_bytes(status: &str) -> Option<usize> {
    let kb = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))?
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
    };
    kb("RssAnon:")?
        .checked_add(kb("RssShmem:")?)?
        .checked_mul(1024)
}

/// This process's `RssAnon` + `RssShmem`; `None` off Linux.
pub fn host_bytes() -> Option<usize> {
    parse_host_bytes(&std::fs::read_to_string("/proc/self/status").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    fn sample(driver: Option<usize>, host: Option<usize>) -> Sample {
        Sample { driver, host }
    }

    #[test]
    fn driver_growth_wins_over_a_smaller_ledger() {
        // Context + modules (0.3 GiB) predate the baseline and cancel out; the
        // driver also sees the 0.6 GiB of workspaces the ledger never did.
        let got = since(
            sample(Some(GIB * 3 / 10), Some(GIB / 10)),
            sample(Some(101 * GIB + GIB * 3 / 10), Some(GIB)),
            100 * GIB + GIB * 4 / 10,
        );
        assert_eq!(got.device, 101 * GIB);
        assert_eq!(got.device_source, DeviceSource::Driver);
        assert_eq!(got.host, GIB - GIB / 10);
        assert_eq!(got.total(), 101 * GIB + GIB - GIB / 10);
    }

    #[test]
    fn ledger_is_the_floor_when_the_driver_is_silent_or_lower() {
        for (base, now) in [
            (None, None),
            (None, Some(90 * GIB)),
            (Some(GIB), None),
            (Some(GIB), Some(50 * GIB)),
        ] {
            let got = since(sample(base, None), sample(now, None), 100 * GIB);
            assert_eq!(got.device, 100 * GIB);
            assert_eq!(got.device_source, DeviceSource::Ledger);
            assert_eq!(got.host, 0, "an unreadable host counter adds nothing");
        }
    }

    #[test]
    fn a_counter_that_shrank_reads_as_zero_not_as_a_wraparound() {
        let got = since(
            sample(Some(2 * GIB), Some(2 * GIB)),
            sample(Some(GIB), Some(GIB)),
            0,
        );
        assert_eq!((got.device, got.host), (0, 0));
    }

    #[test]
    fn host_bytes_are_anon_plus_shmem_and_never_file() {
        let status = "Name:\tspark\nVmRSS:\t  789576 kB\nRssAnon:\t  580508 kB\n\
                      RssFile:\t  101760 kB\nRssShmem:\t  107308 kB\nThreads:\t9\n";
        assert_eq!(parse_host_bytes(status), Some((580508 + 107308) * 1024));
    }

    #[test]
    fn a_status_without_the_split_rss_fields_is_unreadable() {
        assert_eq!(parse_host_bytes("VmRSS:\t  789576 kB\n"), None);
        assert_eq!(parse_host_bytes("RssAnon:\t  1 kB\n"), None);
        assert_eq!(parse_host_bytes("RssAnon:\tx kB\nRssShmem:\t1 kB\n"), None);
        assert_eq!(parse_host_bytes(""), None);
    }
}
