// SPDX-License-Identifier: AGPL-3.0-only

//! The KDA MXFP8 twin on the recording mock: each rank's twin is the matching
//! slice of the checkpoint's bytes, and the switch is inert when off or on a
//! projection the checkpoint does not ship MXFP8.

use super::*;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightTensor};
use std::collections::HashMap;

#[test]
fn the_switches_are_explicit_zero_or_one() {
    assert_eq!(
        KdaMxPolicy::parse(None, None).unwrap(),
        KdaMxPolicy::default()
    );
    let both = KdaMxPolicy::parse(Some("1"), Some("1")).unwrap();
    assert!(both.decode && both.w8a8_prefill);
    let decode = KdaMxPolicy::parse(Some("1"), Some("0")).unwrap();
    assert!(decode.decode && !decode.w8a8_prefill);
    for bad in ["", "true", "2", "on"] {
        assert!(KdaMxPolicy::parse(Some(bad), None).is_err(), "{bad:?}");
        assert!(KdaMxPolicy::parse(None, Some(bad)).is_err(), "{bad:?}");
    }
}

fn upload(gpu: &MockGpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, p).unwrap();
    p
}

fn read(gpu: &MockGpuBackend, p: DevicePtr, len: usize) -> Vec<u8> {
    let mut b = vec![0u8; len];
    gpu.copy_d2h(p, &mut b).unwrap();
    b
}

/// `s.weight` E4M3 `[64, 128]` + UE8M0 `[64, 4]`, distinct bytes per element.
fn store(gpu: &MockGpuBackend) -> (WeightStore, Vec<u8>, Vec<u8>) {
    let (n, k) = (64, 128);
    let w: Vec<u8> = (0..n * k).map(|i| (i * 7 + i / k) as u8).collect();
    let s: Vec<u8> = (0..n * k / 32).map(|i| 100 + (i % 20) as u8).collect();
    let map = HashMap::from([
        (
            "s.weight".to_string(),
            WeightTensor {
                ptr: upload(gpu, &w),
                shape: vec![n, k],
                dtype: WeightDtype::FP8E4M3,
            },
        ),
        (
            "s.weight_scale".to_string(),
            WeightTensor {
                ptr: upload(gpu, &s),
                shape: vec![n, k / 32],
                dtype: WeightDtype::UInt8,
            },
        ),
    ]);
    (WeightStore::from_map(map), w, s)
}

#[test]
fn each_ranks_twin_is_its_slice_of_the_checkpoint_bytes() {
    let (n, k) = (64usize, 128usize);
    let on = KdaMxPolicy {
        decode: true,
        w8a8_prefill: false,
    };
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let (st, w, s) = store(&gpu);
        // q/k/v: column-parallel, the rank's half of the rows.
        let twin = kda_twin(on, &st, "s", TpShardKind::ColumnParallel, rank, 2, 0, &gpu)
            .unwrap()
            .unwrap()
            .decode
            .unwrap();
        let rows = rank * n / 2..(rank + 1) * n / 2;
        assert_eq!(
            read(&gpu, twin.data, n / 2 * k),
            w[rows.start * k..rows.end * k]
        );
        assert_eq!(
            read(&gpu, twin.scales, n / 2 * k / 32),
            s[rows.start * k / 32..rows.end * k / 32]
        );
        // o: row-parallel, the rank's half of every row's K (whole scale blocks).
        let twin = kda_twin(on, &st, "s", TpShardKind::RowParallel, rank, 2, 0, &gpu)
            .unwrap()
            .unwrap()
            .decode
            .unwrap();
        let (lk, g) = (k / 2, k / 32 / 2);
        let want_w: Vec<u8> = (0..n)
            .flat_map(|r| w[r * k + rank * lk..r * k + (rank + 1) * lk].to_vec())
            .collect();
        let want_s: Vec<u8> = (0..n)
            .flat_map(|r| s[r * k / 32 + rank * g..r * k / 32 + (rank + 1) * g].to_vec())
            .collect();
        assert_eq!(read(&gpu, twin.data, n * lk), want_w);
        assert_eq!(read(&gpu, twin.scales, n * lk / 32), want_s);
        // The checkpoint bytes stay for the BF16 expansion and the release.
        assert!(!st.was_reclaimed("s.weight"));
    }
}

#[test]
fn off_or_not_mxfp8_builds_nothing() {
    let gpu = MockGpuBackend::new();
    let (st, _, _) = store(&gpu);
    let allocs = gpu.alloc_count();
    let off = KdaMxPolicy::default();
    assert!(
        kda_twin(off, &st, "s", TpShardKind::ColumnParallel, 0, 2, 0, &gpu)
            .unwrap()
            .is_none()
    );
    let on = KdaMxPolicy {
        decode: true,
        w8a8_prefill: false,
    };
    // The NVFP4 release's BF16 projection, and an absent one.
    let bf16 = WeightStore::from_map(HashMap::from([(
        "b.weight".to_string(),
        WeightTensor {
            ptr: gpu.alloc(64 * 128 * 2).unwrap(),
            shape: vec![64, 128],
            dtype: WeightDtype::BF16,
        },
    )]));
    for (store, stem) in [(&bf16, "b"), (&st, "absent")] {
        assert!(
            kda_twin(on, store, stem, TpShardKind::RowParallel, 0, 2, 0, &gpu)
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(
        gpu.alloc_count(),
        allocs + 1,
        "only the BF16 fixture allocated"
    );
    assert!(kda_twin(on, &st, "s", TpShardKind::Replicated, 0, 2, 0, &gpu).is_err());
}

/// W8A8 reads the twin's E4M3 bytes in place (no copy) with its own tiled
/// scale image; alone it frees the row-major scales, with decode it keeps
/// one untiled twin for both, and only then are NVFP4 copies unneeded.
#[test]
fn w8a8_shares_the_twin_bytes_and_serves_all_only_with_decode() {
    let (n, k) = (64usize, 128usize);
    for decode in [false, true] {
        let gpu = MockGpuBackend::new();
        let (st, w, _) = store(&gpu);
        let policy = KdaMxPolicy {
            decode,
            w8a8_prefill: true,
        };
        // m_max 0: no cuBLASLt tuning at load (the mock has none).
        let mx = kda_twin(policy, &st, "s", TpShardKind::ColumnParallel, 1, 2, 0, &gpu)
            .unwrap()
            .unwrap();
        let op = mx.w8a8.as_ref().unwrap();
        assert_eq!((op.n, op.k), (n / 2, k));
        assert_eq!(read(&gpu, op.data, n / 2 * k), w[n / 2 * k..]);
        assert_eq!(mx.serves_all(), decode);
        match &mx.decode {
            Some(twin) => {
                assert_eq!(twin.data, op.data, "one copy of the E4M3 bytes");
                assert!(!twin.tiled, "W8A8 needs the untiled layout");
                assert!(gpu.read_alloc(twin.scales).is_some());
            }
            None => assert!(!decode),
        }
    }
}
