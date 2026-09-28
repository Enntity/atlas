// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};

fn plan(tokens: &[u32], blocks: &[u32], row: usize) -> anyhow::Result<KvRowsPlan> {
    KvRowsPlan::new(
        tokens,
        DeviceSpan {
            ptr: DevicePtr(0x10000),
            bytes: 80,
        },
        4,
        8,
        row,
        4,
        blocks,
        8,
        2,
        65536,
        &[],
    )
}

#[test]
fn shifted_rows_cross_shuffled_blocks_without_touching_other_slots() {
    let p = plan(&[1, 3, 2, 7, 0], &[3, 1, 6], 3).unwrap();
    assert_eq!(p.slots, [15, 4, 5, 6, 7]);
    assert_eq!(p.embedding_offsets, [8, 24, 16, 56, 0]);
    assert_eq!(p.rows, 5);
    assert_eq!(p.row_bytes, 8);
    assert_eq!(p.chunk_rows, 2);
    assert!(use_cublas(true, 2));
    assert!(!use_cublas(true, 1));
    assert!(!use_cublas(false, 2));
}

#[test]
fn malformed_whole_request_rejected_before_gpu_operations() {
    assert!(plan(&[1, 8], &[3, 1], 0).is_err());
    // Unlike the main target cache, this private MTP pool does not reserve 0.
    assert_eq!(plan(&[1], &[0, 1], 0).unwrap().slots, [0]);
    for blocks in [&[3, 3][..], &[3, 8][..]] {
        assert!(plan(&[1], blocks, 0).is_err());
    }
    assert!(plan(&[1, 2], &[3], 3).is_err());
    assert!(plan(&[1], &[3], usize::MAX).is_err());
    let source = DeviceSpan {
        ptr: DevicePtr(0x10000),
        bytes: 7,
    };
    assert!(KvRowsPlan::new(&[1], source, 4, 8, 0, 4, &[3], 8, 2, 65536, &[]).is_err());
    let source = DeviceSpan {
        ptr: DevicePtr(0x10000),
        bytes: 80,
    };
    assert!(KvRowsPlan::new(&[1], source, 4, 8, 0, 4, &[3], 8, 2, 65536, &[source]).is_err());
    assert!(KvRowsPlan::new(&[1], source, 4, 8, 0, 4, &[3], 8, 2, 0, &[]).is_err());
    assert!(KvRowsPlan::new(&[1], source, usize::MAX, 8, 0, 4, &[3], 8, 2, 65536, &[]).is_err());
    assert!(KvRowsPlan::new(&[1], source, 4, 8, 0, usize::MAX, &[1], 8, 2, 65536, &[]).is_err());
    let adjacent = DeviceSpan {
        ptr: DevicePtr(0x10050),
        bytes: 8,
    };
    assert!(KvRowsPlan::new(&[1], source, 4, 8, 0, 4, &[3], 8, 2, 65536, &[adjacent]).is_ok());
}

#[test]
fn planned_embedding_copies_and_slot_upload_use_real_mock_memory() {
    let gpu = MockGpuBackend::new();
    let embedding = gpu.alloc(64).unwrap();
    let bytes: Vec<u8> = (0..64).collect();
    gpu.copy_h2d(&bytes, embedding).unwrap();
    let output = gpu.alloc(16).unwrap();
    let metadata = gpu.alloc(16).unwrap();
    let p = plan(&[1, 3, 2, 7, 0], &[3, 1, 6], 3).unwrap();
    p.copy_embeddings(&gpu, embedding, output, 2, 2, 0).unwrap();
    p.upload_slots(&gpu, metadata, 2, 2, 0).unwrap();
    assert_eq!(
        gpu.read_alloc(output).unwrap(),
        [bytes[16..24].to_vec(), bytes[56..64].to_vec()].concat()
    );
    assert_eq!(
        gpu.read_alloc(metadata).unwrap(),
        [5i64.to_le_bytes(), 6i64.to_le_bytes()].concat()
    );
    assert_eq!(gpu.d2d_count(), 2);
    assert_eq!(gpu.launch_count(), 0);
    assert!(
        p.copy_embeddings(&gpu, embedding, output, usize::MAX, 1, 0)
            .is_err()
    );
    assert!(p.upload_slots(&gpu, metadata, 4, 2, 0).is_err());
    assert_eq!(gpu.d2d_count(), 2);
}
