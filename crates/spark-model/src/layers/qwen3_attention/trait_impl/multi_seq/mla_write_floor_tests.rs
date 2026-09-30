// SPDX-License-Identifier: AGPL-3.0-only
//! The GLM paged prefill honours the Marconi KV write floor: rows below it
//! replay positions held in shared prefix-cache blocks and must reach neither
//! the latent cache nor the semantic index. Recorded launches establish the
//! written row ranges, not CUDA numerics.
use super::*;

const LATENT_WRITE: u64 = 821;
const TAIL_WRITE: u64 = 801;
const POOL_FINALIZE: u64 = 802;
const KEY_ROW: usize = 128 * 2;
const LATENT_ROW: usize = 512 * 2;

fn ptr(p: DevicePtr) -> Vec<u8> {
    p.0.to_ne_bytes().to_vec()
}

/// One cache write: `(kernel, row-major source pointers, slots, rows)`.
type Write = (u64, Vec<Vec<u8>>, Vec<u8>, u32);

/// Cache writes of one `prefill_attention_paged` call over `rows` rows from
/// `seq_len_start`, in launch order.
fn writes(seq_len_start: usize, rows: usize, floor: usize) -> (Vec<Write>, [DevicePtr; 2]) {
    let mut out = None;
    fixture(|gpu, config, layer| {
        let arena = BufferArena::new(config, 64, 32768, 16, 1, gpu).unwrap();
        let mut dispatch = ops::GemmDispatch::defaults();
        dispatch.cublas_gemm = false;
        let derived = ops::DerivedWeights::new();
        let levers = ops::ModelLevers::defaults();
        let stats = ops::ModelStats::new();
        let base = gpu.alloc(4 * 4096).unwrap();
        let meta = AttnMetadataDev {
            positions: base,
            positions_h: base,
            positions_w: base,
            slot: base.offset(4096),
            seq_len: base.offset(2 * 4096),
            block_table: base.offset(3 * 4096),
            max_blocks_per_seq: 256,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let ctx = ForwardContext {
            buffers: &arena,
            gpu,
            config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            ssm_batch: None,
            attn_metadata: Some(meta),
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: true,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            256,
            gpu,
        )
        .unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        let (typed_before, launch_before) = (gpu.1.lock().unwrap().len(), gpu.launch_count());
        layer
            .prefill_attention_paged(
                &mut crate::layer::EmptyLayerState,
                arena.norm_output(),
                rows,
                seq_len_start,
                &mut cache,
                &Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
                None,
                floor,
                &ctx,
                0,
            )
            .unwrap();
        let is_write = |k: u64| [LATENT_WRITE, TAIL_WRITE, POOL_FINALIZE].contains(&k);
        let grids: Vec<u32> = gpu.launches_snapshot()[launch_before..]
            .iter()
            .filter(|l| is_write(l.func))
            .map(|l| l.grid[0])
            .collect();
        let typed = gpu.1.lock().unwrap();
        let calls: Vec<_> = typed[typed_before..]
            .iter()
            .filter(|(k, _)| is_write(*k))
            .collect();
        assert_eq!(calls.len(), grids.len(), "every cache write is typed");
        let found = calls
            .into_iter()
            .zip(grids)
            .map(|((k, a), rows)| {
                if *k == LATENT_WRITE {
                    assert_eq!(a[2], ptr(cache.k_pool_ptr(0)), "latent pool");
                    return (*k, a[..2].to_vec(), a[4].clone(), rows);
                }
                // The index kernels also bound their rows by `num_tokens`.
                assert_eq!(a[5], rows.to_ne_bytes().to_vec(), "kernel {k} rows");
                // The finalize reads the raw tails, not the scratch keys.
                let sources = if *k == TAIL_WRITE {
                    a[..2].to_vec()
                } else {
                    Vec::new()
                };
                (*k, sources, a[4].clone(), rows)
            })
            .collect();
        out = Some((found, [arena.ssm_qkvz(), meta.slot]));
    });
    out.unwrap()
}

/// The index writes an owner of `rows` rows at stacked row `row0` issues when
/// its first `skip` rows are floored.
fn owner_index_writes(
    [scratch, slot]: [DevicePtr; 2],
    row0: usize,
    rows: usize,
    skip: usize,
) -> Vec<Write> {
    let slots = ptr(slot.offset((row0 + skip) * 8));
    let keys = scratch.offset(skip * KEY_ROW);
    let gates = scratch.offset((rows + skip) * KEY_ROW);
    let n = (rows - skip) as u32;
    vec![
        (TAIL_WRITE, vec![ptr(keys), ptr(gates)], slots.clone(), n),
        (POOL_FINALIZE, Vec::new(), slots, n),
    ]
}

fn latent_write([scratch, slot]: [DevicePtr; 2], total: usize, floor: usize) -> Write {
    (
        LATENT_WRITE,
        vec![
            ptr(scratch.offset(floor * LATENT_ROW)),
            ptr(scratch.offset((total + floor) * LATENT_ROW)),
        ],
        ptr(slot.offset(floor * 8)),
        (total - floor) as u32,
    )
}

#[test]
fn no_floor_writes_every_row_from_the_unshifted_operands() {
    // Floor 0 is every cold prefill: the launches carry the base pointers and
    // the full row count, i.e. exactly what the path issued before the floor.
    let (found, at) = writes(32, 48, 0);
    let mut want = vec![latent_write(at, 48, 0)];
    want.extend(owner_index_writes(at, 0, 48, 0));
    assert_eq!(found, want);
}

#[test]
fn warm_replay_rows_below_the_floor_are_not_written() {
    // Snapshot at 32, radix match at 48: rows [0, 16) replay shared blocks.
    let (found, at) = writes(32, 48, 16);
    let mut want = vec![latent_write(at, 48, 16)];
    want.extend(owner_index_writes(at, 0, 48, 16));
    assert_eq!(found, want);
}

#[test]
fn a_chunk_wholly_below_the_floor_writes_nothing() {
    for floor in [48, 1000] {
        assert!(writes(32, 48, floor).0.is_empty(), "floor {floor}");
    }
}

#[test]
fn the_floor_carries_across_the_dense_and_sparse_pieces() {
    // Rows [0, 32) end at the top-k boundary (dense), rows [32, 64) select.
    // One joint latent write, then each piece's index update.
    let (found, at) = writes(2016, 64, 0);
    let mut want = vec![latent_write(at, 64, 0)];
    want.extend(owner_index_writes(at, 0, 32, 0));
    want.extend(owner_index_writes(at, 32, 32, 0));
    assert_eq!(found, want);

    // Floor inside the first piece.
    let (found, at) = writes(2016, 64, 16);
    let mut want = vec![latent_write(at, 64, 16)];
    want.extend(owner_index_writes(at, 0, 32, 16));
    want.extend(owner_index_writes(at, 32, 32, 0));
    assert_eq!(found, want);

    // Floor past the first piece: it updates no index, the second skips 16.
    let (found, at) = writes(2016, 64, 48);
    let mut want = vec![latent_write(at, 64, 48)];
    want.extend(owner_index_writes(at, 32, 32, 16));
    assert_eq!(found, want);
}
