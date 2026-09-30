// SPDX-License-Identifier: AGPL-3.0-only
//! The GLM paged prefill over a latent-sharded cache (`ATLAS_GLM_KV_SHARD=1`):
//! which pool, slots and block table each launch is handed, and what the pair
//! exchanges. Recorded launches, not CUDA numerics.
use super::*;
use crate::layers::glm_kv_shard::{self as shard, ScratchLayout};
use spark_runtime::kv_cache::LatentShard;
use std::sync::Mutex;

const MAP_SLOTS: u64 = 830;
const LOCALIZE: u64 = 831;
const COPY_BLOCKS: u64 = 832;
const LATENT_WRITE: u64 = 821;
const SPARSE_ATTN: u64 = 840;
const RANK: usize = 1;
const BLOCKS: usize = 512;

/// This rank of a pair: the byte count of every exchange, in order.
#[derive(Default)]
struct Pair(Mutex<Vec<usize>>);

impl spark_comm::CommBackend for Pair {
    fn rank(&self) -> usize {
        RANK
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, _: u64, _: u64, bytes: usize, add: bool, _: u64) -> Result<bool> {
        assert!(!add, "shard exchanges copy, never add");
        self.0.lock().unwrap().push(bytes);
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
}

/// What one sharded `prefill_attention_paged` call did.
struct Pass {
    /// Typed launches `(kernel, arguments)` in order.
    launches: Vec<(u64, Vec<Vec<u8>>)>,
    exchanges: Vec<usize>,
    shard: LatentShard,
    layout: ScratchLayout,
    /// This rank's latent pool, the sequence's slots and its block table.
    pool: DevicePtr,
    slot: DevicePtr,
    block_table: DevicePtr,
}

impl Pass {
    fn of(&self, kernel: u64) -> Vec<&Vec<Vec<u8>>> {
        let launches = self.launches.iter();
        launches.filter(|l| l.0 == kernel).map(|l| &l.1).collect()
    }
    fn scratch(&self, offset: usize) -> Vec<u8> {
        ptr(self.shard.scratch.offset(offset))
    }
}

fn ptr(p: DevicePtr) -> Vec<u8> {
    p.0.to_ne_bytes().to_vec()
}

fn word(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

/// `rows` rows from `seq_len_start` through the paged prefill of rank
/// [`RANK`] on a sharded `dtype` cache, the first `floor` rows unwritten.
fn pass(dtype: KvCacheDtype, seq_len_start: usize, rows: usize, floor: usize) -> Pass {
    let mut out = None;
    fixture_with(dtype, |gpu, config, layer| {
        layer.glm_sparse_attn_k = KernelHandle(SPARSE_ATTN);
        let layer = &*layer;
        let mut arena = BufferArena::new(config, 256, 32768, 16, 1, gpu).unwrap();
        if dtype == KvCacheDtype::Fp8G128 {
            arena.attach_glm_latent_scratch(4096, gpu).unwrap();
        }
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
            max_blocks_per_seq: BLOCKS as u32,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        // Logical block `l` is physical block `l`: every residue matches.
        let table: Vec<u8> = (0..BLOCKS as u32).flat_map(u32::to_le_bytes).collect();
        gpu.copy_h2d(&table, meta.block_table).unwrap();
        let pair = Pair::default();
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
            comm: Some(&pair),
            graph_capture: false,
            gdn_exact_replay: true,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 512,
            num_layers: 1,
            dtype,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let spec = shard::spec(RANK, &kv_config, BLOCKS * 16 - 64, 256);
        let mut cache = PagedKvCache::new_latent_sharded(kv_config, BLOCKS, gpu, spec).unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        let before = gpu.1.lock().unwrap().len();
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
        let shard = cache.latent_shard().unwrap();
        out = Some(Pass {
            launches: gpu.1.lock().unwrap()[before..].to_vec(),
            exchanges: pair.0.lock().unwrap().clone(),
            shard,
            layout: ScratchLayout::of(&shard, cache.config()).unwrap(),
            pool: cache.latent_pool_ptr(0),
            slot: meta.slot,
            block_table: meta.block_table,
        });
    });
    out.unwrap()
}

#[test]
fn a_floored_write_maps_only_the_written_rows_to_this_ranks_slots() {
    // 8 verify-sized rows past the top-k boundary, the first 4 replaying
    // cached positions: the slots of rows [4, 8) are localized and written
    // to this rank's pool, never through the unsharded pool accessors.
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8G128] {
        let p = pass(dtype, 4096, 8, 4);
        let local = p.scratch(p.layout.slots);
        let [map] = p.of(MAP_SLOTS)[..] else {
            panic!("{dtype:?}: one slot mapping");
        };
        let owner = [word(RANK as u32), word(2)];
        assert_eq!(
            map[..3],
            [ptr(p.slot.offset(4 * 8)), local.clone(), word(4)]
        );
        assert_eq!(map[4..], owner, "{dtype:?}");
        let sides = if dtype == KvCacheDtype::Bf16 { 2 } else { 1 };
        let [write] = p.of(LATENT_WRITE)[..] else {
            panic!("{dtype:?}: one latent write");
        };
        assert_eq!(write[sides], ptr(p.pool), "{dtype:?} pool");
        assert_eq!(write[2 * sides], local, "{dtype:?} slots");
    }
}

#[test]
fn few_rows_take_the_merge_form_over_this_ranks_tokens() {
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8G128] {
        let p = pass(dtype, 4096, 8, 0);
        // Queries, then the peer's heads' partial: the merge form's two swaps.
        assert_eq!(
            p.exchanges,
            [8 * 32 * 512 * 2, 8 * 32 * 513 * 4],
            "{dtype:?}"
        );
        let [localize] = p.of(LOCALIZE)[..] else {
            panic!("{dtype:?}: one localize");
        };
        // The selection is read through the sequence's table; the attention
        // reads the localized ids through the identity table over the pool.
        assert_eq!(localize[2], ptr(p.block_table), "{dtype:?}");
        assert!(p.of(COPY_BLOCKS).is_empty(), "{dtype:?}: no view");
        assert!(
            p.of(SPARSE_ATTN).is_empty(),
            "{dtype:?}: no unsharded reader"
        );
        let reads_pool =
            |a: &&Vec<Vec<u8>>| a.len() > 5 && a[1] == ptr(p.pool) && a[5] == ptr(p.shard.identity);
        let partials = p.launches.iter().map(|(_, a)| a).filter(reads_pool).count();
        assert_eq!(partials, 2, "{dtype:?}: peer heads, then own heads");
    }
}

#[test]
fn a_chunk_reads_its_history_assembled_from_both_ranks() {
    // 128 rows (a view-form owner) ending at token 4224 = 264 blocks, of
    // which this rank stores the 132 odd ones.
    let p = pass(KvCacheDtype::Bf16, 4096, 128, 0);
    let block = 16 * 512 * 2;
    assert_eq!(p.exchanges, [132 * block], "one round of the peer's blocks");
    let view = p.scratch(p.layout.view);
    let copies = p.of(COPY_BLOCKS);
    let blocks: Vec<_> = copies.iter().map(|a| a[4].clone()).collect();
    assert_eq!(blocks, [word(132), word(132), word(132)], "own, send, land");
    assert_eq!((&copies[0][0], &copies[0][2]), (&ptr(p.pool), &view));
    assert_eq!(copies[2][2], view, "the peer's blocks land in the view");
    // The unchanged sparse kernel reads the view through the identity table.
    let [attn] = p.of(SPARSE_ATTN)[..] else {
        panic!("one sparse attention launch");
    };
    assert_eq!(attn[1..3], [view.clone(), view]);
    assert_eq!(attn[5], ptr(p.shard.identity));
    assert!(p.of(LOCALIZE).is_empty(), "the view form localizes nothing");
}
