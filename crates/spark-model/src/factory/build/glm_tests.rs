// SPDX-License-Identifier: AGPL-3.0-only

//! The cross-rank KV block agreement over a communicator: what the ranks
//! gather, and that a rank sharding alone stops both at startup.

use super::*;
use crate::layers::glm_kv_shard as shard;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::Mutex;

/// One rank of a pair whose peer gathers the word `peer`; the byte count of
/// every gather.
struct Ranks<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    peer: u64,
    gathers: Mutex<Vec<usize>>,
}

impl spark_comm::CommBackend for Ranks<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        self.gathers.lock().unwrap().push(bytes);
        let mut ours = [0u8; 8];
        self.gpu.copy_d2h(DevicePtr(send), &mut ours)?;
        let (mine, theirs) = (8 * self.rank, 8 * (1 - self.rank));
        self.gpu.copy_h2d(&ours, DevicePtr(recv).offset(mine))?;
        self.gpu
            .copy_h2d(&self.peer.to_le_bytes(), DevicePtr(recv).offset(theirs))
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all-reduce")
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

fn kv_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 11,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

/// The GLM plan of `rank`, latent-sharded or not.
fn plan(rank: usize, sharded: bool) -> GlmCachePlan {
    let (shape, cfg) = (GlmMlaShape::new(512, 0).unwrap(), kv_config());
    let index = shape.bf16_index(4, 128).unwrap();
    let plan = GlmCachePlan::new(shape, &cfg, Some(index))
        .unwrap()
        .aliased_v(&cfg);
    if sharded {
        plan.latent_sharded(&cfg, shard::spec(rank, &cfg, 65_536, 8256))
    } else {
        plan
    }
}

fn agree(rank: usize, blocks: usize, sharded: bool, peer: u64) -> (Result<usize>, Vec<usize>) {
    let gpu = MockGpuBackend::new();
    let ranks = Ranks {
        gpu: &gpu,
        rank,
        peer,
        gathers: Mutex::default(),
    };
    let agreed = agree_kv_blocks(Some(&ranks), &gpu, blocks, Some(plan(rank, sharded)));
    assert_eq!(gpu.alloc_count(), 0, "the gather buffer is freed");
    (agreed, ranks.gathers.into_inner().unwrap())
}

#[test]
fn unsharded_ranks_gather_their_block_counts_and_take_the_smaller() {
    for rank in 0..2 {
        let (agreed, gathers) = agree(rank, 177_000, false, 150_000);
        assert_eq!((agreed.unwrap(), gathers), (150_000, vec![8]));
    }
}

#[test]
fn a_rank_sharding_alone_stops_both_ranks_at_the_gather() {
    // This test process sets no shard variable, so a sharded rank's word
    // carries the shard bit alone.
    let sharded = |blocks| shard::blocks_word(blocks, shard::settings_word(true).unwrap());
    for rank in 0..2 {
        let (agreed, gathers) = agree(rank, 300_000, true, sharded(280_000));
        assert_eq!((agreed.unwrap(), gathers), (280_000, vec![8]));
        // The sharded rank and the unsharded one see the same two words, and
        // each refuses them.
        for (ours, peer) in [(true, 280_000), (false, sharded(280_000))] {
            let (agreed, _) = agree(rank, 300_000, ours, peer);
            let err = agreed.unwrap_err().to_string();
            assert!(err.contains("ATLAS_GLM_KV_SHARD settings"), "{err}");
            assert!(err.contains(&format!("rank {rank} has")), "{err}");
        }
    }
}
