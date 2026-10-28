// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn chain_shard_view_assembles_the_history() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    for (case, &blocks) in [1usize, 5, 4099, 9001].iter().enumerate() {
        let live = Mutex::new(vec![]);
        let mut rng = Rng(0x7e57 + case as u64);
        let slots = blocks.div_ceil(WORLD) + 2;
        let logical: Vec<Vec<u8>> = (0..blocks)
            .map(|_| (0..BB).map(|_| rng.next() as u8).collect())
            .collect();
        let layout = ScratchLayout::for_world(WORLD, blocks, 0, BB);
        let mut per_rank = vec![];
        for r in 0..WORLD {
            let t = table(blocks, slots, &mut rng);
            let pool = alloc(gpu, &live, slots * BB, 0)?;
            let mut mine_slot = vec![];
            let mut mine_dst = vec![];
            for (l, &b) in t
                .iter()
                .enumerate()
                .filter(|(_, b)| **b as usize % WORLD == r)
            {
                gpu.copy_h2d(&logical[l], pool.offset(b as usize / WORLD * BB))?;
                mine_slot.push(b / WORLD as u32);
                mine_dst.push(l as u32);
            }
            let scratch = alloc(gpu, &live, layout.total, 0x5a)?;
            let view = scratch.offset(layout.view);
            let slot_dev = upload(gpu, &live, &u32s(&mine_slot))?;
            let dst_dev = upload(gpu, &live, &u32s(&mine_dst))?;
            ops::glm_kv_shard_copy_blocks(
                gpu,
                pool,
                Some(slot_dev),
                view,
                Some(dst_dev),
                mine_slot.len(),
                BB,
                0,
            )?;
            gpu.synchronize(0)?;
            per_rank.push((pool, scratch, slot_dev));
        }
        let (posts, meet) = (Mutex::new([(0, 0, 0); WORLD]), Barrier::new(WORLD));
        let views: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WORLD)
                .map(|r| {
                    let (per_rank, posts, meet, layout) = (&per_rank, &posts, &meet, &layout);
                    scope.spawn(move || -> Result<Vec<u8>> {
                        gpu.bind_to_thread()?;
                        let stream = gpu.create_stream()?;
                        let comm = MemChain {
                            gpu,
                            rank: r,
                            posts,
                            meet,
                        };
                        let (pool, scratch, mine_slot) = per_rank[r];
                        let s = LatentShard {
                            spec: LatentShardSpec {
                                rank: r,
                                world: WORLD,
                                scratch_bytes: layout.total,
                                view_blocks: blocks,
                                write_rows: 0,
                                lane: false,
                            },
                            local_blocks: slots,
                            scratch,
                            identity: DevicePtr::NULL,
                            lane: None,
                        };
                        let view = scratch.offset(layout.view);
                        let parts = ViewParts {
                            pool,
                            view,
                            mine_slot,
                            blocks,
                            block_bytes: BB,
                        };
                        parts.gather(gpu, &comm, &s, layout, stream)?;
                        gpu.synchronize(stream)?;
                        let mut out = vec![0u8; blocks * BB];
                        gpu.copy_d2h(view, &mut out)?;
                        Ok(out)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect::<Result<Vec<_>>>()
        })?;
        let want: Vec<u8> = logical.concat();
        for (r, v) in views.iter().enumerate() {
            ensure!(*v == want, "{blocks} blocks: rank {r}'s view differs");
        }
        println!("OK view of {blocks} blocks on all {WORLD} ranks");
        for p in live.into_inner().unwrap() {
            gpu.free(p)?;
        }
    }
    Ok(())
}
