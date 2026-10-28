// SPDX-License-Identifier: AGPL-3.0-only

//! NVFP4 row dequant against rows of the real checkpoint.
//!
//! Fixture: `bench/qwen4_exp/ple_nvfp4_fixtures.json`, from
//! `bench/qwen4_exp/ple_nvfp4_golden.py`, which range-reads nine rows (shards
//! 0, 1 and 127, three files) of nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6
//! off Hugging Face and dequantizes them with flashinfer's oracle math in torch.

use super::*;

pub(crate) struct GoldenRow {
    pub global_row: u64,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub bf16: Vec<u16>,
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// `(head_dim, rows_per_shard, global scale, rows)` from the fixture.
pub(crate) fn golden() -> (usize, u64, f32, Vec<GoldenRow>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/qwen4_exp/ple_nvfp4_fixtures.json"
    );
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path)
            .expect("run bench/qwen4_exp/ple_nvfp4_golden.py to generate the fixture"),
    )
    .expect("ple_nvfp4_fixtures.json");
    let rows = v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| GoldenRow {
            global_row: r["global_row"].as_u64().unwrap(),
            packed: hex(r["packed"].as_str().unwrap()),
            scales: hex(r["scales"].as_str().unwrap()),
            bf16: r["bf16_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_u64().unwrap() as u16)
                .collect(),
        })
        .collect();
    (
        v["head_dim"].as_u64().unwrap() as usize,
        v["rows_per_shard"].as_u64().unwrap(),
        f32::from_bits(v["weight_scale_2_bits"].as_u64().unwrap() as u32),
        rows,
    )
}

/// One packed row as the repack writes it.
pub(crate) fn pack_row(r: &GoldenRow, head_dim: usize) -> Vec<u8> {
    let mut row = r.packed.clone();
    row.extend_from_slice(&r.scales);
    row.resize(packed_stride(head_dim), 0);
    row
}

/// BF16 bits equal, with -0 == +0: code 8 is -0 in the kernel and in
/// `atlas_core::numeric::FP4_E2M1`, +0 in the oracle's table.
pub(crate) fn same_bf16(a: u16, b: u16) -> bool {
    a == b || (a & 0x7FFF == 0 && b & 0x7FFF == 0)
}

#[test]
fn packed_geometry_of_the_fp4_local_table() {
    assert_eq!(
        (code_bytes(160), scale_bytes(160), packed_stride(160)),
        (80, 10, 96)
    );
    assert!(check_packed_stride(96, 160, false).is_ok());
    assert!(check_packed_stride(90, 160, false).is_err(), "unpadded");
    assert!(check_packed_stride(160, 160, false).is_err(), "an FP8 row");
    assert!(
        check_packed_stride(96, 160, true).is_err(),
        "FP8 scale array too"
    );
    assert!(
        check_packed_stride(96, 150, false).is_err(),
        "not group-aligned"
    );
}

/// The CPU dequant of every fixture row equals the oracle's BF16, bit for bit
/// (up to the sign of zero). This is the number the GPU gather is held to.
#[test]
fn cpu_dequant_matches_the_checkpoint_golden() {
    let (head_dim, rows_per, global, rows) = golden();
    assert_eq!(head_dim, 160);
    assert_eq!(rows_per, 2_500_012);
    assert_eq!(rows.len(), 9);
    // weight_scale_2 is a real multiplier here (~3.3e-5): a dequant that
    // dropped or inverted it could not pass.
    assert!(global > 0.0 && global < 1e-3, "global {global}");
    let mut nonzero = 0;
    for r in &rows {
        assert_eq!(
            (r.packed.len(), r.scales.len(), r.bf16.len()),
            (80, 10, 160)
        );
        let got = dequant_packed_bf16(&pack_row(r, head_dim), head_dim, global);
        for (i, (&g, &w)) in got.iter().zip(&r.bf16).enumerate() {
            assert!(
                same_bf16(g, w),
                "row {} elem {i}: got {g:#06x} want {w:#06x}",
                r.global_row
            );
        }
        nonzero += got.iter().filter(|b| *b & 0x7FFF != 0).count();
    }
    // Eight live rows (~150 nonzero each) and one all-zero padding row.
    assert!(nonzero > 8 * 140, "only {nonzero} nonzero values");
}

/// `weight_scale_2` MULTIPLIES (ModelOpt: it is the amax-derived global, ~3.3e-5
/// here) — a divide would give the oracle's rows a 1e9x larger scale, and
/// dropping it a 3e4x one. Both must miss the fixture, and the real rows must
/// come out embedding-sized.
///
/// The scale ORDER, by contrast, is not observable in BF16 on these rows:
/// `e2m1 * (block * global)` produced the same bits for all 1440 values (each
/// order rounds once or twice in f32, and BF16 rounding swamps the
/// difference). The kernel keeps the reference's left-to-right order anyway.
#[test]
fn global_scale_multiplies() {
    let (head_dim, _, global, rows) = golden();
    let packed: Vec<u8> = rows.iter().flat_map(|r| pack_row(r, head_dim)).collect();
    let want: Vec<u16> = rows.iter().flat_map(|r| r.bf16.clone()).collect();
    for wrong in [1.0 / global, 1.0] {
        let got = dequant_packed_bf16(&packed, head_dim, wrong);
        let same = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| same_bf16(**g, **w))
            .count();
        // Only the zeros can agree.
        let zeros = want.iter().filter(|w| *w & 0x7FFF == 0).count();
        assert_eq!(
            same,
            zeros,
            "global {wrong}: {same} of {} agree",
            want.len()
        );
    }
    let vals: Vec<f32> = want
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
    let max = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("golden rows: rms {rms:.3e}, max |v| {max:.3e}");
    assert!(rms > 1e-4 && max < 1.0, "rms {rms}, max {max}");
}

/// The whole NVFP4 read path on a GPU: the golden rows, scattered among junk
/// rows in a synthetic two-plane table (three shards over two files, scale
/// plane apart from the code plane), packed by `ngram_pack`, faulted through
/// the O_DIRECT row cache (row 42 straddles a 4 KiB block: 42 x 96 = 4032),
/// and gathered by `batched_embed_nvfp4`. Every value must equal the oracle's
/// BF16 (golden rows) or the CPU dequant (junk rows), bit for bit.
///
/// GPU test: `#[ignore]` per repo convention. Run with
/// ```text
/// cargo test -p spark-model --release nvfp4_gather -- --ignored --nocapture
/// ```
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn nvfp4_gather_matches_the_golden_through_pack_and_cache() {
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_storage::ngram_pack::{PackSpec, Plane, Verify, open_or_pack};

    let (head_dim, _, global, golden_rows) = golden();
    let (cb, sb) = (code_bytes(head_dim), scale_bytes(head_dim));
    let rows_per = 50u64;
    // Golden row k lands at these global rows: shard starts, 42 (the block
    // straddle), shard ends, interior rows.
    let at: [u64; 9] = [0, 42, 49, 50, 77, 99, 100, 123, 149];
    let junk = |g: u64, i: usize| (g.wrapping_mul(0x9E37) ^ (i as u64 * 29)) as u8;
    let row_of = |g: u64| -> (Vec<u8>, Vec<u8>) {
        match at.iter().position(|&a| a == g) {
            Some(k) => (golden_rows[k].packed.clone(), golden_rows[k].scales.clone()),
            None => (
                (0..cb).map(|i| junk(g, i)).collect(),
                (0..sb).map(|i| junk(g, 100 + i)).collect(),
            ),
        }
    };
    let dir = std::env::temp_dir().join(format!("nvfp4_gather_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (fa, fb) = (dir.join("a.safetensors"), dir.join("b.safetensors"));
    let (mut a, mut b) = (vec![7u8; 1001], vec![9u8; 333]);
    let (mut codes, mut scales) = (Vec::new(), Vec::new());
    for shard in 0..3u64 {
        // Shard 1's codes live in file B, everything else in A; scales in B
        // except shard 2's. Offsets are odd on purpose.
        let (cbuf, cpath) = if shard == 1 {
            (&mut b, &fb)
        } else {
            (&mut a, &fa)
        };
        codes.push((cpath.clone(), cbuf.len() as u64));
        for r in 0..rows_per {
            cbuf.extend(row_of(shard * rows_per + r).0);
        }
        let (sbuf, spath) = if shard == 2 {
            (&mut a, &fa)
        } else {
            (&mut b, &fb)
        };
        scales.push((spath.clone(), sbuf.len() as u64));
        for r in 0..rows_per {
            sbuf.extend(row_of(shard * rows_per + r).1);
        }
    }
    std::fs::write(&fa, &a).unwrap();
    std::fs::write(&fb, &b).unwrap();
    let spec = PackSpec {
        planes: vec![
            Plane {
                shards: &codes,
                row_bytes: cb,
            },
            Plane {
                shards: &scales,
                row_bytes: sb,
            },
        ],
        rows_per_shard: rows_per,
        stride: packed_stride(head_dim),
        tag: "nvfp4g16",
    };
    let table = open_or_pack(&spec, &dir.join("packed"), Verify::Full).unwrap();

    // CUDA first: the cache arena is pinned, GPU-addressable memory.
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").unwrap();
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).unwrap();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g
        .kernel("embed_from_argmax", "batched_embed_nvfp4")
        .unwrap();
    let mut cache = table.open_row_cache(64).unwrap();
    cache.set_nvfp4_global_scale(global).unwrap();
    check_packed_stride(cache.row_stride(), head_dim, false).unwrap();

    // Every golden row, some junk rows, and repeats (cache hits), shuffled.
    let ids: Vec<u64> = vec![149, 0, 42, 13, 77, 42, 50, 99, 141, 100, 123, 49, 0, 88];
    let mut slots = Vec::new();
    cache.resolve(&ids, &mut slots).unwrap();
    let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
    let slots_dev = g.alloc(slot_bytes.len()).unwrap();
    g.copy_h2d(&slot_bytes, slots_dev).unwrap();
    let out = g.alloc(ids.len() * head_dim * 2).unwrap();
    crate::layers::ops::batched_embed_nvfp4(
        g,
        k,
        slots_dev,
        DevicePtr(cache.table_dev_va().unwrap()),
        out,
        global,
        ids.len() as u32,
        head_dim as u32,
        table.stride as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    cache.end_batch();
    let mut raw = vec![0u8; ids.len() * head_dim * 2];
    g.copy_d2h(out, &mut raw).unwrap();
    let got: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let (mut golden_checked, mut junk_checked) = (0, 0);
    for (n, &id) in ids.iter().enumerate() {
        let (c, s) = row_of(id);
        let packed = pack_row(
            &GoldenRow {
                global_row: id,
                packed: c,
                scales: s,
                bf16: vec![],
            },
            head_dim,
        );
        let want = match at.iter().position(|&a| a == id) {
            Some(k) => {
                golden_checked += 1;
                golden_rows[k].bf16.clone()
            }
            None => {
                junk_checked += 1;
                dequant_packed_bf16(&packed, head_dim, global)
            }
        };
        for i in 0..head_dim {
            let gv = got[n * head_dim + i];
            assert!(
                same_bf16(gv, want[i]),
                "id {id} elem {i}: got {gv:#06x} want {:#06x}",
                want[i]
            );
        }
    }
    let (hits, misses, _) = cache.stats();
    println!(
        "{} rows gathered ({golden_checked} golden, {junk_checked} junk), \
         cache {hits} hits / {misses} misses: all bit-exact",
        ids.len()
    );
    assert_eq!((golden_checked, junk_checked), (11, 3));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn decodes_rows_with_block_and_global_scales() {
    let mut rows = vec![0u8; 2 * packed_stride(32)];
    rows[..8].fill(0x22);
    rows[8..16].fill(0x77);
    rows[16..18].copy_from_slice(&[0x38, 0x40]);
    rows[packed_stride(32)..packed_stride(32) + 16].fill(0x99);
    rows[packed_stride(32) + 16..packed_stride(32) + 18].fill(0x40);
    let out = dequant_packed_bf16(&rows, 32, 0.25);
    let at = |i: usize| f32::from_bits(u32::from(out[i]) << 16);
    assert_eq!(out.len(), 64);
    assert!((0..16).all(|i| at(i) == 0.25));
    assert!((16..32).all(|i| at(i) == 3.0));
    assert!((32..64).all(|i| at(i) == -0.25));
}

#[test]
fn rejects_mismatched_rows_and_strides() {
    assert!(check_packed_stride(18, 32, true).is_err());
    assert!(check_packed_stride(16, 32, false).is_err());
    assert!(check_packed_stride(32, 32, false).is_ok());
    assert!(check_packed_stride(32, 30, false).is_err());
}
